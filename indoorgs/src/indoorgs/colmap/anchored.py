"""동선 고정(route-anchored) 정합: 그린 동선을 강한 자세 정보로 써서 incremental SfM 없이 자세를 정함 (실험용).

흐름 (stages/register.py 의 register.mode: anchored):
  1) route_headings: 동선 위치(priors.json)의 진행 방향 -> 사진별 요(yaw) 초기값. 멈춤(속도 ~0)·급회전은 표시(flag)
  2) select_pairs: 동선 순서 앞뒤 seq_window 장 + 동선 거리 max_dist_m 안이고 요 차이 max_yaw_deg 안인 짝만
     -> pairs.txt -> COLMAP matches_importer (기하 검증은 COLMAP 그대로)
  3) 자세 그래프 (점 없음): 검증된 짝의 inlier 로 상대 자세 (essential / 순수 회전은 방향 벡터 Procrustes)
     -> rotation_averaging (상대 회전 + 롤 0(짐벌) + 약한 동선 요) -> translation_averaging (이동 방향 + 동선 위치)
  4) write_initial_model (점 없는 cameras/images) -> point_triangulator (자세 고정) -> 필요하면 BA ->
     pose_prior_mapper --input_path (나머지 사진 PnP 등록 + 위치 제약 전역 BA)
초점거리: 영상 프레임에 EXIF 가 없어 feature_extractor 의 초기값(1.2 x 긴 변)이 틀림 -> estimate_focal 로
F 행렬들에서 추정 (K^T F K 의 두 특이값이 같아지는 f, 1차원 탐색). config 로 직접 줄 수도 있음.

numpy + OpenCV + sqlite. 큰 희소 최소제곱은 scipy.sparse 가 있으면 씀 (없으면 numpy 밀집, 사진 수가 많으면 느림).
"""
from __future__ import annotations

import math
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import model as cmodel
from .pair_verify import MAX_ID, pair_id, rot2qvec


# ---------------------------------------------------------------- 1) 동선 -> 초기 자세
def _wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def route_headings(P: np.ndarray, t: np.ndarray, window_s: float = 1.0, stop_speed: float = 0.15,
                   turn_deg_s: float = 45.0, dilate_s: float = 0.5) -> dict:
    """동선 위치 P (N,3), 시간 t (N,) -> {"yaw": (N,) rad, "speed": (N,) m/s, "flag": (N,) bool, "reason": [str]}.
    진행 방향 = t±window_s 사이 위치 차. 속도 stop_speed 미만(멈춤/제자리 회전) 또는 요 변화가 turn_deg_s 넘는 곳은 flag
    (앞뒤 dilate_s 초까지 넓힘). 멈춘 곳의 요는 앞뒤 움직인 곳에서 이어받음."""
    P, t = np.asarray(P, float), np.asarray(t, float)
    n = len(P)
    if n == 0:
        return {"yaw": np.zeros(0), "speed": np.zeros(0), "flag": np.zeros(0, bool), "reason": []}
    lo = np.searchsorted(t, t - window_s, "left")
    hi = np.clip(np.searchsorted(t, t + window_s, "right") - 1, 0, n - 1)
    d = P[hi, :2] - P[lo, :2]
    dt = np.maximum(t[hi] - t[lo], 1e-6)
    speed = np.linalg.norm(d, axis=1) / dt
    yaw = np.arctan2(d[:, 1], d[:, 0])
    moving = speed >= stop_speed
    if moving.any():
        idx = np.flatnonzero(moving)
        for i in np.flatnonzero(~moving):                 # 가장 가까운 움직인 사진의 요
            yaw[i] = yaw[idx[np.argmin(np.abs(idx - i))]]
    rate = np.zeros(n)
    if n > 1:
        dy = _wrap(np.diff(yaw)) / np.maximum(np.diff(t), 1e-6)
        rate[1:] = np.abs(dy)
        rate[:-1] = np.maximum(rate[:-1], np.abs(dy))
    turn = rate > math.radians(turn_deg_s)
    raw = ~moving | turn
    flag = raw.copy()
    for i in np.flatnonzero(raw):
        flag |= np.abs(t - t[i]) <= dilate_s
    reason = ["stop" if not moving[i] else "turn" if turn[i] else ("near" if flag[i] else "") for i in range(n)]
    return {"yaw": yaw, "speed": speed, "flag": flag, "reason": reason}


def yaw_pitch_to_R(yaw: float, pitch: float = 0.0) -> np.ndarray:
    """요·피치(롤 0) -> R world_from_cam (COLMAP 카메라: x 오른쪽, y 아래, z 앞). 월드 z 가 위."""
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    z = np.array([cy * cp, sy * cp, sp])
    x = np.array([sy, -cy, 0.0])
    y = np.cross(z, x)
    return np.stack([x, y, z], 1)


def R_yaw(R_wc: np.ndarray) -> float:
    f = R_wc[:, 2]
    if math.hypot(f[0], f[1]) < 1e-6:                       # 똑바로 위/아래: 오른쪽 축으로
        x = R_wc[:, 0]
        return math.atan2(-x[0], x[1])
    return math.atan2(f[1], f[0])


# ---------------------------------------------------------------- 2) 짝 고르기
def select_pairs(P: np.ndarray, yaw: np.ndarray, seq_window: int = 10, max_dist_m: float = 4.0,
                 max_yaw_deg: float = 60.0, max_per_image: int = 30) -> list[tuple[int, int]]:
    """사진 순서(=시간) 번호 쌍 (i<j). 앞뒤 seq_window 장은 모두 + 그 밖은 수평 거리 max_dist_m 안,
    요 차이 max_yaw_deg 안인 짝을 사진마다 가까운 순 max_per_image 개까지 (양쪽 중 한쪽이라도 고르면 넣음)."""
    P = np.asarray(P, float)
    n = len(P)
    out = set()
    for i in range(n):
        for j in range(i + 1, min(n, i + seq_window + 1)):
            out.add((i, j))
    if n < 2 or max_per_image <= 0 or max_dist_m <= 0:
        return sorted(out)
    yaw = np.asarray(yaw, float)
    ct = math.cos(math.radians(max_yaw_deg))
    for i in range(n):
        d = np.linalg.norm(P[:, :2] - P[i, :2], axis=1)
        ok = (d <= max_dist_m) & (np.cos(yaw - yaw[i]) >= ct) & (np.abs(np.arange(n) - i) > seq_window)
        cand = np.flatnonzero(ok)
        for j in cand[np.argsort(d[cand], kind="stable")][:max_per_image]:
            out.add((min(i, j), max(i, j)))
    return sorted(out)


def write_pairs_file(path: Path, names: list[str], pairs: list[tuple[int, int]]) -> int:
    """COLMAP matches_importer --match_type pairs 형식: 줄마다 '이름1 이름2'."""
    Path(path).write_text("".join(f"{names[i]} {names[j]}\n" for i, j in pairs), encoding="utf-8")
    return len(pairs)


# ---------------------------------------------------------------- DB 읽기
@dataclass
class DbData:
    names: dict[int, str]                       # image_id -> 이름
    camera: tuple                               # (camera_id, model, width, height, params)
    image_camera: dict[int, int]
    keypoints: dict[int, np.ndarray]
    geoms: dict[tuple[int, int], dict] = field(default_factory=dict)    # (id1<id2) -> {"m": (k,2), "config", "F"}


def read_db(db: Path, min_inliers: int = 15) -> DbData:
    with closing(sqlite3.connect(str(db))) as c:
        names = dict(c.execute("SELECT image_id, name FROM images"))
        imcam = dict(c.execute("SELECT image_id, camera_id FROM images"))
        cams = c.execute("SELECT camera_id, model, width, height, params FROM cameras").fetchall()
        kp = {i: np.frombuffer(b, np.float32).reshape(r, cc)[:, :2].astype(np.float64)
              for i, r, cc, b in c.execute("SELECT image_id, rows, cols, data FROM keypoints") if r}
        rows = c.execute("SELECT pair_id, rows, data, config, F FROM two_view_geometries WHERE rows >= ?",
                         (min_inliers,)).fetchall()
    cid, model, w, h, pb = cams[0]
    cam = (cid, model, w, h, np.frombuffer(pb, np.float64).copy())
    geoms = {}
    for pid, r, blob, cfg, F in rows:
        i2 = pid % MAX_ID
        i1 = (pid - i2) // MAX_ID
        geoms[(i1, i2)] = {"m": np.frombuffer(blob, np.uint32).reshape(r, 2), "config": cfg,
                           "F": None if F is None or len(F) != 72 else np.frombuffer(F, np.float64).reshape(3, 3)}
    return DbData(names, cam, imcam, kp, geoms)


# ---------------------------------------------------------------- 초점거리
def estimate_focal(Fs: list[np.ndarray], width: int, height: int, lo: float = 0.3, hi: float = 3.0,
                   steps: int = 200) -> tuple[float | None, dict]:
    """F 행렬들 -> 초점거리(px). 주점 = 화면 가운데. f 마다 E = K^T F K 의 특이값 s1>=s2 로 (s1-s2)/(s1+s2) 의
    중앙값을 구해 가장 작은 f (로그 격자, lo..hi x 긴 변) + 주변 3점 포물선 보정."""
    Fs = [F for F in Fs if F is not None and np.all(np.isfinite(F))]
    if len(Fs) < 5:
        return None, {"n": len(Fs)}
    L = max(width, height)
    fs = L * np.exp(np.linspace(math.log(lo), math.log(hi), steps))
    cx, cy = width / 2.0, height / 2.0
    Fa = np.stack(Fs)
    cost = []
    for f in fs:
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        E = np.einsum("ji,njk,kl->nil", K, Fa, K)
        s = np.linalg.svd(E, compute_uv=False)
        cost.append(float(np.median((s[:, 0] - s[:, 1]) / np.maximum(s[:, 0] + s[:, 1], 1e-12))))
    cost = np.array(cost)
    k = int(np.argmin(cost))
    f = fs[k]
    if 0 < k < steps - 1:
        a, b, c = cost[k - 1], cost[k], cost[k + 1]
        den = a - 2 * b + c
        if den > 0:
            u = 0.5 * (a - c) / den
            f = float(math.exp(math.log(fs[k]) + u * (math.log(fs[1]) - math.log(fs[0]))))
    edge = k in (0, steps - 1)
    return (None if edge else float(f)), {"n": len(Fs), "cost": float(cost[k]), "at_edge": edge, "f_grid": float(fs[k])}


# ---------------------------------------------------------------- 3) 상대 자세
@dataclass
class RelPose:
    i: int                  # 순서 번호 (collect_relposes 는 i < j 로 맞춤)
    j: int
    R: np.ndarray           # cam_j_from_cam_i
    d: np.ndarray | None    # cam_i 좌표의 cam_j 중심 방향 (단위). 시차가 작으면 None
    n: int                  # inlier 수
    parallax_deg: float
    kp: tuple | None = None  # (i 쪽 특징점 번호, j 쪽 특징점 번호) inlier


def _bearings(pts: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    n = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    b = np.hstack([n, np.ones((len(n), 1))])
    return b / np.linalg.norm(b, axis=1, keepdims=True)


def _procrustes(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """b2 ~ R b1 인 회전 (순수 회전일 때 정확)."""
    U, _, Vt = np.linalg.svd(b2.T @ b1)
    D = np.diag([1, 1, np.linalg.det(U @ Vt)])
    return U @ D @ Vt


def relative_pose(p1: np.ndarray, p2: np.ndarray, K: np.ndarray, dist: np.ndarray,
                  min_parallax_deg: float = 1.0) -> tuple[np.ndarray, np.ndarray | None, float] | None:
    """inlier 매칭 -> (R cam2_from_cam1, cam1 좌표의 cam2 중심 방향 또는 None, 시차 중앙값°). 실패하면 None.
    시차(회전을 빼고 남은 방향 차이)가 min_parallax_deg 미만이면 이동 방향을 안 믿고 회전은 Procrustes 로."""
    if len(p1) < 8:
        return None
    b1, b2 = _bearings(p1, K, dist), _bearings(p2, K, dist)
    Rr = _procrustes(b1, b2)
    par_rot = float(np.degrees(np.median(np.arccos(np.clip(np.sum((b1 @ Rr.T) * b2, 1), -1, 1)))))
    if par_rot < min_parallax_deg:
        return Rr, None, par_rot
    n1, n2 = b1[:, :2] / b1[:, 2:], b2[:, :2] / b2[:, 2:]
    try:
        E, mask = cv2.findEssentialMat(n1, n2, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=1.5 / K[0, 0])
        if E is None or E.shape != (3, 3):
            return Rr, None, par_rot
        _, R, t, mask2 = cv2.recoverPose(E, n1, n2, np.eye(3), mask=mask)
    except cv2.error:
        return Rr, None, par_rot
    if mask2 is None or int((mask2.ravel() > 0).sum()) < max(8, len(p1) // 4):
        return Rr, None, par_rot
    return R, (-R.T @ t).ravel() / max(1e-12, float(np.linalg.norm(t))), par_rot


def collect_relposes(dbd: DbData, order: list[str], K: np.ndarray, dist: np.ndarray, seq_graph: int = 5,
                     extra_per_image: int = 5, min_inliers: int = 30, min_parallax_deg: float = 1.0) -> list[RelPose]:
    """자세 그래프의 변: 순서 차 seq_graph 이하 짝 모두 + 그 밖은 사진마다 inlier 많은 순 extra_per_image 개."""
    idx = {n: k for k, n in enumerate(order)}
    id2k = {iid: idx[nm] for iid, nm in dbd.names.items() if nm in idx}
    seq, other = [], {}
    for (a, b), g in dbd.geoms.items():
        if a not in id2k or b not in id2k or len(g["m"]) < min_inliers or g["config"] in (7,):   # 7 = WATERMARK
            continue
        ka, kb = id2k[a], id2k[b]
        if abs(ka - kb) <= seq_graph:
            seq.append((a, b))
        else:
            other.setdefault(ka, []).append((len(g["m"]), a, b))
            other.setdefault(kb, []).append((len(g["m"]), a, b))
    chosen = set(seq)
    for lst in other.values():
        for _, a, b in sorted(lst, reverse=True)[:extra_per_image]:
            chosen.add((a, b))
    out = []
    for a, b in sorted(chosen):
        if a not in dbd.keypoints or b not in dbd.keypoints:
            continue
        m = dbd.geoms[(a, b)]["m"]
        r = relative_pose(dbd.keypoints[a][m[:, 0]], dbd.keypoints[b][m[:, 1]], K, dist, min_parallax_deg)
        if r is None:
            continue
        ka, kb, Rr, d = id2k[a], id2k[b], r[0], r[1]
        ia, ib = m[:, 0], m[:, 1]
        if ka > kb:                                     # 항상 i < j (시간 순)로: R -> R^T, 방향 -> -(R d)
            ka, kb, Rr, d, ia, ib = kb, ka, Rr.T, (None if d is None else -(Rr @ d)), ib, ia
        out.append(RelPose(ka, kb, Rr, d, len(m), r[2], (ia, ib)))
    return out


def _depths(R: np.ndarray, d: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """기선 1 짝 (cam2_from_cam1 = R, cam1 좌표 cam2 중심 = d) 에서 점들의 (cam1 거리, cam2 거리) (k,2). 중점 삼각측량."""
    r2 = b2 @ R                                          # cam2 광선을 cam1 좌표로 (R^T b2)
    a11 = np.sum(b1 * b1, 1)
    a12 = -np.sum(b1 * r2, 1)
    a22 = np.sum(r2 * r2, 1)
    c1, c2 = b1 @ d, -(r2 @ d)
    det = a11 * a22 - a12 * a12
    det = np.where(np.abs(det) < 1e-12, np.nan, det)
    l1 = (c1 * a22 - a12 * c2) / det
    l2 = (a11 * c2 - a12 * c1) / det
    return np.stack([l1, l2], 1)


def scale_ratios(dbd: DbData, rels: list[RelPose], order: list[str], K: np.ndarray, dist: np.ndarray,
                 min_common: int = 15, max_span: int = 12) -> list[tuple[int, int, float, int]]:
    """두 변 (i,j), (j,k) (i<j<k, 둘 다 이동 방향 있음) 이 사진 j 의 같은 특징점을 보면 그 점의 j 거리로 기선 비
    s_jk / s_ij = z_ij / z_jk (중앙값). 돌려줌 [(변 a 번호, 변 b 번호, 비, 공통 점 수)]. 변 번호 = rels 안 위치."""
    byname = {nm: iid for iid, nm in dbd.names.items()}
    idx_of = {k: byname[nm] for k, nm in enumerate(order) if nm in byname}
    into: dict[int, list] = {}
    outof: dict[int, list] = {}
    for e, r in enumerate(rels):
        if r.d is None or getattr(r, "kp", None) is None or r.j - r.i > max_span:
            continue
        into.setdefault(r.j, []).append(e)
        outof.setdefault(r.i, []).append(e)
    bear = {}

    def B(k):
        if k not in bear:
            bear[k] = _bearings(dbd.keypoints[idx_of[k]], K, dist)
        return bear[k]
    out = []
    for j in sorted(set(into) & set(outof)):
        for ea in into[j]:
            ra = rels[ea]
            za = _depths(ra.R, ra.d, B(ra.i)[ra.kp[0]], B(j)[ra.kp[1]])[:, 1]          # j 쪽 거리
            ma = dict(zip(ra.kp[1].tolist(), za.tolist()))
            for eb in outof[j]:
                rb = rels[eb]
                zb = _depths(rb.R, rb.d, B(j)[rb.kp[0]], B(rb.j)[rb.kp[1]])[:, 0]
                common = [(ma[q], z) for q, z in zip(rb.kp[0].tolist(), zb.tolist()) if q in ma]
                if len(common) < min_common:
                    continue
                c = np.array(common)
                ok = np.isfinite(c).all(1) & (c[:, 0] > 0) & (c[:, 1] > 0)
                if ok.sum() < min_common:
                    continue
                out.append((ea, eb, float(np.exp(np.median(np.log(c[ok, 0] / c[ok, 1])))), int(ok.sum())))
    return out


# ---------------------------------------------------------------- 회전 평균
def _log(R: np.ndarray) -> np.ndarray:
    v, _ = cv2.Rodrigues(np.ascontiguousarray(R))
    return v.ravel()


def _exp(w: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(w, float).reshape(3, 1))
    return R


def _solve_ls(rows: list, cols: list, vals: list, rhs: list, n: int) -> np.ndarray:
    """희소 최소제곱 (scipy 있으면 lsqr 대신 정규방정식 spsolve, 없으면 numpy 밀집)."""
    m = len(rhs)
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import spsolve
        A = csr_matrix((vals, (rows, cols)), shape=(m, n))
        AtA = (A.T @ A).tocsc()
        AtA = AtA + 1e-9 * _speye(n)
        return np.asarray(spsolve(AtA, A.T @ np.asarray(rhs, float))).ravel()
    except ImportError:
        A = np.zeros((m, n))
        np.add.at(A, (rows, cols), vals)
        return np.linalg.lstsq(A, np.asarray(rhs, float), rcond=None)[0]


def _speye(n):
    from scipy.sparse import identity
    return identity(n, format="csc")


def chain_rotations(n: int, rels: list[RelPose], R0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """가장 inlier 많은 변부터 넓혀 가는 신장 트리(최대 가중)로 회전 초기값. 돌려줌 (R (n,3,3), 연결됨 (n,) bool)."""
    R = np.tile(np.eye(3), (n, 1, 1))
    done = np.zeros(n, bool)
    adj: dict[int, list] = {}
    for r in rels:
        adj.setdefault(r.i, []).append((r.n, r.i, r.j, r.R))
        adj.setdefault(r.j, []).append((r.n, r.j, r.i, r.R.T))
    import heapq
    for start in range(n):
        if done[start] or start not in adj:
            continue
        R[start] = R0[start]
        done[start] = True
        heap = [(-w, a, b, Rab) for (w, a, b, Rab) in adj[start]]
        heapq.heapify(heap)
        while heap:
            _, a, b, Rba = heapq.heappop(heap)          # Rba = cam_b_from_cam_a
            if done[b]:
                continue
            R[b] = R[a] @ Rba.T                          # world_from_b = world_from_a @ a_from_b
            done[b] = True
            for (w, x, y, Rxy) in adj.get(b, []):
                if not done[y]:
                    heapq.heappush(heap, (-w, x, y, Rxy))
    return R, done


def rotation_averaging(n: int, rels: list[RelPose], yaw_prior: np.ndarray, yaw_w: np.ndarray,
                       R_init: np.ndarray, sigma_rel_deg: float = 1.0, sigma_roll_deg: float = 2.0,
                       sigma_pitch_deg: float = 60.0, huber_deg: float = 2.0, iters: int = 15) -> tuple[np.ndarray, dict]:
    """R world_from_cam (n,3,3) 를 상대 회전 + 롤 0(짐벌: 카메라 오른쪽 축이 수평) + 약한 동선 요(yaw_w 가중, 0 = 안 씀)
    + 아주 약한 피치 0 으로 맞춤 (Gauss-Newton, 월드 쪽 섭동 R <- exp(w) R, Huber IRLS).
    상대 회전 R_ij = cam_j_from_cam_i: (w_i - w_j) = -R_i log(R_ij^T R_j^T R_i)."""
    R = R_init.copy()
    sr, sroll, sp = math.radians(sigma_rel_deg), math.radians(sigma_roll_deg), math.radians(sigma_pitch_deg)
    hub = math.radians(huber_deg)
    res = np.zeros(len(rels))
    for it in range(iters):
        rows, cols, vals, rhs = [], [], [], []
        m = 0
        for k, r in enumerate(rels):
            e = _log(r.R.T @ R[r.j].T @ R[r.i])
            res[k] = np.linalg.norm(e)
            w = math.sqrt(min(1.0, hub / max(res[k], 1e-12))) / sr     # Huber 가중 (sqrt)
            g = -R[r.i] @ e
            for a in range(3):
                rows += [m, m]
                cols += [3 * r.i + a, 3 * r.j + a]
                vals += [w, -w]
                rhs.append(w * g[a])
                m += 1
        for i in range(n):
            x, f = R[i][:, 0], R[i][:, 2]
            # 롤: x_z + (w x x)_z = 0 -> w0 * x1 - w1 * x0 = -x_z
            ws = 1.0 / sroll
            rows += [m, m]; cols += [3 * i, 3 * i + 1]; vals += [ws * x[1], -ws * x[0]]; rhs.append(-ws * x[2]); m += 1
            ws = 1.0 / sp
            rows += [m, m]; cols += [3 * i, 3 * i + 1]; vals += [ws * f[1], -ws * f[0]]; rhs.append(-ws * f[2]); m += 1
            if yaw_w[i] > 0:
                ws = math.sqrt(yaw_w[i])
                rows.append(m); cols.append(3 * i + 2); vals.append(ws)
                rhs.append(ws * float(_wrap(yaw_prior[i] - R_yaw(R[i])))); m += 1
        dw = _solve_ls(rows, cols, vals, rhs, 3 * n).reshape(n, 3)
        for i in range(n):
            R[i] = _exp(dw[i]) @ R[i]
        if np.max(np.linalg.norm(dw, axis=1)) < 1e-6:
            break
    for k, r in enumerate(rels):
        res[k] = np.linalg.norm(_log(r.R.T @ R[r.j].T @ R[r.i]))
    deg = np.degrees(res)
    return R, {"iters": it + 1, "rel_res_median_deg": float(np.median(deg)) if len(deg) else None,
               "rel_res_p90_deg": float(np.percentile(deg, 90)) if len(deg) else None,
               "rel_outliers_5deg": int((deg > 5).sum())}


def translation_averaging(R: np.ndarray, rels: list[RelPose], P: np.ndarray, sigma: np.ndarray,
                          sigma_dir_deg: float = 3.0, min_base_m: float = 0.05, iters: int = 5,
                          huber: float = 2.0, smooth_sigma_m: float = 0.0, ratios: list | None = None,
                          sigma_ratio: float = 0.15) -> tuple[np.ndarray, dict]:
    """카메라 중심 C (n,3): 상대 이동 방향 d_ij (월드 = R_i d) 과 동선 위치 P (sigma (n,3)) 로.
    변마다 미지 길이 s_ij: C_j - C_i - s_ij d_ij = 0 (가중 1/(sigma_dir x 동선 기선)), 위치 (C_i - P_i)/sigma.
    길이 s 가 음수로 나온 변(방향이 뒤집힘)은 다음 반복에서 뺌. 잔차 Huber IRLS."""
    n = len(P)
    eidx = [k for k, r in enumerate(rels) if r.d is not None]
    pos = {k: q for q, k in enumerate(eidx)}
    E = [rels[k] for k in eidx]
    rat = [(pos[a], pos[b], rho, c) for a, b, rho, c in (ratios or []) if a in pos and b in pos]
    sd = math.radians(sigma_dir_deg)
    wts = np.ones(len(E))
    keep = np.ones(len(E), bool)
    C = P.copy()
    for it in range(iters):
        rows, cols, vals, rhs = [], [], [], []
        m = 0
        nv = 3 * n + len(E)
        for k, r in enumerate(E):
            if not keep[k]:
                continue
            b = max(min_base_m, float(np.linalg.norm(P[r.j] - P[r.i])))
            w = wts[k] / (sd * b)
            dw = R[r.i] @ r.d
            for a in range(3):
                rows += [m, m, m]
                cols += [3 * r.j + a, 3 * r.i + a, 3 * n + k]
                vals += [w, -w, -w * dw[a]]
                rhs.append(0.0)
                m += 1
        for qa, qb, rho, c in rat:                   # s_b - rho s_a = 0
            if not (keep[qa] and keep[qb]):
                continue
            ba = max(min_base_m, float(np.linalg.norm(P[E[qa].j] - P[E[qa].i])))
            w = 1.0 / (sigma_ratio * ba * max(rho, 0.1))
            rows += [m, m]; cols += [3 * n + qb, 3 * n + qa]; vals += [w, -w * rho]; rhs.append(0.0); m += 1
        for i in range(n):
            for a in range(3):
                w = 1.0 / sigma[i, a]
                rows.append(m); cols.append(3 * i + a); vals.append(w); rhs.append(w * P[i, a]); m += 1
        if smooth_sigma_m > 0:
            for i in range(1, n - 1):
                for a in range(3):
                    w = 1.0 / smooth_sigma_m
                    rows += [m, m, m]; cols += [3 * (i - 1) + a, 3 * i + a, 3 * (i + 1) + a]
                    vals += [w, -2 * w, w]; rhs.append(0.0); m += 1
        x = _solve_ls(rows, cols, vals, rhs, nv)
        C = x[:3 * n].reshape(n, 3)
        s = x[3 * n:]
        # 잔차 (방향 각도 오차)
        ang = np.zeros(len(E))
        for k, r in enumerate(E):
            v = C[r.j] - C[r.i]
            nv_ = np.linalg.norm(v)
            ang[k] = math.degrees(math.acos(max(-1.0, min(1.0, float(v @ (R[r.i] @ r.d)) / max(nv_, 1e-9)))))
        keep = keep & (s > 0)
        wts = np.minimum(1.0, huber * sigma_dir_deg / np.maximum(ang, 1e-9)) ** 0.5
    dp = np.linalg.norm(C - P, axis=1)
    return C, {"edges": len(E), "ratios": len(rat), "edges_used": int(keep.sum()), "dir_res_median_deg": float(np.median(ang)) if len(E) else None,
               "dir_res_p90_deg": float(np.percentile(ang, 90)) if len(E) else None,
               "moved_from_prior_median_m": float(np.median(dp)), "moved_from_prior_max_m": float(dp.max())}


# ---------------------------------------------------------------- 4) 초기 모델 쓰기
def initial_model(dbd: DbData, order: list[str], R_wc: np.ndarray, C: np.ndarray, use: np.ndarray,
                  focal: float | None) -> cmodel.Model:
    """점 없는 COLMAP 모델 (cameras + images, DB 의 image_id·camera_id 그대로). focal 을 주면 카메라 초점거리·주점(가운데)을
    바꾸고 왜곡은 0 (point_triangulator 가 이 카메라를 씀). frames.bin/rigs.bin 은 안 씀 (4.2.1 은 없으면 사진마다 frame 을 만듦:
    pycolmap 4.2.1 Reconstruction 읽기로 확인)."""
    cid, mdl, w, h, params = dbd.camera
    params = np.array(params, float)
    if focal:
        name = cmodel.CAM_NAMES.get(mdl)
        if name in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            params[0], params[1], params[2] = focal, w / 2.0, h / 2.0
            params[3:] = 0.0
        elif name in ("PINHOLE", "OPENCV", "FULL_OPENCV"):
            params[0] = params[1] = focal
            params[2], params[3] = w / 2.0, h / 2.0
            params[4:] = 0.0
        else:
            raise ValueError(f"카메라 모델 {name} 은 초점거리 바꾸기 미지원")
    cams = {cid: cmodel.Camera(cid, mdl, int(w), int(h), params)}
    byname = {nm: iid for iid, nm in dbd.names.items()}
    images = {}
    for k, nm in enumerate(order):
        if not use[k] or nm not in byname:
            continue
        iid = byname[nm]
        Rcw = R_wc[k].T
        images[iid] = cmodel.Image(iid, rot2qvec(Rcw), -Rcw @ C[k], dbd.image_camera.get(iid, cid), nm)
    return cmodel.Model(cams, images, {})


def camera_K(dbd: DbData, focal: float | None) -> tuple[np.ndarray, np.ndarray]:
    from .pair_verify import camera_to_cv
    cid, mdl, w, h, params = dbd.camera
    if focal:
        return np.array([[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1.0]]), np.zeros(4)
    return camera_to_cv(cmodel.Camera(cid, mdl, w, h, params))


# ---------------------------------------------------------------- 점 줄이기 (BA 크기 줄이기 실험)
def thin_points(m: cmodel.Model, max_per_image: int = 0, min_track: int = 2) -> tuple[cmodel.Model, dict]:
    """트랙 길이 min_track 미만 점을 빼고, max_per_image > 0 이면 사진마다 트랙 긴 순으로 그 개수 안에 드는 점만 남김
    (한 사진이라도 고르면 남김). 빠진 점의 관측은 사진에서 -1."""
    pts = {pid: p for pid, p in m.points.items() if len(p.track) >= min_track}
    if max_per_image > 0:
        tl = {pid: len(p.track) for pid, p in pts.items()}
        keep = set()
        for im in m.images.values():
            ids = [int(x) for x in im.pid if x >= 0 and int(x) in tl]
            keep.update(sorted(ids, key=lambda x: -tl[x])[:max_per_image])
        pts = {pid: p for pid, p in pts.items() if pid in keep}
    ids = np.array(sorted(pts), np.int64)
    images = {}
    for iid, im in m.images.items():
        pid = im.pid.copy()
        if len(pid):
            pid[~np.isin(pid, ids)] = -1
        images[iid] = cmodel.Image(im.id, im.q, im.t, im.camera_id, im.name, im.xy, pid)
    return cmodel.Model(m.cameras, images, pts, m.frames, m.rigs_bytes), {"points_before": len(m.points),
                                                                           "points_after": len(pts)}


def rel_residuals_deg(R: np.ndarray, rels: list[RelPose]) -> np.ndarray:
    return np.array([math.degrees(np.linalg.norm(_log(r.R.T @ R[r.j].T @ R[r.i]))) for r in rels])


def solve_rotations(n: int, rels: list[RelPose], yaw_route: np.ndarray, flag: np.ndarray, seq_graph: int = 5,
                    sigma_yaw_deg: float = 30.0, prune_deg: tuple = (15.0, 5.0), **kw) -> tuple[np.ndarray, np.ndarray, list, dict]:
    """회전 정하기 전체: 앞뒤 짝만으로 신장 트리 초기값 -> 동선 요에 전체 요 맞춤 -> 앞뒤 아닌 짝은 잔차가 prune_deg 넘으면 빼며
    rotation_averaging 반복. 돌려줌 (R world_from_cam, 연결됨, 쓴 변, 통계).
    동선 요는 flag(멈춤·급회전) 아닌 사진에만 sigma_yaw_deg 로 약하게 (영상이 진행 방향에서 흔들리는 만큼 크게)."""
    R0 = np.array([yaw_pitch_to_R(y) for y in yaw_route])
    seq = [r for r in rels if abs(r.i - r.j) <= seq_graph]
    Rc, conn = chain_rotations(n, seq, R0)
    if conn.any():                                       # 연결 조각마다 동선 요에 맞춤 (원형 평균)
        lab = _components(n, seq)
        for c in set(lab[conn]):
            ids = np.flatnonzero((lab == c) & conn)
            w = np.where(flag[ids], 0.1, 1.0)
            off = float(np.angle(np.sum(w * np.exp(1j * _wrap(yaw_route[ids] - np.array([R_yaw(Rc[i]) for i in ids]))))))
            Q = _exp([0, 0, off])
            Rc[ids] = np.einsum("ij,njk->nik", Q, Rc[ids])
    yw = np.where(flag | ~conn, 0.0, 1.0 / math.radians(sigma_yaw_deg) ** 2)
    used = [r for r in rels if conn[r.i] and conn[r.j]]
    R = Rc
    st = {}
    for thr in prune_deg:
        res = rel_residuals_deg(R, used)
        used = [r for r, e in zip(used, res) if abs(r.i - r.j) <= seq_graph or e <= thr]
        R, st = rotation_averaging(n, used, yaw_route, yw, R, **kw)
    st.update({"connected": int(conn.sum()), "edges": len(used), "edges_all": len(rels)})
    return R, conn, used, st


def _components(n: int, rels: list[RelPose]) -> np.ndarray:
    par = list(range(n))

    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for r in rels:
        par[f(r.i)] = f(r.j)
    return np.array([f(i) for i in range(n)])


def pose_error_stats(R_wc: np.ndarray, C: np.ndarray, use: np.ndarray, rels: list[RelPose]) -> dict:
    """그래프 품질 (정답 없이): 쓴 사진 수, 연결 안 된 사진 수."""
    deg = np.zeros(len(R_wc), int)
    for r in rels:
        deg[r.i] += 1
        deg[r.j] += 1
    return {"used": int(use.sum()), "isolated": int((deg == 0).sum())}
