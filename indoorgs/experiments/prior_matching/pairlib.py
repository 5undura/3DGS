"""동선(priors)으로 COLMAP 매칭 짝을 고르고·검증하는 함수들 (시험 구현, numpy + sqlite + OpenCV 만 씀).

DB 의 two_view_geometries (기하 검증을 통과한 짝)에서 '동선과 안 맞는 짝'을 지운다. mapper 는 이 표만 읽는다.
  select_pairs   : 거리·이웃 수로 짝 고르기 (spatial_matcher 와 같은 규칙) + 앞뒤 프레임 짝 유지
  relative_poses : 짝마다 inlier 매칭으로 상대 자세 (R, t 방향, 시차) 추정
  check_heading  : 상대 회전각 vs 동선 진행 방향 차이 (T_rot)
  check_depth    : 동선상 거리가 먼데 시차가 거의 없는 짝 (T_depth, 도면 크기 사용)
  check_poses    : 1차 정합(앞뒤 짝만 + 위치 제약)의 자세와 상대 자세 비교 (T_pose)
"""
from __future__ import annotations

import math
import sqlite3
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np

MAX_ID = 2147483647


def pair_id(i1: int, i2: int) -> int:
    a, b = (i1, i2) if i1 < i2 else (i2, i1)
    return a * MAX_ID + b


def id_pair(pid: int) -> tuple[int, int]:
    i2 = pid % MAX_ID
    return (pid - i2) // MAX_ID, i2


def read_db(db: Path, min_inliers: int = 15) -> dict:
    """images(id->name), order(id->시간 순 번호), keypoints, pairs(pair_id -> inlier 매칭 Nx2)."""
    with closing(sqlite3.connect(str(db))) as c:
        ims = dict(c.execute("SELECT image_id, name FROM images"))
        kp = {i: np.frombuffer(b, np.float32).reshape(r, cc)[:, :2].astype(np.float64)
              for i, r, cc, b in c.execute("SELECT image_id, rows, cols, data FROM keypoints")}
        pairs = {pid: np.frombuffer(b, np.uint32).reshape(r, 2)
                 for pid, r, b in c.execute("SELECT pair_id, rows, data FROM two_view_geometries WHERE rows >= ?",
                                            (min_inliers,))}
        cam = c.execute("SELECT width, height, params FROM cameras LIMIT 1").fetchone()
    order = {i: k for k, (n, i) in enumerate(sorted((n, i) for i, n in ims.items()))}
    return {"images": ims, "order": order, "kp": kp, "pairs": pairs,
            "size": (cam[0], cam[1]), "cam_params": np.frombuffer(cam[2], np.float64)}


def delete_pairs(db: Path, pids) -> int:
    with closing(sqlite3.connect(str(db))) as c, c:
        c.executemany("DELETE FROM two_view_geometries WHERE pair_id=?", [(int(p),) for p in pids])
    return len(pids)


def prior_arrays(priors: dict, images: dict) -> dict[int, np.ndarray]:
    by = {f["image"]: np.asarray(f["xyz"], float) for f in priors["frames"]}
    return {i: by[n] for i, n in images.items() if n in by}


# ------------------------------------------------------------------ (a) 짝 고르기
def select_pairs(pos: dict[int, np.ndarray], order: dict[int, int], max_distance: float, max_num_neighbors: int = 0,
                 seq_overlap: int = 10, ignore_z: bool = True) -> set[int]:
    """둘 수 있는 짝의 pair_id 집합. spatial_matcher 규칙(사진마다 max_distance 안의 가까운 max_num_neighbors 장,
    0 이면 제한 없음) + 시간 순 앞뒤 seq_overlap 장."""
    ids = sorted(pos, key=lambda i: order[i])
    P = np.array([pos[i] for i in ids])
    if ignore_z:
        P = P[:, :2]
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    keep = set()
    for a, ia in enumerate(ids):
        nn = np.argsort(D[a])
        nn = nn[(D[a, nn] <= max_distance) & (nn != a)]
        if max_num_neighbors:
            nn = nn[:max_num_neighbors]
        keep.update(pair_id(ia, ids[b]) for b in nn)
    by_order = sorted(order, key=lambda i: order[i])
    for a, ia in enumerate(by_order):
        for b in range(a + 1, min(len(by_order), a + seq_overlap + 1)):
            keep.add(pair_id(ia, by_order[b]))
    return keep


# ------------------------------------------------------------------ 상대 자세
def relative_poses(d: dict, focal: float, pids=None) -> dict[int, dict]:
    """짝마다 E 분해. 돌려줌 pid -> {R (cam2_from_cam1), c2 (cam1 좌표의 cam2 중심 방향, 단위),
    angle (회전각 °), depth (기선 길이를 1 로 봤을 때 점 깊이 중앙값; 시차가 없으면 큼), front (앞에 놓인 점 비율)}"""
    w, h = d["size"]
    K = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1.0]])
    out = {}
    for pid in (pids if pids is not None else d["pairs"]):
        m = d["pairs"][pid]
        i1, i2 = id_pair(pid)
        p1, p2 = d["kp"][i1][m[:, 0]], d["kp"][i2][m[:, 1]]
        try:
            E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=0.999, threshold=2.0)
            if E is None or E.shape != (3, 3):
                continue
            n, R, t, mask2, X = cv2.recoverPose(E, p1, p2, K, distanceThresh=1e9, mask=mask)
        except cv2.error:
            continue
        X = X.reshape(4, -1)
        ok = (mask2.ravel() > 0) & (np.abs(X[3]) > 1e-12)
        z = X[2, ok] / X[3, ok]
        z = z[z > 0]
        ninl = int(mask.sum())
        depth = float(np.median(z)) if len(z) >= 8 else float("inf")
        ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))
        out[pid] = {"R": R, "c2": (-R.T @ t).ravel(), "angle": ang, "depth": depth,
                    "front": float(n) / max(1, ninl), "n": int(len(m))}
    return out


# ------------------------------------------------------------------ 동선 진행 방향
def path_headings(pos_by_order: np.ndarray, t: np.ndarray, span_s: float = 0.8, min_speed: float = 0.25,
                  turn_window_s: float = 2.0, turn_deg: float = 60.0) -> np.ndarray:
    """프레임별 진행 방향 단위 벡터 (N,2). 모르면 NaN: 느리게 움직이거나(멈춤), 방향이 크게 꺾이는 곳 앞뒤 turn_window_s."""
    n = len(t)
    xy = pos_by_order[:, :2]
    h = np.full((n, 2), np.nan)
    for i in range(n):
        a = np.searchsorted(t, t[i] - span_s)
        b = min(n - 1, np.searchsorted(t, t[i] + span_s))
        dt = t[b] - t[a]
        v = xy[b] - xy[a]
        if dt > 0 and np.linalg.norm(v) / dt >= min_speed:
            h[i] = v / np.linalg.norm(v)
    bad = np.zeros(n, bool)
    for i in range(n):
        a = np.searchsorted(t, t[i] - turn_window_s)
        b = min(n - 1, np.searchsorted(t, t[i] + turn_window_s))
        hh = h[a:b + 1]
        hh = hh[~np.isnan(hh[:, 0])]
        if len(hh) >= 2 and (hh @ hh.T).min() < math.cos(math.radians(turn_deg)):
            bad[i] = True
    h[bad] = np.nan
    return h


def check_heading(rel: dict, order: dict, head: np.ndarray, tol_deg: float = 80.0, seq_keep: int = 10) -> set[int]:
    """T_rot: |상대 회전각 - 진행 방향 차이| > tol 인 짝 (지울 것). 방향을 모르는 프레임·앞뒤 seq_keep 장 짝은 안 건드림."""
    drop = set()
    for pid, r in rel.items():
        a, b = (order[i] for i in id_pair(pid))
        if abs(a - b) <= seq_keep or np.isnan(head[a, 0]) or np.isnan(head[b, 0]):
            continue
        dh = math.degrees(math.acos(max(-1.0, min(1.0, float(head[a] @ head[b])))))
        if abs(r["angle"] - dh) > tol_deg:
            drop.add(pid)
    return drop


def check_depth(rel: dict, order: dict, pos_by_order: np.ndarray, max_depth: float, sigma: float = 0.5,
                min_baseline: float = 2.0, seq_keep: int = 10) -> set[int]:
    """T_depth: 동선상 두 사진이 b (>= min_baseline) 떨어져 있는데 시차가 너무 작은 짝.
    기선을 (b - 2*sqrt(2)*sigma) 로 보면 점 깊이 중앙값이 max_depth (도면 대각선) 를 넘는다 = 실제로는 거의 같은 자리에서 찍힌 모양."""
    drop = set()
    for pid, r in rel.items():
        a, b = (order[i] for i in id_pair(pid))
        if abs(a - b) <= seq_keep:
            continue
        base = float(np.linalg.norm(pos_by_order[a, :2] - pos_by_order[b, :2]))
        if base < min_baseline:
            continue
        if r["depth"] * (base - 2 * math.sqrt(2) * sigma) > max_depth:
            drop.add(pid)
    return drop


def check_poses(rel: dict, order: dict, poses: dict[int, tuple[np.ndarray, np.ndarray]], rot_tol: float = 30.0,
                dir_tol: float = 45.0, min_baseline: float = 1.0, seq_keep: int = 10,
                drop_unknown: bool = True) -> tuple[set[int], dict]:
    """T_pose: 1차 정합 자세 (order 번호 -> (R_world_from_cam, C)) 와 상대 자세 비교.
    회전 차이 > rot_tol, 또는 (예측 기선 >= min_baseline 일 때) 기선 방향 차이 > dir_tol 이면 지움.
    한쪽이라도 1차 자세가 없으면 drop_unknown 이면 지움 (앞뒤 seq_keep 장 짝은 항상 둠)."""
    drop, st = set(), {"rot": 0, "dir": 0, "unknown": 0}
    for pid, r in rel.items():
        a, b = (order[i] for i in id_pair(pid))
        if abs(a - b) <= seq_keep:
            continue
        if a not in poses or b not in poses:
            if drop_unknown:
                drop.add(pid)
                st["unknown"] += 1
            continue
        (R1, C1), (R2, C2) = poses[a], poses[b]
        Rp = R2.T @ R1                                   # cam2_from_cam1 예측
        dR = Rp.T @ r["R"]
        e = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2))))
        if e > rot_tol:
            drop.add(pid)
            st["rot"] += 1
            continue
        c2 = R1.T @ (C2 - C1)                            # cam1 좌표의 cam2 중심
        base = float(np.linalg.norm(c2))
        if base >= min_baseline:
            ang = math.degrees(math.acos(max(-1.0, min(1.0, float(c2 @ r["c2"]) / base))))
            if ang > dir_tol:
                drop.add(pid)
                st["dir"] += 1
    return drop, st


# ------------------------------------------------------------------ 정답으로 짝 라벨 (실험 평가용)
def label_pairs(d: dict, gt: dict[str, tuple[np.ndarray, np.ndarray]], K: np.ndarray, max_px: float) -> dict[int, bool]:
    """pid -> 가짜 짝인가 (정답 F 에 대한 inlier Sampson 오차 중앙값 > max_px)."""
    Ki = np.linalg.inv(K)

    def skew(t):
        return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    out = {}
    for pid, m in d["pairs"].items():
        i1, i2 = id_pair(pid)
        (R1, C1), (R2, C2) = gt[d["images"][i1]], gt[d["images"][i2]]
        F = Ki.T @ skew(R2.T @ (C1 - C2)) @ (R2.T @ R1) @ Ki
        x1 = np.c_[d["kp"][i1][m[:, 0]], np.ones(len(m))]
        x2 = np.c_[d["kp"][i2][m[:, 1]], np.ones(len(m))]
        Fx1, Ftx2 = x1 @ F.T, x2 @ F
        e = np.sqrt(np.sum(x2 * Fx1, 1) ** 2 / (Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2 + 1e-30))
        # 거의 제자리 회전(기선 < 5 cm)인 짝은 F 로 판단 못 함 -> 회전 호모그래피로
        if np.linalg.norm(C1 - C2) < 0.05:
            Hm = K @ (R2.T @ R1) @ Ki
            y = x1 @ Hm.T
            e = np.linalg.norm(y[:, :2] / y[:, 2:3] - x2[:, :2], axis=1)
            out[pid] = bool(np.median(e) > 4 * max_px)
        else:
            out[pid] = bool(np.median(e) > max_px)
    return out
