"""제안 구현 (시험판): 1차 정합(앞뒤 짝만 + 위치 제약)의 자세로 COLMAP DB 의 짝을 검증해 지운다.

src 에 넣을 모양으로 씀: numpy + OpenCV + sqlite + indoorgs.colmap.model 만 사용 (pycolmap 없음).
쓰는 법 (register 단계 안에서):
    ref = load_reference(sparse_pass1, priors)                 # 1차 모델들 (도면 좌표) -> 사진별 자세
    stats = verify_pairs(db_copy, ref)                         # db_copy 의 two_view_geometries 에서 안 맞는 짝 삭제
주의: 짝을 지운 DB 에 다시 *_matcher 를 돌리면 지운 짝이 되살아난다 (4.2.1 feature_matching_utils.cc: matches 는 있고
two_view_geometries 가 없는 짝은 다시 검증함). 그래서 사본에서 지우고, 그 사본으로는 mapper 만 돌린다.
"""
from __future__ import annotations

import math
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from indoorgs.colmap import model as cmodel

MAX_ID = 2147483647          # COLMAP pair_id = id1 * MAX_ID + id2 (id1 < id2)


@dataclass
class Reference:
    poses: dict[str, tuple[np.ndarray, np.ndarray]]      # 사진 이름 -> (R_world_from_cam, 중심)
    K: np.ndarray
    dist: np.ndarray                                     # OpenCV 왜곡 계수 (k1 k2 p1 p2)
    info: dict = field(default_factory=dict)


def camera_to_cv(cam) -> tuple[np.ndarray, np.ndarray]:
    """COLMAP 카메라 -> (K, dist). 모델 번호: 0 SIMPLE_PINHOLE, 1 PINHOLE, 2 SIMPLE_RADIAL, 3 RADIAL, 4 OPENCV."""
    p = np.asarray(cam.params, float)
    if cam.model in (0, 2, 3):
        K = np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1.0]])
        k = list(p[3:5]) + [0.0] * (2 - len(p[3:5]))
        return K, np.array([k[0], k[1], 0.0, 0.0])
    if cam.model in (1, 4):
        K = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1.0]])
        d = np.zeros(4)
        d[:len(p[4:8])] = p[4:8]
        return K, d
    raise ValueError(f"카메라 모델 {cam.model} 은 지원 안 함 (SIMPLE_PINHOLE/PINHOLE/SIMPLE_RADIAL/RADIAL/OPENCV)")


def level_rotation(R_wc: np.ndarray) -> np.ndarray:
    """짐벌 카메라(수평 유지) 가정: 카메라 오른쪽 축들과 가장 수직인 방향을 위(+z)로 돌리는 회전 Q."""
    rx = R_wc[:, :, 0]
    _, v = np.linalg.eigh(rx.T @ rx)
    g = v[:, 0]
    if g @ (-R_wc[:, :, 1].mean(0)) < 0:        # 위 = 카메라 아래축(y)의 반대쪽
        g = -g
    ax = np.cross(g, [0, 0, 1.0])
    sn, cs = float(np.linalg.norm(ax)), float(g[2])
    if sn < 1e-9:
        return np.eye(3)
    ax /= sn
    Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + sn * Kx + (1 - cs) * Kx @ Kx


def fit_to_priors(R_wc: np.ndarray, C: np.ndarray, P: np.ndarray, iters: int = 5, scale_m: float = 1.0):
    """수평을 맞춘 모델을 제약 위치 P 에 맞춤: 수직축 회전 + 축척 + 이동, Cauchy 가중 최소제곱 (P 가 NaN 인 사진은 빼고 계산).
    일직선 동선에서 위치 제약만으로는 복도 축 둘레 기울기(뒤집힘 포함)가 안 정해지므로 수평 맞추기와 꼭 같이 쓴다.
    돌려줌: (R_wc, C, {"scale", "resid_median"})"""
    ok = ~np.isnan(P[:, 0])
    if ok.sum() < 3:
        return R_wc, C, {"scale": 1.0, "resid_median": float("nan")}
    w = ok.astype(float)
    for _ in range(iters):
        ca, pa = (w[:, None] * C).sum(0) / w.sum(), (w[:, None] * np.nan_to_num(P)).sum(0) / w.sum()
        za = (C[:, 0] - ca[0]) + 1j * (C[:, 1] - ca[1])
        zb = np.nan_to_num(P[:, 0] - pa[0]) + 1j * np.nan_to_num(P[:, 1] - pa[1])
        k = (w * np.conj(za)) @ zb / max(1e-12, abs((w * np.conj(za)) @ za))
        s, th = abs(k), float(np.angle(k))
        Rz = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1.0]])
        C2 = s * (C - ca) @ Rz.T + pa
        r = np.linalg.norm(np.nan_to_num(C2 - P), axis=1)
        w = ok / (1.0 + (r / scale_m) ** 2)
    return np.einsum("ij,njk->nik", Rz, R_wc), C2, {"scale": float(s), "resid_median": float(np.median(r[ok]))}


def load_reference(sparse: Path, priors: dict, max_far_m: float = 1.5, min_model: int = 10,
                   level: bool = True) -> Reference:
    """sparse/<n> 의 모든 모델(사진 min_model 장 이상, pose_prior_mapper 가 도면 좌표에 놓은 것)에서 사진 자세를 모음.
    제약에서 max_far_m 넘게 떨어진 사진은 뺌 (1차에서 잘못 놓인 사진). 겹치면 큰 모델 것."""
    pri = {f["image"]: np.asarray(f["xyz"], float) for f in priors["frames"]}
    models = []
    for d in sorted(p for p in Path(sparse).iterdir() if p.is_dir()):
        try:
            m = cmodel.read_model(d)
        except (FileNotFoundError, OSError, ValueError):
            continue
        if len(m.images) >= min_model:
            models.append(m)
    if not models:
        raise ValueError(f"{sparse}: 모델 없음")
    models.sort(key=lambda m: -len(m.images))
    poses, far, tilt = {}, 0, []
    for m in models:
        ims = list(m.images.values())
        R = np.array([im.R.T for im in ims])            # world_from_cam
        C = np.array([im.center for im in ims])
        if level:
            Q = level_rotation(R)
            tilt.append(math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Q) - 1) / 2)))))
            P = np.array([pri.get(im.name, np.full(3, np.nan)) for im in ims])
            R, C, _ = fit_to_priors(np.einsum("ij,njk->nik", Q, R), C @ Q.T, P)
        for im, r, c in zip(ims, R, C):
            if im.name in poses:
                continue
            if im.name in pri and np.linalg.norm(c - pri[im.name]) > max_far_m:
                far += 1
                continue
            poses[im.name] = (r, c)
    K, dist = camera_to_cv(next(iter(models[0].cameras.values())))
    return Reference(poses, K, dist, {"models": [len(m.images) for m in models], "poses": len(poses),
                                      "far_dropped": far, "leveled_deg": tilt, "focal": float(K[0, 0])})


def _relative_pose(p1, p2, K, dist):
    """inlier 매칭 -> (R cam2_from_cam1, cam1 좌표의 cam2 중심 방향). 실패하면 None."""
    n1 = cv2.undistortPoints(p1.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    n2 = cv2.undistortPoints(p2.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    try:
        E, mask = cv2.findEssentialMat(n1, n2, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=2.0 / K[0, 0])
        if E is None or E.shape != (3, 3):
            return None
        _, R, t, _ = cv2.recoverPose(E, n1, n2, np.eye(3), mask=mask)
    except cv2.error:
        return None
    return R, (-R.T @ t).ravel()


def _angle(c: float) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def verify_pairs(db: Path, ref: Reference, seq_keep: int = 10, rot_tol_deg: float = 30.0, dir_tol_deg: float = 45.0,
                 min_baseline_m: float = 1.0, min_inliers: int = 15, drop_unknown: bool = True) -> dict:
    """db 의 two_view_geometries 에서 1차 자세와 안 맞는 짝을 지움. 돌려줌: 통계.
    - 이름 순(=시간 순)으로 앞뒤 seq_keep 장 안의 짝은 그대로 둠
    - 두 사진 다 1차 자세가 있으면: 상대 회전 차이 > rot_tol_deg 이거나, 1차 기선이 min_baseline_m 이상인데
      기선 방향 차이 > dir_tol_deg 이면 삭제
    - 한쪽이라도 1차 자세가 없으면 drop_unknown 일 때 삭제"""
    with closing(sqlite3.connect(str(db))) as c:
        ims = dict(c.execute("SELECT image_id, name FROM images"))
        order = {i: k for k, (n, i) in enumerate(sorted((n, i) for i, n in ims.items()))}
        kp = {i: np.frombuffer(b, np.float32).reshape(r, cc)[:, :2].astype(np.float64)
              for i, r, cc, b in c.execute("SELECT image_id, rows, cols, data FROM keypoints")}
        rows = c.execute("SELECT pair_id, rows, data FROM two_view_geometries WHERE rows >= ?", (min_inliers,)).fetchall()
    st = {"pairs": len(rows), "kept_sequential": 0, "kept": 0, "drop_rot": 0, "drop_dir": 0, "drop_unknown": 0,
          "drop_no_pose": 0}
    drop = []
    for pid, r, blob in rows:
        i2 = pid % MAX_ID
        i1 = (pid - i2) // MAX_ID
        if abs(order[i1] - order[i2]) <= seq_keep:
            st["kept_sequential"] += 1
            continue
        a, b = ref.poses.get(ims[i1]), ref.poses.get(ims[i2])
        if a is None or b is None:
            if drop_unknown:
                drop.append(pid)
                st["drop_unknown"] += 1
            else:
                st["kept"] += 1
            continue
        m = np.frombuffer(blob, np.uint32).reshape(r, 2)
        rel = _relative_pose(kp[i1][m[:, 0]], kp[i2][m[:, 1]], ref.K, ref.dist)
        if rel is None:
            drop.append(pid)
            st["drop_no_pose"] += 1
            continue
        (R1, C1), (R2, C2) = a, b
        Rp = R2.T @ R1
        if _angle((np.trace(Rp.T @ rel[0]) - 1) / 2) > rot_tol_deg:
            drop.append(pid)
            st["drop_rot"] += 1
            continue
        c2 = R1.T @ (C2 - C1)
        base = float(np.linalg.norm(c2))
        if base >= min_baseline_m and _angle(float(c2 @ rel[1]) / base) > dir_tol_deg:
            drop.append(pid)
            st["drop_dir"] += 1
            continue
        st["kept"] += 1
    with closing(sqlite3.connect(str(db))) as c, c:
        c.executemany("DELETE FROM two_view_geometries WHERE pair_id=?", [(p,) for p in drop])
    st["dropped"] = len(drop)
    st["dropped_ids"] = drop
    return st
