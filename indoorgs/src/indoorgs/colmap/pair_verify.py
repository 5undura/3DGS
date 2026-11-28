"""1차 정합 자세로 COLMAP DB 의 짝 검증 + 최종 모델을 위치 제약(도면 좌표)에 직접 맞추기.

근거: experiments/prior_matching/RESULTS.md (합성 복도 v4, 15번 모두 255장 등록·틀린 사진 0·순간이동 0).
시험판 experiments/prior_matching/prior_verify.py, pm.align_to_priors 를 옮김. numpy + OpenCV + sqlite + colmap.model 만 씀.

register 안에서 쓰는 순서:
    ref = load_reference(colmap/sparse_pass1, priors)   # 1차 모델들(앞뒤 짝만 + 위치 제약) -> 사진별 자세 (수평 + 도면 좌표)
    stats = verify_pairs(colmap/db_verified.db, ref)    # DB 사본의 two_view_geometries 에서 안 맞는 짝 삭제
    new, st = align_model_to_priors(model, priors)      # 2차 결과를 수평 맞추고 제약에 다시 맞춤

주의: 짝을 지운 DB 에 *_matcher 를 다시 돌리면 지운 짝이 되살아난다 (COLMAP 4.2.1 controllers/feature_matching_utils.cc:
matches 는 있고 two_view_geometries 가 없는 짝은 다시 검증함). 그래서 사본에서 지우고, 사본으로는 mapper 만 돌린다.
수평 맞추기(level)는 짐벌(Osmo Pocket 3)처럼 화면 수평이 유지되는 영상에서만 맞다.
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

MAX_ID = 2147483647          # COLMAP pair_id = id1 * MAX_ID + id2 (id1 < id2), kMaxNumImages


@dataclass
class Reference:
    poses: dict[str, tuple[np.ndarray, np.ndarray]]      # 사진 이름 -> (R_world_from_cam, 중심), 도면 좌표
    K: np.ndarray
    dist: np.ndarray                                     # OpenCV 왜곡 계수 (k1 k2 p1 p2)
    info: dict = field(default_factory=dict)


def pair_id(i1: int, i2: int) -> int:
    a, b = (i1, i2) if i1 < i2 else (i2, i1)
    return a * MAX_ID + b


def camera_to_cv(cam: cmodel.Camera) -> tuple[np.ndarray, np.ndarray]:
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
    raise ValueError(f"카메라 모델 {cmodel.CAM_NAMES.get(cam.model, cam.model)} 은 지원 안 함 "
                     "(SIMPLE_PINHOLE/PINHOLE/SIMPLE_RADIAL/RADIAL/OPENCV)")


def rot2qvec(R: np.ndarray) -> np.ndarray:
    """회전 행렬 -> 사원수 (w x y z), w >= 0."""
    K = np.array([[R[0, 0] - R[1, 1] - R[2, 2], 0, 0, 0],
                  [R[1, 0] + R[0, 1], R[1, 1] - R[0, 0] - R[2, 2], 0, 0],
                  [R[2, 0] + R[0, 2], R[2, 1] + R[1, 2], R[2, 2] - R[0, 0] - R[1, 1], 0],
                  [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1], R[0, 0] + R[1, 1] + R[2, 2]]]) / 3.0
    w, v = np.linalg.eigh(K)
    q = v[[3, 0, 1, 2], np.argmax(w)]
    return -q if q[0] < 0 else q


def level_rotation(R_wc: np.ndarray) -> np.ndarray:
    """짐벌 카메라(수평 유지) 가정: 카메라 오른쪽 축들과 가장 수직인 방향을 위(+z)로 돌리는 회전 Q.
    R_wc: (N,3,3) world_from_cam. 카메라들이 여러 방향(요)을 봐야 위가 정해진다."""
    rx = R_wc[:, :, 0]
    _, v = np.linalg.eigh(rx.T @ rx)
    g = v[:, 0]
    if g @ (-R_wc[:, :, 1].mean(0)) < 0:        # 위 = 카메라 아래축(y)의 반대쪽
        g = -g
    ax = np.cross(g, [0, 0, 1.0])
    sn, cs = float(np.linalg.norm(ax)), float(g[2])
    if sn < 1e-9:
        return np.eye(3) if cs > 0 else np.diag([1.0, -1.0, -1.0])
    ax /= sn
    Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + sn * Kx + (1 - cs) * Kx @ Kx


def _rot_deg(Q: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Q) - 1) / 2))))


def fit_to_priors(C: np.ndarray, P: np.ndarray, iters: int = 5, scale_m: float = 1.0):
    """(수평을 맞춘) 카메라 중심 C 를 제약 위치 P 에 맞춤: 수직축 회전 + 축척 + 이동, Cauchy 가중 최소제곱.
    P 가 NaN 인 사진은 빼고 계산. 일직선 동선에서 위치 제약만으로는 복도 축 둘레 기울기가 안 정해지므로 수평 맞추기와 같이 쓴다.
    돌려줌: (s, Rz, b, stats) — 새 좌표 = s * Rz @ x + b. 제약 있는 사진이 3장 미만이면 None."""
    ok = ~np.isnan(P[:, 0])
    if ok.sum() < 3:
        return None
    w = ok.astype(float)
    Pn = np.nan_to_num(P)
    for _ in range(iters):
        ca, pa = (w[:, None] * C).sum(0) / w.sum(), (w[:, None] * Pn).sum(0) / w.sum()
        za = (C[:, 0] - ca[0]) + 1j * (C[:, 1] - ca[1])
        zb = (Pn[:, 0] - pa[0]) + 1j * (Pn[:, 1] - pa[1])
        k = (w * np.conj(za)) @ zb / max(1e-12, abs((w * np.conj(za)) @ za))
        s, th = abs(k), float(np.angle(k))
        Rz = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1.0]])
        b = pa - s * Rz @ ca
        r = np.linalg.norm(s * C @ Rz.T + b - Pn, axis=1)
        w = ok / (1.0 + (r / scale_m) ** 2)
    return float(s), Rz, b, {"scale": float(s), "yaw_deg": math.degrees(th),
                             "resid_median_m": float(np.median(r[ok])), "n_priors": int(ok.sum())}


def _prior_map(priors: dict) -> dict[str, np.ndarray]:
    return {f["image"]: np.asarray(f["xyz"], float) for f in priors.get("frames", [])}


def similarity_to_priors(R_wc: np.ndarray, C: np.ndarray, P: np.ndarray, level: bool = True):
    """수평 맞추기(level) 뒤 제약에 맞추는 닮음 변환. 돌려줌: (s, A, b, stats), 새 좌표 = s * A @ x + b. 못 맞추면 None."""
    Q = level_rotation(R_wc) if level else np.eye(3)
    fit = fit_to_priors(C @ Q.T, P)
    if fit is None:
        return None
    s, Rz, b, st = fit
    st["leveled_deg"] = round(_rot_deg(Q), 3)
    return s, Rz @ Q, b, st


def load_reference(sparse_dir: Path, priors: dict, max_far_m: float = 1.5, min_model: int = 10,
                   level: bool = True) -> Reference:
    """sparse_dir/<n> 의 모델(사진 min_model 장 이상) 모두에서 사진 자세를 모음. 모델마다 수평 맞추기 + 제약에 다시 맞추기.
    제약에서 max_far_m 넘게 떨어진 사진은 뺌 (1차에서 잘못 놓인 사진). 겹치면 큰 모델 것."""
    pri = _prior_map(priors)
    models = []
    for d in sorted(p for p in Path(sparse_dir).iterdir() if p.is_dir()):
        try:
            m = cmodel.read_model(d)
        except (FileNotFoundError, OSError, ValueError):
            continue
        if len(m.images) >= min_model:
            models.append(m)
    if not models:
        raise ValueError(f"{sparse_dir}: 사진 {min_model}장 이상인 모델 없음")
    models.sort(key=lambda m: -len(m.images))
    poses, far, tilt, unfit = {}, 0, [], 0
    for m in models:
        ims = list(m.images.values())
        R = np.array([im.R.T for im in ims])            # world_from_cam
        C = np.array([im.center for im in ims])
        P = np.array([pri.get(im.name, np.full(3, np.nan)) for im in ims])
        sim = similarity_to_priors(R, C, P, level)
        if sim is None:
            unfit += 1                                  # 제약에 못 맞춘 조각: 좌표를 못 믿으니 안 씀
            continue
        s, A, b, st = sim
        tilt.append(st["leveled_deg"])
        R, C = np.einsum("ij,njk->nik", A, R), s * C @ A.T + b
        for im, r, c in zip(ims, R, C):
            if im.name in poses:
                continue
            if im.name in pri and np.linalg.norm(c - pri[im.name]) > max_far_m:
                far += 1
                continue
            poses[im.name] = (r, c)
    K, dist = camera_to_cv(next(iter(models[0].cameras.values())))
    return Reference(poses, K, dist, {"models": [len(m.images) for m in models], "poses": len(poses),
                                      "far_dropped": far, "unfit_models": unfit, "leveled_deg": tilt,
                                      "focal": float(K[0, 0])})


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


def verify_pairs(db_path: Path, ref: Reference, seq_keep: int = 10, rot_tol_deg: float = 30.0,
                 dir_tol_deg: float = 45.0, min_baseline_m: float = 1.0, drop_unknown: bool = True,
                 min_inliers: int = 15) -> dict:
    """db_path 의 two_view_geometries 에서 1차 자세와 안 맞는 짝을 지움 (DB 를 고침 -> 사본에만 쓸 것). 돌려줌: 통계.
    - 이름 순(=시간 순)으로 앞뒤 seq_keep 장 안의 짝은 그대로 둠
    - 두 사진 다 기준 자세가 있으면: inlier 로 구한 상대 회전이 기준과 rot_tol_deg 넘게 다르거나,
      기준 기선이 min_baseline_m 이상인데 이동 방향이 dir_tol_deg 넘게 다르면 삭제. 상대 자세를 못 구하면 삭제
    - 한쪽이라도 기준 자세가 없으면 drop_unknown 일 때 삭제
    - inlier 가 min_inliers 미만인 짝은 mapper 도 안 쓰므로 건드리지 않음 (pairs 에도 안 셈)"""
    with closing(sqlite3.connect(str(db_path))) as c:
        ims = dict(c.execute("SELECT image_id, name FROM images"))
        order = {i: k for k, (n, i) in enumerate(sorted((n, i) for i, n in ims.items()))}
        kp = {i: np.frombuffer(b, np.float32).reshape(r, cc)[:, :2].astype(np.float64)
              for i, r, cc, b in c.execute("SELECT image_id, rows, cols, data FROM keypoints") if r}
        rows = c.execute("SELECT pair_id, rows, data FROM two_view_geometries WHERE rows >= ?",
                         (min_inliers,)).fetchall()
    st = {"pairs": len(rows), "kept_sequential": 0, "kept": 0, "drop_rot": 0, "drop_dir": 0, "drop_unknown": 0,
          "drop_no_pose": 0, "ref_poses": len(ref.poses)}
    drop = []
    for pid, r, blob in rows:
        i2 = pid % MAX_ID
        i1 = (pid - i2) // MAX_ID
        if i1 not in order or i2 not in order:
            continue
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
        rel = None
        if i1 in kp and i2 in kp:
            rel = _relative_pose(kp[i1][m[:, 0]], kp[i2][m[:, 1]], ref.K, ref.dist)
        if rel is None:
            drop.append(pid)
            st["drop_no_pose"] += 1
            continue
        (R1, C1), (R2, C2) = a, b
        Rp = R2.T @ R1                                   # 기준 cam2_from_cam1
        if _angle((np.trace(Rp.T @ rel[0]) - 1) / 2) > rot_tol_deg:
            drop.append(pid)
            st["drop_rot"] += 1
            continue
        c2 = R1.T @ (C2 - C1)                            # cam1 좌표의 cam2 중심
        base = float(np.linalg.norm(c2))
        if base >= min_baseline_m and _angle(float(c2 @ rel[1]) / base) > dir_tol_deg:
            drop.append(pid)
            st["drop_dir"] += 1
            continue
        st["kept"] += 1
    with closing(sqlite3.connect(str(db_path))) as c, c:
        c.executemany("DELETE FROM two_view_geometries WHERE pair_id=?", [(p,) for p in drop])
    st["dropped"] = len(drop)
    return st


def transform_model(model: cmodel.Model, s: float, A: np.ndarray, b: np.ndarray) -> cmodel.Model:
    """닮음 변환 x' = s * A @ x + b 를 모델 전체(사진 자세, frames.bin 의 rig 자세, 3D 점)에 적용한 새 모델.
    cam_from_world (R, t) -> (R A^T, s t - R A^T b). frames.bin 의 rig_from_world 도 같은 식
    (COLMAP 4.x 는 frames.bin 이 있으면 그 자세를 씀. 우리 rig 는 카메라 1대짜리라 cam_from_rig = 단위)."""
    images = {}
    for iid, im in model.images.items():
        R2 = im.R @ A.T
        images[iid] = cmodel.Image(im.id, rot2qvec(R2), s * im.t - R2 @ b, im.camera_id, im.name, im.xy, im.pid.copy())
    points = {pid: cmodel.Point3D(p.id, s * A @ p.xyz + b, p.rgb, p.error, p.track) for pid, p in model.points.items()}
    frames = None
    if model.frames is not None:
        frames = []
        for fr in model.frames:
            R = cmodel.qvec2rot(fr.pose[:4])
            R2 = R @ A.T
            t2 = s * np.asarray(fr.pose[4:7], float) - R2 @ b
            frames.append(cmodel.Frame(fr.id, fr.rig_id, tuple(map(float, (*rot2qvec(R2), *t2))), list(fr.data)))
    return cmodel.Model(dict(model.cameras), images, points, frames, model.rigs_bytes)


def align_model_to_priors(model: cmodel.Model, priors: dict, level: bool = True) -> tuple[cmodel.Model, dict]:
    """최종 모델을 도면 좌표로: 수평 맞추기(level, 카메라 오른쪽 축들과 가장 수직인 방향 = 위) + 수직축 회전·축척·이동을
    제약 위치에 강건 최소제곱으로. 점·카메라 모두 변환한 새 모델과 통계를 돌려줌. 제약 있는 사진이 3장 미만이면 ValueError."""
    pri = _prior_map(priors)
    ims = list(model.images.values())
    if not ims:
        raise ValueError("모델에 사진 없음")
    R = np.array([im.R.T for im in ims])
    C = np.array([im.center for im in ims])
    P = np.array([pri.get(im.name, np.full(3, np.nan)) for im in ims])
    sim = similarity_to_priors(R, C, P, level)
    if sim is None:
        raise ValueError(f"제약이 있는 사진이 3장 미만 ({int((~np.isnan(P[:, 0])).sum())}장)")
    s, A, b, st = sim
    new = transform_model(model, s, A, b)
    C2 = s * C @ A.T + b
    d = np.linalg.norm(C2 - P, axis=1)
    ok = ~np.isnan(d)
    st.update({"level": level, "far_1m": int((d[ok] > 1.0).sum()),
               "transform": {"scale": s, "rotation": A.tolist(), "translation": b.tolist()}})
    return new, st
