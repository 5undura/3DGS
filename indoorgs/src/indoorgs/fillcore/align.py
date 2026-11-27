"""문 근처 미세 정렬: 점-점 ICP (강체, 회전·이동 작게 제한). merge 가 씀."""
from __future__ import annotations

import numpy as np


def kabsch(src, dst) -> np.ndarray:
    """dst ~= R src + t (강체) 4x4."""
    ms, md = src.mean(0), dst.mean(0)
    H = (src - ms).T @ (dst - md)
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = md - R @ ms
    return T


def apply(M, P):
    M = np.asarray(M, float)
    return P @ M[:3, :3].T + M[:3, 3]


def rot_deg(M) -> float:
    R = np.asarray(M)[:3, :3]
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


# 짝 최대 거리. 예전 0.3: 문 난 벽의 복도 쪽 면(복도 구간만 봄)과 방 쪽 면(방 구간만 봄)이 벽 두께(0.2m)만큼
# 떨어져 짝이 되어, 이미 맞은 구간을 벽 두께만큼 옮겨 붙였음 (합성 220회 중 121회 나빠짐 -> 0.08 에서 8회,
# experiments/accuracy/merge_eval.py). 벽 두께의 절반보다 작아야 함. 0.1m 벽은 0.08 로도 못 막음 (RESULTS.md)
MAX_CORR = 0.08


def icp(src, dst, max_rot_deg: float = 3.0, max_trans: float = 0.3, max_corr: float = MAX_CORR,
        iters: int = 40, min_gain: float = 0.1, min_inlier: float = 0.3) -> tuple[np.ndarray | None, dict]:
    """src 를 dst 에 맞추는 4x4 (dst ~= T src). 받아들일 수 없으면 (None, 이유).
    - 짝: 최근접, 거리 < min(max_corr, 3 x 중앙값) 만 (겹치지 않는 부분은 버림)
    - 받아들임: 회전 <= max_rot_deg, 이동 <= max_trans, 짝 거리 중앙값이 min_gain 이상 줄고 짝 비율 >= min_inlier"""
    from scipy.spatial import cKDTree

    src = np.asarray(src, float); dst = np.asarray(dst, float)
    info: dict = {"n_src": int(len(src)), "n_dst": int(len(dst))}
    if len(src) < 30 or len(dst) < 30:
        info["reason"] = "점이 너무 적음"
        return None, info
    tree = cKDTree(dst)
    d0 = tree.query(src, distance_upper_bound=max_corr)[0]
    fin0 = np.isfinite(d0)
    if fin0.mean() < min_inlier:
        info["reason"] = f"겹치는 점 비율 {fin0.mean():.2f} < {min_inlier}"
        return None, info
    med0 = float(np.median(d0[fin0]))
    T = np.eye(4)
    for _ in range(iters):
        cur = apply(T, src)
        d, j = tree.query(cur, distance_upper_bound=max_corr)
        fin = np.isfinite(d)
        if fin.sum() < 30:
            break
        m = fin & (d < max(3 * np.median(d[fin]), 1e-9))
        dT = kabsch(cur[m], dst[j[m]])
        T = dT @ T
        if np.abs(dT[:3, 3]).max() < 1e-6 and rot_deg(dT) < 1e-5:
            break
    d1 = tree.query(apply(T, src), distance_upper_bound=max_corr)[0]
    fin1 = np.isfinite(d1)
    med1 = float(np.median(d1[fin1])) if fin1.any() else np.inf
    info.update(median_before=round(med0, 4), median_after=round(med1, 4), inlier=round(float(fin1.mean()), 3),
                rot_deg=round(rot_deg(T), 3), trans=round(float(np.linalg.norm(T[:3, 3])), 4))
    if rot_deg(T) > max_rot_deg:
        info["reason"] = f"회전 {rot_deg(T):.2f}도 > {max_rot_deg}"
    elif np.linalg.norm(T[:3, 3]) > max_trans:
        info["reason"] = f"이동 {np.linalg.norm(T[:3, 3]):.3f} > {max_trans}"
    elif fin1.mean() < min_inlier:
        info["reason"] = f"맞춘 뒤 겹치는 점 비율 {fin1.mean():.2f} < {min_inlier}"
    elif not med1 < (1 - min_gain) * med0:
        info["reason"] = f"나아지지 않음 (짝 거리 중앙값 {med0:.4f} -> {med1:.4f})"
    else:
        return T, info
    return None, info
