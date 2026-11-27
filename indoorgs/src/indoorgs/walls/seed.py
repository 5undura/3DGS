"""학습 전에 도면 면(맞춘 면) 위에 초기 점 심기.

Brush·LichtFeld 는 sparse/0/points3D.bin 의 xyz·rgb 로 가우시안을 시작함. 흰 벽에는 점이 거의 없어
구멍·안개가 생기므로, 면 위를 격자로 뽑은 점에 사진 색을 입혀 더함.

공개 API
  sample_faces(planes, spacing, rng)        -> (xyz, 면 번호)   격자 + 작은 흔들림, 문 구멍·다각형 밖 제외
  seed_points(planes, model, image_dir, ...) -> (xyz, rgb uint8, info)
  write_seeded_model(model, xyz, rgb, out_dir) 새 모델 폴더 (원래 점 + 새 점; 원래 모델은 건드리지 않음)

보임 판정 (점 하나 x 카메라 하나)
  카메라 앞 (z_cam > 0.1), 사진 안에 투영, 거리 < max_dist, 면 법선과 입사각 < max_incidence_deg,
  카메라가 면 안쪽 (signed_dist(C) > 0), 다른 면에 가리지 않음: 카메라->점 선분이 다른 면(contains_uv,
  여백 0, 문 구멍은 열림)을 점보다 2 cm 넘게 앞에서 지나면 가림.
  카메라가 max_cameras 보다 많으면 k 번째마다 하나씩만 씀 (k = ceil(개수 / max_cameras)), 이름 순.
  min_views 이상에서 보인 점만 남기고, 색 = 점수(cos 입사각 / 거리) 가 높은 max_views 개 보기의 채널별 중앙값
  (쌍선형 보간, BGR->RGB).
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from ..colmap import model as cm
from .planes import Plane

__all__ = ["sample_faces", "seed_points", "write_seeded_model", "camera_intrinsics"]

EPS_OCC = 0.02


def sample_faces(planes: list[Plane], spacing: float = 0.08, rng=0, jitter: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """면마다 spacing 격자 (칸 가운데) + ±jitter·spacing 흔들림. contains_uv 밖(문 구멍, 다각형 밖)은 버림."""
    rng = np.random.default_rng(rng)
    P, F = [], []
    for k, p in enumerate(planes):
        su, sv = p.size
        if su <= 0 or sv <= 0:
            continue
        us = np.arange(spacing / 2, su, spacing)
        vs = np.arange(spacing / 2, sv, spacing)
        U, V = np.meshgrid(us, vs, indexing="ij")
        uv = np.stack([U.ravel(), V.ravel()], 1)
        uv = uv + rng.uniform(-jitter, jitter, uv.shape) * spacing
        uv[:, 0] = np.clip(uv[:, 0], 0, su)
        uv[:, 1] = np.clip(uv[:, 1], 0, sv)
        uv = uv[p.contains_uv(uv)]
        if len(uv) == 0:
            continue
        X = p.point(uv[:, 0], uv[:, 1])
        X = X - (p.signed_dist(X))[:, None] * p.n     # 수치 오차 제거: 평면 위로 정확히
        P.append(X)
        F.append(np.full(len(X), k, np.int64))
    if not P:
        return np.zeros((0, 3)), np.zeros(0, np.int64)
    return np.concatenate(P), np.concatenate(F)


def camera_intrinsics(cam: cm.Camera) -> tuple[float, float, float, float]:
    """fx, fy, cx, cy. 왜곡 없는 SIMPLE_PINHOLE(0)·PINHOLE(1) 만 (image_undistorter 결과)."""
    if cam.model == 0:
        f, cx, cy = cam.params[:3]
        return float(f), float(f), float(cx), float(cy)
    if cam.model == 1:
        fx, fy, cx, cy = cam.params[:4]
        return float(fx), float(fy), float(cx), float(cy)
    raise ValueError(f"왜곡 없는 카메라 모델만 지원 (SIMPLE_PINHOLE/PINHOLE): {cm.CAM_NAMES.get(cam.model, cam.model)}"
                     " - image_undistorter 결과 모델을 쓰세요")


def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    im = img.astype(np.float32)
    top = im[y0, x0] * (1 - fx) + im[y0, x1] * fx
    bot = im[y1, x0] * (1 - fx) + im[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def _near_existing(existing: np.ndarray, X: np.ndarray, r: float) -> np.ndarray:
    """X 중 existing 점에서 r 안인 것 (scipy 있으면 cKDTree, 없으면 복셀 해시)."""
    if len(existing) == 0 or len(X) == 0:
        return np.zeros(len(X), bool)
    try:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(existing).query(X, k=1, distance_upper_bound=r)
        return d <= r
    except ImportError:
        pass
    key = np.floor(existing / r).astype(np.int64)
    cells: dict = {}
    for i, kk in enumerate(map(tuple, key)):
        cells.setdefault(kk, []).append(i)
    out = np.zeros(len(X), bool)
    kx = np.floor(X / r).astype(np.int64)
    offs = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
    for i, (x, y, z) in enumerate(map(tuple, kx)):
        for a, b, c in offs:
            ids = cells.get((x + a, y + b, z + c))
            if ids and np.min(np.sum((existing[ids] - X[i]) ** 2, 1)) <= r * r:
                out[i] = True
                break
    return out


def seed_points(planes: list[Plane], model: cm.Model, image_dir, *, spacing: float = 0.08, min_views: int = 2,
                max_views: int = 8, max_dist: float = 8.0, max_incidence_deg: float = 75.0,
                skip_near_existing: float = 0.0, max_points: int = 1_500_000, max_cameras: int = 300,
                rng=0) -> tuple[np.ndarray, np.ndarray, dict]:
    """면 위 점 + 사진 색. 돌려줌 (xyz (M,3) float64, rgb (M,3) uint8, info)."""
    image_dir = Path(image_dir)
    gen = np.random.default_rng(rng)
    X, face = sample_faces(planes, spacing, gen)
    n_sampled = len(X)
    info: dict = {"n_sampled": n_sampled, "spacing": spacing}
    if skip_near_existing > 0 and model.points and len(X):
        ex = np.array([p.xyz for p in model.points.values()], float)
        near = _near_existing(ex, X, skip_near_existing)
        info["n_skipped_near_existing"] = int(near.sum())
        X, face = X[~near], face[~near]
    if len(X) > max_points:
        keep = np.sort(gen.choice(len(X), max_points, replace=False))
        info["n_subsampled_away"] = int(len(X) - max_points)
        X, face = X[keep], face[keep]
    N, M = len(X), len(planes)

    # 면 법선·부호 거리 (점마다 자기 면)
    normals = np.array([p.n for p in planes]) if M else np.zeros((0, 3))
    Nf = normals[face] if N else np.zeros((0, 3))
    SD = np.stack([p.signed_dist(X) for p in planes], 1) if (M and N) else np.zeros((N, M))  # 점 x 면

    names = sorted(model.images.values(), key=lambda im: im.name)
    step = max(1, math.ceil(len(names) / max_cameras)) if max_cameras else 1
    used = names[::step]
    info.update(n_cameras=len(names), camera_step=step, n_cameras_used=0, missing_images=[])

    K = max(1, max_views)
    score = np.full((N, K), -np.inf, np.float32)
    col = np.full((N, K, 3), np.nan, np.float32)
    nview = np.zeros(N, np.int32)
    cos_min = math.cos(math.radians(max_incidence_deg))

    for im in used:
        path = image_dir / im.name
        img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR) if path.is_file() else None
        if img is None:
            info["missing_images"].append(im.name)
            continue
        info["n_cameras_used"] += 1
        cam = model.cameras[im.camera_id]
        fx, fy, cx, cy = camera_intrinsics(cam)
        sx, sy = img.shape[1] / cam.width, img.shape[0] / cam.height   # 사진을 줄여 둔 경우
        R, t = im.R, np.asarray(im.t, float)
        C = -R.T @ t
        sdC = np.array([p.signed_dist(C) for p in planes])               # 카메라의 면별 부호 거리
        # 1) 싼 판정
        D = X - C
        dist = np.linalg.norm(D, axis=1)
        cand = (dist < max_dist) & (dist > 1e-6) & (sdC[face] > 0)
        idx = np.flatnonzero(cand)
        if len(idx) == 0:
            continue
        cosi = -np.einsum("ij,ij->i", D[idx], Nf[idx]) / dist[idx]
        ok = cosi > cos_min
        idx, cosi = idx[ok], cosi[ok]
        Xc = X[idx] @ R.T + t
        z = Xc[:, 2]
        front = z > 0.1
        idx, cosi, Xc, z = idx[front], cosi[front], Xc[front], z[front]
        px = fx * Xc[:, 0] / z + cx
        py = fy * Xc[:, 1] / z + cy
        inside = (px >= 0) & (px <= cam.width) & (py >= 0) & (py <= cam.height)   # COLMAP: 픽셀 가운데 = i + 0.5
        idx, cosi, px, py = idx[inside], cosi[inside], px[inside], py[inside]
        if len(idx) == 0:
            continue
        # 2) 가림: 선분이 면 q 를 지나려면 카메라와 점이 q 의 서로 반대편
        occl = np.zeros(len(idx), bool)
        L = dist[idx]
        for q, pq in enumerate(planes):
            sp = SD[idx, q]
            cross = (np.sign(sp) != np.sign(sdC[q])) & (face[idx] != q) & ~occl
            if sdC[q] == 0:
                continue
            j = np.flatnonzero(cross)
            if len(j) == 0:
                continue
            tt = sdC[q] / (sdC[q] - sp[j])                      # 선분 위 교점 비율 (0..1)
            before = tt * L[j] < L[j] - EPS_OCC
            j, tt = j[before], tt[before]
            if len(j) == 0:
                continue
            H = C + tt[:, None] * (X[idx[j]] - C)
            hit = pq.contains_uv(pq.uv(H))
            occl[j[hit]] = True
        vis = ~occl
        idx, cosi, px, py, L = idx[vis], cosi[vis], px[vis], py[vis], L[vis]
        if len(idx) == 0:
            continue
        nview[idx] += 1
        rgb = _bilinear(img, px * sx - 0.5, py * sy - 0.5)[:, ::-1]   # BGR -> RGB
        sc = (cosi / np.maximum(L, 0.3)).astype(np.float32)
        slot = np.argmin(score[idx], axis=1)
        better = sc > score[idx, slot]
        ii, ss = idx[better], slot[better]
        score[ii, ss] = sc[better]
        col[ii, ss] = rgb[better]

    keep = nview >= min_views
    info["n_rejected_visibility"] = int((~keep).sum())
    Xk = X[keep]
    with np.errstate(all="ignore"):
        rgb = np.nanmedian(col[keep], axis=1) if keep.any() else np.zeros((0, 3))
    rgb = np.clip(np.round(np.nan_to_num(rgb)), 0, 255).astype(np.uint8)
    fk = face[keep]
    info["n_added"] = int(len(Xk))
    info["planes"] = {p.id: {"sampled": int((face == k).sum()), "added": int((fk == k).sum())}
                      for k, p in enumerate(planes)}
    return Xk, rgb, info


def write_seeded_model(model: cm.Model, xyz: np.ndarray, rgb: np.ndarray, out_dir, overwrite: bool = False) -> Path:
    """새 모델 폴더에 씀: cameras/images/frames/rigs 그대로, points3D = 원래 점 + 새 점 (새 id, error 0, 빈 트랙).
    원래 모델은 바꾸지 않음. out_dir 에 이미 모델이 있으면 overwrite=True 일 때만 덮어씀."""
    out = Path(out_dir)
    if not overwrite and ((out / "points3D.bin").exists() or (out / "images.bin").exists()):
        raise FileExistsError(f"이미 모델이 있음 (덮어쓰기 금지): {out}")
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    rgb = np.asarray(rgb).reshape(-1, 3).astype(np.uint8)
    if len(xyz) != len(rgb):
        raise ValueError("xyz 와 rgb 개수가 다름")
    points = dict(model.points)
    start = (max(points) + 1) if points else 1
    empty = np.zeros((0, 2), np.uint32)
    for i in range(len(xyz)):
        pid = start + i
        points[pid] = cm.Point3D(pid, xyz[i].copy(), rgb[i].copy(), 0.0, empty)
    cm.write_model(cm.Model(model.cameras, model.images, points, model.frames, model.rigs_bytes), out)
    return out
