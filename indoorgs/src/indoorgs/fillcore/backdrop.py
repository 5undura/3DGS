"""바탕층(backdrop) 만들기 + 구멍만 고르기. tools/fill/fill_section.py 1~5단계, merge_holes.py 이식.

면 정하기 두 방식
- 점에서 추정 (fill_section 그대로): 카메라 아래 방향 + 낮은 점 RANSAC 으로 바닥 평면, 바닥 점·카메라로 평면도 격자,
  천장 = 평면도 안 카메라보다 높은 점의 99 백분위. COLMAP 축척은 구간마다 달라 길이는 '바닥 위 카메라 높이' 비례.
- 도면(plan.json) 방: 구간이 도면 좌표로 정합됐을 때(priors.json). 위 = +z, 바닥 = floor_z, 천장 = ceiling_z,
  평면도 = 방 바닥 다각형. 천장 높이가 없으면 점에서 추정.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .. import plyio

FLOOR, CEILING, WALL = 0, 1, 2
REF_CAM_HEIGHT = 0.67      # 복도 1 COLMAP 에서 바닥 위 카메라 높이. 모든 길이의 기준 (fill_section.py)
OFFSET = 0.012             # 면 바로 뒤(바깥)로 미는 거리 (x 축척)


@dataclass
class Samples:
    """면 위 샘플 점 (모델 좌표)."""
    P: np.ndarray              # (N,3) 면 위 위치
    N: np.ndarray              # (N,3) 방 안쪽을 향한 법선
    kind: np.ndarray           # (N,) FLOOR / CEILING / WALL
    up: np.ndarray             # 위 방향
    ax1: np.ndarray            # 바닥·천장 스플랫의 첫 축 방향 (복도 방향)
    S: float                   # 축척 (카메라 높이 / 0.67)
    SP: float                  # 샘플 간격
    info: dict = field(default_factory=dict)


# ---------------- 점에서 추정 ----------------
def _pca_axis(Cc, up):
    if len(Cc) >= 2 and np.linalg.norm(Cc) > 0:
        ax1 = np.linalg.svd(Cc, full_matrices=False)[2][0]
    else:
        ax1 = np.array([1.0, 0, 0])
    ax1 = ax1 - (ax1 @ up) * up
    if np.linalg.norm(ax1) < 1e-9:
        ax1 = np.cross(up, [1.0, 0, 0]) if abs(up[0]) < 0.9 else np.cross(up, [0, 1.0, 0])
    return ax1 / np.linalg.norm(ax1)


def samples_from_points(C, cam_down, X, err, seed: int = 0, log: Callable = lambda m: None) -> Samples:
    """C: 카메라 중심 (M,3), cam_down: 카메라 y축(사진 아래 방향)의 세계 방향 (M,3), X: 점 (K,3), err: 재투영 오차"""
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    up0 = -np.mean(cam_down, 0); up0 /= np.linalg.norm(up0)
    good = err < 2.0
    spread = np.median(np.linalg.norm(C - np.median(C, 0), axis=1))
    dist, _ = cKDTree(C).query(X)
    near = good & (dist < 3 * max(spread, 1e-9))
    Xn = X[near]
    if len(Xn) < 50:
        raise ValueError(f"카메라 근처 점이 너무 적음 ({len(Xn)})")
    h0 = (Xn - np.median(C, 0)) @ up0
    low = Xn[h0 < np.percentile(h0, 30)]
    # 1) 바닥 평면 RANSAC
    best = (0, None)
    thr0 = np.percentile(np.abs(h0 - np.median(h0)), 50) * 0.05
    for _ in range(3000):
        s = low[rng.choice(len(low), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0]); nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn
        if abs(n @ up0) < np.cos(np.radians(25)):
            continue
        k = (np.abs((low - s[0]) @ n) < thr0).sum()
        if k > best[0]:
            best = (k, (n, s[0]))
    if best[1] is None:
        raise ValueError("바닥 평면을 찾지 못함")
    n, p0 = best[1]
    Q = low[np.abs((low - p0) @ n) < thr0]
    c0 = Q.mean(0)
    up = np.linalg.svd(Q - c0, full_matrices=False)[2][2]
    if up @ up0 < 0:
        up = -up
    hcam = float(np.median((C - c0) @ up))
    if hcam <= 0:
        raise ValueError("카메라가 바닥 아래에 있음 (바닥 평면 추정 실패)")
    S = hcam / REF_CAM_HEIGHT
    log(f"바닥 평면: 점 {len(Q)}개, 카메라 높이 {hcam:.3f} -> 축척 {S:.2f}")
    Cc = C - C.mean(0); Cc -= np.outer(Cc @ up, up)
    ax1 = _pca_axis(Cc, up); ax2 = np.cross(up, ax1)
    Mw = np.stack([ax1, ax2, up], 1)
    F = lambda p: (p - c0) @ Mw
    Xf, Cf = F(Xn), F(C)
    # 2) 평면도 격자 + 천장
    cell = 0.04 * S
    xy = Xf[np.abs(Xf[:, 2]) < 0.03 * S, :2]
    allxy = np.concatenate([xy, Cf[:, :2]])
    mn = allxy.min(0) - 0.5 * S; mx = allxy.max(0) + 0.5 * S
    Wd = np.ceil((mx - mn) / cell).astype(int)
    g = np.zeros(Wd[::-1], bool)
    ij = ((allxy - mn) / cell).astype(int); g[ij[:, 1], ij[:, 0]] = True
    g = ndi.binary_closing(g, np.ones((7, 7)), iterations=2); g = ndi.binary_fill_holes(g)
    lab, nl = ndi.label(g)
    g = lab == (1 + np.argmax(ndi.sum(g, lab, range(1, nl + 1))))
    g = ndi.binary_opening(g, np.ones((5, 5)))
    gi = np.clip(((Xf[:, :2] - mn) / cell).astype(int), 0, np.array(g.shape[::-1]) - 1)
    inside = g[gi[:, 1], gi[:, 0]]
    zin = Xf[inside & (Xf[:, 2] > hcam), 2]
    H = float(np.percentile(zin, 99)) if len(zin) > 100 else 2.4 * hcam
    H = float(np.clip(H, 1.6 * hcam, 4.0 * hcam))
    log(f"평면도 넓이 {g.sum() * cell * cell:.2f}, 천장 {H:.3f} (카메라 높이 x {H / hcam:.2f})")
    # 3) 샘플
    SP = 0.025 * S
    xs = np.arange(mn[0], mn[0] + g.shape[1] * cell, SP); ys = np.arange(mn[1], mn[1] + g.shape[0] * cell, SP)
    XX, YY = np.meshgrid(xs, ys)
    ins = g[np.clip(((YY - mn[1]) / cell).astype(int), 0, g.shape[0] - 1),
            np.clip(((XX - mn[0]) / cell).astype(int), 0, g.shape[1] - 1)]
    fx, fy = XX[ins], YY[ins]
    P = [np.stack([fx, fy, np.zeros_like(fx)], 1), np.stack([fx, fy, np.full_like(fx, H)], 1)]
    N = [np.tile([0, 0, 1.], (len(fx), 1)), np.tile([0, 0, -1.], (len(fx), 1))]
    K = [np.full(len(fx), FLOOR), np.full(len(fx), CEILING)]
    zs = np.arange(SP / 2, H, SP); wp, wn = [], []
    for di, dj, nrm in [(0, 1, (-1, 0)), (0, -1, (1, 0)), (1, 0, (0, -1)), (-1, 0, (0, 1))]:
        edge = g & ~np.roll(g, (-di, -dj), (0, 1)); r, c = np.nonzero(edge)
        cx = mn[0] + (c + 0.5) * cell + dj * cell / 2; cy = mn[1] + (r + 0.5) * cell + di * cell / 2
        for t in np.arange(-cell / 2 + SP / 4, cell / 2, SP):
            ex = cx + (t if di != 0 else 0); ey = cy + (t if dj != 0 else 0)
            for zz in zs:
                wp.append(np.stack([ex, ey, np.full_like(ex, zz)], 1)); wn.append(np.tile([nrm[0], nrm[1], 0.], (len(ex), 1)))
    if wp:
        Wp = np.concatenate(wp); P.append(Wp); N.append(np.concatenate(wn)); K.append(np.full(len(Wp), WALL))
    P = np.concatenate(P); N = np.concatenate(N)
    return Samples(c0 + P @ Mw.T, N @ Mw.T, np.concatenate(K), up, ax1, S, SP,
                   dict(mode="points", camera_height=hcam, ceiling=H, footprint_area=float(g.sum() * cell * cell)))


# ---------------- 도면 방 ----------------
def points_in_polygon(x, y, poly) -> np.ndarray:
    """짝홀 규칙 (schemas.Room.contains 와 같음), 벡터화."""
    inside = np.zeros(np.shape(x), bool)
    pts = list(poly)
    for i in range(len(pts)):
        (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % len(pts)]
        cond = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cond & (x < xc)
    return inside


def rooms_of_cameras(rooms, C, min_frac: float = 0.05) -> list:
    """카메라가 min_frac 이상(최소 1대) 들어 있는 방."""
    out = []
    for r in rooms:
        k = int(points_in_polygon(C[:, 0], C[:, 1], r.floor).sum())
        if k >= max(1, min_frac * len(C)):
            out.append(r)
    return out


def samples_from_plan(rooms, C, X=None, log: Callable = lambda m: None) -> Samples:
    """도면 좌표(z 위)로 정합된 구간. rooms: schemas.Room 목록 (이 구간이 찍은 방)."""
    if not rooms:
        raise ValueError("카메라가 들어 있는 방이 없음")
    up = np.array([0, 0, 1.0])
    hs = []
    for r in rooms:
        m = points_in_polygon(C[:, 0], C[:, 1], r.floor)
        if m.any():
            hs.append(np.median(C[m, 2]) - r.floor_z)
    hcam = float(np.median(hs)) if hs else 1.4
    if hcam <= 0:
        raise ValueError(f"카메라가 바닥(floor_z) 아래에 있음 (높이 {hcam:.2f})")
    S = hcam / REF_CAM_HEIGHT
    SP = 0.025 * S
    Cc = C - C.mean(0); Cc[:, 2] = 0
    ax1 = _pca_axis(Cc, up)
    P, N, K, ceil_info = [], [], [], {}
    for r in rooms:
        poly = np.asarray(r.floor, float)
        fz = r.floor_z
        if r.ceiling_z is not None:
            cz = float(r.ceiling_z)
        else:
            cz = fz + 2.4 * hcam
            if X is not None:
                m = points_in_polygon(X[:, 0], X[:, 1], poly) & (X[:, 2] > fz + hcam)
                if m.sum() > 100:
                    cz = float(np.percentile(X[m, 2], 99))
            cz = float(np.clip(cz, fz + 1.6 * hcam, fz + 4.0 * hcam))
            log(f"방 {r.id}: ceiling_z 없음 -> 점에서 추정 {cz:.2f}")
        ceil_info[r.id] = (fz, cz)
        mn, mx = poly.min(0), poly.max(0)
        XX, YY = np.meshgrid(np.arange(mn[0] + SP / 2, mx[0], SP), np.arange(mn[1] + SP / 2, mx[1], SP))
        ins = points_in_polygon(XX, YY, poly)
        fx, fy = XX[ins], YY[ins]
        P += [np.stack([fx, fy, np.full_like(fx, fz)], 1), np.stack([fx, fy, np.full_like(fx, cz)], 1)]
        N += [np.tile([0, 0, 1.], (len(fx), 1)), np.tile([0, 0, -1.], (len(fx), 1))]
        K += [np.full(len(fx), FLOOR), np.full(len(fx), CEILING)]
        zs = np.arange(fz + SP / 2, cz, SP)
        for i in range(len(poly)):                         # 반시계 다각형 -> 안쪽 법선 = 변 방향 왼쪽
            a, b = poly[i], poly[(i + 1) % len(poly)]
            L = np.linalg.norm(b - a)
            if L < 1e-9:
                continue
            d = (b - a) / L; nin = np.array([-d[1], d[0], 0.0])
            ts = np.arange(SP / 2, L, SP)
            ex, ey = a[0] + d[0] * ts, a[1] + d[1] * ts
            EX, ZZ = np.meshgrid(ex, zs); EY, _ = np.meshgrid(ey, zs)
            P.append(np.stack([EX.ravel(), EY.ravel(), ZZ.ravel()], 1))
            N.append(np.tile(nin, (EX.size, 1))); K.append(np.full(EX.size, WALL))
    log(f"도면 방 {[r.id for r in rooms]}, 카메라 높이 {hcam:.2f} -> 축척 {S:.2f}")
    return Samples(np.concatenate(P), np.concatenate(N), np.concatenate(K), up, ax1, S, SP,
                   dict(mode="plan", camera_height=hcam, rooms={k: list(v) for k, v in ceil_info.items()}))


# ---------------- 사진 투영으로 색 ----------------
def colorize(sm: Samples, images: list[dict], cam: dict, image_dir: Path, step: int = 6,
             log: Callable = lambda m: None) -> np.ndarray:
    """사진 1/4 크기로 샘플마다 가장 정면·가까운 6장의 색 중간값. 못 본 샘플은 가장 가까운 색. (N,3) 0..1"""
    import cv2
    from scipy.spatial import cKDTree

    if cam["model"] not in (0, 1):
        raise ValueError("왜곡 보정된 PINHOLE/SIMPLE_PINHOLE 카메라만 지원 (data/ 의 모델을 쓸 것)")
    pp = cam["params"]
    fxp, fyp, cxp, cyp = (pp[0], pp[0], pp[1], pp[2]) if cam["model"] == 0 else pp
    Pw, Nw, S = sm.P, sm.N, sm.S
    bestS = np.full((len(Pw), 6), np.inf); cols = np.zeros((len(Pw), 6, 3), np.float32)
    views = sorted(images, key=lambda d: d["name"])[::max(1, step)]
    used = 0
    for im in views:
        fp = Path(image_dir) / im["name"]
        if not fp.is_file():
            continue
        # cv2.imread 는 한글 경로를 못 읽음 -> fromfile + imdecode
        img = cv2.imdecode(np.fromfile(str(fp), np.uint8), cv2.IMREAD_REDUCED_COLOR_4)
        if img is None:
            continue
        used += 1
        h, w = img.shape[:2]; rs = cam["w"] / w
        Xc = Pw @ im["R"].T + im["t"]; zc = Xc[:, 2]; ok = zc > 0.05 * S
        zsafe = np.maximum(zc, 1e-6)
        u = (fxp * Xc[:, 0] / zsafe + cxp) / rs; v = (fyp * Xc[:, 1] / zsafe + cyp) / rs
        ok &= (u > 2) & (u < w - 3) & (v > 2) & (v < h - 3)
        Cw = -im["R"].T @ im["t"]; vd = Cw - Pw; dd = np.linalg.norm(vd, axis=1)
        cs = np.einsum("ij,ij->i", vd, Nw) / np.maximum(dd, 1e-12)
        ok &= cs > 0.15
        idx = np.nonzero(ok)[0]
        if len(idx) == 0:
            continue
        sc = dd[idx] / cs[idx] ** 0.5
        cc = img[v[idx].astype(int), u[idx].astype(int)][:, ::-1].astype(np.float32) / 255
        worst = np.argmax(bestS[idx], 1); bt = sc < bestS[idx, worst]
        bestS[idx[bt], worst[bt]] = sc[bt]; cols[idx[bt], worst[bt]] = cc[bt]
    have = np.isfinite(bestS)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            col = np.nanmedian(np.where(have[..., None], cols, np.nan), 1)
    okc = ~np.isnan(col[:, 0])
    if not okc.any():
        log(f"경고: 사진 {used}장에서 색을 못 얻음 -> 회색")
        return np.full((len(Pw), 3), 0.5, np.float32)
    if (~okc).any():
        col[~okc] = col[okc][cKDTree(Pw[okc]).query(Pw[~okc])[1]]
    log(f"색: 사진 {used}장, 못 본 샘플 {(~okc).mean() * 100:.1f}%")
    return col.astype(np.float32)


# ---------------- 바탕층 스플랫 ----------------
def to_splats(sm: Samples, col: np.ndarray, names: list[str] | None = None) -> plyio.GaussianPly:
    """불투명 납작 스플랫 (면 바로 뒤). names 를 주면 그 속성(학습 PLY 와 같게), 없으면 Brush SH2 순서."""
    Nw = sm.N
    pos = sm.P - Nw * OFFSET * sm.S
    t1 = np.where(np.abs(Nw @ sm.up)[:, None] > 0.9, np.tile(sm.ax1, (len(Nw), 1)), np.tile(sm.up, (len(Nw), 1)))
    t1 = t1 - np.einsum("ij,ij->i", t1, Nw)[:, None] * Nw; t1 /= np.linalg.norm(t1, axis=1)[:, None]
    Rm = np.stack([t1, np.cross(Nw, t1), Nw], 2)
    q = plyio.mat_to_quat(Rm)
    ply = plyio.make(len(pos), names or plyio.brush_names(2),
                     ["comment IndoorGS backdrop (tools/fill/fill_section.py)"])
    d = ply.data
    for i in range(3):
        d[f"f_dc_{i}"] = (col[:, i] - 0.5) / plyio.SH_C0
    d["opacity"] = np.log(0.99 / 0.01)
    for i in range(4):
        d[f"rot_{i}"] = q[:, i]
    d["scale_0"] = np.log(sm.SP * 0.7); d["scale_1"] = np.log(sm.SP * 0.7); d["scale_2"] = np.log(2e-4 * sm.S)
    for i, k in enumerate("xyz"):
        d[k] = pos[:, i]
    if "nx" in d.dtype.names:
        pass   # 법선은 학습기들이 0 으로 씀. 그대로 0
    return ply


# ---------------- 구멍만 고르기 (merge_holes.py) ----------------
def hole_mask(train: plyio.GaussianPly, backdrop_xyz: np.ndarray, kind: np.ndarray, thr: float = 1.0,
              rmul: float = 3.0, keep_walls: bool = False, log: Callable = lambda m: None) -> np.ndarray:
    """바탕층 점마다 반경(바탕층 점 간격 x rmul) 안 학습 스플랫(불투명도 > 0.05)의 불투명도 합 < thr 이면 구멍.
    벽 바탕층은 기본 제외 (복도 1: 평면도 경계 벽이 실제 벽과 80% 어긋나 반투명 벽 뒤로 비쳐 뭉개짐)."""
    from scipy.spatial import cKDTree

    Q = backdrop_xyz
    if len(Q) < 2:
        return np.zeros(len(Q), bool)
    sp = float(np.median(cKDTree(Q).query(Q, k=2)[0][:, 1]))
    r = sp * rmul
    op = train.opacity()
    keep = op > 0.05
    cover = np.zeros(len(Q))
    if keep.any():
        tree = cKDTree(train.xyz[keep]); opk = op[keep]
        for s in range(0, len(Q), 20000):
            for j, idx in enumerate(tree.query_ball_point(Q[s:s + 20000], r)):
                cover[s + j] = opk[idx].sum()
    hole = cover < thr
    if not keep_walls:
        hole &= kind != WALL
    log(f"구멍 고르기: 간격 {sp:.4f}, 반경 {r:.4f}, 기준 {thr} -> {hole.sum():,}/{len(Q):,} "
        f"(바닥 {(hole & (kind == FLOOR)).sum():,}, 천장 {(hole & (kind == CEILING)).sum():,}, 벽 {(hole & (kind == WALL)).sum():,})")
    return hole
