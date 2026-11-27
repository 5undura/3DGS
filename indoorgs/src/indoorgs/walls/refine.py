"""정합 점(COLMAP points3D)으로 도면 면의 실제 위치를 다시 맞추기.

줄자·박스 도면은 몇 cm~수십 cm 틀리고, 위치 사전값(σ≈0.5 m)으로 맞춘 정합 축척도 몇 % 틀릴 수 있음.
그래서 면마다 성긴 점으로 위치(법선 방향 이동)와 작은 기울기를 다시 맞춘 뒤 씀.

공개 API
  assign_points(planes, xyz, search, uv_margin)  -> (N,) 면 번호 (-1 = 어느 면에도 안 속함)
  refine_planes(planes, xyz, ...)                 -> (맞춘 면 목록, 면별 정보 dict 목록)
  room_widths(planes_before, planes_after)        마주 보는 벽 사이 폭 (도면 vs 맞춤) - 전체 축척 오차 확인용
  flatness(xyz, planes, band)                     면 근처 점의 |거리| RMS·중앙값 (점 기준 평탄도)

맞추기 방법 (면마다)
  면 2D 좌표 (u, v) 와 도면 면까지의 부호 거리 s 로 s = a + b·u + c·v 를 맞춤 (u, v 는 면 가운데 기준).
  1) RANSAC: 점 3개 가설, 기울기 atan(√(b²+c²)) 가 2·max_tilt 넘는 가설은 버림, |잔차| < inlier_thr 개수 최대.
  2) IRLS (Huber, k = inlier_thr/2) 로 다듬기.
  3) 기울기가 max_tilt_deg 를 넘으면 (b, c) 를 그 각도로 줄이고 a 만 다시 맞춤 (clamped=True).
  결과 평면의 법선 = n - b·U - c·V (정규화). shift_m = 면 가운데에서의 이동량 a (도면 법선 방향, + = 방 안쪽).
  덮임(coverage): 면을 cell 크기 격자로 나눠 면 안(contains_uv) 칸 중 inlier 가 있는 칸 비율.
  모서리 띠(edge_band, 바깥 경계에서 0.15 m) 안의 점은 이웃 면과 섞이므로 맞추기·덮임에 안 씀.
  inlier < min_points 또는 coverage < min_coverage 또는 |shift| ≥ 0.9·search 면 도면 면 그대로 (source "plan").
"""
from __future__ import annotations

import math

import numpy as np

from .planes import Plane, WALL, better_face

__all__ = ["assign_points", "refine_planes", "room_widths", "flatness", "face_coverage", "edge_distance"]


def assign_points(planes: list[Plane], xyz: np.ndarray, search: float = 0.30, uv_margin: float = 0.10) -> np.ndarray:
    """점마다 |부호 거리| < search 이고 면 범위(uv_margin 허용, 문 구멍 제외) 안인 면 중 가장 가까운 면 번호.
    두 방이 공유하는 벽(등을 맞댄 두 면)은 거리 대신 점이 어느 쪽에 있는지로 나눔 (planes.better_face)."""
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    best_sd = np.full(len(xyz), np.nan)
    idx = np.full(len(xyz), -1, np.int64)
    normals = np.array([p.n for p in planes], float).reshape(-1, 3)
    for k, p in enumerate(planes):
        sd = p.signed_dist(xyz)
        cand = np.flatnonzero(np.abs(sd) < search)
        if len(cand) == 0:
            continue
        cur = idx[cand]
        has = cur >= 0
        win = better_face(sd[cand], p.n, best_sd[cand], normals[np.maximum(cur, 0)], has)
        cand = cand[win]
        if len(cand) == 0:
            continue
        ok = p.contains_uv(p.uv(xyz[cand]), uv_margin)
        cand = cand[ok]
        best_sd[cand] = sd[cand]
        idx[cand] = k
    return idx


def face_coverage(p: Plane, uv: np.ndarray, cell: float = 0.5) -> float:
    """면 안 격자 칸 중 uv 점이 하나라도 있는 칸 비율."""
    nu = max(1, int(math.ceil(p.size[0] / cell)))
    nv = max(1, int(math.ceil(p.size[1] / cell)))
    cu = (np.arange(nu) + 0.5) * p.size[0] / nu
    cv = (np.arange(nv) + 0.5) * p.size[1] / nv
    CU, CV = np.meshgrid(cu, cv, indexing="ij")
    valid = p.contains_uv(np.stack([CU.ravel(), CV.ravel()], 1)).reshape(nu, nv)
    if not valid.any():
        return 0.0
    uv = np.asarray(uv, float).reshape(-1, 2)
    if len(uv) == 0:
        return 0.0
    iu = np.clip((uv[:, 0] / p.size[0] * nu).astype(int), 0, nu - 1)
    iv = np.clip((uv[:, 1] / p.size[1] * nv).astype(int), 0, nv - 1)
    hit = np.zeros((nu, nv), bool)
    hit[iu, iv] = True
    return float((hit & valid).sum() / valid.sum())


def edge_distance(p: Plane, uv: np.ndarray) -> np.ndarray:
    """면 바깥 경계(벽: 직사각형, 바닥·천장: 방 다각형)까지 거리. 문 구멍 가장자리는 안 셈."""
    uv = np.atleast_2d(np.asarray(uv, float))
    if p.polygon is None:
        return np.minimum.reduce([uv[:, 0], p.size[0] - uv[:, 0], uv[:, 1], p.size[1] - uv[:, 1]])
    P = p.point(uv[:, 0], uv[:, 1])
    x, y = P[:, 0], P[:, 1]
    best = np.full(len(uv), np.inf)
    n = len(p.polygon)
    for i in range(n):
        a = np.array(p.polygon[i], float)
        b = np.array(p.polygon[(i + 1) % n], float)
        ab = b - a
        t = np.clip(((x - a[0]) * ab[0] + (y - a[1]) * ab[1]) / max(ab @ ab, 1e-12), 0, 1)
        best = np.minimum(best, np.hypot(x - a[0] - t * ab[0], y - a[1] - t * ab[1]))
    return best


def _wlsq(A: np.ndarray, s: np.ndarray, w: np.ndarray) -> np.ndarray:
    sw = np.sqrt(w)
    sol, *_ = np.linalg.lstsq(A * sw[:, None], s * sw, rcond=None)
    return sol


def _irls(A, s, w, sol, thr, iters=15):
    """Huber IRLS (k = thr/2, 3·thr 넘는 잔차는 버림). sol 에서 시작."""
    k = thr / 2
    for _ in range(iters):
        r = np.abs(A @ sol - s)
        hw = np.where(r <= k, 1.0, k / np.maximum(r, 1e-12)) * (r < 3 * thr)
        if (hw > 0).sum() < A.shape[1]:
            break
        new = _wlsq(A, s, w * hw)
        if np.allclose(new, sol, atol=1e-7):
            sol = new
            break
        sol = new
    return sol


def _clamp_tilt(A, s, w, u, v, sol, thr, max_tan):
    t = math.hypot(sol[1], sol[2])
    if t <= max_tan:
        return sol, False
    sol = sol.copy()
    sol[1:] *= max_tan / t
    for _ in range(10):
        r = s - sol[1] * u - sol[2] * v
        m = np.abs(r - sol[0]) < thr
        if not m.any():
            break
        sol[0] = float(np.average(r[m], weights=w[m]))
    return sol, True


def _cell_keys(u, v, cell):
    iu = np.floor((u - u.min()) / cell).astype(np.int64)
    iv = np.floor((v - v.min()) / cell).astype(np.int64)
    return iu, iv, int(iu.max()) + 1, int(iv.max()) + 1


def _refit(A, s, w, keep, sol, thr, cols):
    """keep 점으로 IRLS, cols (예: [0], [0, 1], [0, 1, 2]) 열만 풀고 나머지 계수는 0."""
    cols = list(cols)
    sub = _irls(A[keep][:, cols], s[keep], w[keep], sol[cols], thr)
    out = np.zeros(3)
    out[cols] = sub
    return out


def _front_patches(u, v, s, w, sol, *, cell, min_cell_pts, front_min, max_protrude, max_front_area):
    """1단계 (덩어리 단위): 점이 있는 칸을 8-이웃으로 이은 덩어리마다 잔차 중앙값. 다른 덩어리들(사각형 넓이 가중
    중앙값)보다 뚜렷이 앞이고 (차 > max(front_min, 2·(se_k + se_rest))), 사각형 넓이가 전체의 max_front_area 미만인
    덩어리 = 튀어나온 물체. 벽 점이 성기고 포스터 점이 촘촘하면 기울기까지 포스터에 맞춰져 칸 단위 잔차로는 안 보이므로
    (wall_ab corr:w0: 기울기 0.74° 가 포스터를 비탈로 설명) 기울기 없는 잔차로 덩어리끼리 비교."""
    from scipy import ndimage

    r = s - sol[0]
    cand = np.abs(r - np.median(r)) < max_protrude
    iu, iv, nu, nv = _cell_keys(u, v, cell)
    cnt = np.zeros((nu, nv), np.int64)
    np.add.at(cnt, (iu[cand], iv[cand]), 1)
    occ = cnt >= min_cell_pts
    lab, nl = ndimage.label(occ, structure=np.ones((3, 3)))
    excl = np.zeros(len(s), bool)
    if nl < 2:
        return excl
    oi, oj = np.nonzero(occ)
    box_all = (oi.max() - oi.min() + 1) * (oj.max() - oj.min() + 1)
    pl = lab[iu, iv]
    meds, ses, boxes = [], [], []
    for li, sl in enumerate(ndimage.find_objects(lab), start=1):
        m = cand & (pl == li)
        rk = r[m]
        meds.append(float(np.median(rk)))
        sig = 1.4826 * float(np.median(np.abs(rk - np.median(rk)))) + 0.003
        ses.append(1.25 * sig / math.sqrt(len(rk)))
        boxes.append((sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start))
    meds, ses, boxes = np.array(meds), np.array(ses), np.array(boxes, float)
    for k in range(nl):
        if boxes[k] >= max_front_area * box_all:
            continue
        o = np.arange(nl) != k
        mo = meds[o]
        order = np.argsort(mo)
        cw = np.cumsum(boxes[o][order])
        jm = order[np.searchsorted(cw, 0.5 * cw[-1])]
        rest = mo[jm]
        se_rest = ses[o][jm]
        if meds[k] - rest > max(front_min, 2 * (ses[k] + se_rest)):
            excl |= cand & (pl == k + 1)
    return excl


def _backmost(u, v, s, w, sol, thr, *, cell=0.25, min_cell_pts=3, front_min=0.006, max_protrude=0.12,
              max_front_area=0.5, iters=12, offset_only=False):
    """맞춘 면 앞(방 안쪽)에 몰린 점 무리(포스터·액자 등 튀어나온 물체) 를 빼고 가장 뒤쪽 층에 다시 맞춤.
    (11-28 wall_ab 버그 2: 흰 벽은 점이 없어 포스터 앞면 점이 벽 점의 절반 넘음 -> 면이 0.76° 기울고 1 cm 앞으로)

    1단계 _front_patches (덩어리 단위, 기울기 없는 잔차). 2단계 (칸 단위, 지금 맞춤 기준): 면을 cell(m) 칸으로
    나눠 칸마다 잔차 중앙값 m_c (점 min_cell_pts 개 이상). m_c > max(front_min, 2·se_c) (se_c = 1.25·σ/√n_c,
    σ = 잔차 MAD) 이면 '앞 칸'. 앞 칸을 8-이웃으로 묶은 덩어리마다, 그 덩어리를 감싸는 사각형 넓이가 점이 있는
    모든 칸을 감싸는 사각형 넓이의 max_front_area 미만일 때만 뺌. 뺄 때마다 IRLS 로 다시 맞춤 (안 바뀔 때까지).
    즉 '면 일부에만 있는 앞쪽 층' 만 물체로 봄: 벽은 면에 넓게 퍼져 있고 포스터·액자는 좁은 곳에 몰림. 점 개수는
    보지 않음 (흰 벽은 점이 적어 포스터 점이 더 많을 수 있음). 2단계는 같은 규칙으로 면 일부에만 있는 '뒤 칸'
    (움푹 들어간 문·창) 도 뺌 -> 넓게 퍼진 층이 벽. 벽 대부분에 걸친 층은 앞이든 뒤든 빼지 않음. 남은 점이 면 가로(세로)의 절반 미만에만 있으면 그 방향 기울기는 0 으로 묶음.
    offset_only: 기울기 0 으로 묶고 a 만. 반환 (sol, 뺀 점 마스크, 뺀 앞 칸 수)."""
    from scipy import ndimage

    n = len(s)
    A = np.stack([np.ones(n), u, v], 1)
    su0 = (u.max() - u.min()) if n else 0.0
    sv0 = (v.max() - v.min()) if n else 0.0

    def cols_for(keep):
        if offset_only:
            return [0]
        c = [0]
        if su0 > 0 and np.ptp(u[keep]) >= 0.5 * su0:
            c.append(1)
        if sv0 > 0 and np.ptp(v[keep]) >= 0.5 * sv0:
            c.append(2)
        return c

    excl = _front_patches(u, v, s, w, sol, cell=cell, min_cell_pts=min_cell_pts, front_min=front_min,
                          max_protrude=max_protrude, max_front_area=max_front_area)
    if excl.any() and (~excl).sum() >= 10:
        sol = _refit(A, s, w, ~excl, sol, thr, cols_for(~excl))
    else:
        excl = np.zeros(n, bool)
    patch_excl = excl.copy()
    iu, iv, nu, nv = _cell_keys(u, v, cell)
    key = iu * nv + iv
    n_front = 0
    for _ in range(iters):
        r = s - A @ sol
        cand = (np.abs(r) < max_protrude) & ~patch_excl
        if cand.sum() < 10:
            break
        sig = 1.4826 * np.median(np.abs(r[cand] - np.median(r[cand]))) + 1e-4
        ks, inv, cnt = np.unique(key[cand], return_inverse=True, return_counts=True)
        rc = r[cand]
        med = np.array([np.median(rc[inv == i]) for i in range(len(ks))])
        occ = cnt >= min_cell_pts
        if occ.sum() < 2:
            break
        se = 1.25 * sig / np.sqrt(np.maximum(cnt, 1))
        thr_c = np.maximum(front_min, 2 * se)
        occ_g = np.zeros((nu, nv), bool)
        occ_g[ks[occ] // nv, ks[occ] % nv] = True
        oi, oj = np.nonzero(occ_g)
        box_all = (oi.max() - oi.min() + 1) * (oj.max() - oj.min() + 1)
        drop_g = np.zeros((nu, nv), bool)
        # 앞쪽 층(튀어나온 물체)과, 같은 규칙으로 면 일부에만 있는 뒤쪽 층(움푹 들어간 문·창)도 뺌: 넓게 퍼진 층 = 벽
        for side in (occ & (med > thr_c), occ & (med < -thr_c)):
            fr_g = np.zeros((nu, nv), bool)
            fr_g[ks[side] // nv, ks[side] % nv] = True
            lab, nl = ndimage.label(fr_g, structure=np.ones((3, 3)))
            for li, sl in enumerate(ndimage.find_objects(lab), start=1):
                if sl is None:
                    continue
                box = (sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start)
                if box < max_front_area * box_all:
                    drop_g[sl][lab[sl] == li] = True
        new_excl = patch_excl.copy()
        ci = np.flatnonzero(cand)
        new_excl[ci] |= drop_g[iu[ci], iv[ci]]
        if np.array_equal(new_excl, excl):
            break
        keep = ~new_excl
        if keep.sum() < 10:
            break
        sol = _refit(A, s, w, keep, sol, thr, cols_for(keep))
        excl = new_excl
        n_front = int(drop_g.sum())
    return sol, excl, n_front


def _fit_face(u, v, s, w, thr, max_tan, rng, iters=200, backmost=True, offset_only=False):
    """s = a + b u + c v 강건 맞춤. 돌려줌: (a, b, c, inlier 마스크, clamped, 뺀 앞쪽 점 마스크).
    offset_only: 기울기 0 으로 a 만 (점이 면 한쪽 줄에만 있어 기울기를 알 수 없을 때)."""
    n = len(s)
    A = np.stack([np.ones(n), u, v], 1)
    if offset_only:
        A1 = A[:, :1]
        best = np.array([np.median(s)])
        best_cnt = -1
        for _ in range(min(iters, n)):
            c = s[rng.integers(n)]
            cnt = int((np.abs(s - c) < thr).sum())
            if cnt > best_cnt:
                best_cnt, best = cnt, np.array([c])
        sol1 = _irls(A1, s, w, best, thr)
        sol = np.array([sol1[0], 0.0, 0.0])
        excl = np.zeros(n, bool)
        if backmost:
            sol, excl, _ = _backmost(u, v, s, w, sol, thr, offset_only=True)
        inl = np.abs(A @ sol - s) < thr
        return float(sol[0]), 0.0, 0.0, inl, False, excl
    best_cnt, best = -1, None
    for _ in range(iters):
        i3 = rng.choice(n, 3, replace=False)
        try:
            sol = np.linalg.solve(A[i3], s[i3])
        except np.linalg.LinAlgError:
            continue
        if math.hypot(sol[1], sol[2]) > 2 * max_tan:
            continue
        cnt = int((np.abs(A @ sol - s) < thr).sum())
        if cnt > best_cnt:
            best_cnt, best = cnt, sol
    if best is None:   # 모든 가설이 기울기 제한에 걸림: 기울기 0, 중앙값
        best = np.array([np.median(s), 0.0, 0.0])
    sol = _irls(A, s, w, best, thr)
    excl = np.zeros(n, bool)
    if backmost:
        sol, excl, _ = _backmost(u, v, s, w, sol, thr)
    keep = ~excl
    sol, clamped = _clamp_tilt(A[keep], s[keep], w[keep], u[keep], v[keep], sol, thr, max_tan)
    inl = np.abs(A @ sol - s) < thr
    return float(sol[0]), float(sol[1]), float(sol[2]), inl, clamped, excl


def _span(x: np.ndarray, size: float) -> float:
    """점들이 면 한 축에서 차지하는 범위 (5~95 백분위) / 면 크기."""
    if len(x) < 2 or size <= 0:
        return 0.0
    lo, hi = np.percentile(x, [5, 95])
    return float((hi - lo) / size)


def _fit_plane(p: Plane, P: np.ndarray, w: np.ndarray, *, relaxed: bool, inlier_thr: float, max_tan: float, rng,
               min_points: int, min_coverage: float, cell: float, search: float, backmost: bool,
               relax_min_points: int, relax_min_coverage: float, relax_max_se: float, relax_min_span: float,
               relax_min_inlier_frac: float) -> tuple[Plane | None, dict]:
    """면 하나 맞추기. 반환 (맞춘 면 또는 None, 정보). relaxed: 점이 적어도 면에 넓게 퍼져 있고 잘 맞으면 받아줌."""
    info: dict = {"n_points": 0, "coverage": 0.0, "shift_m": 0.0, "tilt_deg": 0.0, "clamped": False,
                  "rms_before_m": None, "rms_after_m": None}
    need = relax_min_points if relaxed else min_points
    if len(P) < max(3, need):
        info["reason"] = f"점 부족 ({len(P)} < {need})"
        return None, info
    uv = p.uv(P)
    s = p.signed_dist(P)
    uc, vc = p.size[0] / 2, p.size[1] / 2
    su_all, sv_all = _span(uv[:, 0], p.size[0]), _span(uv[:, 1], p.size[1])
    offset_only = relaxed and min(su_all, sv_all) < 0.5      # 한 줄(걸레받이 등)에만 점: 기울기는 모름 -> 0
    a, b, c, inl, clamped, excl = _fit_face(uv[:, 0] - uc, uv[:, 1] - vc, s, w, inlier_thr, max_tan, rng,
                                           backmost=backmost, offset_only=offset_only)
    res = s - (a + b * (uv[:, 0] - uc) + c * (uv[:, 1] - vc))
    cov = face_coverage(p, uv[inl], cell)
    n_in = int(inl.sum())
    tilt = math.degrees(math.atan(math.hypot(b, c)))
    rms = float(np.sqrt(np.mean(res[inl] ** 2))) if n_in else None
    info.update(n_points=n_in, coverage=round(cov, 4), shift_m=a, tilt_deg=tilt, clamped=clamped,
                rms_before_m=float(np.sqrt(np.mean(s[inl] ** 2))) if n_in else None, rms_after_m=rms,
                n_front_excluded=int(excl.sum()), offset_only=bool(offset_only))
    if abs(a) >= 0.9 * search:
        info["reason"] = f"이동량이 탐색 한계에 닿음 ({a:+.3f} m, search {search})"
        info["at_bound"] = True
        return None, info
    ok = n_in >= min_points and cov >= min_coverage
    if not ok and relaxed:
        su, sv = (_span(uv[inl, 0], p.size[0]), _span(uv[inl, 1], p.size[1])) if n_in else (0.0, 0.0)
        se = (rms / math.sqrt(n_in)) if n_in else 1.0
        frac = n_in / max(1, len(P))
        info.update(span_u=round(su, 3), span_v=round(sv, 3), offset_se_m=se, inlier_frac=round(frac, 3))
        ok = (n_in >= relax_min_points and cov >= relax_min_coverage and se <= relax_max_se
              and frac >= relax_min_inlier_frac and max(su, sv) >= relax_min_span)
        if not ok:
            info["reason"] = (f"느슨한 기준도 못 넘음 (inlier {n_in}, 덮임 {cov:.2f}, 범위 {max(su, sv):.2f}, "
                              f"se {se * 100:.2f}cm, inlier 비율 {frac:.2f})")
            return None, info
        info["relaxed"] = True
    elif not ok:
        info["reason"] = (f"inlier 부족 ({n_in} < {min_points})" if n_in < min_points
                          else f"덮임 부족 ({cov:.2f} < {min_coverage})")
        return None, info
    n_new = p.n - b * p.U - c * p.V
    a0 = a - b * uc - c * vc          # 면 원점에서의 이동량 (Plane.shifted 는 원점 기준)
    return p.shifted(a0, n_new), info


def _infer_from_neighbours(k: int, planes: list[Plane], out: list[Plane], infos: list[dict], xyz: np.ndarray,
                           owner: np.ndarray, *, max_dev: float, inlier_thr: float, max_angle_deg: float = 2.0,
                           min_overlap: float = 0.3) -> tuple[Plane | None, dict]:
    """도면 치수를 믿을 때(trust_plan_dims) 못 맞춘 면 k 를 맞춘 평행 면(마주 보는 벽, 등을 맞댄 공유 벽, 같은 줄 벽)이
    옮겨 간 만큼 똑같이 옮김 (방이 통째로 평행 이동했다고 봄 = 도면 폭·벽 두께가 맞다고 가정).
    후보: source "fit" 인 면 중 법선이 max_angle_deg 안으로 평행(같은 쪽 또는 반대쪽)하고, 면 가로 범위가 min_overlap
    넘게 겹치는 것. 도면상 거리가 가장 가까운 것을 씀 (도면 치수 오차는 거리만큼 쌓이므로).
    면 k 에 점이 조금이라도 있으면 (추정 위치 ±3·inlier_thr 안 10 개 이상) 그 점들의 중앙값이 추정과 max_dev 넘게
    다르면 안 씀 (도면을 덜 믿음)."""
    p = planes[k]
    cos_min = math.cos(math.radians(max_angle_deg))
    c_p = p.point(p.size[0] / 2, p.size[1] / 2)
    best = None
    for j, q in enumerate(planes):
        if j == k or infos[j].get("source") != "fit" or q.kind != p.kind:
            continue
        cs = float(p.n @ q.n)
        if abs(cs) < cos_min:
            continue
        # 겹침: p 의 가로 범위를 q 의 U 축으로 옮겨 비교 (바닥·천장은 다각형이라 겹침 검사 생략)
        if p.kind == WALL:
            ends = np.array([p.point(0, 0), p.point(p.size[0], 0)])
            qu = np.sort(q.uv(ends)[:, 0])
            ov = min(qu[1], q.size[0]) - max(qu[0], 0.0)
            if ov < min_overlap * min(p.size[0], q.size[0]):
                continue
        dist = abs(float(q.signed_dist(c_p)))
        if best is None or dist < best[0]:
            best = (dist, j, cs)
    info: dict = {}
    if best is None:
        info["reason_infer"] = "평행한 맞춘 면 없음"
        return None, info
    dist, j, cs = best
    q, q2 = planes[j], out[j]
    x0 = c_p - q.signed_dist(c_p) * q.n                  # p 가운데를 도면 q 위로
    shift_q = -float(q2.signed_dist(x0))                  # q 가 자기 안쪽 법선으로 옮겨 간 양 (그 자리에서)
    shift = shift_q * cs
    pts = xyz[owner == k]
    if len(pts):
        sd = p.signed_dist(pts) - shift
        near = np.abs(sd) < 3 * inlier_thr
        if near.sum() >= 10 and abs(float(np.median(sd[near]))) > max_dev:
            info["reason_infer"] = (f"면 점 {int(near.sum())}개의 중앙값이 추정과 {float(np.median(sd[near])) * 100:+.1f}cm "
                                    f"다름 (> {max_dev * 100:.1f}cm)")
            return None, info
        info["n_check_points"] = int(near.sum())
    info.update(inferred_from=q.id, inferred_dist_m=round(dist, 3), shift_m=shift, tilt_deg=0.0)
    return p.shifted(shift), info


def refine_planes(planes: list[Plane], xyz: np.ndarray, *, weights: np.ndarray | None = None,
                  search: float = 0.30, max_tilt_deg: float = 3.0, min_points: int = 40,
                  min_coverage: float = 0.25, uv_margin: float = 0.10, inlier_thr: float = 0.03,
                  cell: float = 0.5, edge_band: float = 0.15, backmost: bool = True, relaxed: bool = True,
                  relax_edge_band: float = 0.05, relax_min_points: int = 20, relax_min_coverage: float = 0.08,
                  relax_max_se: float = 0.004, relax_min_span: float = 0.6, relax_min_inlier_frac: float = 0.6,
                  trust_plan_dims: bool = False, infer_max_dev: float = 0.015,
                  rng=0) -> tuple[list[Plane], list[dict]]:
    """도면 면 -> 점에 맞춘 면. 맞출 수 없는 면은 도면 면 그대로 (info["source"] == "plan").

    11-28 wall_ab 수정:
    - backmost (기본 켬): 튀어나온 물체(포스터·액자) 앞면 점을 빼고 가장 뒤쪽 층(벽)에 맞춤 (_backmost).
    - relaxed (기본 켬): 기본 기준(min_points, min_coverage, 모서리 띠 edge_band)으로 못 맞추면 모서리 띠를
      relax_edge_band 로 줄여(걸레받이·모서리 점) 다시. 그래도 덮임이 모자라면 inlier >= relax_min_points,
      덮임 >= relax_min_coverage, 이동량 표준오차(rms/√n) <= relax_max_se, inlier 비율 >= relax_min_inlier_frac,
      면 가로 또는 세로의 relax_min_span 이상에 퍼져 있을 때만 받음 (info["relaxed"]). 점이 한 줄에만 있으면
      (가로·세로 중 하나가 면의 절반 미만) 기울기 0 으로 이동량만 (info["offset_only"]).
    - trust_plan_dims (기본 끔, 줄자 도면을 믿을 때만): 그래도 못 맞춘 면을 맞춘 평행 면(마주 보는 벽, 공유 벽)이
      옮겨 간 만큼 옮김 (source "infer", info["inferred_from"]). 면에 점이 조금 있고 추정과 다르면 안 함 (plan 유지)."""
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    w_all = np.ones(len(xyz)) if weights is None else np.asarray(weights, float).reshape(-1)
    rng = np.random.default_rng(rng)
    max_tan = math.tan(math.radians(max_tilt_deg))
    owner = assign_points(planes, xyz, search, uv_margin)
    kw = dict(inlier_thr=inlier_thr, max_tan=max_tan, rng=rng, min_points=min_points, min_coverage=min_coverage,
              cell=cell, search=search, backmost=backmost, relax_min_points=relax_min_points,
              relax_min_coverage=relax_min_coverage, relax_max_se=relax_max_se, relax_min_span=relax_min_span,
              relax_min_inlier_frac=relax_min_inlier_frac)
    out, infos = [], []
    for k, p in enumerate(planes):                       # 1) 기본 기준
        sel = owner == k
        base = {"id": p.id, "kind": p.kind, "room": p.room, "source": "plan", "n_assigned": int(sel.sum())}
        if edge_band > 0 and sel.any():     # 모서리 띠의 점은 두 면 어느 쪽인지 애매 -> 맞추기에 안 씀
            j = np.flatnonzero(sel)
            sel[j[edge_distance(p, p.uv(xyz[j])) < edge_band]] = False
        fitted, info = _fit_plane(p, xyz[sel], w_all[sel], relaxed=False, **kw)
        info = {**base, **info, "edge_band": edge_band}
        if fitted is not None:
            info["source"] = "fit"
        out.append(fitted if fitted is not None else p)
        infos.append(info)
    if relaxed:                                          # 2) 모서리 띠 점 더 쓰기 (+ 못 맞춘 면은 느슨하게)
        fit1 = [(q, k) for k, q in enumerate(out) if infos[k]["source"] == "fit"]
        for k, p in enumerate(planes):
            was_fit = infos[k]["source"] == "fit"
            if infos[k].get("at_bound"):
                continue
            sel = owner == k
            band_k = min(edge_band, relax_edge_band)
            if sel.any():
                j = np.flatnonzero(sel)
                ed = edge_distance(p, p.uv(xyz[j]))
                drop = ed < band_k
                # 모서리 띠(edge_band) 안 점은 이미 맞춘 이웃 면 위(±inlier_thr)에 있으면 그 면 것으로 보고 뺌.
                # 걸레받이·문틀처럼 이웃 면에서 떨어진 점은 남음.
                mid = np.flatnonzero((ed < edge_band) & ~drop)
                for q, kq in fit1:
                    if kq == k:
                        continue
                    if len(mid) == 0:
                        break
                    X = xyz[j[mid]]
                    on = (np.abs(q.signed_dist(X)) < inlier_thr) & q.contains_uv(q.uv(X), uv_margin)
                    drop[mid[on]] = True
                    mid = mid[~on]
                sel[j[drop]] = False
            fitted, info = _fit_plane(p, xyz[sel], w_all[sel], relaxed=not was_fit, **kw)
            if fitted is None:
                if not was_fit:
                    infos[k]["reason_relaxed"] = info.get("reason")
                continue
            infos[k] = {**{x: infos[k][x] for x in ("id", "kind", "room", "n_assigned")}, **info,
                        "edge_band": band_k, "source": "fit"}
            infos[k].pop("reason", None)
            out[k] = fitted
    if trust_plan_dims:
        for k, p in enumerate(planes):
            if infos[k]["source"] != "plan" or infos[k].get("at_bound"):
                continue
            q, inf = _infer_from_neighbours(k, planes, out, infos, xyz, owner, max_dev=infer_max_dev,
                                            inlier_thr=inlier_thr)
            if q is None:
                infos[k]["reason_infer"] = inf.get("reason_infer")
                continue
            infos[k].update(inf)
            infos[k]["source"] = "infer"
            out[k] = q
    return out, infos


def room_widths(planes_before: list[Plane], planes_after: list[Plane], min_angle_deg: float = 175.0) -> list[dict]:
    """같은 방의 마주 보는(반대 법선) 벽 쌍마다 도면 폭과 맞춘 폭. ratio = fit/plan (1 에서 멀면 축척 오차 의심).
    폭 = 벽 a 의 면 가운데점에서 벽 b 까지 부호 거리."""
    cos_min = math.cos(math.radians(min_angle_deg))
    after = {p.id: p for p in planes_after}
    out = []
    walls = [p for p in planes_before if p.kind == WALL]
    for i, a in enumerate(walls):
        for b in walls[i + 1:]:
            if a.room != b.room or float(a.n @ b.n) > cos_min:
                continue
            ca = a.point(a.size[0] / 2, a.size[1] / 2)
            plan_w = float(b.signed_dist(ca))
            if plan_w <= 0:
                continue
            a2, b2 = after.get(a.id, a), after.get(b.id, b)
            fit_w = float(b2.signed_dist(a2.point(a2.size[0] / 2, a2.size[1] / 2)))
            out.append({"room": a.room, "a": a.id, "b": b.id, "plan_m": plan_w, "fit_m": fit_w,
                        "ratio": fit_w / plan_w})
    return out


def flatness(xyz: np.ndarray, planes: list[Plane], band: float = 0.05, uv_margin: float = 0.0) -> dict:
    """band 안 점의 가장 가까운 면 |거리| 통계. {"overall": {...}, "planes": {id: {...}}}; 값은 n, rms_m, median_m."""
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    owner = assign_points(planes, xyz, band, uv_margin)

    def stats(d):
        if len(d) == 0:
            return {"n": 0, "rms_m": None, "median_m": None}
        return {"n": int(len(d)), "rms_m": float(np.sqrt(np.mean(d ** 2))), "median_m": float(np.median(np.abs(d)))}

    per, alld = {}, []
    for k, p in enumerate(planes):
        d = p.signed_dist(xyz[owner == k])
        per[p.id] = stats(d)
        alld.append(d)
    return {"overall": stats(np.concatenate(alld) if alld else np.zeros(0)), "planes": per}
