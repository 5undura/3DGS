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

from .planes import Plane, WALL

__all__ = ["assign_points", "refine_planes", "room_widths", "flatness", "face_coverage", "edge_distance"]


def assign_points(planes: list[Plane], xyz: np.ndarray, search: float = 0.30, uv_margin: float = 0.10) -> np.ndarray:
    """점마다 |부호 거리| < search 이고 면 범위(uv_margin 허용, 문 구멍 제외) 안인 면 중 가장 가까운 면 번호."""
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    best = np.full(len(xyz), np.inf)
    idx = np.full(len(xyz), -1, np.int64)
    for k, p in enumerate(planes):
        ad = np.abs(p.signed_dist(xyz))
        cand = np.flatnonzero((ad < search) & (ad < best))
        if len(cand) == 0:
            continue
        ok = p.contains_uv(p.uv(xyz[cand]), uv_margin)
        cand = cand[ok]
        best[cand] = ad[cand]
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


def _fit_face(u, v, s, w, thr, max_tan, rng, iters=200):
    """s = a + b u + c v 강건 맞춤. 돌려줌: (a, b, c, inlier 마스크, clamped)."""
    n = len(s)
    A = np.stack([np.ones(n), u, v], 1)
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
    sol = best
    k = thr / 2
    for _ in range(15):
        r = np.abs(A @ sol - s)
        hw = np.where(r <= k, 1.0, k / np.maximum(r, 1e-12)) * (r < 3 * thr)
        if (hw > 0).sum() < 3:
            break
        new = _wlsq(A, s, w * hw)
        if np.allclose(new, sol, atol=1e-7):
            sol = new
            break
        sol = new
    clamped = False
    t = math.hypot(sol[1], sol[2])
    if t > max_tan:
        clamped = True
        sol = sol.copy()
        sol[1:] *= max_tan / t
        for _ in range(10):
            r = s - sol[1] * u - sol[2] * v
            m = np.abs(r - sol[0]) < thr
            if not m.any():
                break
            sol[0] = float(np.average(r[m], weights=w[m]))
    inl = np.abs(A @ sol - s) < thr
    return float(sol[0]), float(sol[1]), float(sol[2]), inl, clamped


def refine_planes(planes: list[Plane], xyz: np.ndarray, *, weights: np.ndarray | None = None,
                  search: float = 0.30, max_tilt_deg: float = 3.0, min_points: int = 40,
                  min_coverage: float = 0.25, uv_margin: float = 0.10, inlier_thr: float = 0.03,
                  cell: float = 0.5, edge_band: float = 0.15, rng=0) -> tuple[list[Plane], list[dict]]:
    """도면 면 -> 점에 맞춘 면. 맞출 수 없는 면은 도면 면 그대로 (info["source"] == "plan")."""
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    w_all = np.ones(len(xyz)) if weights is None else np.asarray(weights, float).reshape(-1)
    rng = np.random.default_rng(rng)
    max_tan = math.tan(math.radians(max_tilt_deg))
    owner = assign_points(planes, xyz, search, uv_margin)
    out, infos = [], []
    for k, p in enumerate(planes):
        sel = owner == k
        if edge_band > 0 and sel.any():     # 모서리 띠의 점은 두 면 어느 쪽인지 애매 -> 맞추기에 안 씀
            j = np.flatnonzero(sel)
            sel[j[edge_distance(p, p.uv(xyz[j])) < edge_band]] = False
        P = xyz[sel]
        w = w_all[sel]
        info = {"id": p.id, "kind": p.kind, "room": p.room, "source": "plan", "n_assigned": int(sel.sum()),
                "n_points": 0, "coverage": 0.0, "shift_m": 0.0, "tilt_deg": 0.0, "clamped": False,
                "rms_before_m": None, "rms_after_m": None}
        if len(P) < max(3, min_points):
            info["reason"] = f"점 부족 ({len(P)} < {min_points})"
            out.append(p)
            infos.append(info)
            continue
        uv = p.uv(P)
        s = p.signed_dist(P)
        uc, vc = p.size[0] / 2, p.size[1] / 2
        a, b, c, inl, clamped = _fit_face(uv[:, 0] - uc, uv[:, 1] - vc, s, w, inlier_thr, max_tan, rng)
        res = s - (a + b * (uv[:, 0] - uc) + c * (uv[:, 1] - vc))
        cov = face_coverage(p, uv[inl], cell)
        n_in = int(inl.sum())
        tilt = math.degrees(math.atan(math.hypot(b, c)))
        info.update(n_points=n_in, coverage=round(cov, 4), shift_m=a, tilt_deg=tilt, clamped=clamped,
                    rms_before_m=float(np.sqrt(np.mean(s[inl] ** 2))) if n_in else None,
                    rms_after_m=float(np.sqrt(np.mean(res[inl] ** 2))) if n_in else None)
        if n_in < min_points:
            info["reason"] = f"inlier 부족 ({n_in} < {min_points})"
        elif cov < min_coverage:
            info["reason"] = f"덮임 부족 ({cov:.2f} < {min_coverage})"
        elif abs(a) >= 0.9 * search:
            info["reason"] = f"이동량이 탐색 한계에 닿음 ({a:+.3f} m, search {search})"
            info["at_bound"] = True
        if "reason" in info:
            out.append(p)
            infos.append(info)
            continue
        n_new = p.n - b * p.U - c * p.V
        a0 = a - b * uc - c * vc          # 면 원점에서의 이동량 (Plane.shifted 는 원점 기준)
        out.append(p.shifted(a0, n_new))
        info["source"] = "fit"
        infos.append(info)
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
