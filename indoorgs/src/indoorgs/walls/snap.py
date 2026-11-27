"""학습이 끝난 스플랫을 도면 면(벽·바닥·천장)에 붙이기 (벽 평탄화 마지막 단계).

입력 면(planes)은 refine.py 가 실제 벽 위치로 맞춘 Plane 목록 (planes.py). 좌표는 도면 좌표 (m, z 위).

공개 API
  assign(xyz, planes, band, uv_margin=0.05)          -> (면 번호 int32, -1 = 없음), 부호 거리
  snap_to_planes(ply, planes, band=.., thickness=..) -> (새 GaussianPly, info)   중심을 면 위로 + 납작하게
  remove_behind_walls(ply, planes, plan, margin=..)  -> (새 GaussianPly, info)   벽 뒤·바닥 아래·천장 위 떠다니는 것 지움
  verify_planes(ply, planes)                         -> {면 id: ok, peak_m}       바닥·천장을 스플랫 밀도 봉우리로 확인
  fraction_in_rooms(ply, plan)                       -> 0..1                     도면 방 안 불투명 스플랫 비율
  splat_flatness(ply, planes, band=.., min_opacity=..) -> dict                   "벽 평탄도" 지표
  snap_file(in_ply, out_ply, planes, plan, **opts)   -> info                     읽기 -> 지우기 -> 붙이기 -> 쓰기

정한 것 (왜 그렇게 했는지)
- 배정: 가우시안 **중심**이 면에서 |부호 거리| < band 이고 면 범위 안(contains_uv, uv_margin 만큼 가장자리 허용,
  문 구멍 제외)이면 그 면. 여러 면이면 |부호 거리|가 가장 작은 면 (모서리는 가까운 쪽 하나).
  두 방이 공유하는 벽(등을 맞댄 두 면)은 어느 쪽에 있는지로 나눔 (planes.better_face).
- 색 검사 (11-28, 지운 의견): 기본 band 3 cm, 그리고 색이 그 자리 벽 색(면을 1 m 칸으로 나눈 칸별 넓이 가중 중앙값,
  면 텍스처가 있으면 그 칸 사진 픽셀 중앙값)과
  채널 최대 0.12 넘게 다르면 안 붙임 -> 액자·게시판처럼 벽에 붙은 물체는 입체감 그대로.
- 벽에 걸친(straddle) 가우시안: 중심이 band 안이면 법선 방향 두께와 상관없이 납작하게 만든다. 벽 앞의 뿌연 층
  (법선 방향으로 길쭉한 가우시안)이 바로 없애려는 "라이다 메쉬 같은" 울퉁불퉁함이라서. 법선 방향 축이 길어도
  그 축만 thickness 로 줄이고, 면 안쪽 두 축(색·무늬를 담당)은 그대로 둔다.
- 큰 배경 덩어리: 면 안쪽 두 축 중 긴 축의 표준편차(exp(scale)) > max_inplane 이면 **아예 건드리지 않음**
  (옮기지도 납작하게 하지도 않음). 이런 것은 벽 무늬가 아니라 여러 면에 걸친 배경 채움이라 옮기면 다른 곳이 깨짐.
  "면 안쪽 축" = 법선과 가장 덜 정렬된 두 축 (납작하게 만든 뒤에도 같은 두 축이 남음).
- 회전: 법선과 가장 정렬된 축 a 를 ±n (a 와 같은 쪽 부호) 으로 보내는 **최소 회전**(로드리게스) Q 를 세 축 모두에
  곱함. 나머지 두 축은 자동으로 면 위에 놓이고 직교·오른손 좌표가 유지됨 ("면에 투영 후 다시 직교화"와 같은 결과를
  가장 작은 회전으로). 쿼터니언은 정규화해서 씀 (w >= 0). 축 번호가 안 바뀌므로 scale_k 는 그 축 그대로.
- 바늘 (11-28 검토): 법선과 가장 정렬된 축이 가장 긴 축(중간 축의 2배 넘게)이고 법선과 30도 넘게 기울면
  (최대 약 55도까지 가능) 그 축을 4 mm 로 줄이면 길이 20 cm 바늘이 점이 됨. 그런 가우시안은 옮기기만 하고 납작하게
  하지 않음 -> snap 뒤에 도는 바늘 쪼개기(walls/cleanup.py)가 처리.
- 색(f_dc, f_rest)·불투명도·법선(nx..) 은 건드리지 않음 (SH 는 세계 좌표 기준이라 가우시안 자기 축 회전과 무관).
  배정 안 된 가우시안은 비트 그대로. 속성 순서·형식도 입력 그대로.
- 벽 뒤 지우기: 벽 면 범위 안(문 구멍 **제외**)으로 투영되고 부호 거리 < -margin 이면서, 도면의 **어느 방 안에도
  없으면** 지움. 그래서 문 너머 다른 방(도면에 있는 방)의 내용은 남고, 문 구멍으로 보이는 도면 밖 공간도 남음
  (문 구멍에 투영되면 벽 뒤로 안 침). "방 안"은 방을 margin 만큼 키워서 봄: 안 그러면 이 방 벽 바로 바깥 몇 cm 의
  점(학습이 벽을 조금 뒤에 만든 것)이 **다른 방의 먼 벽** 뒤로 잡혀 지워짐. 대신 두께 2*margin 보다 얇은 벽 속의
  점은 양쪽 방 어느 쪽 margin 안이라 남음 (벽 뒤 margin 넘는 것만 지운다는 약속과 같음). 바닥 아래·천장 위: 그 방 다각형 안에서 부호 거리 < -margin 이면 같은 규칙.
  방 안 판정은 plan 을 주면 3D (다각형 + floor_z..ceiling_z). 다각형 목록만 주면 2D 이고, 이때 바닥 아래·천장 위
  판정에는 방 예외를 쓰지 않음 (2D 로는 위층·아래층 방을 알 수 없어서 그 방 다각형 안이면 지움).
- 지우기 안전장치 (11-28 검토: 도면 천장 2.6 m / 실제 2.8 m 면 실제 천장이 100% 지워짐):
  ① 면 하나가 그 이웃(면에서 ±0.5 m, 면 범위 안) 불투명 스플랫의 20% 넘게 지우면 그 면은 지우기에 안 씀,
  ② 전체로 불투명 스플랫의 2% 넘게 지우면 가장 많이 지우는 면부터 뺌. 뺀 면은 info["skipped_faces"] 와 경고.
  ③ 바닥·천장은 snap_file 이 실제 스플랫의 법선 방향 밀도 봉우리가 면 ±5 cm 안인지 먼저 확인 (verify_planes),
     아니면 지우기·붙이기 모두에 안 씀. 맞추지 못한 면(planes.json fit source "plan")은 stages/snap.py 가 아예 안 넘김.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .. import plyio
from ..plyio import GaussianPly
from .planes import CEILING, FLOOR, WALL, Plane, _in_polygon, better_face

CHUNK = 1 << 20          # 한 번에 다루는 가우시안 수 (메모리 상한)
KINDS = (WALL, FLOOR, CEILING)
NEEDLE_MIN_DEG = 30.0    # 법선 쪽 축 = 가장 긴 축이고 법선과 이 각도 넘게 기울면 납작하게 안 함 (바늘)
NEEDLE_COS = float(np.cos(np.radians(NEEDLE_MIN_DEG)))
NEEDLE_RATIO = 2.0       # 가장 긴 축 / 중간 축 이 이보다 커야 바늘로 봄 (거의 둥근 덩어리는 그대로 납작하게)
MAX_REMOVE_FRAC = 0.02   # 벽 뒤 지우기가 불투명 스플랫의 이 비율 넘게 지우면 가장 많이 지우는 면부터 뺌
MAX_FACE_FRAC = 0.20     # 한 면 이웃(면에서 ±NBHD_M) 불투명 스플랫 중 이 비율 넘게 지우면 그 면은 지우기에 안 씀
NBHD_M = 0.5
MIN_IN_ROOMS = 0.5       # 불투명 스플랫 중 도면 방(0.3 m 키움) 안 비율이 이보다 작으면 도면 좌표가 아닌 것으로 보고 건너뜀


# ---------------- 배정 ----------------
def assign(xyz: np.ndarray, planes: list[Plane], band: float, uv_margin: float = 0.05,
           chunk: int = CHUNK) -> tuple[np.ndarray, np.ndarray]:
    """각 점의 가장 가까운 면 (|부호 거리| < band 이고 면 범위 안). 반환 (idx int32 (-1 없음), 부호 거리 float64
    (배정 안 된 점은 nan))."""
    xyz = np.asarray(xyz)
    N = len(xyz)
    idx = np.full(N, -1, np.int32)
    sdist = np.full(N, np.nan)
    if N == 0 or not planes:
        return idx, sdist
    normals = np.array([p.n for p in planes], np.float64).reshape(-1, 3)
    for s in range(0, N, chunk):
        P = np.asarray(xyz[s:s + chunk], np.float64)
        bi = np.full(len(P), -1, np.int32)
        bsd = np.full(len(P), np.nan)
        for j, pl in enumerate(planes):
            sd = P @ pl.n - pl.d
            cand = np.nonzero(np.abs(sd) < band)[0]
            if len(cand) == 0:
                continue
            cur = bi[cand]
            win = better_face(sd[cand], pl.n, bsd[cand], normals[np.maximum(cur, 0)], cur >= 0)
            cand = cand[win]
            if len(cand) == 0:
                continue
            ok = pl.contains_uv(pl.uv(P[cand]), uv_margin)
            c = cand[ok]
            bi[c] = j
            bsd[c] = sd[c]
        idx[s:s + chunk] = bi
        sdist[s:s + chunk] = bsd
    return idx, sdist


# ---------------- 회전 도움 ----------------
def _quats(d: np.ndarray, sel: np.ndarray) -> np.ndarray:
    return np.stack([d[f"rot_{i}"][sel] for i in range(4)], 1).astype(np.float64)


def _scales(d: np.ndarray, sel: np.ndarray) -> np.ndarray:
    return np.stack([d[f"scale_{i}"][sel] for i in range(3)], 1).astype(np.float64)


def _normal_axis(R: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R (M,3,3) 의 열(가우시안 축) 과 n (M,3) 의 내적 -> (가장 정렬된 축 번호, 그 내적 (부호 포함))."""
    dots = np.einsum("mik,mi->mk", R, n)
    k = np.argmax(np.abs(dots), 1)
    return k, dots[np.arange(len(k)), k]


def _min_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """단위 벡터 a -> b 로 보내는 최소 회전 (M,3,3). a·b >= 0 가정 (여기서는 >= 1/sqrt(3))."""
    v = np.cross(a, b)
    c = np.einsum("mi,mi->m", a, b)
    K = np.zeros((len(a), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -v[:, 2], v[:, 1]
    K[:, 1, 0], K[:, 1, 2] = v[:, 2], -v[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -v[:, 1], v[:, 0]
    return np.eye(3)[None] + K + (K @ K) / (1.0 + c)[:, None, None]


# ---------------- 붙이기 ----------------
def _dc_rgb(d: np.ndarray, g: np.ndarray) -> np.ndarray:
    """SH DC -> 0..1 색 (3DGS: rgb = 0.5 + C0 * f_dc)."""
    return 0.5 + plyio.SH_C0 * np.stack([d[f"f_dc_{i}"][g] for i in range(3)], 1).astype(np.float64)


def _footprint(d: np.ndarray, g: np.ndarray, pl: Plane) -> np.ndarray:
    """g 번 가우시안들을 면에 정사영한 1σ 타원 넓이 π·sqrt(det Σ2) (m²). 회전이 깨진 것은 0."""
    q = _quats(d, g)
    qn = np.linalg.norm(q, axis=1)
    good = np.isfinite(qn) & (qn > 1e-12)
    q[~good] = (1.0, 0.0, 0.0, 0.0)
    R = plyio.quat_to_mat(q)
    S2 = np.exp(2.0 * _scales(d, g))
    MU = np.einsum("j,njk->nk", pl.U, R)
    MV = np.einsum("j,njk->nk", pl.V, R)
    a = np.einsum("nk,nk->n", MU * MU, S2)
    b = np.einsum("nk,nk->n", MU * MV, S2)
    c = np.einsum("nk,nk->n", MV * MV, S2)
    area = np.pi * np.sqrt(np.maximum(a * c - b * b, 0.0))
    area[~good | ~np.isfinite(area)] = 0.0
    return area


def _wmedian(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """채널별 가중 중앙값. x (n, 3), w (n,) >= 0. 가중 합이 0 이면 그냥 중앙값."""
    if w.sum() <= 0:
        return np.median(x, 0)
    out = np.empty(x.shape[1])
    for ch in range(x.shape[1]):
        o = np.argsort(x[:, ch], kind="stable")
        cw = np.cumsum(w[o])
        out[ch] = x[o[np.searchsorted(cw, 0.5 * cw[-1])], ch]
    return out


def _texture_grid(tex, pl: Plane, cell: float, nu: int, nv: int, min_frac: float = 0.25):
    """면 텍스처(walls.facetex.FaceTexture) -> 칸별 사진 픽셀 색 중앙값 (nu, nv, 3) 0..1 과 쓸 수 있는 칸 (nu, nv).
    텍스처 픽셀은 면 위 같은 넓이라 칸 중앙값 = 넓이 기준 중앙값. 색 있는 픽셀(source > 0)이 칸 안 픽셀의 min_frac 미만이면 안 씀."""
    H, W = tex.image.shape[:2]
    jj, ii = np.meshgrid(np.arange(W), np.arange(H))
    u = (jj + 0.5) * tex.res
    v = (ii + 0.5) * tex.res
    iu = np.clip((u / cell).astype(int), 0, nu - 1)
    iv = np.clip((v / cell).astype(int), 0, nv - 1)
    grid = np.zeros((nu, nv, 3))
    ok = np.zeros((nu, nv), bool)
    have = (tex.source > 0) & tex.inside
    img = tex.image.astype(np.float64) / 255.0
    for a in range(nu):
        for b in range(nv):
            cellm = (iu == a) & (iv == b) & tex.inside
            m = cellm & have
            if m.sum() >= max(4, min_frac * max(1, cellm.sum())):
                grid[a, b] = np.median(img[m], 0)
                ok[a, b] = True
    return grid, ok


def wall_colors(ply: GaussianPly, planes: list[Plane], band: float, uv_margin: float = 0.05, cell: float = 1.0,
                min_opacity: float = 0.5, chunk: int = CHUNK, texture: dict | None = None,
                weighting: str = "area") -> dict:
    """면별 "그 자리 벽 색": 면을 cell(m) 칸으로 나눠 칸마다 기준 색 (조명이 점점 변해도 따라감).
    - weighting="area" (기본, 11-28 wall_ab 버그 1 수정): 불투명 가우시안 DC 색의 **넓이 가중** 중앙값.
      가중치 = 면에 정사영한 1σ 타원 넓이(칸 넓이로 자름) × 불투명도. 무늬 있는 물체(포스터·액자)는 흰 벽보다 m² 당
      가우시안이 훨씬 많아서 개수 중앙값이면 칸 넓이의 48% 인 포스터가 가우시안의 85% 를 차지해 '벽 색' 이 포스터 색이
      되고 색 검사가 거꾸로 동작했음 (포스터 붙임, 흰 벽 건너뜀). 넓이로 세면 칸의 절반 넘게 덮는 쪽이 기준 색.
    - weighting="count": 예전 방식 (가우시안 개수 중앙값, 비교용).
    - texture ({면 id: walls.facetex.FaceTexture}, 선택): 있으면 그 면은 사진 투영 텍스처의 칸별 픽셀 중앙값
      (넓이 기준과 같은 뜻, 사진 색 그대로) 을 우선 쓰고, 색 있는 픽셀이 모자란 칸만 가우시안 기준.
    가우시안이 적은 칸(<8)은 면 전체 값. 반환 {면 번호: (칸 색 (nu, nv, 3), 면 전체 색 (3,))}"""
    if weighting not in ("area", "count"):
        raise ValueError("weighting 은 area | count")
    d = ply.data
    N = len(d)
    acc: dict[int, list] = {}
    cell_area = cell * cell
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        P = np.stack([d["x"][s:e], d["y"][s:e], d["z"][s:e]], 1).astype(np.float64)
        idx, _ = assign(P, planes, band, uv_margin, chunk=chunk)
        sel = np.nonzero(idx >= 0)[0]
        if len(sel) == 0:
            continue
        g = s + sel
        op = 1.0 / (1.0 + np.exp(-d["opacity"][g].astype(np.float64)))
        m = op >= min_opacity
        sel, g, op = sel[m], g[m], op[m]
        rgb = _dc_rgb(d, g)
        for j in np.unique(idx[sel]):
            mm = idx[sel] == j
            uv = planes[j].uv(P[sel[mm]])
            if weighting == "area":
                w = np.minimum(_footprint(d, g[mm], planes[j]), cell_area) * op[mm]
            else:
                w = np.ones(int(mm.sum()))
            acc.setdefault(int(j), []).append((uv, rgb[mm], w))
    out = {}
    for j, pl in enumerate(planes):
        tex = texture.get(pl.id) if texture else None
        if j not in acc and tex is None:
            continue
        nu = max(1, int(np.ceil(pl.size[0] / cell)))
        nv = max(1, int(np.ceil(pl.size[1] / cell)))
        if j in acc:
            parts = acc[j]
            uv = np.concatenate([a for a, _, _ in parts])
            rgb = np.concatenate([b for _, b, _ in parts])
            w = np.concatenate([c for _, _, c in parts])
            glob = _wmedian(rgb, w)
        else:
            uv, rgb, w = np.zeros((0, 2)), np.zeros((0, 3)), np.zeros(0)
            glob = None
        tgrid, tok = _texture_grid(tex, pl, cell, nu, nv) if tex is not None else (None, np.zeros((nu, nv), bool))
        if glob is None:
            if not tok.any():
                continue
            glob = np.median(tgrid[tok], 0)
        grid = np.tile(glob, (nu, nv, 1))
        if len(uv):
            iu = np.clip((uv[:, 0] / cell).astype(int), 0, nu - 1)
            iv = np.clip((uv[:, 1] / cell).astype(int), 0, nv - 1)
            key = iu * nv + iv
            order = np.argsort(key, kind="stable")
            ks, starts = np.unique(key[order], return_index=True)
            ends = np.append(starts[1:], len(order))
            for k_, a, b in zip(ks, starts, ends):
                if b - a >= 8:
                    o = order[a:b]
                    grid[k_ // nv, k_ % nv] = _wmedian(rgb[o], w[o])
        if tex is not None:
            grid[tok] = tgrid[tok]
        out[j] = (grid, glob)
    return out


def snap_to_planes(ply: GaussianPly, planes: list[Plane], *, band: float = 0.03, thickness: float = 0.004,
                   max_inplane: float = 0.6, min_opacity: float = 0.0, uv_margin: float = 0.05,
                   flatten: bool = True, move: bool = True, color_tol: float | None = 0.12, color_cell: float = 1.0,
                   color_texture: dict | None = None, color_weighting: str = "area",
                   chunk: int = CHUNK) -> tuple[GaussianPly, dict]:
    """면 근처 가우시안을 면 위로 옮기고(move) 법선 방향으로 납작하게(flatten). 규칙은 모듈 주석.
    max_inplane: 면 안쪽 긴 축 표준편차(m) 상한, 넘으면 안 건드림. min_opacity: 이보다 투명하면 안 건드림 (0..1).
    color_tol (지운 의견, 11-28): 벽 앞 ±band(기본 3 cm) 안이라도 색이 그 자리 벽 색(wall_colors)과 채널 최대 차이
    color_tol(0..1) 넘게 다르면 안 건드림 -> 액자·게시판·스위치 같은 벽에 붙은 물체의 입체감을 지킴. None 이면 색 검사 없음.
    color_texture / color_weighting: wall_colors 의 texture / weighting (기본 넓이 가중)."""
    plyio.check_required(ply.names)
    d = ply.data.copy()
    wcol = (wall_colors(ply, planes, band, uv_margin, color_cell, chunk=chunk, texture=color_texture,
                        weighting=color_weighting) if color_tol is not None else {})
    N = len(d)
    logt = np.log(thickness)
    normals = np.array([p.n for p in planes], np.float64).reshape(-1, 3)
    offs = np.array([p.d for p in planes], np.float64)
    info = {"n_total": int(N), "n_assigned": {k: 0 for k in KINDS}, "n_assigned_plane": {},
            "n_skipped_big": 0, "n_skipped_transparent": 0, "n_skipped_color": 0, "n_skipped_needle": 0,
            "n_moved": 0, "n_flattened": 0,
            "mean_abs_shift": 0.0, "max_abs_shift": 0.0,
            "params": {"band": band, "thickness": thickness, "max_inplane": max_inplane,
                       "min_opacity": min_opacity, "uv_margin": uv_margin, "flatten": flatten, "move": move,
                       "color_tol": color_tol, "color_cell": color_cell, "color_weighting": color_weighting,
                       "color_texture": sorted(color_texture) if color_texture else None}}
    shift_sum = 0.0
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        P = np.stack([d["x"][s:e], d["y"][s:e], d["z"][s:e]], 1).astype(np.float64)
        idx, _ = assign(P, planes, band, uv_margin, chunk=chunk)
        sel = np.nonzero(idx >= 0)[0]
        if len(sel) == 0:
            continue
        pi = idx[sel]
        for j in np.unique(pi):
            cnt = int(np.sum(pi == j))
            info["n_assigned"][planes[j].kind] += cnt
            info["n_assigned_plane"][planes[j].id] = info["n_assigned_plane"].get(planes[j].id, 0) + cnt
        g = s + sel                                        # 전체 배열 번호
        if min_opacity > 0:
            op = 1.0 / (1.0 + np.exp(-d["opacity"][g].astype(np.float64)))
            keep = op >= min_opacity
            info["n_skipped_transparent"] += int(np.sum(~keep))
            sel, pi, g = sel[keep], pi[keep], g[keep]
        if color_tol is not None and len(g):
            rgb = _dc_rgb(d, g)
            ref = np.empty_like(rgb)
            for j in np.unique(pi):
                mm = pi == j
                if int(j) not in wcol:
                    ref[mm] = rgb[mm]
                    continue
                grid, _ = wcol[int(j)]
                uv = planes[j].uv(P[sel[mm]])
                nu, nv = grid.shape[:2]
                iu = np.clip((uv[:, 0] / color_cell).astype(int), 0, nu - 1)
                iv = np.clip((uv[:, 1] / color_cell).astype(int), 0, nv - 1)
                ref[mm] = grid[iu, iv]
            keep = np.abs(rgb - ref).max(1) <= color_tol
            info["n_skipped_color"] += int(np.sum(~keep))
            sel, pi, g = sel[keep], pi[keep], g[keep]
        qq = _quats(d, g)
        qn = np.linalg.norm(qq, axis=1)
        good = np.isfinite(qn) & (qn > 1e-12)              # 회전이 깨진 가우시안은 안 건드림
        sel, pi, g, qq = sel[good], pi[good], g[good], qq[good]
        n = normals[pi]
        R = plyio.quat_to_mat(qq)
        k, dot = _normal_axis(R, n)
        sc = _scales(d, g)
        inpl = sc.copy()
        inpl[np.arange(len(k)), k] = -np.inf
        big = np.exp(inpl.max(1)) > max_inplane
        info["n_skipped_big"] += int(big.sum())
        ok = ~big
        sel, pi, g, n, R, k, dot, sc = sel[ok], pi[ok], g[ok], n[ok], R[ok], k[ok], dot[ok], sc[ok]
        if len(g) == 0:
            continue
        if move:
            sd = np.einsum("mi,mi->m", P[sel], n) - offs[pi]
            Q = P[sel] - sd[:, None] * n
            for i, c in enumerate("xyz"):
                d[c][g] = Q[:, i]
            info["n_moved"] += int(len(g))
            a = np.abs(sd)
            shift_sum += float(a.sum())
            info["max_abs_shift"] = max(info["max_abs_shift"], float(a.max()))
        if flatten:
            # 바늘: 법선 쪽 축이 가장 긴 축이고 (중간 축의 NEEDLE_RATIO 배 넘게 김) 법선과 NEEDLE_MIN_DEG 넘게 기울면
            # 그 축을 thickness 로 줄이면 점이 됨 -> 납작하게 하지 않고 (옮기기만) snap 뒤 바늘 쪼개기(cleanup)에 넘김
            ss = np.sort(sc, 1)
            needle = ((k == np.argmax(sc, 1)) & (np.abs(dot) < NEEDLE_COS)
                      & (ss[:, 2] - ss[:, 1] > np.log(NEEDLE_RATIO)))
            info["n_skipped_needle"] += int(needle.sum())
            fl = ~needle
            g, k, dot, n, R, sc = g[fl], k[fl], dot[fl], n[fl], R[fl], sc[fl]
            ar = np.arange(len(k))
            axis = R[ar, :, k]                             # (M,3) 법선 쪽 축
            target = np.where(dot[:, None] < 0, -n, n)
            Rn = _min_rotation(axis, target) @ R
            q = plyio.mat_to_quat(Rn)
            for i in range(4):
                d[f"rot_{i}"][g] = q[:, i]
            newk = np.minimum(sc[ar, k], logt)
            for i in range(3):
                m = k == i
                if m.any():
                    d[f"scale_{i}"][g[m]] = newk[m]
            info["n_flattened"] += int(len(g))
    if info["n_moved"]:
        info["mean_abs_shift"] = shift_sum / info["n_moved"]
    return GaussianPly(d, list(ply.comments)), info


# ---------------- 벽 뒤 지우기 ----------------
def _rooms_3d(plan_or_polys) -> list[tuple[list, float, float]]:
    """Plan / Room 목록 / 다각형 목록 -> [(다각형, z_min, z_max)]. 다각형만 주면 z 는 무한."""
    rooms = getattr(plan_or_polys, "rooms", plan_or_polys)
    out = []
    for r in rooms or []:
        if hasattr(r, "floor"):
            cz = r.ceiling_z if r.ceiling_z is not None else np.inf
            out.append(([tuple(map(float, p)) for p in r.floor], float(r.floor_z), float(cz)))
        else:
            out.append(([tuple(map(float, p)) for p in r], -np.inf, np.inf))
    return out


def _inside_any_room(P: np.ndarray, rooms, margin: float = 0.0) -> np.ndarray:
    """방(다각형 + 높이)을 margin 만큼 키워서 그 안인지."""
    ins = np.zeros(len(P), bool)
    for poly, z0, z1 in rooms:
        zc = (P[:, 2] >= z0 - margin) & (P[:, 2] <= z1 + margin) & ~ins
        if zc.any():
            ii = np.nonzero(zc)[0]
            ins[ii] = _in_polygon(P[ii, 0], P[ii, 1], poly, margin)
    return ins


def behind_faces(xyz: np.ndarray, planes: list[Plane], plan_or_polys, margin: float = 0.15,
                 chunk: int = CHUNK) -> np.ndarray:
    """점마다 지우게 만든 면 번호 (-1 = 안 지움; 여러 면이면 벽 > 바닥 > 천장 순서의 앞 것)."""
    rooms = _rooms_3d(plan_or_polys)
    rooms3 = [r for r in rooms if np.isfinite(r[1]) or np.isfinite(r[2])]
    N = len(xyz)
    face = np.full(N, -1, np.int32)
    code = {WALL: 1, FLOOR: 2, CEILING: 3}
    order = sorted(range(len(planes)), key=lambda j: code[planes[j].kind])
    for s in range(0, N, chunk):
        P = np.asarray(xyz[s:s + chunk], np.float64)
        f = np.full(len(P), -1, np.int32)
        for j in order:
            pl = planes[j]
            cand = np.nonzero((f < 0) & ((P @ pl.n - pl.d) < -margin))[0]
            if len(cand) == 0:
                continue
            ok = pl.contains_uv(pl.uv(P[cand]), 0.0)      # 문 구멍 안이면 False
            f[cand[ok]] = j
        kind = np.array([code[p.kind] for p in planes] + [0], np.int8)[f]   # f = -1 -> 0
        hit = np.nonzero(kind == 1)[0]
        if len(hit) and rooms:
            f[hit[_inside_any_room(P[hit], rooms, margin)]] = -1
        hit = np.nonzero(kind > 1)[0]                     # 바닥·천장: 높이를 아는 방만 (2D 다각형은 판정 불가)
        if len(hit) and rooms3:
            f[hit[_inside_any_room(P[hit], rooms3, margin)]] = -1
        face[s:s + chunk] = f
    return face


def behind_mask(xyz: np.ndarray, planes: list[Plane], plan_or_polys, margin: float = 0.15,
                chunk: int = CHUNK) -> tuple[np.ndarray, np.ndarray]:
    """지울 점 표시와 이유 코드 (0 없음, 1 벽, 2 바닥, 3 천장; 여러 개면 앞의 것)."""
    face = behind_faces(xyz, planes, plan_or_polys, margin, chunk)
    code = {WALL: 1, FLOOR: 2, CEILING: 3}
    reason = np.array([code[p.kind] for p in planes] + [0], np.int8)[face]
    return reason > 0, reason


def remove_behind_walls(ply: GaussianPly, planes: list[Plane], plan_or_polys, *, margin: float = 0.15,
                        max_remove_frac: float | None = MAX_REMOVE_FRAC, max_face_frac: float | None = MAX_FACE_FRAC,
                        nbhd: float = NBHD_M, min_opacity: float = 0.5, min_nbhd: int = 20,
                        chunk: int = CHUNK) -> tuple[GaussianPly, dict]:
    """벽 뒤(문 구멍 제외)·바닥 아래·천장 위에 있고 도면의 어느 방에도 없는 가우시안을 뺀 새 PLY.
    안전장치(모듈 설명): 면 이웃의 max_face_frac 넘게, 또는 전체 불투명 스플랫의 max_remove_frac 넘게 지우는 면은
    빼고 다시 계산. None 이면 그 검사 안 함."""
    xyz = _XYZ(ply)
    op = 1.0 / (1.0 + np.exp(-ply.data["opacity"].astype(np.float64)))
    opq = op >= min_opacity
    n_opq = max(int(opq.sum()), 1)
    active = list(range(len(planes)))
    skipped: dict[str, str] = {}
    warnings: list[str] = []
    face = np.full(len(ply.data), -1, np.int32)
    for _ in range(len(planes) + 1):
        sub = [planes[j] for j in active]
        face = behind_faces(xyz, sub, plan_or_polys, margin, chunk)
        drop: dict[int, str] = {}
        rm_opq = {}
        for jj in np.unique(face[face >= 0]):
            j = active[int(jj)]
            pl = planes[j]
            mine = (face == jj) & opq
            rm_opq[j] = int(mine.sum())
            if max_face_frac is None or rm_opq[j] == 0:
                continue
            nb, nrm = 0, 0
            for s in range(0, len(xyz), chunk):
                P = xyz[s:s + chunk]
                sd = P @ pl.n - pl.d
                c = np.nonzero((np.abs(sd) < nbhd) & opq[s:s + chunk])[0]
                if len(c):
                    c = c[pl.contains_uv(pl.uv(P[c]), 0.0)]
                nb += len(c)
                nrm += int(mine[s:s + chunk][c].sum())
            if nb >= min_nbhd and nrm / nb > max_face_frac:
                drop[j] = f"면 이웃 불투명 스플랫의 {nrm / nb * 100:.0f}% 를 지움 (> {max_face_frac * 100:.0f}%)"
        if not drop and max_remove_frac is not None and sum(rm_opq.values()) / n_opq > max_remove_frac:
            j = max(rm_opq, key=rm_opq.get)
            drop[j] = (f"전체 불투명 스플랫의 {sum(rm_opq.values()) / n_opq * 100:.1f}% 를 지움 "
                       f"(> {max_remove_frac * 100:.0f}%), 이 면이 {rm_opq[j]:,}개로 가장 많음")
        if not drop:
            break
        for j, why in drop.items():
            skipped[planes[j].id] = why
            warnings.append(f"경고: {planes[j].id} 뒤 지우기 건너뜀 - {why} (도면 면이 실제와 다를 수 있음)")
        active = [j for j in active if j not in drop]
    rm = face >= 0
    code = {WALL: 1, FLOOR: 2, CEILING: 3}
    sub = [planes[j] for j in active]
    reason = np.array([code[p.kind] for p in sub] + [0], np.int8)[face]
    info = {"n_total": int(ply.n), "n_removed": int(rm.sum()), "n_kept": int(ply.n - rm.sum()),
            "removed_by": {WALL: int(np.sum(reason == 1)), FLOOR: int(np.sum(reason == 2)),
                           CEILING: int(np.sum(reason == 3))},
            "removed_by_plane": {sub[int(j)].id: int(np.sum(face == j)) for j in np.unique(face[face >= 0])},
            "skipped_faces": skipped, "warnings": warnings,
            "margin": margin, "max_remove_frac": max_remove_frac, "max_face_frac": max_face_frac}
    return GaussianPly(ply.data[~rm].copy(), list(ply.comments)), info


def verify_planes(ply: GaussianPly, planes: list[Plane], kinds=(FLOOR, CEILING), *, search: float = 0.30,
                  tol: float = 0.05, bin_m: float = 0.02, min_count: int = 30, min_opacity: float = 0.5,
                  chunk: int = CHUNK) -> dict:
    """kinds 면마다 실제 불투명 스플랫의 법선 방향 밀도 봉우리(면 범위 안, ±search, bin_m 칸)가 면에서 tol 안인지.
    반환 {면 id: {"ok", "peak_m", "n"}} (kinds 아닌 면은 없음). 스플랫이 min_count 보다 적으면 ok=False (확인 불가)."""
    out = {}
    js = [j for j, p in enumerate(planes) if p.kind in kinds]
    if not js:
        return out
    edges = np.arange(-search, search + 1e-9, bin_m)
    hist = {j: np.zeros(len(edges) - 1) for j in js}
    d = ply.data
    for s in range(0, len(d), chunk):
        e = min(len(d), s + chunk)
        op = 1.0 / (1.0 + np.exp(-d["opacity"][s:e].astype(np.float64)))
        P = np.stack([d["x"][s:e], d["y"][s:e], d["z"][s:e]], 1).astype(np.float64)[op >= min_opacity]
        for j in js:
            pl = planes[j]
            sd = P @ pl.n - pl.d
            c = np.nonzero(np.abs(sd) < search)[0]
            if len(c):
                c = c[pl.contains_uv(pl.uv(P[c]), 0.0)]
            hist[j] += np.histogram(sd[c], edges)[0]
    for j in js:
        h = hist[j]
        n = int(h.sum())
        if n < min_count:
            out[planes[j].id] = {"ok": False, "peak_m": None, "n": n, "reason": f"스플랫 부족 ({n} < {min_count})"}
            continue
        hs = np.convolve(h, np.ones(3) / 3, mode="same")              # 칸 경계에 걸친 봉우리
        k = int(np.argmax(hs))
        peak = float(0.5 * (edges[k] + edges[k + 1]))
        r = {"ok": abs(peak) <= tol, "peak_m": peak, "n": n}
        if not r["ok"]:
            r["reason"] = f"스플랫 밀도 봉우리가 면에서 {peak * 100:+.0f}cm (허용 ±{tol * 100:.0f}cm)"
        out[planes[j].id] = r
    return out


def fraction_in_rooms(ply: GaussianPly, plan_or_polys, margin: float = 0.3, min_opacity: float = 0.5,
                      chunk: int = CHUNK) -> float:
    """불투명 스플랫 중 도면 방(margin 만큼 키움, 높이 포함) 안 비율. 학습 PLY 가 도면 좌표인지 보는 데 씀."""
    rooms = _rooms_3d(plan_or_polys)
    d = ply.data
    n_in = n = 0
    for s in range(0, len(d), chunk):
        e = min(len(d), s + chunk)
        op = 1.0 / (1.0 + np.exp(-d["opacity"][s:e].astype(np.float64)))
        P = np.stack([d["x"][s:e], d["y"][s:e], d["z"][s:e]], 1).astype(np.float64)[op >= min_opacity]
        n += len(P)
        n_in += int(_inside_any_room(P, rooms, margin).sum()) if rooms else 0
    return n_in / n if n else 1.0


class _XYZ:
    """큰 PLY 의 xyz 를 덩어리로 float64 로 꺼내 주는 얇은 보기 (3x8 바이트 x N 복사를 피함)."""

    def __init__(self, ply: GaussianPly):
        self.d = ply.data

    def __len__(self):
        return len(self.d)

    def __getitem__(self, sl):
        return np.stack([self.d["x"][sl], self.d["y"][sl], self.d["z"][sl]], 1).astype(np.float64)


# ---------------- 평탄도 ----------------
def splat_flatness(ply: GaussianPly, planes: list[Plane], band: float = 0.05, min_opacity: float = 0.5,
                   uv_margin: float = 0.05, chunk: int = CHUNK) -> dict:
    """"벽 평탄도". 불투명(sigmoid(opacity) >= min_opacity)하고 면에서 band 안인 가우시안만.
    - rms_dist: 불투명도 가중 RMS |부호 거리| (중심이 면에서 얼마나 흩어졌나), m
    - rms_thick: 불투명도 가중 RMS sqrt(부호거리^2 + 법선 방향 표준편차^2) (중심은 면 위라도 두꺼우면 뿌옇다), m
    - within_1cm: 그중 |부호 거리| <= 1 cm 비율
    전체(all)와 종류별(wall/floor/ceiling). count = 해당 가우시안 수."""
    d = ply.data
    acc = {k: np.zeros(5) for k in ("all",) + KINDS}       # sum w, sum w sd^2, sum w t^2, count, n<=1cm
    kinds = [p.kind for p in planes]
    normals = np.array([p.n for p in planes], np.float64).reshape(-1, 3)
    for s in range(0, len(d), chunk):
        e = min(len(d), s + chunk)
        op = 1.0 / (1.0 + np.exp(-d["opacity"][s:e].astype(np.float64)))
        cand = np.nonzero(op >= min_opacity)[0]
        if len(cand) == 0:
            continue
        P = np.stack([d["x"][s:e][cand], d["y"][s:e][cand], d["z"][s:e][cand]], 1).astype(np.float64)
        idx, sd = assign(P, planes, band, uv_margin, chunk=chunk)
        m = idx >= 0
        if not m.any():
            continue
        g = s + cand[m]
        pi, sd, w = idx[m], sd[m], op[cand[m]]
        R = plyio.quat_to_mat(_quats(d, g))
        S = np.exp(_scales(d, g))
        # 법선 방향 분산 = sum_k (a_k·n)^2 s_k^2
        var_n = np.einsum("mk,mk->m", np.einsum("mik,mi->mk", R, normals[pi]) ** 2, S ** 2)
        for key in ("all",) + KINDS:
            mk = np.ones(len(pi), bool) if key == "all" else np.array([kinds[j] == key for j in pi], bool)
            if not mk.any():
                continue
            ww = w[mk]
            acc[key] += [ww.sum(), (ww * sd[mk] ** 2).sum(), (ww * (sd[mk] ** 2 + var_n[mk])).sum(),
                         mk.sum(), np.sum(np.abs(sd[mk]) <= 0.01)]
    out = {"band": band, "min_opacity": min_opacity}
    for key, (sw, s2, t2, cnt, n1) in acc.items():
        out[key] = {"count": int(cnt),
                    "rms_dist": float(np.sqrt(s2 / sw)) if sw > 0 else None,
                    "rms_thick": float(np.sqrt(t2 / sw)) if sw > 0 else None,
                    "within_1cm": float(n1 / cnt) if cnt else None}
    a = out["all"]                                          # 요약 (stages/snap.py 메시지용, 없으면 nan)
    out["rms_m"] = a["rms_dist"] if a["rms_dist"] is not None else float("nan")
    out["thick_m"] = a["rms_thick"] if a["rms_thick"] is not None else float("nan")
    out["within_1cm"] = a["within_1cm"] if a["within_1cm"] is not None else float("nan")
    return out


# ---------------- 파일 단위 ----------------
_SNAP_KEYS = {"band", "thickness", "max_inplane", "min_opacity", "uv_margin", "flatten", "move", "color_tol",
              "color_cell", "color_texture", "color_weighting"}


def snap_file(in_ply, out_ply, planes: list[Plane], plan, *, remove_behind: bool = True, margin: float = 0.15,
              flat_band: float = 0.05, verify: bool = True, min_in_rooms: float | None = MIN_IN_ROOMS,
              max_remove_frac: float | None = MAX_REMOVE_FRAC, max_face_frac: float | None = MAX_FACE_FRAC,
              **opts) -> dict:
    """읽기 -> (검사) -> (remove_behind 면) remove_behind_walls -> snap_to_planes -> 새 파일에 쓰기. 입력 파일은 안 바꿈.
    opts 는 snap_to_planes 인자. plan 은 Plan (또는 remove_behind_walls 가 받는 방 목록).
    planes 는 믿을 수 있는 면(정합 점으로 맞춘 면)만 줄 것 (stages/snap.py). 여기서 더 거르는 것:
    - min_in_rooms: 불투명 스플랫 중 도면 방(0.3 m 키움) 안이 이 비율 미만이면 도면 좌표가 아닌 PLY 로 보고 아무것도
      안 씀 (info["skipped"] = 이유, 파일 안 만듦).
    - verify: 바닥·천장은 스플랫 밀도 봉우리로 확인(verify_planes), 실패한 면은 지우기·붙이기에 안 씀.
    반환 요약 키: n_removed, n_moved, n_flattened, flatness_before/after["rms_m"], planes_used, warnings."""
    bad = set(opts) - _SNAP_KEYS
    if bad:
        raise TypeError(f"모르는 옵션 {sorted(bad)}")
    in_ply, out_ply = Path(in_ply), Path(out_ply)
    if in_ply.resolve() == out_ply.resolve():
        raise ValueError("out_ply 가 in_ply 와 같음 - 원본을 덮어쓰지 않음")
    ply = plyio.read(in_ply)
    info: dict = {"in": str(in_ply), "out": str(out_ply), "warnings": []}
    if min_in_rooms is not None and plan is not None:
        frac = fraction_in_rooms(ply, plan)
        info["in_rooms_frac"] = frac
        if frac < min_in_rooms:
            info["skipped"] = (f"불투명 스플랫 중 도면 방 안이 {frac * 100:.0f}% (< {min_in_rooms * 100:.0f}%): "
                               "학습 PLY 가 도면 좌표가 아닌 것 같음 -> 벽 붙이기 건너뜀")
            info["warnings"].append("경고: " + info["skipped"])
            return info
    if verify:
        info["verify"] = verify_planes(ply, planes)
        bad_ids = {k for k, v in info["verify"].items() if not v["ok"]}
        for k in sorted(bad_ids):
            info["warnings"].append(f"경고: {k} 안 씀 - {info['verify'][k].get('reason')}")
        planes = [p for p in planes if p.id not in bad_ids]
    info["planes_used"] = [p.id for p in planes]
    info["stats_before"] = plyio.stats(ply)
    info["flatness_before"] = splat_flatness(ply, planes, band=flat_band)
    info["n_removed"] = 0
    if remove_behind and plan is not None:
        ply, info["remove"] = remove_behind_walls(ply, planes, plan, margin=margin, max_remove_frac=max_remove_frac,
                                                  max_face_frac=max_face_frac)
        info["n_removed"] = info["remove"]["n_removed"]
        info["warnings"] += info["remove"]["warnings"]
    ply, info["snap"] = snap_to_planes(ply, planes, **opts)
    info["n_moved"], info["n_flattened"] = info["snap"]["n_moved"], info["snap"]["n_flattened"]
    info["flatness_after"] = splat_flatness(ply, planes, band=flat_band)
    info["stats_after"] = plyio.stats(ply)
    plyio.write(out_ply, ply)
    return info
