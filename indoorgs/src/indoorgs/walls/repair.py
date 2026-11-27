"""학습 뒤 면(벽·바닥·천장)의 비치는 곳·빈 곳을 가까운 스플랫 복사로 메우기 (snap 다음, splat_walls.ply 에).

# ---------------------------------------------------------------------------
# 출처 (Apache License 2.0): 이 파일의 일부는 splat-repair 를 고쳐 옮긴 것이다.
#   splat-repair, Copyright 2026 Nicolas Drum, https://github.com/NicolasDrum/splat-repair
#   core/splatcore/repair.py ("Fast Fill" = rim donor 전략). 라이선스 고지는 indoorgs/NOTICE,
#   본문은 indoorgs/licenses/Apache-2.0.txt.
# 옮기고 고친 것 (모두 수정됨):
#   _pick_rim_donors   <- repair._donors_from_rim : 역거리 가중 k=10 이웃 중 하나를 확률적으로 고름 (그대로).
#                         색 보정은 IDW 평균 쪽으로 당기던 것을, 면의 저주파 색장 차이 F(목표)-F(기증자)
#                         만큼 옮기는 것으로 바꿈 (기증자의 고주파 무늬는 남김).
#   _rim_density       <- repair._repair_chart_body 의 per_cell (구멍 둘레 셀 개수의 중앙값). 중앙값 대신
#                         절사 평균 (셀당 개수가 작을 때 정수 중앙값이 밀도를 낮게 잡아서).
#   _deficit_fraction  <- repair._scale_opacity 의 '광학 깊이로 더한다' 생각: 셀마다 남은 덮임을 빼고
#                         모자란 광학 깊이만큼만 새로 넣음 (repair 는 opacity 를 줄였고, 여기는 개수를 줄임).
#   _cells             <- repair._cells (셀 번호 계산) 그대로.
#   통째 복사 원칙       <- repair 모듈 설명: 새 가우시안은 기증자 하나의 scale·rotation·SH·opacity 를 통째로
#                         받고 위치(와 DC 색)만 바꾼다. 여러 이웃의 공분산을 평균하면 '솜뭉치'가 됨.
#   안 옮긴 것: repair._turn_copies (같은 평면 위라 법선이 어디나 같아 회전이 필요 없음, 면을 넘어 복사하지 않음),
#               fit_height_field (면이 평면이라 h = 0), _cell_outliers·patch 전략 (흰 벽은 rim 으로 충분),
#               cleanup.py (벽 뒤 부유물은 snap.remove_behind_walls 가 지움).
# 덮임 지도(coverage), 카메라 가시성, 면 경계·문 구멍 처리는 IndoorGS 에서 새로 씀.
# ---------------------------------------------------------------------------

입력 면(planes)은 walls/planes.py 의 Plane (n·x = d, n 방 안쪽). 좌표는 도면 좌표 (m).

공개 API
  coverage_map(ply, plane, cell=.., band=..)      -> (덮임 0..1 (nv,nu), 면 안 셀 bool (nv,nu))
  repair_faces(ply, planes, *, cell=0.05, min_cover=0.5, max_hole_m2=4.0, cameras=None, rng=0, texture=None, ...)
                                                   -> (새 GaussianPly, info)
  repair_file(in_ply, out_ply, planes, **opts)    -> info   (in == out 이면 거부)

정한 것
- 덮임 지도: 면 (u, v) 를 cell 크기 격자로. 면에서 |부호 거리| < band 이고 면 범위 안인 가우시안 (snap.assign 을
  그 면 하나로 = 다른 면과 겹쳐도 셈. 모서리 셀이 이웃 면에 가우시안을 뺏겨 구멍으로 보이지 않게) 각각의
  면 위 2D 공분산 Σ2 = B R S² Rᵀ Bᵀ (B = [U; V], 법선 방향으로 정사영 = 정면에서 본 모양).
  셀 평균 알파 ≈ sigmoid(opacity) · sqrt(det Σ2 / det Σ2') · exp(-½ dᵀ Σ2'⁻¹ d), Σ2' = Σ2 + cell²/12 · I
  (상자 평균을 가우시안으로 근사: 셀보다 작은 가우시안도 면적만큼 셈). 알파는 0.99 로 자름 (3DGS 래스터라이저와 같음).
  셀 덮임 = 1 - Π(1 - ā) = 1 - exp(-τ), τ = Σ -log(1 - ā).
- 구멍: 면 안 셀(contains_uv, 문 구멍 제외) 중 덮임 < min_cover. 8-이웃 연결 성분 하나 = 구멍 하나.
  cameras(도면 좌표 카메라 중심 (M,3)) 를 주면 어떤 카메라도 못 본 셀은 메우지 않음 (방 안쪽, max_view_dist 안,
  입사각 < max_view_deg, 다른 면에 가리지 않음). 넓이 > max_hole_m2 인 구멍은 메우지 않음 (카메라 유무와 무관;
  None 이면 제한 없음). 사진이 '벽이 있다' 고 말하지 않는 큰 빈 곳(유리, 열린 공간, 잘못 맞춘 면)을 막으려고.
- 기증자: 같은 면에 (모든 면 중 가장 가까운 면으로 배정된) 가우시안 중 sigmoid(opacity) >= 0.2, 면 안쪽 긴 축
  표준편차 <= max_donor_sigma, 중심이 구멍 둘레 rim_m 안의 '잘 덮인' 셀에 있는 것. 모자라면 둘레를 넓힘.
- 개수: 둘레 셀의 기증자 개수 절사 평균 × (모자란 광학 깊이 / 둘레 광학 깊이 중앙값). 확률적 반올림.
  놓은 뒤 덮임을 다시 계산해 아직 모자란 셀만 다시 (최대 passes 번).
- 새 가우시안: 기증자 행을 통째로 복사 (속성 순서 그대로, 법선 nx.. 도). 위치만 목표 셀 안 무작위 점 (면 위,
  부호 거리 0), DC 색 += color_smooth · (F(목표 셀) - F(기증자 셀)). F 는 면의 저주파 색장: 잘 덮인 셀의 불투명도
  가중 평균 DC 를 (중앙값 ± 3 MAD 로 자른 뒤) 가우시안 정규화 합성곱(tone_sigma) 으로 펴고, 빈 곳은 가장 가까운 값.
  같은 면 = 같은 법선이라 회전·SH 는 기증자 그대로.
- texture (선택, {면 id: walls.facetex.FaceTexture}): 주면 새 가우시안의 DC 색은 텍스처의 그 (u, v) 색
  (쌍선형, source > 0 인 픽셀만) 으로: f_dc = (rgb/255 - 0.5) / SH_C0. 기증자는 scale·rotation·opacity·고차 SH 만
  줌. 텍스처가 없는 곳(source 0, 면 id 가 사전에 없음)은 위의 기증자 색 + 색장 차이 그대로.
  texture_rest="keep"(기본): 고차 SH 는 기증자 것 (이웃과 같은 시점 의존 밝기 변화 - 텍스처는 여러 시점의 중앙값이라
  SH 평균 0 과 맞음). "zero": 텍스처 색으로 칠한 가우시안의 f_rest 를 0 으로 (모든 방향에서 텍스처 색 그대로).
  합성 데이터에서는 패치 차이를 잴 수 없어 (기증자 f_rest 평균 0) 기본은 keep, 실제 장면에서 비교 후 정할 것.
- 원본 가우시안은 비트 그대로, 새 것은 뒤에 붙음. 주석(comments)도 그대로.
"""
from __future__ import annotations

import inspect
import time
from pathlib import Path

import numpy as np

from .. import plyio
from ..plyio import GaussianPly
from .planes import Plane
from .snap import assign

ALPHA_MAX = 0.99            # 3DGS 래스터라이저의 알파 상한
DONOR_MIN_ALPHA = 0.2       # repair._rim_band 의 alpha > 0.2
KNN = 10                    # repair._donors_from_rim 의 k
MAX_R_CELLS = 24            # 한 가우시안이 덮임 지도에 닿는 반경 상한 (셀)
RASTER_BATCH = 1 << 22      # 래스터 한 번에 다루는 (가우시안 x 셀) 쌍 수


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def _rng(rng):
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


# ---------------- 격자 ----------------
class _Grid:
    def __init__(self, pl: Plane, cell: float):
        self.cell = cell
        self.nu = max(1, int(np.ceil(pl.size[0] / cell - 1e-9)))
        self.nv = max(1, int(np.ceil(pl.size[1] / cell - 1e-9)))
        uc = (np.arange(self.nu) + 0.5) * cell
        vc = (np.arange(self.nv) + 0.5) * cell
        UU, VV = np.meshgrid(uc, vc)
        self.centers = np.stack([UU.ravel(), VV.ravel()], 1)
        self.inside = pl.contains_uv(self.centers).reshape(self.nv, self.nu)


def _cells(u, v, cell, nu, nv):
    """repair._cells: (u, v) -> 셀 번호, 범위 안 여부."""
    iu = np.floor(u / cell).astype(np.int64)
    iv = np.floor(v / cell).astype(np.int64)
    ok = (iu >= 0) & (iu < nu) & (iv >= 0) & (iv < nv)
    return iu, iv, ok


# ---------------- 가우시안 -> 면 위 모양 ----------------
def _inplane(d: np.ndarray, g: np.ndarray, pl: Plane):
    """g 번 가우시안들의 (u, v), 면 위 2D 공분산 (a, b, c) = (Σuu, Σuv, Σvv), 알파, 쓸 수 있는지."""
    P = np.stack([d["x"][g], d["y"][g], d["z"][g]], 1).astype(np.float64)
    uv = pl.uv(P)
    q = np.stack([d[f"rot_{i}"][g] for i in range(4)], 1).astype(np.float64)
    qn = np.linalg.norm(q, axis=1)
    good = np.isfinite(qn) & (qn > 1e-12)
    q[~good] = (1.0, 0.0, 0.0, 0.0)
    R = plyio.quat_to_mat(q)
    S2 = np.exp(2.0 * np.stack([d[f"scale_{i}"][g] for i in range(3)], 1).astype(np.float64))
    MU = np.einsum("j,njk->nk", pl.U, R)       # U 를 가우시안 축 좌표로
    MV = np.einsum("j,njk->nk", pl.V, R)
    a = np.einsum("nk,nk->n", MU * MU, S2)
    b = np.einsum("nk,nk->n", MU * MV, S2)
    c = np.einsum("nk,nk->n", MV * MV, S2)
    alpha = _sigmoid(d["opacity"][g])
    good &= np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & np.isfinite(alpha) & np.isfinite(uv).all(1)
    return uv, a, b, c, alpha, good


def _raster_tau(uv, a, b, c, alpha, cell, nu, nv) -> np.ndarray:
    """셀마다 광학 깊이 τ = Σ -log(1 - ā) (nv, nu). 모듈 설명의 상자 평균 근사."""
    tau = np.zeros(nv * nu)
    if len(uv) == 0:
        return tau.reshape(nv, nu)
    s2 = cell * cell / 12.0
    A, C, B = a + s2, c + s2, b
    det = np.maximum(A * C - B * B, 1e-30)
    det0 = np.maximum(a * c - b * b, 0.0)
    amp = np.minimum(alpha * np.sqrt(det0 / det), ALPHA_MAX)
    iA, iB, iC = C / det, -B / det, A / det
    lam = 0.5 * (A + C) + np.sqrt(0.25 * (A - C) ** 2 + B * B)
    r = np.clip(np.ceil(3.0 * np.sqrt(lam) / cell), 0, MAX_R_CELLS).astype(np.int64)
    iu0 = np.floor(uv[:, 0] / cell).astype(np.int64)
    iv0 = np.floor(uv[:, 1] / cell).astype(np.int64)
    use = amp > 1e-4
    for rr in np.unique(r[use]):
        gi = np.nonzero(use & (r == rr))[0]
        o = np.arange(-rr, rr + 1)
        ox, oy = [m.ravel() for m in np.meshgrid(o, o)]
        K = len(ox)
        step = max(1, RASTER_BATCH // K)
        for s in range(0, len(gi), step):
            h = gi[s:s + step]
            iu = iu0[h, None] + ox[None]
            iv = iv0[h, None] + oy[None]
            ok = (iu >= 0) & (iu < nu) & (iv >= 0) & (iv < nv)
            du = (iu + 0.5) * cell - uv[h, 0, None]
            dv = (iv + 0.5) * cell - uv[h, 1, None]
            q = iA[h, None] * du * du + 2.0 * iB[h, None] * du * dv + iC[h, None] * dv * dv
            al = amp[h, None] * np.exp(-0.5 * q)
            ok &= al > 1e-4
            tau += np.bincount((iv * nu + iu)[ok], weights=-np.log1p(-al[ok]), minlength=nv * nu)
    return tau.reshape(nv, nu)


def _face_members(xyz, pl: Plane, band: float, uv_margin: float) -> np.ndarray:
    idx, _ = assign(xyz, [pl], band, uv_margin)
    return np.nonzero(idx == 0)[0]


def coverage_map(ply: GaussianPly, plane: Plane, cell: float = 0.05, band: float = 0.06,
                 uv_margin: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """면 하나의 덮임 지도 (nv, nu) 0..1 와 면 안 셀 표시. 행 = v, 열 = u, 셀 (i, j) 중심 = ((j+.5)cell, (i+.5)cell)."""
    gr = _Grid(plane, cell)
    g = _face_members(ply.xyz, plane, band, uv_margin)
    uv, a, b, c, al, good = _inplane(ply.data, g, plane)
    tau = _raster_tau(uv[good], a[good], b[good], c[good], al[good], cell, gr.nu, gr.nv)
    return 1.0 - np.exp(-tau), gr.inside


# ---------------- 카메라 가시성 ----------------
def _seen(pl: Plane, j: int, planes: list[Plane], targets: np.ndarray, cams: np.ndarray, *, max_dist: float,
          max_deg: float, cell: float, max_cams: int) -> np.ndarray:
    """targets (면 위 3D 점) 마다 어떤 카메라라도 가리지 않고 볼 수 있는지."""
    seen = np.zeros(len(targets), bool)
    sdc = cams @ pl.n - pl.d
    cams = cams[sdc > 0.05]
    if len(cams) == 0 or len(targets) == 0:
        return seen
    if len(cams) > max_cams:
        cams = cams[np.linspace(0, len(cams) - 1, max_cams).round().astype(int)]
    sdc = cams @ pl.n - pl.d
    min_cos = np.cos(np.radians(max_deg))
    others = [(o, k) for k, o in enumerate(planes) if k != j]
    step = max(1, 200_000 // len(cams))
    for s in range(0, len(targets), step):
        T = targets[s:s + step]
        D = T[:, None, :] - cams[None]                    # 카메라 -> 목표
        dist = np.linalg.norm(D, axis=2)
        ok = (dist < max_dist) & (sdc[None] / np.maximum(dist, 1e-9) > min_cos)
        for o, _ in others:
            if not ok.any():
                break
            den = D @ o.n
            num = o.d - cams @ o.n
            with np.errstate(divide="ignore", invalid="ignore"):
                t = num[None] / den
            hit = ok & (t > 0) & (t < 1) & ((1 - t) * dist > 2 * cell)
            if not hit.any():
                continue
            ii, kk = np.nonzero(hit)
            X = cams[kk] + t[ii, kk, None] * D[ii, kk]
            blk = o.contains_uv(o.uv(X), 0.0)
            ok[ii[blk], kk[blk]] = False
        seen[s:s + step] = ok.any(1)
    return seen


# ---------------- 채우기 도움 (splat-repair 에서 고쳐 옮김) ----------------
def _rim_density(cnt: np.ndarray, rim: np.ndarray) -> float:
    """repair._repair_chart_body 의 per_cell (둘레 셀 기증자 개수의 중앙값) 을 10~90 백분위 절사 평균으로 바꿈:
    셀당 2~3 개처럼 개수가 작으면 정수 중앙값이 밀도를 30% 까지 낮게 잡음."""
    v = cnt[rim]
    if not v.size:
        return 1.0
    lo, hi = np.percentile(v, [10, 90])
    m = v[(v >= lo) & (v <= hi)]
    return float(np.clip(m.mean() if m.size else np.mean(v), 0.05, 64.0))


def _deficit_fraction(tau_have: np.ndarray, tau_rim: float) -> np.ndarray:
    """모자란 광학 깊이 비율 (repair._scale_opacity 의 광학 깊이 덧셈). 0 = 이미 둘레만큼, 1 = 비어 있음."""
    return np.clip((tau_rim - tau_have) / max(tau_rim, 1e-9), 0.0, 1.0)


def _pick_rim_donors(tree, pool_uv_n: int, targets_uv: np.ndarray, rng) -> np.ndarray:
    """repair._donors_from_rim: k 이웃 중 역거리 가중으로 하나씩 확률적으로. 반환 = pool 안 번호."""
    k = int(min(KNN, pool_uv_n))
    dd, jj = tree.query(targets_uv, k=k)
    if k == 1:
        dd, jj = dd[:, None], jj[:, None]
    w = 1.0 / np.maximum(dd, 1e-9)
    w /= w.sum(1, keepdims=True)
    cum = np.cumsum(w, axis=1)
    pick = (cum < rng.random((len(targets_uv), 1))).sum(1).clip(0, k - 1)
    return jj[np.arange(len(targets_uv)), pick]


def _tone_field(dc_sum: np.ndarray, w_sum: np.ndarray, known: np.ndarray, sigma_cells: float) -> np.ndarray:
    """면의 저주파 DC 색장 (nv, nu, 3). 잘 덮인 셀의 가중 평균을 중앙값 ± 3 MAD 로 자르고 정규화 합성곱, 빈 곳은 가장 가까운 값."""
    from scipy import ndimage

    nv, nu = known.shape
    F = np.zeros((nv, nu, 3))
    have = known & (w_sum > 1e-9)
    if not have.any():
        return F
    mean = np.where(have[..., None], dc_sum / np.maximum(w_sum, 1e-9)[..., None], 0.0)
    vals = mean[have]
    med = np.median(vals, 0)
    mad = np.median(np.abs(vals - med), 0) * 1.4826 + 1e-6
    mean = np.clip(mean, med - 3 * mad, med + 3 * mad)
    m = have.astype(np.float64)
    den = ndimage.gaussian_filter(m, sigma_cells, mode="nearest")
    for ch in range(3):
        F[..., ch] = ndimage.gaussian_filter(mean[..., ch] * m, sigma_cells, mode="nearest") / np.maximum(den, 1e-12)
    ok = den > 1e-3
    if not ok.all():
        ind = ndimage.distance_transform_edt(~ok, return_distances=False, return_indices=True)
        F = F[tuple(ind)]
    return F


# ---------------- 메인 ----------------
def repair_faces(ply: GaussianPly, planes: list[Plane], *, cell: float = 0.05, min_cover: float = 0.5,
                 max_hole_m2: float | None = 4.0, cameras=None, rng=0, band: float = 0.06,
                 uv_margin: float = 0.05, rim_m: float = 0.25, max_donor_sigma: float = 0.25,
                 min_donors: int = 8, color_smooth: float = 1.0, tone_sigma: float = 0.3, passes: int = 6,
                 max_view_dist: float = 12.0, max_view_deg: float = 80.0, max_cams: int = 256,
                 texture: dict | None = None, texture_rest: str = "keep"
                 ) -> tuple[GaussianPly, dict]:
    """면마다 덮임 < min_cover 인 셀(문 구멍 제외)을 같은 면 둘레 기증자의 통째 복사로 메운 새 PLY. 규칙은 모듈 설명.

    info: n_added, n_before, n_after, seconds, params, faces[id] = {hole_m2_before, hole_m2_after, cover_before,
    cover_after (면 안 셀 중 덮임 >= min_cover 비율), n_added, n_holes, n_holes_filled, skipped_large_m2,
    unseen_m2, no_donor_m2, n_textured (texture 를 줬을 때)}."""
    from scipy import ndimage
    from scipy.spatial import cKDTree

    t0 = time.time()
    plyio.check_required(ply.names)
    if not 0 < min_cover < 1:
        raise ValueError("min_cover 는 0..1 사이")
    if texture_rest not in ("keep", "zero"):
        raise ValueError("texture_rest 는 keep | zero")
    rest_names = [k for k in ply.names if k.startswith("f_rest_")]
    rng = _rng(rng)
    d = ply.data
    N = len(d)
    cams = None if cameras is None else np.asarray(cameras, np.float64).reshape(-1, 3)
    tau_thr = -np.log1p(-min_cover)
    xyz = ply.xyz
    own, _ = assign(xyz, planes, band, uv_margin)        # 기증자: 가장 가까운 면 하나에만
    info: dict = {"n_before": int(N), "n_added": 0, "faces": {},
                  "params": {"cell": cell, "min_cover": min_cover, "max_hole_m2": max_hole_m2,
                             "cameras": None if cams is None else int(len(cams)), "band": band,
                             "rim_m": rim_m, "max_donor_sigma": max_donor_sigma, "color_smooth": color_smooth,
                             "tone_sigma": tone_sigma, "passes": passes,
                             "texture": None if not texture else sorted(texture), "texture_rest": texture_rest}}
    new_parts: list[np.ndarray] = []
    cell_a = cell * cell
    for j, pl in enumerate(planes):
        gr = _Grid(pl, cell)
        nu, nv, inside = gr.nu, gr.nv, gr.inside
        fi = {"hole_m2_before": 0.0, "hole_m2_after": 0.0, "cover_before": None, "cover_after": None,
              "n_added": 0, "n_holes": 0, "n_holes_filled": 0, "skipped_large_m2": 0.0, "unseen_m2": 0.0,
              "no_donor_m2": 0.0}
        info["faces"][pl.id] = fi
        tex = texture.get(pl.id) if texture else None
        if tex is not None:
            fi["n_textured"] = 0
        n_in = int(inside.sum())
        if n_in == 0:
            continue
        g = _face_members(xyz, pl, band, uv_margin)
        uv, a, b, c, al, good = _inplane(d, g, pl)
        tau = _raster_tau(uv[good], a[good], b[good], c[good], al[good], cell, nu, nv)
        bad = inside & (tau < tau_thr)
        fi["hole_m2_before"] = float(bad.sum() * cell_a)
        fi["cover_before"] = float(1.0 - bad.sum() / n_in)
        if not bad.any():
            fi["cover_after"] = fi["cover_before"]
            continue

        # 기증자 후보 (이 면에 배정된 것) 와 셀별 개수·색 합
        mine = own[g] == j
        big = np.sqrt(np.maximum(0.5 * (a + c) + np.sqrt(0.25 * (a - c) ** 2 + b * b), 0))
        don = mine & good & (al >= DONOR_MIN_ALPHA) & (big <= max_donor_sigma)
        iu, iv, okc = _cells(uv[:, 0], uv[:, 1], cell, nu, nv)
        don &= okc
        cnt = np.bincount((iv * nu + iu)[don], minlength=nv * nu).reshape(nv, nu).astype(np.float64)
        well = inside & (tau >= tau_thr)
        tm = mine & good & okc
        lin_t = (iv * nu + iu)[tm]
        w_t = al[tm]
        dc_t = np.stack([d[f"f_dc_{k}"][g[tm]] for k in range(3)], 1).astype(np.float64)
        w_sum = np.bincount(lin_t, w_t, nv * nu).reshape(nv, nu)
        dc_sum = np.stack([np.bincount(lin_t, w_t * dc_t[:, k], nv * nu) for k in range(3)], 1).reshape(nv, nu, 3)
        F = _tone_field(dc_sum, w_sum, well, tone_sigma / cell)
        don_i = np.nonzero(don)[0]
        don_cell = (iv * nu + iu)[don_i]

        # 구멍 성분
        fill = bad.copy()
        if cams is not None:
            cv_, cu_ = np.nonzero(bad)
            T = pl.point((cu_ + 0.5) * cell, (cv_ + 0.5) * cell)
            vis = _seen(pl, j, planes, T, cams, max_dist=max_view_dist, max_deg=max_view_deg, cell=cell,
                        max_cams=max_cams)
            fill[cv_[~vis], cu_[~vis]] = False
            fi["unseen_m2"] = float((~vis).sum() * cell_a)
        lab, nlab = ndimage.label(fill, structure=np.ones((3, 3)))
        fi["n_holes"] = int(nlab)
        objs = ndimage.find_objects(lab)
        added_face: list[np.ndarray] = []
        for li, sl in enumerate(objs, start=1):
            if sl is None:
                continue
            area = float((lab[sl] == li).sum() * cell_a)
            if max_hole_m2 is not None and area > max_hole_m2:
                fi["skipped_large_m2"] += area
                continue
            # 둘레: 성분을 rim_m 만큼 키운 곳의 잘 덮인 셀. 기증자가 모자라면 넓힘
            pool = None
            for grow in (1, 2, 4, 8):
                rc = int(np.ceil(grow * rim_m / cell))
                y0, y1 = max(sl[0].start - rc, 0), min(sl[0].stop + rc, nv)
                x0, x1 = max(sl[1].start - rc, 0), min(sl[1].stop + rc, nu)
                comp = lab[y0:y1, x0:x1] == li
                ring = ndimage.binary_dilation(comp, np.ones((3, 3), bool), iterations=rc) & well[y0:y1, x0:x1]
                if not ring.any():
                    continue
                rim = np.zeros((nv, nu), bool)
                rim[y0:y1, x0:x1] = ring
                sel = rim.ravel()[don_cell]
                if sel.sum() >= min_donors:
                    pool = don_i[sel]
                    break
            if pool is None:
                fi["no_donor_m2"] += area
                continue
            fi["n_holes_filled"] += 1
            per_cell = _rim_density(cnt, rim)
            tau_rim = float(np.median(tau[rim]))
            puv = uv[pool]
            tree = cKDTree(puv)
            pcell_u = np.clip(np.floor(puv[:, 0] / cell).astype(np.int64), 0, nu - 1)
            pcell_v = np.clip(np.floor(puv[:, 1] / cell).astype(np.int64), 0, nv - 1)
            cy, cx = np.nonzero(comp)
            cy, cx = cy + y0, cx + x0
            todo = np.ones(len(cy), bool)
            for ps in range(passes):
                if not todo.any():
                    break
                ty, tx = cy[todo], cx[todo]
                f = _deficit_fraction(tau[ty, tx], tau_rim)
                if ps > 0:
                    f = np.maximum(f, 0.25)               # 둘레만큼 됐는데도 기준 아래면 조금씩 더
                want = per_cell * f
                k = np.floor(want).astype(np.int64)
                k += rng.random(len(want)) < (want - k)
                if ps > 0:
                    k = np.maximum(k, 1)
                if k.sum() == 0:
                    k[:] = 1
                ty, tx = np.repeat(ty, k), np.repeat(tx, k)
                M = len(ty)
                tu = (tx + rng.random(M)) * cell
                tv = (ty + rng.random(M)) * cell
                out = ~pl.contains_uv(np.stack([tu, tv], 1))   # 셀이 면 가장자리·문 구멍에 걸치면 셀 중심으로
                tu[out] = (tx[out] + 0.5) * cell
                tv[out] = (ty[out] + 0.5) * cell
                loc = _pick_rim_donors(tree, len(pool), np.stack([tu, tv], 1), rng)
                pick = pool[loc]
                rows = d[g[pick]].copy()
                X = pl.point(tu, tv)
                for ci, cn in enumerate("xyz"):
                    rows[cn] = X[:, ci]
                use_tex = np.zeros(M, bool)
                if tex is not None:
                    trgb, tsrc = tex.sample(np.stack([tu, tv], 1))
                    use_tex = tsrc > 0
                    tdc = (trgb / 255.0 - 0.5) / plyio.SH_C0
                    for ci in range(3):
                        rows[f"f_dc_{ci}"][use_tex] = tdc[use_tex, ci]
                    if texture_rest == "zero":
                        for k in rest_names:
                            rows[k][use_tex] = 0
                    fi["n_textured"] += int(use_tex.sum())
                if color_smooth:
                    shift = color_smooth * (F[ty, tx] - F[pcell_v[loc], pcell_u[loc]])
                    shift[use_tex] = 0
                    for ci in range(3):
                        rows[f"f_dc_{ci}"] = rows[f"f_dc_{ci}"].astype(np.float64) + shift[:, ci]
                added_face.append(rows)
                fi["n_added"] += int(M)
                nuv, na, nb_, nc, nal, ngood = _inplane(rows, np.arange(M), pl)
                tau += _raster_tau(nuv[ngood], na[ngood], nb_[ngood], nc[ngood], nal[ngood], cell, nu, nv)
                todo = tau[cy, cx] < tau_thr
        new_parts.extend(added_face)
        after = inside & (tau < tau_thr)
        fi["hole_m2_after"] = float(after.sum() * cell_a)
        fi["cover_after"] = float(1.0 - after.sum() / n_in)
    if new_parts:
        out = np.concatenate([d] + new_parts)
    else:
        out = d.copy()
    info["n_added"] = int(len(out) - N)
    info["n_after"] = int(len(out))
    info["hole_m2_before"] = float(sum(f["hole_m2_before"] for f in info["faces"].values()))
    info["hole_m2_after"] = float(sum(f["hole_m2_after"] for f in info["faces"].values()))
    info["seconds"] = round(time.time() - t0, 2)
    return GaussianPly(out, list(ply.comments)), info


_REPAIR_KEYS = {k for k, p in inspect.signature(repair_faces).parameters.items() if p.kind is p.KEYWORD_ONLY}


def repair_file(in_ply, out_ply, planes: list[Plane], **opts) -> dict:
    """읽기 -> repair_faces -> 새 파일에 쓰기. in_ply 와 out_ply 가 같으면 거부 (원본 안 덮음)."""
    bad = set(opts) - _REPAIR_KEYS
    if bad:
        raise TypeError(f"모르는 옵션 {sorted(bad)}")
    in_ply, out_ply = Path(in_ply), Path(out_ply)
    if in_ply.resolve() == out_ply.resolve():
        raise ValueError("out_ply 가 in_ply 와 같음 - 원본을 덮어쓰지 않음")
    ply = plyio.read(in_ply)
    out, info = repair_faces(ply, planes, **opts)
    info = {"in": str(in_ply), "out": str(out_ply), "stats_before": plyio.stats(ply), **info,
            "stats_after": plyio.stats(out)}
    plyio.write(out_ply, out)
    return info
