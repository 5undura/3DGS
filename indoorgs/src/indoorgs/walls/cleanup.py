"""학습 뒤 스플랫 자동 정리 (사용자 손 안 감, CPU 만, 300만~600만 개를 수십 초에).

지운이 VR 에서 본 문제 (복도 1, Brush·LichtFeld): 벽을 비스듬히 보면 줄무늬(바늘형 ~16%), 비치는 벽, 방 안에 뜬
부유물, 시점을 바꾸면 색이 튐. 이 모듈은 그중 면(도면)과 상관없이 스플랫 자체로 고칠 수 있는 것을 고친다.
(벽 뒤 부유물은 snap.remove_behind_walls, 비치는 면은 repair 가 맡음.)

# ---------------------------------------------------------------------------
# 출처 (Apache License 2.0): remove_floaters 의 일부 생각은 splat-repair 에서 가져와 고친 것이다.
#   splat-repair, Copyright 2026 Nicolas Drum, https://github.com/NicolasDrum/splat-repair
#   core/splatcore/cleanup.py. 라이선스 고지는 indoorgs/NOTICE, 본문은 indoorgs/licenses/Apache-2.0.txt.
# 가져와 고친 것:
#   - "고립(isolation) 혼자는 증거가 아니다": kNN 거리로 잰 고립은 흐림(낮은 불투명도) 또는 큼(bulk) 과
#     **함께**일 때만 지움 (cleanup.analyse 의 v_iso x max(v_faint, v_bulk)). 여기서는 부드러운 점수 대신
#     문턱 두 개의 AND (보수적, 결과를 설명하기 쉬움).
#   - "따로 떨어진 작은 무리(detached)": cleanup._detached 의 연결 성분 생각. 원본은 가우시안 kNN 그래프
#     (거리 < reach x 중앙 간격 또는 2σ 가 닿음) 에서 전체의 0.1% (최대 5000) 보다 작은 성분을 지움.
#     여기서는 voxel(기본 10 cm) 26-이웃 그래프의 성분이고, 상한을 훨씬 작게 (기본 min(300, max(20, 5e-5 N)))
#     잡음: 실내에서 매달린 표지판·조명은 수백~수천 개라서.
#   안 옮긴 것: GPU 렌더로 재는 hidden/fringe/reveal (아래서 본 안개 - 실내에는 '아래' 가 바닥이라 해당 없음).
# 카메라 앞 안개(cameras), 면 보호(planes), 방 안만(plan), voxel kNN 고립은 IndoorGS 에서 새로 씀.
# 같은 저장소 tools/fill/clean_section.py (복도 1 실제 데이터에서 쓰던 정리) 의 기준 C(투과)·D(허공) 와 값을 옮김
# (see_through_votes, remove_floaters model=). 축척 S = 카메라 높이/0.67 ≈ 2.1 m 를 곱해 미터로 고정 (도면 좌표라서).
# split_needles 의 target 은 tools/flatten.py --aniso (벽 따라 길쭉함 상한, 도움말 6 / 코드 기본 10) 를 참고:
# flatten 은 긴 축을 잘라 길이를 잃지만 여기서는 조각으로 나눠 길이를 지킴.
# ---------------------------------------------------------------------------

공개 API
  split_needles(ply, *, ratio=10, max_pieces=8, keep_volume=True, target=3, ...)  -> (ply, info)  바늘형 쪼개기
  remove_floaters(ply, *, planes=None, plan=None, cameras=None, ...)              -> (ply, info)  부유물 지우기
  fix_huge(ply, *, max_axis=0.5, ...)                                             -> (ply, info)  아주 큰 것
  damp_sh(ply, *, max_ratio=1.0, factor=0.3, ...)                                 -> (ply, info)  색 튐 줄이기
  ply_quality(ply)                                                                -> dict  전후 비교 요약
  cleanup_file(in_ply, out_ply, *, planes, plan, cameras, steps, needles={..}, ...) -> info

공통
- 속성은 이름으로 (plyio). rot_0..3 = (w,x,y,z) 정규화 안 됐을 수 있음 -> 축 계산할 때만 정규화, 값은 그대로 복사.
  scale_* = log(표준편차 σ, m), opacity = logit. 축 a 의 방향 = R(q) 의 a 번째 열 (Σ = R diag(σ²) Rᵀ).
- 건드리지 않은 행은 비트 그대로·원래 순서. 쪼갠 행은 빼고, 조각은 맨 뒤에 붙음. 속성 순서·주석 그대로.

1) 바늘 쪼개기 (split_needles)
  왜: 바늘형(긴 축/중간 축 > 10, plyio.stats 와 같은 정의) 은 정면에서는 벽을 잘 덮지만, 비스듬히 보면
  (a) 3DGS 래스터라이저(Unity 플러그인 포함)는 가우시안을 **중심의 원근 야코비안 하나**로 평면에 투영(EWA
  아핀 근사)하는데, 긴 가우시안의 먼 끝은 실제 원근과 크게 어긋나 이웃 바늘과 틈·겹침이 생기고 (줄무늬),
  (b) 정렬도 중심 깊이 하나라 바늘 길이 방향으로 앞뒤가 뒤바뀜. 짧은 조각으로 나누면 두 오차 모두 조각 길이에
  비례해 줄어든다. (합성 측정은 tests/test_h_cleanup.py 와 보고서 참고)
  어떻게 (긴 축 길이 방향 1D 프로파일만 바꿈, 나머지 두 축·회전·색·SH 그대로):
  - σ_L = 긴 축 σ, σ_M = 중간 축 σ. 조각 σ_p = target·σ_M (조각 종횡비 ≤ target).
  - 가우시안 합성곱 항등식 N(0, σ_L²) = N(0, σ_p²) * N(0, s²), s² = σ_L² - σ_p² 를 이산화:
      g_L(t) = exp(-t²/2σ_L²) ≈ Σ_i f_i · exp(-(t - t_i)²/2σ_p²)
    t_i = u_i · s, u_i = 등간격 [-span, +span] 의 k 점 (span 기본 1.8 = ±1.8 s),
    w_i = exp(-u_i²/2) (N(0, s²) 의 밀도), f_i = σ_L·w_i / (σ_p·Σw)  -> Σ f_i·σ_p = σ_L
    즉 축 방향으로 적분한 알파 ∫ o·g dt 가 정확히 보존됨 (o 는 원래 불투명도).
  - 조각 수 k = ceil(2·span·s / (spacing·σ_p)) + 1 (조각 간격 ≤ spacing·σ_p, spacing 기본 2.0:
    간격 2σ 의 가우시안 줄은 물결이 진폭의 ~1.4%). k > max_pieces 면 k = max_pieces 로 두고 간격이 맞게
    σ_p 를 키움: σ_p = 2·span·σ_L / sqrt(spacing²(k-1)² + 4·span²) (이때 조각 종횡비 > target, info 에 셈).
    정면 프로파일 최대 오차 (σ_L/σ_p = 3.3): k=6 9%, k=7 ~6%, k=8 3% (나머지는 꼬리 자름 + 물결).
  - 불투명도 (keep_volume=True, piece_opacity): 두 극한의 평균
      o_i = ½·[ min(f_i·o, 0.99)  +  (1 - (1-o)^{f_i}) ]
    앞 항 = 알파 합 보존 (투명한 쪽·바늘 가장자리에서 정확, 겹친 조각의 1-Π(1-α) 합성 때문에 불투명한 쪽은
    덜 덮임), 뒤 항 = 광학 깊이 τ = -ln(1-o) 를 f_i 로 나눔 (불투명한 중심에서 정확, 꼬리는 더 진해짐).
    o 가 작으면 둘 다 f_i·o -> 축 방향 적분 알파 Σ o_i σ_p = o σ_L 보존. 합성 벽(o = 0.3/0.7/0.95)에서 정면 덮임
    차이: 알파 합 -3~-7%, 광학 깊이 +2~+10%, 평균 -1~+3% (tests/test_h_cleanup.py, 보고서).
    keep_volume=False 면 조각이 원래 o 그대로 (더 진해짐, 비교용).
  - 주의: 조각은 원래보다 투명하므로 plyio.stats 의 translucent_ratio(< 0.5 비율) 는 **올라감** (보이는 덮임은 같음).
    개수도 늘어남 (바늘 1개 -> 보통 7~8 조각; max_pieces 로 줄일 수 있으나 그만큼 조각이 길어짐).
2) 부유물 (remove_floaters) - 아래 함수 주석.
3) 아주 큰 것 (fix_huge): 가장 긴 축 σ > max_axis. sigmoid(opacity) < translucent(0.3) 면 지움 (방 전체에 깔린
   옅은 안개 = 비치는 벽·뿌연 화면의 원인). 아니면 1) 의 같은 식으로 σ > max_axis 인 축마다 차례로 쪼갬
   (σ_p 상한 = max_axis). 축 둘·셋이 크면 조각 수는 곱 (축마다 max_pieces 까지).
4) SH 튐 (damp_sh) - 아래 함수 주석.
"""
from __future__ import annotations

import inspect
import time
from pathlib import Path

import numpy as np

from .. import plyio
from ..plyio import GaussianPly

CHUNK = 1 << 20              # 한 번에 다루는 가우시안 수 (메모리 상한)
HUGE_AXIS = 0.5              # ply_quality 의 "아주 큰" 기준 (가장 긴 축 σ, m)
_O_EPS = 1e-7
_KEY_BITS = 21               # voxel 번호 축마다 21 비트 (5 cm 면 105 km)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def _logit(o):
    o = np.clip(o, _O_EPS, 1 - _O_EPS)
    return np.log(o) - np.log1p(-o)


def _scales(d, sel=slice(None)) -> np.ndarray:
    return np.exp(np.stack([d[f"scale_{i}"][sel] for i in range(3)], 1).astype(np.float64))


def _xyz(d, sel=slice(None)) -> np.ndarray:
    return np.stack([d["x"][sel], d["y"][sel], d["z"][sel]], 1).astype(np.float64)


def _axis_dirs(d, sel: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """sel 행의 axis 번째 축 방향 (M,3) = R(q) 의 열. 쿼터니언은 여기서만 정규화."""
    q = np.stack([d[f"rot_{i}"][sel] for i in range(4)], 1).astype(np.float64)
    nq = np.linalg.norm(q, axis=1, keepdims=True)
    q = np.where(nq > 1e-12, q, np.array([1.0, 0, 0, 0]))
    R = plyio.quat_to_mat(q)
    return R[np.arange(len(sel)), :, axis]


def _replace(ply: GaussianPly, drop: np.ndarray, new: list[np.ndarray]) -> GaussianPly:
    """drop 표시된 행을 빼고 (나머지는 비트 그대로·순서 그대로) new 를 뒤에 붙인 새 PLY. 한 번만 할당."""
    d = ply.data
    keep = ~drop
    n_new = sum(len(a) for a in new)
    out = np.empty(int(keep.sum()) + n_new, d.dtype)
    pos = 0
    for s in range(0, len(d), CHUNK):
        part = d[s:s + CHUNK][keep[s:s + CHUNK]]
        out[pos:pos + len(part)] = part
        pos += len(part)
    for a in new:
        out[pos:pos + len(a)] = a
        pos += len(a)
    return GaussianPly(out, list(ply.comments))


# ---------------- 1D 쪼개기 공통 수학 ----------------
def split_layout(sigma_long, sigma_piece_max, max_pieces: int = 8, span: float = 1.8, spacing: float = 2.0):
    """조각 수 k (정수 배열), 조각 σ_p (배열). 식은 모듈 주석 1).  sigma_long <= sigma_piece_max 면 k = 1."""
    sL = np.asarray(sigma_long, np.float64)
    sp = np.minimum(np.asarray(sigma_piece_max, np.float64), sL)
    s = np.sqrt(np.maximum(sL ** 2 - sp ** 2, 0.0))
    k = np.ceil(2 * span * s / (spacing * np.maximum(sp, 1e-30)) - 1e-9).astype(np.int64) + 1
    k = np.where(sL > sp * (1 + 1e-9), np.maximum(k, 2), 1)
    cap = k > max_pieces
    k = np.minimum(k, max(1, int(max_pieces)))
    sp = np.where(cap, 2 * span * sL / np.sqrt(spacing ** 2 * (k - 1) ** 2 + 4 * span ** 2), sp)
    return k, sp


def split_weights(k: int, sigma_long, sigma_piece, span: float = 1.8):
    """k 조각의 위치 비율 u (k,) 와 (M,k) 위치 t_i, 광학 깊이 비율 f_i. Σ f_i σ_p = σ_L."""
    u = np.linspace(-span, span, k) if k > 1 else np.zeros(1)
    w = np.exp(-0.5 * u ** 2)
    sL = np.asarray(sigma_long, np.float64)[:, None]
    sp = np.asarray(sigma_piece, np.float64)[:, None]
    s = np.sqrt(np.maximum(sL ** 2 - sp ** 2, 0.0))
    t = u[None] * s
    f = sL * w[None] / (sp * w.sum())
    return t, f


OPACITY_MODE = "mix"


def piece_opacity(o: np.ndarray, f: np.ndarray, mode: str | None = None) -> np.ndarray:
    """원래 불투명도 o (M,) 와 비율 f (M,k) -> 조각 불투명도 (M,k)."""
    mode = mode or OPACITY_MODE
    lin = np.minimum(o[:, None] * f, 0.99)
    dep = -np.expm1(np.log1p(-o)[:, None] * f)
    if mode == "alpha":
        return lin
    if mode == "depth":
        return dep
    return 0.5 * (lin + dep)


def _split_rows(d: np.ndarray, idx: np.ndarray, axis: np.ndarray, sp_max: np.ndarray, *, max_pieces: int,
                span: float, spacing: float, keep_volume: bool) -> tuple[np.ndarray, dict]:
    """d[idx] 를 axis(행마다 0..2) 방향으로 쪼갠 조각 행 (구조화 배열). 다른 두 축·회전·색·SH 는 그대로."""
    st = {"n_split": 0, "n_pieces": 0, "n_capped": 0}
    if len(idx) == 0:
        return d[:0].copy(), st
    S = _scales(d, idx)
    sL = S[np.arange(len(idx)), axis]
    k, sp = split_layout(sL, sp_max, max_pieces, span, spacing)
    k0, _ = split_layout(sL, sp_max, 1 << 30, span, spacing)
    st["n_capped"] = int(np.sum(k0 > k))
    outs = []
    for kk in np.unique(k):
        if kk < 2:
            continue
        g = np.nonzero(k == kk)[0]
        for s0 in range(0, len(g), CHUNK // 8):
            gg = g[s0:s0 + CHUNK // 8]
            rows = idx[gg]
            a = _axis_dirs(d, rows, axis[gg])                        # (M,3)
            t, f = split_weights(int(kk), sL[gg], sp[gg], span)       # (M,k)
            P = _xyz(d, rows)[:, None, :] + t[..., None] * a[:, None, :]
            new = np.repeat(d[rows], kk)
            for i, c in enumerate("xyz"):
                new[c] = P[..., i].ravel()
            ax = np.repeat(axis[gg], kk)
            lsp = np.repeat(np.log(sp[gg]), kk)
            for j in range(3):
                m = ax == j
                if m.any():
                    col = new[f"scale_{j}"]
                    col[m] = lsp[m]
            if keep_volume:
                o = np.minimum(_sigmoid(d["opacity"][rows]), 1 - _O_EPS)
                new["opacity"] = _logit(piece_opacity(o, f)).ravel()
            outs.append(new)
            st["n_split"] += len(gg)
            st["n_pieces"] += len(gg) * int(kk)
    return (np.concatenate(outs) if outs else d[:0].copy()), st


def _needle_ratio(S: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(긴 축/중간 축, 긴 축 번호, 중간 축 σ)."""
    order = np.argsort(S, 1)
    r = np.arange(len(S))
    sl, sm = S[r, order[:, 2]], S[r, order[:, 1]]
    return sl / np.maximum(sm, 1e-12), order[:, 2], sm


# ---------------- 1) 바늘 쪼개기 ----------------
def split_needles(ply: GaussianPly, *, ratio: float = 10.0, max_pieces: int = 8, keep_volume: bool = True,
                  target: float = 3.0, span: float = 1.8, spacing: float = 2.0,
                  min_opacity: float = 0.0) -> tuple[GaussianPly, dict]:
    """긴 축/중간 축 > ratio 인 가우시안을 긴 축 위 조각 k 개로 (모듈 주석 1). 조각 종횡비 ≤ target
    (max_pieces 에 걸리면 넘을 수 있음, info n_capped). min_opacity(0..1) 보다 투명한 바늘은 그대로 둠."""
    plyio.check_required(ply.names)
    if target >= ratio:
        raise ValueError("target 은 ratio 보다 작아야 함")
    d = ply.data
    N = len(d)
    needle = np.zeros(N, bool)
    axis = np.zeros(N, np.int8)
    spm = np.zeros(N)
    for s in range(0, N, CHUNK):
        S = _scales(d, slice(s, s + CHUNK))
        r, a, sm = _needle_ratio(S)
        m = r > ratio
        if min_opacity > 0:
            m &= _sigmoid(d["opacity"][s:s + CHUNK]) >= min_opacity
        needle[s:s + CHUNK] = m
        axis[s:s + CHUNK] = a
        spm[s:s + CHUNK] = target * sm
    idx = np.nonzero(needle)[0]
    new, st = _split_rows(d, idx, axis[idx].astype(np.int64), spm[idx], max_pieces=max_pieces, span=span,
                          spacing=spacing, keep_volume=keep_volume)
    out = _replace(ply, needle, [new])
    info = {"n_in": N, "n_needles": int(len(idx)), "n_pieces": st["n_pieces"], "n_capped": st["n_capped"],
            "n_out": out.n, "params": {"ratio": ratio, "max_pieces": max_pieces, "keep_volume": keep_volume,
                                       "target": target, "span": span, "spacing": spacing,
                                       "min_opacity": min_opacity}}
    return out, info


# ---------------- 3) 아주 큰 것 ----------------
def fix_huge(ply: GaussianPly, *, max_axis: float = HUGE_AXIS, translucent: float = 0.3, max_pieces: int = 8,
             span: float = 1.8, spacing: float = 2.0, keep_volume: bool = True,
             return_removed: bool = False) -> tuple[GaussianPly, dict]:
    """가장 긴 축 σ > max_axis(m): 투명(sigmoid < translucent) 하면 지우고, 아니면 σ > max_axis 인 축마다 쪼갬."""
    plyio.check_required(ply.names)
    d = ply.data
    N = len(d)
    huge = np.zeros(N, bool)
    for s in range(0, N, CHUNK):
        huge[s:s + CHUNK] = _scales(d, slice(s, s + CHUNK)).max(1) > max_axis
    hi = np.nonzero(huge)[0]
    op = _sigmoid(d["opacity"][hi])
    rm = hi[op < translucent]
    sp_idx = hi[op >= translucent]
    work = d[sp_idx].copy()
    n_pieces_capped = 0
    for _ in range(3):                                   # 축마다 한 번씩 (가장 긴 축부터)
        if len(work) == 0:
            break
        S = _scales(work)
        a = np.argmax(S, 1)
        big = S[np.arange(len(work)), a] > max_axis * (1 + 1e-6)
        if not big.any():
            break
        bi = np.nonzero(big)[0]
        pieces, st = _split_rows(work, bi, a[bi], np.full(len(bi), max_axis), max_pieces=max_pieces, span=span,
                                 spacing=spacing, keep_volume=keep_volume)
        n_pieces_capped += st["n_capped"]
        work = np.concatenate([work[~big], pieces])
    out = _replace(ply, huge, [work])
    info = {"n_in": N, "n_huge": int(len(hi)), "n_removed": int(len(rm)), "n_split": int(len(sp_idx)),
            "n_pieces": int(len(work)), "n_capped": int(n_pieces_capped), "n_out": out.n,
            "params": {"max_axis": max_axis, "translucent": translucent, "max_pieces": max_pieces}}
    if return_removed:
        info["removed"] = GaussianPly(d[rm].copy(), list(ply.comments))
    return out, info


# ---------------- 2) 부유물 ----------------
def _voxel_index(P: np.ndarray, v: float, origin: np.ndarray) -> np.ndarray:
    lim = (1 << _KEY_BITS) - 1
    return np.clip(np.floor((P - origin) / v), 0, lim).astype(np.int64)


def _pack(I: np.ndarray) -> np.ndarray:
    return (I[:, 0] << (2 * _KEY_BITS)) | (I[:, 1] << _KEY_BITS) | I[:, 2]


def _unpack(k: np.ndarray) -> np.ndarray:
    m = (1 << _KEY_BITS) - 1
    return np.stack([(k >> (2 * _KEY_BITS)) & m, (k >> _KEY_BITS) & m, k & m], 1)


def _voxel_components(keys_u: np.ndarray) -> np.ndarray:
    """정렬된 고유 voxel 번호 -> 26-이웃 연결 성분 번호."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    V = len(keys_u)
    I = _unpack(keys_u)
    lim = (1 << _KEY_BITS) - 1
    offs = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) > (0, 0, 0)]
    ii, jj = [], []
    for o in offs:                                       # 13 방향 (반대 방향은 대칭)
        J = I + np.array(o)
        ok = np.all((J >= 0) & (J <= lim), 1)
        nk = _pack(J[ok])
        src = np.nonzero(ok)[0]
        pos = np.searchsorted(keys_u, nk)
        pos = np.minimum(pos, V - 1)
        hit = keys_u[pos] == nk
        ii.append(src[hit])
        jj.append(pos[hit])
    ii = np.concatenate(ii) if ii else np.zeros(0, np.int64)
    jj = np.concatenate(jj) if jj else np.zeros(0, np.int64)
    g = coo_matrix((np.ones(len(ii), np.int8), (ii, jj)), shape=(V, V))
    _, lab = connected_components(g, directed=False)
    return lab


def _intrinsics(cam) -> tuple[float, float, float, float]:
    """COLMAP 카메라 -> fx, fy, cx, cy (왜곡 계수는 무시: 학습 입력은 왜곡 보정된 PINHOLE)."""
    p = np.asarray(cam.params, np.float64)
    if cam.model in (1, 4, 5, 6):                         # PINHOLE, OPENCV, OPENCV_FISHEYE, FULL_OPENCV: fx fy cx cy ..
        return p[0], p[1], p[2], p[3]
    return p[0], p[0], p[1], p[2]                         # SIMPLE_*, RADIAL*, FOV: f cx cy ..


def _model_points(model, max_error: float) -> np.ndarray:
    pts = [pt.xyz for pt in model.points.values() if pt.error < max_error]
    return np.asarray(pts, np.float64).reshape(-1, 3)


def see_through_votes(P: np.ndarray, model, *, max_views: int = 100, grid=(48, 27), min_pts: int = 3,
                      near: float = 0.10, front_frac: float = 0.7, front_gap: float = 0.21,
                      agree_frac: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """기준 C (tools/fill/clean_section.py 의 '투과 부유물' 을 옮김, 값은 축척 S≈2.1 m 를 미터로 푼 것).
    사진마다 그 사진에서 실제로 관측된 COLMAP 점의 깊이를 화면 grid 칸별 최솟값으로 (점 < min_pts 칸은 모름),
    스플랫 중심 깊이 z 가 z < front_frac·d 이고 d - z > front_gap 이면 through (보인 표면보다 확실히 앞 = 허공),
    |z - d| < agree_frac·d 면 agree. 사진은 고르게 최대 max_views 장 (원본은 3장마다 1장). 반환 (through, agree) int16."""
    N = len(P)
    through = np.zeros(N, np.int16)
    agree = np.zeros(N, np.int16)
    ims = [im for im in model.images.values() if len(im.pid) and np.sum(im.pid >= 0) >= 50]
    if not ims:
        return through, agree
    if len(ims) > max_views:
        ims = [ims[i] for i in np.linspace(0, len(ims) - 1, max_views).round().astype(int)]
    pid_xyz = {pid: pt.xyz for pid, pt in model.points.items()}
    GX, GY = grid
    P32 = P.astype(np.float32)
    for im in ims:
        cam = model.cameras[im.camera_id]
        fx, fy, cx, cy = _intrinsics(cam)
        W_, H_ = float(cam.width), float(cam.height)
        R, t = im.R, np.asarray(im.t, np.float64)
        ok = np.array([p in pid_xyz for p in im.pid.tolist()], bool) & (im.pid >= 0)
        if ok.sum() < 50:
            continue
        X = np.array([pid_xyz[p] for p in im.pid[ok].tolist()])
        dz = (X @ R.T + t)[:, 2]
        uv = im.xy[ok]
        cid = (np.clip((uv[:, 1] / H_ * GY).astype(int), 0, GY - 1) * GX
               + np.clip((uv[:, 0] / W_ * GX).astype(int), 0, GX - 1))
        dmin = np.full(GX * GY, np.inf)
        np.minimum.at(dmin, cid, dz)
        dmin[np.bincount(cid, minlength=GX * GY) < min_pts] = np.inf
        R32, t32 = R.astype(np.float32), t.astype(np.float32)
        for s in range(0, N, CHUNK):
            Sc = P32[s:s + CHUNK] @ R32.T + t32
            z = Sc[:, 2]
            ii = np.nonzero(z > near)[0]
            if len(ii) == 0:
                continue
            zi = z[ii]
            u = fx * Sc[ii, 0] / zi + cx
            v = fy * Sc[ii, 1] / zi + cy
            vis = (u >= 0) & (u < W_) & (v >= 0) & (v < H_)
            ii, zi, u, v = ii[vis], zi[vis], u[vis], v[vis]
            ds = dmin[np.clip((v / H_ * GY).astype(int), 0, GY - 1) * GX + np.clip((u / W_ * GX).astype(int), 0, GX - 1)]
            known = np.isfinite(ds)
            th = known & (zi < ds * front_frac) & (ds - zi > front_gap)
            ag = known & (np.abs(zi - ds) < ds * agree_frac)
            through[s + ii[th]] += 1
            agree[s + ii[ag]] += 1
    return through, agree


def remove_floaters(ply: GaussianPly, *, planes=None, plan=None, cameras=None, model=None,
                    through_min: int = 4, through_ratio: float = 3.0, through_views: int = 100,
                    air_gap: float = 0.17, air_wall: float = 0.25, point_max_error: float = 2.0,
                    return_removed: bool = False,
                    faint: float = 0.1, iso_voxel: float = 0.05, solid_opacity: float = 0.25, iso_k: int = 4,
                    iso_m: float = 0.15, big_rel: float = 4.0, big_min: float = 0.05, big_max_opacity: float = 0.5,
                    cluster_voxel: float = 0.10, cluster_min_opacity: float = 0.1, cluster_max: int | None = None,
                    cam_radius: float = 0.2, face_keep: float = 0.10, face_uv_margin: float = 0.05
                    ) -> tuple[GaussianPly, dict]:
    """방 안에 떠 있는 부유물 지우기. 이유 (앞의 것부터 하나로 셈, info["removed_by"]):
    model (indoorgs.colmap.model.Model, 도면 좌표의 정합 결과) 을 주면 tools/fill/clean_section.py 의 실제 데이터
    기준 C·D 도 씀 (복도 1 에서 쓰던 값. A 바닥 아래·B 천장 위는 snap.remove_behind_walls 가 하므로 안 함):
    - through (C): see_through_votes 의 through >= through_min 이고 through > through_ratio x agree
                (여러 사진에서 실제 관측 표면보다 확실히 앞 허공). 점이 없는 칸(흰 천장 등)은 판정 안 함.
    - air (D):  관측 점(재투영 오차 < point_max_error)에서 air_gap(17 cm = 0.08 S) 넘게 떨어지고, 어느 면에서도
                air_wall(25 cm) 안쪽이 아닌 것 (무늬 없는 흰 벽은 점이 없으니 보호). 면이 필요해서 planes 나 plan 이
                있을 때만 (plan 만 있으면 planes_from_plan). 바닥·천장 근처도 면 거리로 보호됨 (원본: 바닥 위·천장 아래 0.1 S).
    cameras 를 안 주고 model 을 주면 카메라 중심은 model 에서.
    return_removed: info["removed"] 에 지운 행만 모은 GaussianPly (사람이 열어 확인용, cleanup_file removed_ply).
    - camera:   카메라 중심(cameras (M,3), 도면 좌표) 에서 cam_radius 안. 카메라(와 든 사람)가 있던 자리라 실제
                물체일 수 없음 -> "카메라 앞 안개" (학습 사진 한두 장만 설명하려고 렌즈 앞에 생긴 것).
    - detached: cluster_voxel 크기 voxel 의 26-이웃 그래프에서 작은 성분 (가우시안 수 < cluster_max, 기본
                min(300, max(20, 5e-5 N))). sigmoid >= cluster_min_opacity 인 것만 그래프를 만듦 (옅은 안개 줄이
                부유물을 벽에 이어 주지 않게). splat-repair cleanup._detached 를 고친 것.
    - faint:    sigmoid(opacity) < faint 이고 고립: sigmoid >= solid_opacity 인 가우시안이 하나라도 있는 iso_voxel
                voxel 중심(cKDTree) 까지 iso_k 번째 거리 > iso_m. (불투명도 '합' 문턱은 성긴 곳(어두운 구석)의
                진짜 면을 고립으로 봐서 버림: 하나라도 있으면 그 자리에 무언가 있음)
    - big:      가장 긴 축 σ > max(big_min, big_rel x 장면 중앙값) 이고 sigmoid < big_max_opacity 이고 같은 고립.
    보호 (지우지 않음): planes 를 주면 어느 면에서든 |부호 거리| < face_keep 이고 면 범위 안(uv_margin 허용)인 것
    (면 근처는 snap·repair 담당). plan 을 주면 어느 방(다각형 + 바닥~천장) 안에도 없는 것 (벽 뒤는
    snap.remove_behind_walls 담당, 문 너머 도면 밖은 남김).
    고립만으로는 지우지 않음 (splat-repair: 어두운 구석·얇은 물체도 성기다)."""
    from scipy.spatial import cKDTree

    plyio.check_required(ply.names)
    d = ply.data
    N = len(d)
    if cameras is None and model is not None and model.images:
        cameras = np.array([im.center for im in model.images.values()])
    info = {"n_in": N, "removed_by": {"camera": 0, "detached": 0, "through": 0, "air": 0, "faint": 0, "big": 0},
            "protected_face": 0, "protected_outside": 0, "n_removed": 0}
    if N == 0:
        info["n_out"] = 0
        return GaussianPly(d.copy(), list(ply.comments)), info
    P = _xyz(d)
    op = _sigmoid(d["opacity"])
    smax = np.empty(N)
    for s in range(0, N, CHUNK):
        smax[s:s + CHUNK] = _scales(d, slice(s, s + CHUNK)).max(1)
    origin = P.min(0) - 1.0
    reason = np.zeros(N, np.int8)                         # 0 없음, 1 camera, 2 detached, 3 faint, 4 big

    # camera
    if cameras is not None and len(cameras) and cam_radius > 0:
        ct = cKDTree(np.asarray(cameras, np.float64).reshape(-1, 3))
        for s in range(0, N, CHUNK):
            dd, _ = ct.query(P[s:s + CHUNK], k=1, distance_upper_bound=cam_radius)
            r = reason[s:s + CHUNK]
            r[(dd < cam_radius) & (r == 0)] = 1

    # detached
    if cluster_max is None:
        cluster_max = int(min(300, max(20, 5e-5 * N)))
    g = np.nonzero(op >= cluster_min_opacity)[0]
    n_comp = 0
    if len(g):
        keys = _pack(_voxel_index(P[g], cluster_voxel, origin))
        ku, inv = np.unique(keys, return_inverse=True)
        lab = _voxel_components(ku)
        n_comp = int(lab.max()) + 1
        cnt = np.bincount(lab[inv], minlength=n_comp)
        small = cnt[lab[inv]] < cluster_max
        sel = g[small]
        reason[sel[reason[sel] == 0]] = 2
        del keys, ku, inv, lab

    # C 투과, D 허공 (model 이 있을 때, clean_section.py)
    if model is not None and model.points:
        through, agree = see_through_votes(P, model, max_views=through_views)
        rc = (through >= through_min) & (through > through_ratio * agree) & (reason == 0)
        reason[rc] = 5
        info["n_through_votes"] = int(np.sum(through > 0))
        face_planes = list(planes) if planes else None
        if face_planes is None and plan is not None:
            from .planes import planes_from_plan
            face_planes = planes_from_plan(plan)
        X = _model_points(model, point_max_error)
        if face_planes and len(X):
            ct = cKDTree(X)
            cand_d = np.nonzero(reason == 0)[0]
            far = np.zeros(len(cand_d), bool)
            for s in range(0, len(cand_d), CHUNK):
                dd, _ = ct.query(P[cand_d[s:s + CHUNK]], k=1, distance_upper_bound=air_gap * 3)
                far[s:s + CHUNK] = dd > air_gap
            cd = cand_d[far]
            if len(cd):
                from .snap import assign
                idx, _ = assign(P[cd], face_planes, air_wall, face_uv_margin)
                reason[cd[idx < 0]] = 6

    # faint / big + 고립
    big_thr = max(big_min, big_rel * float(np.median(smax)))
    cand = np.nonzero((reason == 0) & ((op < faint) | ((smax > big_thr) & (op < big_max_opacity))))[0]
    if len(cand):
        I = _voxel_index(P, iso_voxel, origin)
        keys = _pack(I)
        ku, inv = np.unique(keys, return_inverse=True)
        top = np.zeros(len(ku))
        np.maximum.at(top, inv, op)
        solid = _unpack(ku[top >= solid_opacity]).astype(np.float64)
        del keys, I, inv
        if len(solid) >= iso_k:
            tree = cKDTree((solid + 0.5) * iso_voxel + origin)
            iso = np.empty(len(cand))
            for s in range(0, len(cand), CHUNK):
                dd, _ = tree.query(P[cand[s:s + CHUNK]], k=iso_k)
                iso[s:s + CHUNK] = dd[:, -1] if iso_k > 1 else dd
        else:
            iso = np.full(len(cand), np.inf)
        isolated = iso > iso_m
        c = cand[isolated]
        reason[c[op[c] < faint]] = 3
        c = c[reason[c] == 0]
        reason[c] = 4

    # 보호
    hit = np.nonzero(reason > 0)[0]
    if len(hit) and plan is not None:
        from .snap import _inside_any_room, _rooms_3d
        rooms = _rooms_3d(plan)
        if rooms:
            ins = _inside_any_room(P[hit], rooms, 0.0)
            info["protected_outside"] = int(np.sum(~ins))
            reason[hit[~ins]] = 0
            hit = hit[ins]
    if len(hit) and planes:
        from .snap import assign
        idx, _ = assign(P[hit], list(planes), face_keep, face_uv_margin)
        near = idx >= 0
        info["protected_face"] = int(near.sum())
        reason[hit[near]] = 0
    rm = reason > 0
    for code, name in ((1, "camera"), (2, "detached"), (3, "faint"), (4, "big"), (5, "through"), (6, "air")):
        info["removed_by"][name] = int(np.sum(reason == code))
    info["n_removed"] = int(rm.sum())
    out = _replace(ply, rm, [])
    if return_removed:
        info["removed"] = GaussianPly(ply.data[rm].copy(), list(ply.comments))
    info["n_out"] = out.n
    info["n_components"] = n_comp
    info["params"] = {"faint": faint, "iso_voxel": iso_voxel, "solid_opacity": solid_opacity, "iso_k": iso_k,
                      "iso_m": iso_m, "big_thr": big_thr, "big_max_opacity": big_max_opacity,
                      "cluster_voxel": cluster_voxel, "cluster_min_opacity": cluster_min_opacity,
                      "cluster_max": cluster_max, "cam_radius": cam_radius, "face_keep": face_keep,
                      "model": model is not None, "through_min": through_min, "through_ratio": through_ratio,
                      "air_gap": air_gap, "air_wall": air_wall}
    return out, info


# ---------------- 4) SH 튐 ----------------
def _sh_ratio(d, sel=slice(None)) -> tuple[np.ndarray, np.ndarray]:
    """(q = 시점에 따른 색 변화 상한 / 밝기, 고차 SH 가 있는지).
    색 변화 상한: 채널 c 의 고차 계수 벡터 e_c 에 대해 |Σ_k e_ck Y_k(d)| <= ||e_c||·||Y(d)||,
    ||Y(d)||² = Σ_{l>=1} (2l+1)/4π = K/4π (차수와 무관하게 방향마다 같음). swing = max_c ||e_c||·sqrt(K/4π).
    밝기 = DC 색 (0.5 + C0 f_dc) 세 채널 평균을 [0.2, 1] 로 자른 것 (어두운 구석에서 0 으로 나누지 않게)."""
    names = d.dtype.names
    K = sum(1 for k in names if k.startswith("f_rest_")) // 3
    n = len(d[sel])
    if K == 0:
        return np.zeros(n), np.zeros(n, bool)
    e = np.zeros(n)
    for c in range(3):
        ec = np.zeros(n)
        for k in range(K):
            v = d[f"f_rest_{c * K + k}"][sel].astype(np.float64)
            ec += v * v
        e = np.maximum(e, ec)
    swing = np.sqrt(e) * np.sqrt(K / (4 * np.pi))
    rgb = 0.5 + plyio.SH_C0 * np.stack([d[f"f_dc_{i}"][sel] for i in range(3)], 1).astype(np.float64)
    bright = np.clip(rgb.mean(1), 0.2, 1.0)
    return swing / bright, swing > 0


def _sh_outliers(d, max_ratio: float, z_max: float, min_ratio: float) -> tuple[np.ndarray, dict]:
    N = len(d)
    q = np.empty(N)
    has = np.empty(N, bool)
    for s in range(0, N, CHUNK):
        q[s:s + CHUNK], has[s:s + CHUNK] = _sh_ratio(d, slice(s, s + CHUNK))
    out = q > max_ratio
    st = {"median_ratio": None, "z_thr_ratio": None}
    if has.sum() >= 20:
        lq = np.log(q[has])
        med = float(np.median(lq))
        mad = float(np.median(np.abs(lq - med))) * 1.4826
        if mad > 1e-9:
            thr = float(np.exp(med + z_max * mad))
            out |= (q > thr) & (q > min_ratio)
            st["z_thr_ratio"] = thr
        st["median_ratio"] = float(np.exp(med))
    return out, st


def damp_sh(ply: GaussianPly, *, max_ratio: float = 1.0, factor: float = 0.3, z_max: float = 5.0,
            min_ratio: float = 0.25) -> tuple[GaussianPly, dict]:
    """시점 의존 색 튐 줄이기. 왜: 학습 카메라는 동선(복도 가운데 1.4 m) 위에만 있어서, 고차 SH 가 그 몇 방향의
    사진만 맞추도록 크게 자라도 벌이 없다. VR 에서는 머리를 옆·위로 움직여 학습에 없던 방향으로 보므로 그런
    가우시안이 갑자기 밝거나 다른 색으로 튐 (흰 벽에 반짝이는 점·얼룩). 고차 SH 를 0 으로 하면 반사·조명 느낌까지
    사라지므로 튀는 것만 factor 배로 줄임.
    튐 판정: q = 색 변화 상한 / 밝기 (_sh_ratio) 가 max_ratio 넘음 (변화가 바탕색보다 큼), 또는 장면 전체 log q 의
    강건 z (중앙값·1.4826 MAD) > z_max 이면서 q > min_ratio. f_rest 만 바꾸고 나머지(f_dc 포함)는 비트 그대로."""
    plyio.check_required(ply.names)
    if not 0 <= factor <= 1:
        raise ValueError("factor 는 0..1")
    d = ply.data.copy()
    rest = [k for k in d.dtype.names if k.startswith("f_rest_")]
    out_mask, st = _sh_outliers(d, max_ratio, z_max, min_ratio)
    idx = np.nonzero(out_mask)[0]
    for k in rest:
        col = d[k]
        col[idx] = (col[idx].astype(np.float64) * factor).astype(col.dtype)
    info = {"n_in": len(d), "n_damped": int(len(idx)), **st,
            "params": {"max_ratio": max_ratio, "factor": factor, "z_max": z_max, "min_ratio": min_ratio}}
    return GaussianPly(d, list(ply.comments)), info


# ---------------- 요약 ----------------
def ply_quality(ply: GaussianPly, *, huge_axis: float = HUGE_AXIS) -> dict:
    """전후 비교용: needle_ratio·translucent_ratio (plyio.stats 와 같은 정의), 아주 큰 것 수, SH 튐 수 (damp_sh 기본값)."""
    d = ply.data
    N = len(d)
    nd, hg = 0, 0
    for s in range(0, N, CHUNK):
        S = _scales(d, slice(s, s + CHUNK))
        nd += int(np.sum(_needle_ratio(S)[0] > 10))
        hg += int(np.sum(S.max(1) > huge_axis))
    op = _sigmoid(d["opacity"]) if N else np.zeros(0)
    sh, _ = _sh_outliers(d, 1.0, 5.0, 0.25) if N else (np.zeros(0, bool), {})
    return {"count": int(N),
            "needle_ratio": round(nd / N, 4) if N else 0.0, "n_needles": nd,
            "translucent_ratio": round(float(np.mean(op < 0.5)), 4) if N else 0.0,
            "n_huge": hg, "n_sh_outliers": int(sh.sum())}


# ---------------- 파일 단위 ----------------
STEPS = {"floaters": remove_floaters, "huge": fix_huge, "needles": split_needles, "sh": damp_sh}
_CTX = {"planes", "plan", "cameras", "model", "return_removed"}


def _step_keys(fn) -> set[str]:
    return {p.name for p in inspect.signature(fn).parameters.values()
            if p.kind == p.KEYWORD_ONLY} - _CTX


def cleanup_file(in_ply, out_ply, *, planes=None, plan=None, cameras=None, model=None, removed_ply=None,
                 steps=("floaters", "huge", "needles", "sh"), **opts) -> dict:
    """읽기 -> steps 차례로 -> 새 파일에 쓰기. in_ply == out_ply 면 거부 (원본 안 덮음).
    opts: 단계별 옵션 사전, 예) needles={"max_pieces": 4}, floaters={"cam_radius": 0.3}. 모르는 단계·옵션은 TypeError.
    planes·plan·cameras·model 은 floaters 에만 전달. removed_ply 를 주면 floaters·huge 가 지운 스플랫만 모아 따로 씀
    (clean_section.py 처럼 열어서 진짜 물체가 지워지지 않았는지 확인용). 반환: quality_before/after, 단계별 info, 걸린 초."""
    bad = set(opts) - set(STEPS)
    if bad:
        raise TypeError(f"모르는 옵션 {sorted(bad)} (단계 이름: {sorted(STEPS)})")
    for name, o in opts.items():
        if not isinstance(o, dict):
            raise TypeError(f"{name} 옵션은 dict 여야 함")
        b = set(o) - _step_keys(STEPS[name])
        if b:
            raise TypeError(f"{name}: 모르는 옵션 {sorted(b)}")
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        raise ValueError(f"모르는 단계 {unknown}")
    in_ply, out_ply = Path(in_ply), Path(out_ply)
    if in_ply.resolve() == out_ply.resolve():
        raise ValueError("out_ply 가 in_ply 와 같음 - 원본을 덮어쓰지 않음")
    if removed_ply is not None and Path(removed_ply).resolve() in (in_ply.resolve(), out_ply.resolve()):
        raise ValueError("removed_ply 가 in_ply 또는 out_ply 와 같음")
    t0 = time.perf_counter()
    ply = plyio.read(in_ply)
    info: dict = {"in": str(in_ply), "out": str(out_ply), "steps": list(steps),
                  "quality_before": ply_quality(ply), "seconds": {"read": round(time.perf_counter() - t0, 2)}}
    removed: list[np.ndarray] = []
    for name in steps:
        t = time.perf_counter()
        kw = dict(opts.get(name, {}))
        if name == "floaters":
            kw.update(planes=planes, plan=plan, cameras=cameras, model=model)
        if name in ("floaters", "huge") and removed_ply is not None:
            kw["return_removed"] = True
        ply, info[name] = STEPS[name](ply, **kw)
        if "removed" in info[name]:
            removed.append(info[name].pop("removed").data)
        info["seconds"][name] = round(time.perf_counter() - t, 2)
    info["quality_after"] = ply_quality(ply)
    t = time.perf_counter()
    plyio.write(out_ply, ply)
    if removed_ply is not None:
        rem = np.concatenate(removed) if removed else ply.data[:0]
        plyio.write(removed_ply, GaussianPly(rem, list(ply.comments)))
        info["removed_ply"] = str(removed_ply)
        info["n_removed_total"] = int(len(rem))
    info["seconds"]["write"] = round(time.perf_counter() - t, 2)
    return info
