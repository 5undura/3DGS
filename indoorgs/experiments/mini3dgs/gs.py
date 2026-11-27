"""mini3dgs 핵심: 순수 torch(CPU, autograd) 3DGS 래스터라이저 + 가우시안 파라미터 + PLY 입출력.

수식은 원본 3DGS (Kerbl et al. 2023, diff-gaussian-rasterization) 를 따른다.
- Σ = R S Sᵀ Rᵀ  (쿼터니언 w,x,y,z 정규화, S = diag(exp(log_scale)))
- EWA 투영: J = [[fx/z, 0, -fx·x/z²], [0, fy/z, -fy·y/z²]] (x/z, y/z 는 1.3·tan(fov/2) 로 자름),
  Σ' = J W Σ Wᵀ Jᵀ + 0.3·I  (저역 통과 팽창), conic = Σ'⁻¹, 반경 = ceil(3·sqrt(λmax))
- 불투명도 sigmoid, α = min(0.99, o·exp(-½ dᵀ conic d)), α < 1/255 은 버림
- 앞->뒤 합성: C = Σ cᵢ αᵢ Tᵢ + T_N · bg,  Tᵢ = Π_{j<i}(1-αⱼ)
- 색: SH 차수 0..2 (3DGS utils/sh_utils 와 같은 기저·부호), c = max(SH(dir) + 0.5, 0)
- 좌표: 픽셀 (i, j) 의 중심 = (i+0.5, j+0.5) (COLMAP 규약). 근평면 z > 0.2 (3DGS 와 같음)

원본과 다른 점 (정확도에는 거의 영향 없음, 속도·단순함 때문)
- 타일 안 조기 종료(T < 1e-4) 없음: 모든 가우시안을 다 합성 (뒤쪽 기여는 T 가 작아 사실상 0).
- 깊이 정렬은 가우시안 중심의 view-space z 로 전역 정렬 (원본과 같음, 타일별 정렬 대신 안정 정렬로 유지).
- 타일 목록은 3σ 경계 상자 기준 (원본과 같음). 타일들을 가우시안 수 순으로 묶어 패딩을 줄임.
- 역전파는 torch autograd (수동 CUDA 커널 아님). 메모리가 많으면 묶음마다 checkpoint.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)


# ---------------------------------------------------------------- 카메라
@dataclass
class Camera:
    """핀홀 카메라. R, t = cam_from_world (COLMAP 규약: x 오른쪽, y 아래, z 앞)."""
    R: np.ndarray
    t: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    name: str = ""

    @property
    def center(self) -> np.ndarray:
        return -np.asarray(self.R).T @ np.asarray(self.t)

    def scaled(self, width: int) -> "Camera":
        """가로 width 로 줄인 카메라 (세로는 비율 유지, 반올림)."""
        if width == self.width:
            return self
        h = int(round(self.height * width / self.width))
        sx, sy = width / self.width, h / self.height
        return replace(self, fx=self.fx * sx, fy=self.fy * sy, cx=self.cx * sx, cy=self.cy * sy, width=width, height=h)

    @staticmethod
    def from_world(R_wc: np.ndarray, C: np.ndarray, **kw) -> "Camera":
        """world_from_cam 회전 + 카메라 중심으로 만들기."""
        R = np.asarray(R_wc, np.float64).T
        return Camera(R=R, t=-R @ np.asarray(C, np.float64), **kw)


# ---------------------------------------------------------------- 파라미터
PARAM_KEYS = ("means", "quats", "log_scales", "opacity", "sh_dc", "sh_rest")


def n_rest(deg: int) -> int:
    return (deg + 1) ** 2 - 1


def init_params(xyz: np.ndarray, rgb: np.ndarray, sh_degree: int = 2, init_opacity: float = 0.1,
                knn: int = 3) -> dict[str, torch.Tensor]:
    """3DGS 초기화: 스케일 = sqrt(가장 가까운 knn 개 이웃 거리² 평균) (등방), 회전 단위, 불투명도 0.1.
    rgb 는 0..255 또는 0..1."""
    from scipy.spatial import cKDTree
    xyz = np.asarray(xyz, np.float64)
    rgb = np.asarray(rgb, np.float64)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    d, _ = cKDTree(xyz).query(xyz, k=knn + 1)
    dist2 = np.maximum((d[:, 1:] ** 2).mean(1), 1e-7)
    n = len(xyz)
    p = {
        "means": torch.tensor(xyz, dtype=torch.float32),
        "quats": torch.tensor(np.tile([1.0, 0, 0, 0], (n, 1)), dtype=torch.float32),
        "log_scales": torch.tensor(np.repeat(np.log(np.sqrt(dist2))[:, None], 3, 1), dtype=torch.float32),
        "opacity": torch.full((n,), math.log(init_opacity / (1 - init_opacity)), dtype=torch.float32),
        "sh_dc": torch.tensor((rgb - 0.5) / SH_C0, dtype=torch.float32),
        "sh_rest": torch.zeros((n, n_rest(sh_degree), 3), dtype=torch.float32),
    }
    return p


def eval_sh(deg: int, sh_dc: torch.Tensor, sh_rest: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """3DGS eval_sh (차수 0..2). sh_rest (N, K, 3), dirs 정규화된 (N, 3). 반환 RGB (+0.5, 0 이상)."""
    res = SH_C0 * sh_dc
    if deg > 0:
        x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
        res = res - SH_C1 * y * sh_rest[:, 0] + SH_C1 * z * sh_rest[:, 1] - SH_C1 * x * sh_rest[:, 2]
        if deg > 1:
            xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
            res = (res + SH_C2[0] * xy * sh_rest[:, 3] + SH_C2[1] * yz * sh_rest[:, 4]
                   + SH_C2[2] * (2 * zz - xx - yy) * sh_rest[:, 5] + SH_C2[3] * xz * sh_rest[:, 6]
                   + SH_C2[4] * (xx - yy) * sh_rest[:, 7])
    return torch.clamp_min(res + 0.5, 0.0)


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


# ---------------------------------------------------------------- 투영
def project(p: dict, cam: Camera, sh_degree: int, near: float = 0.2):
    """보이는 가우시안만 투영. 반환 dict: idx(원래 번호), means2d(N,2 픽셀, 미분 가능), conic(N,3), depth, radius(int),
    color(N,3), opacity(N, sigmoid 후)."""
    dt = p["means"].dtype
    R = torch.tensor(cam.R, dtype=dt)
    t = torch.tensor(cam.t, dtype=dt)
    tanx, tany = 0.5 * cam.width / cam.fx, 0.5 * cam.height / cam.fy
    with torch.no_grad():
        pv = p["means"] @ R.T + t
        z = pv[:, 2]
        ok = (z > near) & ((pv[:, 0] / z).abs() < 1.3 * tanx + 0.3) & ((pv[:, 1] / z).abs() < 1.3 * tany + 0.3)
        idx = torch.nonzero(ok).squeeze(1)
    means = p["means"][idx]
    pv = means @ R.T + t
    x, y, z = pv.unbind(-1)
    limx, limy = 1.3 * tanx, 1.3 * tany
    txz = torch.clamp(x / z, -limx, limx) * z
    tyz = torch.clamp(y / z, -limy, limy) * z
    zero = torch.zeros_like(z)
    J = torch.stack([cam.fx / z, zero, -cam.fx * txz / (z * z),
                     zero, cam.fy / z, -cam.fy * tyz / (z * z)], -1).reshape(-1, 2, 3)
    M = quat_to_rotmat(p["quats"][idx]) * torch.exp(p["log_scales"][idx])[:, None, :]   # R·S
    T = J @ R                                   # (n,2,3)
    TM = T @ M
    cov = TM @ TM.transpose(1, 2)
    a = cov[:, 0, 0] + 0.3
    b = cov[:, 0, 1]
    c = cov[:, 1, 1] + 0.3
    det = a * c - b * b
    conic = torch.stack([c / det, -b / det, a / det], -1)
    means2d = torch.stack([cam.fx * x / z + cam.cx, cam.fy * y / z + cam.cy], -1)
    with torch.no_grad():
        mid = 0.5 * (a + c)
        lam = mid + torch.sqrt(torch.clamp_min(mid * mid - det, 0.1))
        radius = torch.ceil(3 * torch.sqrt(lam))
    C = torch.tensor(cam.center, dtype=dt)
    dirs = F.normalize(means - C, dim=-1)
    color = eval_sh(sh_degree, p["sh_dc"][idx], p["sh_rest"][idx], dirs)
    return {"idx": idx, "means2d": means2d, "conic": conic, "depth": z, "radius": radius,
            "color": color, "opacity": torch.sigmoid(p["opacity"][idx])}


# ---------------------------------------------------------------- 래스터화
def _composite(pix, m2, con, op, feat, valid):
    """한 묶음의 타일. pix (B,P,2), m2 (B,K,2), con (B,K,3), op (B,K), feat (B,K,C), valid (B,K) float.
    반환 (B,P,C) 가중 합, (B,P) 마지막 투과율."""
    dx = pix[:, :, None, 0] - m2[:, None, :, 0]
    dy = pix[:, :, None, 1] - m2[:, None, :, 1]
    power = -0.5 * (con[:, None, :, 0] * dx * dx + con[:, None, :, 2] * dy * dy) - con[:, None, :, 1] * dx * dy
    alpha = torch.clamp_max(op[:, None, :] * torch.exp(torch.clamp_max(power, 0.0)), 0.99)
    keep = (alpha.detach() >= 1.0 / 255.0) & (valid[:, None, :] > 0)
    alpha = alpha * keep
    l1m = torch.log1p(-alpha)
    cs = torch.cumsum(l1m, -1)
    w = alpha * torch.exp(cs - l1m)
    out = torch.bmm(w, feat)
    return out, torch.exp(cs[..., -1])


def rasterize(proj: dict, width: int, height: int, bg: torch.Tensor, tile: int = 16,
              chunk_elems: int = 1_000_000, use_checkpoint: bool | None = None, with_depth: bool = False):
    """타일 래스터화. 반환 dict: image (H,W,3), alpha (H,W) = 1-T_N, depth (H,W, 가중 깊이/alpha, with_depth 일 때),
    n_pairs (가우시안-타일 쌍 수)."""
    TX, TY = (width + tile - 1) // tile, (height + tile - 1) // tile
    NT, P = TX * TY, tile * tile
    m2, r = proj["means2d"], proj["radius"]
    feat = proj["color"]
    if with_depth:
        feat = torch.cat([feat, proj["depth"][:, None]], -1)
    C = feat.shape[1]
    with torch.no_grad():
        mx, my = m2[:, 0].detach(), m2[:, 1].detach()
        inside = (mx + r > 0) & (mx - r < width) & (my + r > 0) & (my - r < height) & (r > 0)
        vis = torch.nonzero(inside).squeeze(1)
        vis = vis[torch.argsort(proj["depth"].detach()[vis], stable=True)]       # 앞 -> 뒤
        x0 = torch.clamp(torch.floor((mx[vis] - r[vis]) / tile), 0, TX - 1).long()
        x1 = torch.clamp(torch.floor((mx[vis] + r[vis]) / tile), 0, TX - 1).long()
        y0 = torch.clamp(torch.floor((my[vis] - r[vis]) / tile), 0, TY - 1).long()
        y1 = torch.clamp(torch.floor((my[vis] + r[vis]) / tile), 0, TY - 1).long()
        w_ = x1 - x0 + 1
        cnt = w_ * (y1 - y0 + 1)
        total = int(cnt.sum())
        g = torch.repeat_interleave(torch.arange(len(vis)), cnt)
        start = torch.cumsum(cnt, 0) - cnt
        loc = torch.arange(total) - start[g]
        tid = (y0[g] + loc // w_[g]) * TX + x0[g] + loc % w_[g]
        tid_sorted, order = torch.sort(tid, stable=True)          # 같은 타일 안에서는 깊이 순 유지
        gid = vis[g[order]]                                       # proj 안 번호
        tcount = torch.bincount(tid_sorted, minlength=NT)
        tstart = torch.cumsum(tcount, 0) - tcount
        tiles = torch.nonzero(tcount).squeeze(1)
        tiles = tiles[torch.argsort(tcount[tiles], descending=True)]
        ly, lx = torch.meshgrid(torch.arange(tile), torch.arange(tile), indexing="ij")
        local = torch.stack([lx.reshape(-1), ly.reshape(-1)], -1).to(m2.dtype) + 0.5
    if use_checkpoint is None:
        use_checkpoint = total * P > 30_000_000 and torch.is_grad_enabled()
    outs, Ts, ids = [], [], []
    i = 0
    nt = len(tiles)
    while i < nt:
        K = int(tcount[tiles[i]])
        B = max(1, min(nt - i, chunk_elems // max(1, K * P)))
        tb = tiles[i:i + B]
        with torch.no_grad():
            kk = torch.arange(K)
            pos = tstart[tb][:, None] + kk[None, :]
            valid = kk[None, :] < tcount[tb][:, None]
            gi = gid[torch.clamp(pos, max=len(gid) - 1)]
            gi = torch.where(valid, gi, torch.zeros_like(gi))
            org = torch.stack([(tb % TX) * tile, (tb // TX) * tile], -1).to(m2.dtype)
            pix = org[:, None, :] + local[None]
        args = (pix, m2[gi], proj["conic"][gi], proj["opacity"][gi], feat[gi], valid.to(m2.dtype))
        if use_checkpoint:
            o, T = checkpoint(_composite, *args, use_reentrant=False)
        else:
            o, T = _composite(*args)
        outs.append(o)
        Ts.append(T)
        ids.append(tb)
        i += B
    full = torch.zeros(NT, P, C, dtype=m2.dtype)
    Tfull = torch.ones(NT, P, dtype=m2.dtype)
    if ids:
        ids_t = torch.cat(ids)
        full = full.index_copy(0, ids_t, torch.cat(outs))
        Tfull = Tfull.index_copy(0, ids_t, torch.cat(Ts))

    def untile(x):
        x = x.reshape(TY, TX, tile, tile, -1).permute(0, 2, 1, 3, 4).reshape(TY * tile, TX * tile, -1)
        return x[:height, :width]
    feat_img = untile(full)
    Timg = untile(Tfull[..., None])[..., 0]
    out = {"image": feat_img[..., :3] + Timg[..., None] * bg, "alpha": 1 - Timg, "n_pairs": total}
    if with_depth:
        out["depth"] = feat_img[..., 3] / torch.clamp_min(1 - Timg, 1e-6)
    return out


def render_params(p: dict, cam: Camera, sh_degree: int, bg=(0.0, 0.0, 0.0), tile: int = 16, with_depth=False):
    """파라미터 dict 로 렌더. 반환 (rasterize 결과 dict, project 결과 dict)."""
    bg = torch.as_tensor(bg, dtype=p["means"].dtype)
    proj = project(p, cam, sh_degree)
    return rasterize(proj, cam.width, cam.height, bg, tile=tile, with_depth=with_depth), proj


# ---------------------------------------------------------------- PLY
def params_to_ply(p: dict, sh_degree: int, comments=None):
    """Brush 속성 순서 PLY (indoorgs.plyio). f_rest_{c*K+k} = 채널 c 의 k 번째 계수."""
    from indoorgs import plyio
    n = p["means"].shape[0]
    ply = plyio.make(n, plyio.brush_names(sh_degree), comments or ["mini3dgs"])
    d = ply.data
    g = {k: v.detach().cpu().numpy() for k, v in p.items()}
    for i, a in enumerate("xyz"):
        d[a] = g["means"][:, i]
    for c in range(3):
        d[f"f_dc_{c}"] = g["sh_dc"][:, c]
    K = n_rest(sh_degree)
    rest = g["sh_rest"][:, :K, :]
    for c in range(3):
        for k in range(K):
            d[f"f_rest_{c * K + k}"] = rest[:, k, c]
    d["opacity"] = g["opacity"]
    q = g["quats"] / np.linalg.norm(g["quats"], axis=1, keepdims=True)
    q *= np.where(q[:, :1] < 0, -1.0, 1.0)
    for i in range(4):
        d[f"rot_{i}"] = q[:, i]
    for i in range(3):
        d[f"scale_{i}"] = g["log_scales"][:, i]
    return ply


def ply_to_params(ply_or_path) -> tuple[dict, int]:
    """PLY (경로 또는 plyio.GaussianPly) -> (파라미터 dict, SH 차수). 속성은 이름으로 읽음 (Brush/LichtFeld 모두)."""
    from indoorgs import plyio
    ply = plyio.read(ply_or_path) if isinstance(ply_or_path, (str, Path)) else ply_or_path
    d, deg = ply.data, ply.sh_degree
    deg = min(deg, 2)
    K_file = n_rest(ply.sh_degree)
    K = n_rest(deg)
    f = lambda *ks: np.stack([d[k].astype(np.float32) for k in ks], 1)
    rest = np.zeros((len(d), K, 3), np.float32)
    for c in range(3):
        for k in range(K):
            rest[:, k, c] = d[f"f_rest_{c * K_file + k}"]
    p = {"means": f("x", "y", "z"), "quats": f("rot_0", "rot_1", "rot_2", "rot_3"),
         "log_scales": f("scale_0", "scale_1", "scale_2"), "opacity": d["opacity"].astype(np.float32),
         "sh_dc": f("f_dc_0", "f_dc_1", "f_dc_2"), "sh_rest": rest}
    return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in p.items()}, deg


def render(ply_or_params, camera: Camera, bg=(0.0, 0.0, 0.0), sh_degree: int | None = None,
           tile: int = 16, with_depth: bool = False) -> np.ndarray | dict:
    """공개 API: PLY 경로 / GaussianPly / (params, deg) / params dict 를 camera 로 렌더 -> (H,W,3) float32 0..1.
    with_depth=True 면 {"image","alpha","depth"} numpy dict."""
    if isinstance(ply_or_params, tuple):
        p, deg = ply_or_params
    elif isinstance(ply_or_params, dict):
        p, deg = ply_or_params, n_deg_from(ply_or_params)
    else:
        p, deg = ply_to_params(ply_or_params)
    if sh_degree is not None:
        deg = min(deg, sh_degree)
    with torch.no_grad():
        out, _ = render_params(p, camera, deg, bg, tile=tile, with_depth=with_depth)
    img = out["image"].clamp(0, 1).numpy()
    if not with_depth:
        return img
    return {"image": img, "alpha": out["alpha"].numpy(), "depth": out["depth"].numpy()}


def n_deg_from(p: dict) -> int:
    K = p["sh_rest"].shape[1]
    return {0: 0, 3: 1, 8: 2}.get(K, 2)


# ---------------------------------------------------------------- 지표
def _gauss_win(size=11, sigma=1.5):
    x = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-x * x / (2 * sigma * sigma))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)


_WIN = _gauss_win()


def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(H,W,3) 0..1. 3DGS loss_utils.ssim 과 같은 11x11 가우시안 창."""
    x, y = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    mu1 = F.conv2d(x, _WIN, padding=5, groups=3)
    mu2 = F.conv2d(y, _WIN, padding=5, groups=3)
    s11 = F.conv2d(x * x, _WIN, padding=5, groups=3) - mu1 * mu1
    s22 = F.conv2d(y * y, _WIN, padding=5, groups=3) - mu2 * mu2
    s12 = F.conv2d(x * y, _WIN, padding=5, groups=3) - mu1 * mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 * mu1 + mu2 * mu2 + C1) * (s11 + s22 + C2))
    return m.mean()


def psnr(a, b) -> float:
    a = torch.as_tensor(a, dtype=torch.float32).clamp(0, 1)
    b = torch.as_tensor(b, dtype=torch.float32)
    mse = torch.mean((a - b) ** 2).item()
    return float(10 * math.log10(1.0 / max(mse, 1e-12)))
