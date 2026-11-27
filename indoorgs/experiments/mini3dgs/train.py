"""mini3dgs 학습기 (CPU, 평가용). 원본 3DGS train.py 를 작게 줄인 것.

  python train.py --model <sparse/0 또는 모델 폴더> --images <사진 폴더> --out <결과 폴더> --iters 3000 --res 200

공개 API
  load_scene(model_dir, image_dir, res)               -> Scene (카메라·사진·점, 8번째마다 테스트)
  train(model_dir, image_dir, out_dir, iters, res, init_points=None, novel_views=None, **opts) -> metrics dict
  evaluate(params, deg, views, bg)                    -> 시점별 PSNR/SSIM/투과(see-through) 지표
  vr_cameras(cams, offset=0.3, grazing_deg=60, up=None) -> 동선에서 비켜난 "VR 같은" 새 시점 카메라 목록
  (렌더는 gs.render(ply_or_params, camera))

novel_views (새 시점 정답) 는 장면 제공자가 준다: callable(train_cams, test_cams) -> [{"name", "camera": gs.Camera,
"image": (H,W,3) RGB uint8 또는 0..1 float}]. 카메라는 COLMAP 모델 좌표 (제공자가 자기 정답 좌표와 맞춰야 함).
사진·카메라 해상도가 res 와 다르면 학습기가 줄여서 씀.

학습 (원본 3DGS 기본값, 반복 수에 비례해 줄임; 괄호는 30k 기준 원본 값)
  Adam lr: means 1.6e-4·extent -> 1.6e-6·extent 지수 감소, sh_dc 2.5e-3, sh_rest 1.25e-4, opacity 0.05 (원본 초기판),
           log_scales 5e-3, quats 1e-3.   손실 0.8·L1 + 0.2·(1-SSIM).
  SH 차수: 0 에서 시작해 sh_every 마다 +1 (원본 1000).
  밀도 조절: densify_from(500)..densify_until(15000) 사이 100 반복마다 화면 위치 기울기 평균(NDC 단위) > grad_thresh
           이면 작은 것은 복제, 큰 것(최대 축 > 0.01·extent)은 2개로 분할. 불투명도 < 0.005 지움.
           opacity_reset(3000) 마다 불투명도를 0.01 로 내림. 그 뒤로는 화면 반경 > 20px 또는 크기 > 0.1·extent 도 지움.
           cap (최대 개수) 를 넘으면 기울기가 큰 후보부터 남은 자리만큼만 늘림.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_SRC = HERE.parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import gs  # noqa: E402
from gs import Camera  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- 데이터
@dataclass
class View:
    name: str
    camera: Camera
    image: torch.Tensor          # (H,W,3) float 0..1 (RGB)


@dataclass
class Scene:
    train: list[View]
    test: list[View]
    xyz: np.ndarray
    rgb: np.ndarray
    extent: float
    info: dict = field(default_factory=dict)


def _intrinsics_and_dist(cam):
    """COLMAP 카메라 -> (fx, fy, cx, cy, OpenCV 왜곡 계수 또는 None)."""
    m, q = cam.model, cam.params
    if m == 0:
        return q[0], q[0], q[1], q[2], None
    if m == 1:
        return q[0], q[1], q[2], q[3], None
    if m == 2:   # SIMPLE_RADIAL f cx cy k
        return q[0], q[0], q[1], q[2], np.array([q[3], 0, 0, 0])
    if m == 3:   # RADIAL f cx cy k1 k2
        return q[0], q[0], q[1], q[2], np.array([q[3], q[4], 0, 0])
    if m == 4:   # OPENCV fx fy cx cy k1 k2 p1 p2
        return q[0], q[1], q[2], q[3], np.array(q[4:8])
    raise ValueError(f"지원하지 않는 카메라 모델 {m} (PINHOLE/SIMPLE_RADIAL/RADIAL/OPENCV 만)")


def to_tensor_image(img: np.ndarray, width: int, height: int) -> torch.Tensor:
    """RGB uint8 또는 float -> (H,W,3) float32, 크기 다르면 INTER_AREA 로 줄임."""
    img = np.asarray(img)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    img = img.astype(np.float32)
    if img.shape[1] != width or img.shape[0] != height:
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(np.ascontiguousarray(img[..., :3]))


def load_scene(model_dir, image_dir, res: int = 200, test_every: int = 8, max_images: int | None = None) -> Scene:
    """COLMAP 모델 + 사진. 이름순 정렬 후 i % test_every == 0 이 테스트 (원본 3DGS llffhold 와 같음).
    왜곡 있는 모델은 cv2.undistort 로 같은 K 에 펴서 씀."""
    from indoorgs.colmap import model as cm
    d, _ = cm.find_model(Path(model_dir))
    M = cm.read_model(d)
    ims = sorted(M.images.values(), key=lambda im: im.name)
    if max_images:
        ims = ims[:max_images]
    views = []
    n_dist = 0
    for im in ims:
        c = M.cameras[im.camera_id]
        fx, fy, cx, cy, dist = _intrinsics_and_dist(c)
        path = Path(image_dir) / im.name
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        if bgr.shape[1] != c.width or bgr.shape[0] != c.height:
            raise ValueError(f"{im.name}: 사진 크기 {bgr.shape[1]}x{bgr.shape[0]} != 카메라 {c.width}x{c.height}")
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        if dist is not None and np.abs(dist).max() > 1e-6:
            bgr = cv2.undistort(bgr, K, dist)
            n_dist += 1
        cam = Camera(R=cm.qvec2rot(im.q), t=np.asarray(im.t, np.float64), fx=fx, fy=fy, cx=cx, cy=cy,
                     width=c.width, height=c.height, name=im.name).scaled(res)
        views.append(View(im.name, cam, to_tensor_image(bgr[..., ::-1], cam.width, cam.height)))
    train = [v for i, v in enumerate(views) if i % test_every != 0]
    test = [v for i, v in enumerate(views) if i % test_every == 0]
    pts = list(M.points.values())
    xyz = np.array([p.xyz for p in pts]) if pts else np.zeros((0, 3))
    rgb = np.array([p.rgb for p in pts]) if pts else np.zeros((0, 3))
    C = np.array([v.camera.center for v in views])
    extent = float(1.1 * np.linalg.norm(C - C.mean(0), axis=1).max())
    return Scene(train, test, xyz, rgb, extent, {"n_images": len(views), "n_points": len(pts), "undistorted": n_dist})


def load_init_points(init_points):
    """(xyz, rgb) 튜플, COLMAP 모델 폴더, 또는 점 PLY(x,y,z[,red,green,blue]) 경로."""
    if isinstance(init_points, tuple):
        return np.asarray(init_points[0]), np.asarray(init_points[1])
    p = Path(init_points)
    if p.is_dir():
        from indoorgs.colmap import model as cm
        d, _ = cm.find_model(p)
        M = cm.read_model(d)
        pts = list(M.points.values())
        return np.array([q.xyz for q in pts]), np.array([q.rgb for q in pts])
    from indoorgs import plyio
    ply = plyio.read(p)
    d = ply.data
    xyz = np.stack([d["x"], d["y"], d["z"]], 1)
    if "red" in d.dtype.names:
        rgb = np.stack([d["red"], d["green"], d["blue"]], 1)
    elif "f_dc_0" in d.dtype.names:
        rgb = np.clip(np.stack([d["f_dc_0"], d["f_dc_1"], d["f_dc_2"]], 1) * gs.SH_C0 + 0.5, 0, 1)
    else:
        rgb = np.full((len(d), 3), 0.5)
    return xyz, rgb


# ---------------------------------------------------------------- 새 시점
def estimate_up(cams: list[Camera]) -> np.ndarray:
    """카메라들 -y 축(사진 위쪽) 평균 = 세계 위 방향 추정 (사람이 들고 찍은 영상은 대체로 수평)."""
    u = np.mean([-np.asarray(c.R)[1] for c in cams], 0)
    return u / np.linalg.norm(u)


def _rot_axis(axis, ang):
    axis = np.asarray(axis, np.float64) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def vr_cameras(cams: list[Camera], offset: float = 0.3, grazing_deg: float = 60.0, up=None,
               every: int = 1) -> list[Camera]:
    """동선 카메라마다 (every 개마다) 새 시점 6개: 옆으로 ±offset, 위로 +offset / 아래로 -offset (m, 모델 좌표 단위),
    그리고 위 축 기준으로 ±grazing_deg 돌린 것 (복도에서 앞을 보던 카메라 -> 벽을 비스듬히 보는 시점).
    COLMAP 좌표의 축척이 m 가 아니면 offset 을 그 단위로 줘야 함. 이름은 "<원래>_<종류>"."""
    up = estimate_up(cams) if up is None else np.asarray(up, np.float64) / np.linalg.norm(up)
    out = []
    for c in cams[::every]:
        Rwc = np.asarray(c.R).T          # world_from_cam
        C = c.center
        right = Rwc[:, 0] - up * (Rwc[:, 0] @ up)
        right /= np.linalg.norm(right)
        kw = dict(fx=c.fx, fy=c.fy, cx=c.cx, cy=c.cy, width=c.width, height=c.height)
        for tag, dC in [("R+", right * offset), ("R-", -right * offset), ("U+", up * offset), ("U-", -up * offset)]:
            out.append(Camera.from_world(Rwc, C + dC, name=f"{c.name}_{tag}", **kw))
        for tag, ang in [("Y+", grazing_deg), ("Y-", -grazing_deg)]:
            out.append(Camera.from_world(_rot_axis(up, math.radians(ang)) @ Rwc, C, name=f"{c.name}_{tag}", **kw))
    return out


# ---------------------------------------------------------------- 최적화 도우미
LR = {"means": 1.6e-4, "sh_dc": 2.5e-3, "sh_rest": 2.5e-3 / 20, "opacity": 0.05, "log_scales": 5e-3, "quats": 1e-3}


class Gaussians:
    """파라미터 + Adam 상태 (밀도 조절 때 행 단위로 고르기·붙이기)."""

    def __init__(self, p: dict, extent: float):
        self.p = {k: v.detach().clone().requires_grad_(True) for k, v in p.items()}
        self.extent = extent
        self.lr = dict(LR)
        self.lr["means"] = LR["means"] * extent
        self.opt = torch.optim.Adam([{"params": [self.p[k]], "name": k, "lr": self.lr[k]} for k in gs.PARAM_KEYS],
                                    lr=0.0, eps=1e-15)
        self.reset_stats()

    @property
    def n(self):
        return self.p["means"].shape[0]

    def reset_stats(self):
        self.grad_acc = torch.zeros(self.n)
        self.denom = torch.zeros(self.n)
        self.max_r = torch.zeros(self.n)

    def _replace(self, fn):
        """fn(name, tensor) -> 새 tensor. 파라미터와 Adam exp_avg/exp_avg_sq 에 같이 적용."""
        for g in self.opt.param_groups:
            k = g["name"]
            old = g["params"][0]
            st = self.opt.state.pop(old, None)
            new = fn(k, old.detach()).requires_grad_(True)
            if st is not None:
                st["exp_avg"] = fn(k, st["exp_avg"], zero=True)
                st["exp_avg_sq"] = fn(k, st["exp_avg_sq"], zero=True)
                self.opt.state[new] = st
            g["params"][0] = new
            self.p[k] = new

    def keep(self, mask: torch.Tensor):
        self._replace(lambda k, t, zero=False: t[mask])
        self.grad_acc, self.denom, self.max_r = self.grad_acc[mask], self.denom[mask], self.max_r[mask]

    def append(self, newp: dict):
        m = next(iter(newp.values())).shape[0]
        self._replace(lambda k, t, zero=False: torch.cat([t, torch.zeros_like(newp[k]) if zero else newp[k]]))
        self.grad_acc = torch.cat([self.grad_acc, torch.zeros(m)])
        self.denom = torch.cat([self.denom, torch.zeros(m)])
        self.max_r = torch.cat([self.max_r, torch.zeros(m)])

    def reset_opacity(self, value=0.01):
        v = math.log(value / (1 - value))

        def fn(k, t, zero=False):
            if k != "opacity":
                return t
            return torch.zeros_like(t) if zero else torch.clamp_max(t, v)
        self._replace(fn)

    def densify(self, grad_thresh: float, cap: int, percent_dense: float = 0.01) -> dict:
        with torch.no_grad():
            g = self.grad_acc / self.denom.clamp_min(1)
            g[self.denom == 0] = 0
            smax = torch.exp(self.p["log_scales"]).max(1).values
            cand = torch.nonzero(g >= grad_thresh).squeeze(1)
            room = max(0, cap - self.n)
            if len(cand) > room:                      # 자리 부족: 기울기 큰 것부터
                cand = cand[torch.argsort(g[cand], descending=True)[:room]]
            small = smax[cand] <= percent_dense * self.extent
            clone_idx, split_idx = cand[small], cand[~small]
            newp = {k: v.detach()[clone_idx].clone() for k, v in self.p.items()}
            # 분할: 원래 가우시안 분포에서 2점 뽑고 크기 1/1.6
            if len(split_idx):
                s = torch.exp(self.p["log_scales"][split_idx]).repeat(2, 1)
                R = gs.quat_to_rotmat(self.p["quats"][split_idx]).repeat(2, 1, 1)
                off = (R @ (torch.randn_like(s) * s)[..., None])[..., 0]
                sp = {k: v.detach()[split_idx].repeat(2, *[1] * (v.dim() - 1)) for k, v in self.p.items()}
                sp["means"] = sp["means"] + off
                sp["log_scales"] = torch.log(s / 1.6)
                newp = {k: torch.cat([newp[k], sp[k]]) for k in newp}
            n0 = self.n
            self.append(newp)
            if len(split_idx):
                keep = torch.ones(self.n, dtype=torch.bool)
                keep[split_idx] = False
                self.keep(keep)
        return {"clone": int(len(clone_idx)), "split": int(len(split_idx)), "n_before": n0}

    def prune(self, min_opacity=0.005, max_screen=None, max_world=None) -> int:
        with torch.no_grad():
            bad = torch.sigmoid(self.p["opacity"]) < min_opacity
            if max_screen:
                bad |= self.max_r > max_screen
            if max_world:
                bad |= torch.exp(self.p["log_scales"]).max(1).values > max_world
            nb = int(bad.sum())
            if nb:
                self.keep(~bad)
        return nb


# ---------------------------------------------------------------- 평가
def _safe(name: str) -> str:
    import re
    return re.sub(r"[^0-9A-Za-z_-]+", "_", Path(name).name.replace(".png", "").replace(".jpg", "").replace("+", "p"))


def evaluate(p: dict, deg: int, views: list[View], bg=(0, 0, 0), save_dir: Path | None = None,
             tag: str = "test", save_max: int = 6) -> dict:
    """시점마다 PSNR, SSIM, see_through = 누적 alpha < 0.95 인 픽셀 비율 (비침). save_dir 이 있으면
    앞의 save_max 장을 [정답 | 렌더 | alpha] 로 저장."""
    rows = []
    with torch.no_grad():
        for i, v in enumerate(views):
            out, _ = gs.render_params(p, v.camera, deg, bg)
            img = out["image"].clamp(0, 1)
            rows.append({"name": v.name, "psnr": round(gs.psnr(img, v.image), 3),
                         "ssim": round(float(gs.ssim(img, v.image)), 4),
                         "see_through": round(float((out["alpha"] < 0.95).float().mean()), 4)})
            if save_dir is not None and i < save_max:
                save_dir.mkdir(parents=True, exist_ok=True)
                a = out["alpha"][..., None].expand(-1, -1, 3)
                strip = torch.cat([v.image, img, a], 1).numpy()
                cv2.imwrite(str(save_dir / f"{tag}_{i:02d}_{_safe(v.name)}.png"),
                            (strip[..., ::-1] * 255).round().astype(np.uint8))
    mean = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in ("psnr", "ssim", "see_through")} if rows else {}
    return {"mean": mean, "views": rows}


def _novel_list(novel_views, scene: Scene, res: int) -> list[View]:
    if novel_views is None:
        return []
    items = novel_views([v.camera for v in scene.train], [v.camera for v in scene.test]) if callable(novel_views) else novel_views
    out = []
    for it in items:
        cam = it["camera"].scaled(res)
        out.append(View(it.get("name", cam.name), cam, to_tensor_image(it["image"], cam.width, cam.height)))
    return out


# ---------------------------------------------------------------- 학습
DEFAULTS = dict(sh_degree=2, cap=100_000, bg=(0.0, 0.0, 0.0), random_bg=False, grad_thresh=0.0002,
                densify_from=None, densify_until=None, densify_every=100, opacity_reset=None, sh_every=None,
                lambda_dssim=0.2, tile=16, seed=0, log_every=100, eval_every=0, test_every=8, max_images=None,
                save_renders=6, threads=None, finetune=False, checkpoints=(), flat_planes=None, flat_lambda=0.0,
                flat_band=0.03)


def pick_threads(threads=None) -> int:
    """torch 스레드 수. None = 자동: 빈 코어 수 (os.getloadavg 기준, 최소 1). CPU 가 바쁠 때 4 스레드로 두면
    스레드끼리 기다리느라 1 스레드보다 10배 넘게 느려짐 (측정: 부하 11 / 4코어에서 0.27 s/it -> 4.5 s/it)."""
    import os
    if threads:
        return int(threads)
    n = os.cpu_count() or 1
    try:
        load = os.getloadavg()[0]
    except OSError:
        load = 0.0
    return max(1, min(n, int(n - load + 0.5)))


def schedule(iters: int, o: dict) -> dict:
    """원본 30k 일정을 iters 에 비례해 줄임 (명시한 값은 그대로)."""
    f = iters / 30000.0
    s = dict(o)
    if s["densify_from"] is None:
        s["densify_from"] = max(100, int(500 * f))
    if s["densify_until"] is None:
        s["densify_until"] = int(15000 * f)
    if s["opacity_reset"] is None:
        s["opacity_reset"] = max(1000, int(3000 * f)) if iters >= 4000 else 0
    if s["sh_every"] is None:
        s["sh_every"] = max(100, int(1000 * f))
    return s


def train(model_dir, image_dir, out_dir, iters: int = 3000, res: int = 200, init_points=None,
          novel_views=None, scene: Scene | None = None, init_ply=None, **opts) -> dict:
    """학습 + 평가. 반환 metrics dict (out_dir/metrics.json 에도 씀). PLY 는 out_dir/point_cloud.ply (Brush 순서).
    opts: DEFAULTS 참고. bg=(r,g,b) 0..1 (학습·평가 배경), random_bg=True 면 학습 때만 매 반복 무작위 배경.
    scene 을 주면 model_dir/image_dir 대신 그것을 씀 (이미 읽은 장면 재사용).

    이어 학습 (fine-tune, 11-28 wall_ab RESULTS2): init_ply = 학습된(후처리한) 가우시안 PLY 경로. 그 가우시안을 그대로
    시작점으로 같은 손실로 iters 번 더 학습. finetune=True (init_ply 를 주면 기본) 이면 원본 30k 일정의 끝 상태처럼:
    SH 는 처음부터 PLY 차수, 밀도 조절·불투명도 리셋 없음, means 학습률은 끝 값(초기의 1%) 고정, 나머지 학습률 그대로.
    checkpoints=(300, 600) 이면 그 반복에서 point_cloud_<it>.ply 도 씀.
    flat_planes=[(n(3,), d)], flat_lambda > 0: 중심이 면 ±flat_band 안인 가우시안의 가장 짧은 축 길이(m) 평균에
    flat_lambda 를 곱해 손실에 더함 (납작하게 하는 정규화, 실험용)."""
    if init_ply is not None and "finetune" not in opts:
        opts = {**opts, "finetune": True}
    o = schedule(iters, {**DEFAULTS, **opts})
    if o["finetune"]:
        o["densify_until"] = 0
        o["opacity_reset"] = 0
    unknown = set(opts) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"알 수 없는 옵션: {sorted(unknown)}")
    torch.set_num_threads(pick_threads(o["threads"]))
    torch.manual_seed(o["seed"])
    rng = np.random.default_rng(o["seed"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if scene is None:
        scene = load_scene(model_dir, image_dir, res, o["test_every"], o["max_images"])
    if init_ply is not None:
        p0, ply_deg = gs.ply_to_params(init_ply)
        xyz = p0["means"].numpy()
        if ply_deg < o["sh_degree"]:          # PLY 차수가 낮으면 고차 SH 0 으로 채움
            K = gs.n_rest(o["sh_degree"])
            r = torch.zeros((len(xyz), K, 3))
            r[:, :p0["sh_rest"].shape[1]] = p0["sh_rest"]
            p0["sh_rest"] = r
        elif ply_deg > o["sh_degree"]:
            p0["sh_rest"] = p0["sh_rest"][:, :gs.n_rest(o["sh_degree"])]
    elif init_points is not None:
        xyz, rgb = load_init_points(init_points)
    else:
        xyz, rgb = scene.xyz, scene.rgb
    if len(xyz) < 4:
        raise ValueError("초기 점이 너무 적음")
    novel = _novel_list(novel_views, scene, res)
    W, H = scene.train[0].camera.width, scene.train[0].camera.height
    log(f"장면: 학습 {len(scene.train)} / 테스트 {len(scene.test)} / 새 시점 {len(novel)} 장, {W}x{H}, "
        f"초기 점 {len(xyz)}, extent {scene.extent:.2f}, 스레드 {torch.get_num_threads()}, 읽기 {time.time() - t0:.1f}s")
    G = Gaussians(p0 if init_ply is not None else gs.init_params(xyz, rgb, o["sh_degree"]), scene.extent)
    if o["finetune"]:
        G.lr["means"] = G.lr["means"] * 0.01
    bg = torch.tensor(o["bg"], dtype=torch.float32)
    deg = o["sh_degree"] if o["finetune"] else 0
    fl_n = fl_d = None
    if o["flat_planes"] and o["flat_lambda"] > 0:
        fl_n = torch.tensor(np.array([q[0] for q in o["flat_planes"]]), dtype=torch.float32)
        fl_d = torch.tensor(np.array([q[1] for q in o["flat_planes"]]), dtype=torch.float32)
    hist = []
    order = []
    t_train = time.time()
    t_win, pairs_win = time.time(), 0
    after_reset = False
    for it in range(1, iters + 1):
        # lr (means 지수 감소)
        tfrac = (it - 1) / max(1, iters - 1)
        for g in G.opt.param_groups:
            if g["name"] == "means":
                g["lr"] = G.lr["means"] if o["finetune"] else \
                    math.exp(math.log(G.lr["means"]) * (1 - tfrac) + math.log(G.lr["means"] * 0.01) * tfrac)
        if it % o["sh_every"] == 0 and deg < o["sh_degree"]:
            deg += 1
        if not order:
            order = list(rng.permutation(len(scene.train)))
        v = scene.train[order.pop()]
        bgi = torch.rand(3) if o["random_bg"] else bg
        out, proj = gs.render_params(G.p, v.camera, deg, bgi, tile=o["tile"])
        proj["means2d"].retain_grad()
        img = out["image"]
        l1 = (img - v.image).abs().mean()
        loss = (1 - o["lambda_dssim"]) * l1 + o["lambda_dssim"] * (1 - gs.ssim(img, v.image))
        if fl_n is not None:
            with torch.no_grad():
                sdist = (G.p["means"] @ fl_n.T - fl_d).abs().min(1).values
                near = sdist < o["flat_band"]
            if near.any():
                loss = loss + o["flat_lambda"] * torch.exp(G.p["log_scales"][near]).min(1).values.mean()
        loss.backward()
        pairs_win += out["n_pairs"]
        with torch.no_grad():
            # 화면 위치 기울기 (NDC 단위 = 픽셀 기울기 x 0.5·W, 0.5·H : 원본 CUDA 와 같은 단위)
            gm = proj["means2d"].grad
            idx = proj["idx"]
            if gm is not None and it < o["densify_until"]:
                gn = torch.sqrt((gm[:, 0] * 0.5 * W) ** 2 + (gm[:, 1] * 0.5 * H) ** 2)
                seen = proj["radius"] > 0
                G.grad_acc[idx[seen]] += gn[seen]
                G.denom[idx[seen]] += 1
                G.max_r[idx] = torch.maximum(G.max_r[idx], proj["radius"])
        G.opt.step()
        G.opt.zero_grad(set_to_none=True)
        # 밀도 조절
        if it < o["densify_until"]:
            if it > o["densify_from"] and it % o["densify_every"] == 0:
                info = G.densify(o["grad_thresh"], o["cap"])
                big = 20 if (after_reset and o["opacity_reset"]) else None
                npr = G.prune(0.005, big, 0.1 * scene.extent if big else None)
                G.reset_stats()
                hist.append({"it": it, "n": G.n, **info, "pruned": npr})
            if o["opacity_reset"] and it % o["opacity_reset"] == 0:
                G.reset_opacity()
                after_reset = True
        if it % o["log_every"] == 0 or it == iters:
            dt = (time.time() - t_win) / o["log_every"]
            log(f"it {it:5d}  loss {loss.item():.4f}  l1 {l1.item():.4f}  n {G.n:6d}  deg {deg}  "
                f"{dt:.3f}s/it  pairs {pairs_win / o['log_every']:.0f}")
            t_win, pairs_win = time.time(), 0
        if it in tuple(o["checkpoints"]) and it != iters:
            from indoorgs import plyio
            plyio.write(out_dir / f"point_cloud_{it}.ply", gs.params_to_ply(G.p, o["sh_degree"], ["mini3dgs", f"iters {it}"]))
        if o["eval_every"] and it % o["eval_every"] == 0 and it != iters:
            e = evaluate(G.p, deg, scene.test, bg)
            log(f"  [test] it {it}: PSNR {e['mean']['psnr']:.2f}  SSIM {e['mean']['ssim']:.3f}")
            hist.append({"it": it, "test": e["mean"]})
    train_s = time.time() - t_train
    # 결과
    deg_out = o["sh_degree"]
    from indoorgs import plyio
    ply = gs.params_to_ply(G.p, deg_out, ["mini3dgs", f"iters {iters}", f"res {W}x{H}"])
    plyio.write(out_dir / "point_cloud.ply", ply)
    rd = out_dir / "renders" if o["save_renders"] else None
    ev_test = evaluate(G.p, deg, scene.test, bg, rd, "test", o["save_renders"])
    ev_train = evaluate(G.p, deg, scene.train[::max(1, len(scene.train) // 8)], bg)
    ev_novel = evaluate(G.p, deg, novel, bg, rd, "novel", o["save_renders"]) if novel else None
    metrics = {
        "version": 1, "iters": iters, "res": [W, H], "n_init": int(len(xyz)), "count": G.n,
        "init_ply": str(init_ply) if init_ply is not None else None,
        "ply": str(out_dir / "point_cloud.ply"), "ply_stats": plyio.stats(ply),
        "test": ev_test, "train_subset": ev_train["mean"], "novel": ev_novel,
        "time": {"train_s": round(train_s, 1), "s_per_iter": round(train_s / iters, 4),
                 "total_s": round(time.time() - t0, 1)},
        "scene": {**scene.info, "extent": scene.extent, "n_train": len(scene.train), "n_test": len(scene.test)},
        "options": {k: (list(v) if isinstance(v, tuple) else v) for k, v in o.items()},
        "history": hist,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False), encoding="utf-8")
    nv = f"  새 시점 PSNR {ev_novel['mean']['psnr']:.2f} SSIM {ev_novel['mean']['ssim']:.3f}" if ev_novel else ""
    log(f"끝: 테스트 PSNR {ev_test['mean']['psnr']:.2f} SSIM {ev_test['mean']['ssim']:.3f} "
        f"비침 {ev_test['mean']['see_through']:.3f}{nv}  개수 {G.n}  {train_s / iters:.3f}s/it  -> {out_dir}")
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser(description="mini3dgs: CPU 3DGS 평가용 학습기")
    ap.add_argument("--model", required=True, help="COLMAP 모델 폴더 (sparse/0 또는 그 위)")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--res", type=int, default=200, help="학습 가로 해상도 (px)")
    ap.add_argument("--init", default=None, help="초기 점: COLMAP 모델 폴더 또는 점 PLY (없으면 --model 의 점)")
    ap.add_argument("--init-ply", default=None, help="이어 학습: 학습된 가우시안 PLY 에서 시작 (fine-tune, 밀도 조절 없음)")
    ap.add_argument("--checkpoints", default="", help="쉼표로 반복 수, 그때 point_cloud_<it>.ply 저장")
    ap.add_argument("--sh", type=int, default=2)
    ap.add_argument("--cap", type=int, default=100_000)
    ap.add_argument("--bg", default="0,0,0", help="배경 r,g,b (0..1)")
    ap.add_argument("--random-bg", action="store_true")
    ap.add_argument("--grad-thresh", type=float, default=0.0002)
    ap.add_argument("--opacity-reset", type=int, default=None, help="0 = 끔 (기본: 30k 기준 3000 을 비례, 4000 반복 미만이면 끔)")
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--threads", type=int, default=0, help="0 = 자동 (빈 코어 수)")
    a = ap.parse_args(argv)
    train(a.model, a.images, a.out, a.iters, a.res, init_points=a.init, sh_degree=a.sh, cap=a.cap,
          bg=tuple(float(x) for x in a.bg.split(",")), random_bg=a.random_bg, grad_thresh=a.grad_thresh,
          opacity_reset=a.opacity_reset, eval_every=a.eval_every, max_images=a.max_images,
          threads=a.threads or None, init_ply=a.init_ply,
          checkpoints=tuple(int(x) for x in a.checkpoints.split(",") if x))


if __name__ == "__main__":
    main()
