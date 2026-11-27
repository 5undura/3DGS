"""mini3dgs 확인용 작은 실내 장면: 4 x 3 x 2.6 m 상자 방 (흰 벽 + 포스터·문·걸레받이, 타일 바닥, 천장 패널).

numpy 광선 추적으로 정답 사진을 만들고 (시점 무관 색 = 3DGS 가 원리상 완벽히 맞출 수 있는 장면),
COLMAP 비슷한 초기 점 (무늬 있는 곳에 몰리고 흰 벽에는 드묾, 1.5 cm 잡음, 2% 허공 점) 을
진짜 COLMAP 모델 파일 (cameras/images/points3D.bin, PINHOLE) 로 써서 train() 을 실제 데이터와 같은 경로로 돌림.

  python toyroom.py --out <폴더> [--iters 1500] [--res 200]      # 장면 만들기 + 학습 + 새 시점 평가

공개 API
  make_scene(out_dir, n_views=40, width=320, height=240) -> dict(model_dir, image_dir, planes)
  render_gt(camera) -> (H,W,3) uint8 RGB      (아무 카메라나 정답 렌더)
  novel_views(train_cams, test_cams)          -> train(novel_views=...) 에 넘기는 콜백 (vr_cameras + render_gt)
  PLANES  벽·바닥·천장 평면 (n·p = d, n 안쪽)
좌표: x 0..4, y 0..3, z 0..2.6 (z 위, 바닥 z=0). 축척 = m.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import gs  # noqa: E402
from gs import Camera  # noqa: E402

LX, LY, LZ = 4.0, 3.0, 2.6
PPM = 120                     # 텍스처 px/m
HFOV = 75.0
PLANES = [
    {"name": "wall_x0", "n": [1, 0, 0], "d": 0.0}, {"name": "wall_x4", "n": [-1, 0, 0], "d": -LX},
    {"name": "wall_y0", "n": [0, 1, 0], "d": 0.0}, {"name": "wall_y3", "n": [0, -1, 0], "d": -LY},
    {"name": "floor", "n": [0, 0, 1], "d": 0.0}, {"name": "ceiling", "n": [0, 0, -1], "d": -LZ},
]


# ---------------------------------------------------------------- 텍스처
def _lowfreq(rng, h, w, cells, amp):
    g = rng.normal(0, 1, (cells[0], cells[1], 3)).astype(np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC) * amp


def _wall(rng, w_m, seed_items):
    """흰 벽 (0.88 근처, 아주 약한 얼룩) + 걸레받이 + 물건들."""
    h, w = int(LZ * PPM), int(w_m * PPM)
    img = np.full((h, w, 3), 0.86, np.float32) + _lowfreq(rng, h, w, (3, max(2, int(w_m))), 0.015)
    img[int((LZ - 0.10) * PPM):] = (0.35, 0.33, 0.30)                     # 걸레받이 10 cm
    for kind, x0, z0, ww, hh, col in seed_items:
        r0, r1 = int((LZ - z0 - hh) * PPM), int((LZ - z0) * PPM)
        c0, c1 = int(x0 * PPM), int((x0 + ww) * PPM)
        if kind == "poster":
            p = np.empty((r1 - r0, c1 - c0, 3), np.float32)
            p[:] = col
            p += _lowfreq(rng, r1 - r0, c1 - c0, (4, 3), 0.08)
            for k in range(5):                                            # 글자 줄 흉내
                y = int((0.15 + 0.15 * k) * (r1 - r0))
                cv2.line(p, (int(0.1 * (c1 - c0)), y), (int((0.5 + 0.4 * rng.random()) * (c1 - c0)), y),
                         (0.1, 0.1, 0.1), 2)
            cv2.circle(p, (int(0.7 * (c1 - c0)), int(0.8 * (r1 - r0))), int(0.1 * (c1 - c0)), (0.95, 0.9, 0.2), -1)
            img[r0:r1, c0:c1] = p
            cv2.rectangle(img, (c0, r0), (c1 - 1, r1 - 1), (0.15, 0.15, 0.15), 2)
        elif kind == "door":
            img[r0:r1, c0:c1] = col
            img[r0:r1, c0:c1] += _lowfreq(rng, r1 - r0, c1 - c0, (6, 2), 0.03)
            cv2.rectangle(img, (c0, r0), (c1 - 1, r1 - 1), (0.2, 0.2, 0.2), 3)
            cv2.circle(img, (c1 - int(0.12 * PPM), (r0 + r1) // 2), 4, (0.8, 0.8, 0.75), -1)
        elif kind == "switch":
            img[r0:r1, c0:c1] = col
            cv2.rectangle(img, (c0, r0), (c1 - 1, r1 - 1), (0.5, 0.5, 0.5), 1)
    return np.clip(img, 0, 1)


def make_textures(seed=0):
    rng = np.random.default_rng(seed)
    T = {
        "x0": _wall(rng, LY, [("door", 1.0, 0.0, 0.9, 2.05, (0.45, 0.30, 0.18)), ("switch", 2.1, 1.15, 0.08, 0.12, (0.95, 0.95, 0.92))]),
        "x4": _wall(rng, LY, [("poster", 0.6, 1.1, 0.6, 0.8, (0.2, 0.45, 0.75)), ("poster", 1.8, 1.2, 0.5, 0.7, (0.8, 0.3, 0.25))]),
        "y0": _wall(rng, LX, [("poster", 1.5, 1.0, 1.0, 0.7, (0.25, 0.6, 0.35))]),
        "y3": _wall(rng, LX, [("door", 2.6, 0.0, 0.9, 2.05, (0.30, 0.32, 0.38)), ("poster", 0.7, 1.3, 0.5, 0.5, (0.9, 0.75, 0.3))]),
    }
    # 바닥: 30 cm 타일 + 줄눈, 천장: 60 cm 패널 + 조명
    h, w = int(LY * PPM), int(LX * PPM)
    fl = np.full((h, w, 3), (0.55, 0.52, 0.48), np.float32) + _lowfreq(rng, h, w, (12, 16), 0.04)
    for k in range(0, w, int(0.3 * PPM)):
        fl[:, k:k + 2] = 0.35
    for k in range(0, h, int(0.3 * PPM)):
        fl[k:k + 2] = 0.35
    ce = np.full((h, w, 3), 0.92, np.float32) + _lowfreq(rng, h, w, (4, 5), 0.01)
    for k in range(0, w, int(0.6 * PPM)):
        ce[:, k:k + 2] = 0.6
    for k in range(0, h, int(0.6 * PPM)):
        ce[k:k + 2] = 0.6
    for cx, cy in [(1.2, 1.5), (2.8, 1.5)]:
        ce[int((cy - 0.3) * PPM):int((cy + 0.3) * PPM), int((cx - 0.3) * PPM):int((cx + 0.3) * PPM)] = 1.0
    T["floor"], T["ceil"] = np.clip(fl, 0, 1), np.clip(ce, 0, 1)
    return T


_TEX = None


def _tex():
    global _TEX
    if _TEX is None:
        _TEX = make_textures(0)
    return _TEX


def _bilinear(tex, u, v):
    h, w = tex.shape[:2]
    u = np.clip(u, 0, w - 1.001)
    v = np.clip(v, 0, h - 1.001)
    u0, v0 = u.astype(np.int32), v.astype(np.int32)
    fu, fv = (u - u0)[:, None], (v - v0)[:, None]
    a = tex[v0, u0] * (1 - fu) + tex[v0, u0 + 1] * fu
    b = tex[v0 + 1, u0] * (1 - fu) + tex[v0 + 1, u0 + 1] * fu
    return a * (1 - fv) + b * fv


def shade_points(P, ax, d_sign):
    """표면 점 P (n,3), 맞은 축 ax, 그 축 방향 부호 -> 색 (n,3) float. 조명: 천장 등 두 개에서 거리 감쇠 (시점 무관)."""
    T = _tex()
    col = np.zeros((len(P), 3), np.float32)
    m = (ax == 0) & (d_sign < 0)
    col[m] = _bilinear(T["x0"], P[m, 1] * PPM, (LZ - P[m, 2]) * PPM)
    m = (ax == 0) & (d_sign > 0)
    col[m] = _bilinear(T["x4"], (LY - P[m, 1]) * PPM, (LZ - P[m, 2]) * PPM)
    m = (ax == 1) & (d_sign < 0)
    col[m] = _bilinear(T["y0"], (LX - P[m, 0]) * PPM, (LZ - P[m, 2]) * PPM)
    m = (ax == 1) & (d_sign > 0)
    col[m] = _bilinear(T["y3"], P[m, 0] * PPM, (LZ - P[m, 2]) * PPM)
    m = (ax == 2) & (d_sign < 0)
    col[m] = _bilinear(T["floor"], P[m, 0] * PPM, P[m, 1] * PPM)
    m = (ax == 2) & (d_sign > 0)
    col[m] = _bilinear(T["ceil"], P[m, 0] * PPM, P[m, 1] * PPM)
    lamp = np.zeros(len(P))
    for L in ([1.2, 1.5, 2.55], [2.8, 1.5, 2.55]):
        lamp += 1.0 / (1.0 + 0.25 * np.sum((P - np.array(L)) ** 2, 1))
    light = 0.55 + 0.35 * lamp / 1.6
    return col * light[:, None].astype(np.float32)


def raycast(o, d):
    """상자 안쪽 광선. 반환 (P, t, ax, 부호)."""
    lo, hi = np.zeros(3), np.array([LX, LY, LZ])
    with np.errstate(divide="ignore", invalid="ignore"):
        tb = np.where(d > 0, (hi - o) / d, (lo - o) / d)
    tb[~np.isfinite(tb)] = np.inf
    ax = np.argmin(tb, 1)
    t = tb[np.arange(len(d)), ax]
    P = o + d * t[:, None]
    return P, t, ax, np.sign(d[np.arange(len(d)), ax])


def render_gt(cam: Camera, ss: int = 2, rng=None) -> np.ndarray:
    """정답 렌더 (ss x ss 초과 표본 평균). uint8 RGB."""
    w, h = cam.width * ss, cam.height * ss
    u, v = np.meshgrid((np.arange(w) + 0.5) / ss, (np.arange(h) + 0.5) / ss)
    dc = np.stack([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, np.ones_like(u)], -1).reshape(-1, 3)
    d = dc @ np.asarray(cam.R)          # = (Rᵀ dc)ᵀ : world 방향
    o = np.broadcast_to(cam.center, d.shape)
    P, t, ax, sg = raycast(o, d)
    img = shade_points(P, ax, sg).reshape(h, w, 3)
    if ss > 1:
        img = img.reshape(cam.height, ss, cam.width, ss, 3).mean((1, 3))
    if rng is not None:
        img = img + rng.normal(0, 1.0 / 255, img.shape)
    return (np.clip(img, 0, 1) * 255).round().astype(np.uint8)


# ---------------------------------------------------------------- 카메라 경로
def _look(yaw, pitch, roll=0.0):
    f = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    cr, sr = math.cos(roll), math.sin(roll)
    r2, u2 = cr * r + sr * u, -sr * r + cr * u
    return np.stack([r2, -u2, f], 1)    # world_from_cam


def path_cameras(n=40, width=320, height=240) -> list[Camera]:
    """방 가운데를 타원으로 한 바퀴 반 돌며 바깥(벽)을 둘러봄. 높이 1.5 m 근처, 고개 ±12도."""
    fx = width / 2 / math.tan(math.radians(HFOV / 2))
    cams = []
    for i in range(n):
        s = i / n
        th = 2 * math.pi * 1.5 * s
        C = np.array([2.0 + 1.0 * math.cos(th), 1.5 + 0.65 * math.sin(th), 1.5 + 0.08 * math.sin(7 * th)])
        yaw = th + math.radians(40) * math.sin(2 * math.pi * 3 * s) + (math.pi * 0.5 if i % 2 else 0.0)
        pitch = math.radians(12) * math.sin(2 * math.pi * 5 * s)
        cams.append(Camera.from_world(_look(yaw, pitch, math.radians(2) * math.sin(9 * th)), C, fx=fx, fy=fx,
                                      cx=width / 2, cy=height / 2, width=width, height=height,
                                      name=f"img_{i:03d}.png"))
    return cams


# ---------------------------------------------------------------- COLMAP 같은 초기 점
def colmap_like_points(cams: list[Camera], images: list[np.ndarray], n_target=6000, noise=0.015,
                       outlier_frac=0.02, seed=0):
    """무늬(밝기 기울기)가 있는 픽셀에 몰리게 뽑아 표면으로 역투영 + 잡음. 흰 벽에는 드물게 (SfM 처럼).
    반환 xyz (n,3), rgb uint8 (n,3)."""
    rng = np.random.default_rng(seed)
    per = n_target // len(cams)
    X, Cl = [], []
    for cam, img in zip(cams, images):
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)).ravel()
        prob = np.where(mag > 20, mag, 0.0) + 0.02 * mag.mean()
        prob /= prob.sum()
        pix = rng.choice(len(prob), per, replace=False, p=prob)
        u = pix % cam.width + 0.5
        v = pix // cam.width + 0.5
        dc = np.stack([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, np.ones_like(u)], -1)
        d = dc @ np.asarray(cam.R)
        P, t, ax, sg = raycast(np.broadcast_to(cam.center, d.shape), d)
        X.append(P + rng.normal(0, noise, P.shape))
        Cl.append(img.reshape(-1, 3)[pix])
    X, Cl = np.concatenate(X), np.concatenate(Cl)
    k = int(outlier_frac * len(X))
    X[:k] = rng.uniform([-0.3, -0.3, -0.3], [LX + 0.3, LY + 0.3, LZ + 0.3], (k, 3))   # 허공·벽 뒤 점
    return X, Cl.astype(np.uint8)


def make_scene(out_dir, n_views=40, width=320, height=240, n_points=6000, seed=0) -> dict:
    """사진 (PNG) + COLMAP 모델 (sparse/0, PINHOLE) + planes.json."""
    from indoorgs import plyio
    from indoorgs.colmap import model as cm
    out = Path(out_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    cams = path_cameras(n_views, width, height)
    rng = np.random.default_rng(seed)
    imgs = []
    for c in cams:
        im = render_gt(c, rng=rng)
        imgs.append(im)
        cv2.imwrite(str(img_dir / c.name), im[..., ::-1])
    # 학습 시점에서만 점 만들기 (테스트 시점 정보 새지 않게)
    tr = [i for i in range(n_views) if i % 8 != 0]
    xyz, rgb = colmap_like_points([cams[i] for i in tr], [imgs[i] for i in tr], n_points, seed=seed)
    M = cm.Model(cameras={1: cm.Camera(1, 1, width, height, np.array([cams[0].fx, cams[0].fy, cams[0].cx, cams[0].cy]))},
                 images={}, points={})
    for i, c in enumerate(cams):
        M.images[i + 1] = cm.Image(i + 1, plyio.mat_to_quat(c.R), np.asarray(c.t), 1, c.name)
    for j in range(len(xyz)):
        M.points[j + 1] = cm.Point3D(j + 1, xyz[j], rgb[j], 0.5, np.zeros((0, 2), np.uint32))
    md = out / "sparse" / "0"
    cm.write_model(M, md)
    (out / "planes.json").write_text(json.dumps({"version": 1, "note": "n·p = d, n 방 안쪽, m",
                                                 "box": {"x": [0, LX], "y": [0, LY], "z": [0, LZ]},
                                                 "planes": PLANES}, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"model_dir": str(md), "image_dir": str(img_dir), "planes": str(out / "planes.json"),
            "n_points": int(len(xyz))}


def novel_views(train_cams, test_cams, offset=0.3, grazing_deg=60.0):
    """train(novel_views=...) 콜백: 테스트 카메라마다 vr_cameras 6개 (옆 ±, 위·아래, 벽 비스듬히 ±60도) + 정답 렌더.
    방 밖이나 벽에 너무 붙은 카메라 (0.25 m 안) 는 뺌."""
    from train import vr_cameras
    out = []
    for c in vr_cameras(test_cams, offset, grazing_deg, up=[0, 0, 1]):
        C = c.center
        if np.any(C < 0.25) or np.any(C > np.array([LX, LY, LZ]) - 0.25):
            continue
        out.append({"name": c.name, "camera": c, "image": render_gt(c)})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="mini3dgs 상자 방 확인 실행")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--res", type=int, default=200)
    ap.add_argument("--cap", type=int, default=60000)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--random-bg", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    sc = make_scene(Path(a.out) / "scene")
    print(f"장면 만듦 {time.time() - t0:.1f}s, 점 {sc['n_points']}", flush=True)
    from train import train
    m = train(sc["model_dir"], sc["image_dir"], Path(a.out) / "train", a.iters, a.res, novel_views=novel_views,
              cap=a.cap, eval_every=a.eval_every, random_bg=a.random_bg)
    print(json.dumps({"test": m["test"]["mean"], "novel": m["novel"]["mean"], "count": m["count"],
                      "ply_stats": m["ply_stats"], "time": m["time"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
