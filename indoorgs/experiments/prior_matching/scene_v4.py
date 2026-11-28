"""v4 합성 복도: v3(synth_corridor/scene.py, ends_unique)에 실제 지하 복도에 있는 것들을 더함.

v3 와 같은 것: 복도 크기(24 x 2.4 x 2.7 m), 카메라 경로(gt_trajectory), 그린 동선(drawn_path),
              양 끝 8 m 의 문·게시물·끝 벽이 180도 대칭 복사 (헷갈리는 구조는 그대로).
v4 에서 더한 것 (실제 사진: 흰 벽, 천장 배관·보, 게시판, 소화기, 테라조 바닥 참고):
  - 벽: 위치마다 다른 얼룩·긁힘, 작은 공지문(고유), 콘센트(같은 모양 반복), 유도등(같은 모양)
  - 바닥: 테라조 알갱이(위치마다 다름) + 0.6 m 줄눈
  - 천장: 맨 콘크리트 얼룩(고유) + 선형 조명(2.4 m 반복)
  - 입체물(상자): 천장 배관 2줄(한쪽으로 치우침, 실제처럼 비대칭) + 걸이, 보(4.8 m 반복), 벽 기둥(대칭),
    소화전함·소화기·쓰레기통·벤치·상자(고유 위치)
  - 해상도 960x540 (v3 640x360)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "synth_corridor"))
import scene as v3  # noqa: E402

L, W, H, PPM, END_ZONE = v3.L, v3.W, v3.H, v3.PPM, v3.END_ZONE
IMG_W, IMG_H = 960, 540
HFOV_DEG = 80.0
FX = IMG_W / 2 / math.tan(math.radians(HFOV_DEG / 2))
K = np.array([[FX, 0, IMG_W / 2], [0, FX, IMG_H / 2], [0, 0, 1.0]])
SRC_FPS, STEP = v3.SRC_FPS, v3.STEP
gt_trajectory, drawn_path, planes_json = v3.gt_trajectory, v3.drawn_path, v3.planes_json


def _notice(seed, w_m, h_m):
    """작은 공지문: 흰 종이 + 글자 줄 + 색 띠 (seed 마다 다름)."""
    r = np.random.default_rng(seed)
    h, w = int(h_m * PPM), int(w_m * PPM)
    img = np.ones((h, w, 3), np.float32) * r.uniform(0.82, 0.97, 3)
    band = int(h * r.uniform(0.12, 0.3))
    img[:band] = r.uniform(0.1, 0.8, 3)
    v3._text(img, "".join(chr(int(r.integers(65, 90))) for _ in range(3)), (3, band - 3), band / 30, (0.95, 0.95, 0.95), 1)
    y = band + 6
    while y < h - 6:
        x1 = int(r.integers(w // 2, w - 3))
        cv2.line(img, (4, y), (x1, y), (0.15, 0.15, 0.15), 1)
        y += int(r.integers(4, 9))
    if r.random() < 0.6:
        p = (int(r.integers(5, w - 15)), int(r.integers(band, h - 15)))
        cv2.rectangle(img, p, (p[0] + int(r.integers(8, 25)), p[1] + int(r.integers(8, 25))),
                      tuple(float(v) for v in r.uniform(0.1, 0.9, 3)), -1)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0.3, 0.3, 0.3), 1)
    return img


def _stains(rng, T, per_m2, amp=(0.04, 0.14)):
    """위치마다 다른 얼룩(가우스 덩이)·긁힘(가는 선)."""
    h, w = T.shape[:2]
    n = int(per_m2 * h * w / PPM ** 2)
    for _ in range(n):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        if rng.random() < 0.7:
            rad = int(rng.integers(3, 14))
            x0, x1, y0, y1 = max(0, cx - 3 * rad), min(w, cx + 3 * rad), max(0, cy - 3 * rad), min(h, cy + 3 * rad)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            ex = rng.uniform(0.5, 2.0)
            g = np.exp(-(((xx - cx) / (rad * ex)) ** 2 + ((yy - cy) / rad) ** 2))
            T[y0:y1, x0:x1] *= (1 - rng.uniform(*amp) * g)[..., None]
        else:
            ang = rng.uniform(0, math.pi)
            ln = int(rng.integers(8, 50))
            m = np.zeros((h, w), np.uint8) if False else None
            p2 = (int(cx + ln * math.cos(ang)), int(cy + ln * math.sin(ang)))
            x0, x1, y0, y1 = max(0, min(cx, p2[0]) - 2), min(w, max(cx, p2[0]) + 3), max(0, min(cy, p2[1]) - 2), min(h, max(cy, p2[1]) + 3)
            if x1 <= x0 or y1 <= y0:
                continue
            m = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.line(m, (cx - x0, cy - y0), (p2[0] - x0, p2[1] - y0), 255, 1)
            T[y0:y1, x0:x1] *= (1 - rng.uniform(0.08, 0.2) * (m / 255.0))[..., None]


def make_textures(seed: int = 0) -> dict[str, np.ndarray]:
    t = v3.make_textures(seed, "ends_unique")      # A, B, E, F, C (양 끝 대칭 구조는 v3 그대로)
    A, B = t["A"].copy(), t["B"].copy()
    rng = np.random.default_rng(seed + 100)
    # 같은 모양이 반복되는 것: 콘센트(3 m 마다), 유도등(대칭 위치)
    outlet = np.ones((int(0.12 * PPM), int(0.08 * PPM), 3), np.float32) * 0.93
    cv2.rectangle(outlet, (0, 0), (outlet.shape[1] - 1, outlet.shape[0] - 1), (0.5, 0.5, 0.5), 1)
    cv2.circle(outlet, (5, 8), 2, (0.1, 0.1, 0.1), -1)
    cv2.circle(outlet, (10, 8), 2, (0.1, 0.1, 0.1), -1)
    cv2.circle(outlet, (5, 17), 2, (0.1, 0.1, 0.1), -1)
    cv2.circle(outlet, (10, 17), 2, (0.1, 0.1, 0.1), -1)
    exit_sign = np.ones((int(0.15 * PPM), int(0.36 * PPM), 3), np.float32) * np.array([0.1, 0.6, 0.25], np.float32)
    v3._text(exit_sign, "EXIT", (6, 22), 0.7, (0.95, 0.95, 0.95), 2)

    def is_plain(T, x_m, z_m, w_m, h_m):
        h, w = int(h_m * PPM), int(w_m * PPM)
        x0, y1 = int(x_m * PPM), T.shape[0] - int(z_m * PPM)
        reg = T[y1 - h:y1, x0:x0 + w]
        return reg.size and reg.min() > 0.7 and reg.std() < 0.03

    for T in (A, B):
        for x in np.arange(0.5, L - 0.2, 3.0):
            if is_plain(T, x, 0.3, 0.08, 0.12):
                v3._paste(T, outlet, float(x), 0.3)
        for x in (4.0, 12.0 - 0.18, L - 4.0 - 0.36):
            if is_plain(T, x, 2.2, 0.36, 0.15):
                v3._paste(T, exit_sign, float(x), 2.2)
    # 고유한 것: 공지문 (벽마다 약 1.3 m 에 하나), 얼룩·긁힘
    k = 0
    for T in (A, B):
        x = 0.3
        while x < L - 0.6:
            w_m, h_m = (0.21, 0.30) if rng.random() < 0.6 else (0.30, 0.42)
            z = float(rng.uniform(1.05, 1.75))
            if is_plain(T, x, z, w_m, h_m):
                v3._paste(T, _notice(seed + 500 + k, w_m, h_m), float(x), z)
                k += 1
            x += float(rng.uniform(0.7, 1.9))
        _stains(rng, T, per_m2=4.0)
    # 끝 벽: 둘 다 같은 구조(v3 E) + 같은 창틀, 얼룩만 다름
    E = t["E"].copy()
    wy0, wy1 = E.shape[0] - int(2.3 * PPM), E.shape[0] - int(1.55 * PPM)
    wx0, wx1 = int(1.15 * PPM), int(2.25 * PPM)
    E[wy0:wy1, wx0:wx1] = np.array([0.45, 0.55, 0.52], np.float32)
    cv2.rectangle(E, (wx0, wy0), (wx1, wy1), (0.95, 0.95, 0.95), 5)
    cv2.line(E, ((wx0 + wx1) // 2, wy0), ((wx0 + wx1) // 2, wy1), (0.95, 0.95, 0.95), 5)
    E0, EL = E.copy(), E.copy()
    _stains(rng, E0, per_m2=4.0)
    _stains(rng, EL, per_m2=4.0)
    v3._paste(E0, _notice(seed + 480, 0.21, 0.30), 0.15, 1.2)
    v3._paste(EL, _notice(seed + 481, 0.30, 0.42), 0.6, 1.0)
    # 바닥: 테라조 (알갱이 위치마다 다름) + 0.6 m 줄눈
    fr = np.random.default_rng(seed + 200)
    fh, fw = int(W * PPM), int(L * PPM)
    F = 0.66 + v3._lowfreq(fr, fh, fw, 50, 0.03)
    u = fr.random((fh, fw))
    F[u < 0.045] = 0.25
    F[u > 0.965] = 0.92
    big = (fr.random((fh // 2, fw // 2)) < 0.012).astype(np.float32)
    big = cv2.resize(big, (fw, fh), interpolation=cv2.INTER_NEAREST) > 0
    F[big] = 0.33
    tpx = int(0.6 * PPM)
    F[::tpx] = F[1::tpx] = 0.4
    F[:, ::tpx] = F[:, 1::tpx] = 0.4
    F = F[..., None] * np.array([1.0, 0.98, 0.93], np.float32)
    _stains(fr, F, per_m2=2.0, amp=(0.05, 0.2))
    # 천장: 맨 콘크리트 + 선형 조명 (2.4 m 반복)
    cr = np.random.default_rng(seed + 300)
    C = 0.80 + v3._lowfreq(cr, fh, fw, 40, 0.03) + v3._lowfreq(cr, fh, fw, 6, 0.01) \
        + cr.standard_normal((fh, fw)).astype(np.float32) * 0.006
    C = C[..., None] * np.ones(3, np.float32)
    _stains(cr, C, per_m2=5.0, amp=(0.04, 0.12))
    for kx in range(10):
        xc = 1.2 + 2.4 * kx
        x0, x1 = int((xc - 0.04) * PPM), int((xc + 0.04) * PPM)
        y0, y1 = int((W / 2 - 0.6) * PPM), int((W / 2 + 0.6) * PPM)
        C[y0:y1, x0:x1] = 1.0
    out = {"A": A, "B": B, "E0": E0, "EL": EL, "F": F, "C": C}
    return {k_: np.clip(v, 0, 1).astype(np.float32) for k_, v in out.items()}


def make_boxes(seed: int = 0) -> list[dict]:
    """입체물: 축에 나란한 상자 (lo, hi, 색, 잔무늬 세기)."""
    r = np.random.default_rng(seed + 400)
    bx = []

    def add(lo, hi, color, tex=0.03):
        bx.append({"lo": np.array(lo, np.float32), "hi": np.array(hi, np.float32),
                   "color": np.array(color, np.float32), "tex": tex, "seed": int(r.integers(1 << 30))})
    # 천장 배관 2줄: A 벽 쪽으로 치우침 (실제처럼 비대칭) + 걸이 (1.6 m 마다, 같은 모양)
    add([0, -1.02, 2.38], [L, -0.90, 2.50], [0.85, 0.85, 0.83], 0.02)
    add([0, -0.80, 2.42], [L, -0.72, 2.50], [0.15, 0.2, 0.55], 0.02)
    for x in np.arange(0.8, L, 1.6):
        add([x, -1.05, 2.36], [x + 0.04, -0.68, 2.40], [0.4, 0.4, 0.4], 0.0)
    # 보: 4.8 m 마다 (x -> L-x 대칭 위치)
    for x in (2.4, 7.2, 12.0, 16.8, 21.6):
        add([x - 0.15, -W / 2, 2.52], [x + 0.15, W / 2, H], [0.82, 0.81, 0.78], 0.02)
    # 벽 기둥: 대칭 위치 (A 벽 x, B 벽 L-x)
    for x in (4.8, 9.6):
        add([x - 0.2, -W / 2, 0], [x + 0.2, -W / 2 + 0.12, H], [0.84, 0.83, 0.80], 0.02)
        add([L - x - 0.2, W / 2 - 0.12, 0], [L - x + 0.2, W / 2, H], [0.84, 0.83, 0.80], 0.02)
    # 고유 위치: 소화전함, 소화기, 쓰레기통, 벤치, 상자
    add([10.6, W / 2 - 0.10, 0.5], [11.2, W / 2, 1.3], [0.75, 0.12, 0.10], 0.05)
    add([8.6, -W / 2, 0.0], [8.76, -W / 2 + 0.16, 0.5], [0.8, 0.1, 0.1], 0.05)
    add([20.5, W / 2 - 0.16, 0.0], [20.66, W / 2, 0.5], [0.8, 0.1, 0.1], 0.05)
    add([3.3, W / 2 - 0.38, 0.0], [3.65, W / 2 - 0.03, 0.6], [0.25, 0.3, 0.35], 0.08)
    add([17.2, -W / 2 + 0.03, 0.0], [17.55, -W / 2 + 0.38, 0.6], [0.3, 0.45, 0.3], 0.08)
    add([13.6, W / 2 - 0.42, 0.0], [14.8, W / 2 - 0.02, 0.45], [0.5, 0.36, 0.2], 0.10)
    add([6.4, W / 2 - 0.45, 0.0], [6.9, W / 2 - 0.05, 0.4], [0.62, 0.5, 0.36], 0.12)
    add([6.5, W / 2 - 0.42, 0.4], [6.85, W / 2 - 0.1, 0.65], [0.6, 0.48, 0.33], 0.12)
    add([22.0, -W / 2 + 0.05, 0.0], [22.6, -W / 2 + 0.5, 0.5], [0.58, 0.47, 0.33], 0.12)
    for b in bx:   # 면 잔무늬 (64 px/m, 상자마다 다름)
        b["noise"] = np.random.default_rng(b["seed"]).random((256, 256)).astype(np.float32)
    return bx


_FACE_SHADE = np.array([[0.80, 0.86], [0.92, 0.74], [0.62, 1.0]], np.float32)   # [축][-면, +면]


def render(tex, boxes, R_wc, C, ss: int = 2, rng=None):
    """world_from_cam 회전 R_wc, 중심 C. uint8 BGR."""
    w, h = IMG_W * ss, IMG_H * ss
    xs = ((np.arange(w) + 0.5) / ss).astype(np.float32)
    ys = ((np.arange(h) + 0.5) / ss).astype(np.float32)
    u, v = np.meshgrid(xs, ys)
    d_cam = np.stack([(u - K[0, 2]) / FX, (v - K[1, 2]) / FX, np.ones_like(u)], -1).reshape(-1, 3)
    d = (d_cam @ R_wc.T.astype(np.float32)).astype(np.float32)
    d[np.abs(d) < 1e-9] = 1e-9
    o = np.asarray(C, np.float32)
    inv = 1.0 / d
    lo, hi = np.array([0, -W / 2, 0], np.float32), np.array([L, W / 2, H], np.float32)
    tb = np.where(d > 0, (hi - o) * inv, (lo - o) * inv)
    ax = np.argmin(tb, 1)
    N = len(d)
    t = tb[np.arange(N), ax]
    P = o + d * t[:, None]
    col = np.zeros((N, 3), np.float32)
    m = ax == 0
    pos = d[:, 0] > 0
    for sel, uu, key in [(m & ~pos, W / 2 - P[:, 1], "E0"), (m & pos, P[:, 1] + W / 2, "EL")]:
        col[sel] = v3._bilinear(tex[key], uu[sel] * PPM, (H - P[sel, 2]) * PPM)
    m = ax == 1
    pos = d[:, 1] > 0
    for sel, key in [(m & ~pos, "A"), (m & pos, "B")]:
        col[sel] = v3._bilinear(tex[key], P[sel, 0] * PPM, (H - P[sel, 2]) * PPM)
    m = ax == 2
    pos = d[:, 2] > 0
    for sel, key in [(m & ~pos, "F"), (m & pos, "C")]:
        col[sel] = v3._bilinear(tex[key], P[sel, 0] * PPM, (P[sel, 1] + W / 2) * PPM)
    # 상자
    for b in boxes:
        t1, t2 = (b["lo"] - o) * inv, (b["hi"] - o) * inv
        tn, tf = np.minimum(t1, t2), np.maximum(t1, t2)
        a = np.argmax(tn, 1)
        tmin = tn[np.arange(N), a]
        hit = (tf.min(1) >= tmin) & (tmin > 1e-4) & (tmin < t)
        if not hit.any():
            continue
        idx = np.nonzero(hit)[0]
        ah = a[idx]
        Pb = o + d[idx] * tmin[idx, None]
        sign = (d[idx, ah] < 0).astype(np.int64)        # 광선이 -방향이면 +면에 맞음
        shade = _FACE_SHADE[ah, sign]
        i1, i2 = (ah + 1) % 3, (ah + 2) % 3
        uu = (Pb[np.arange(len(idx)), i1] * 64).astype(np.int64) % 256
        vv = (Pb[np.arange(len(idx)), i2] * 64).astype(np.int64) % 256
        nz = 1 + b["tex"] * (b["noise"][vv, uu] - 0.5) * 2
        col[idx] = b["color"] * (shade * nz)[:, None]
        t[idx] = tmin[idx]
        P[idx] = Pb
    phase = np.cos(2 * np.pi * (P[:, 0] - 1.2) / 2.4)
    light = 0.85 + 0.08 * phase - 0.012 * np.minimum(t, 15)
    col *= light[:, None].astype(np.float32)
    img = col.reshape(h, w, 3)
    if ss > 1:
        img = img.reshape(IMG_H, ss, IMG_W, ss, 3).mean((1, 3))
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]
    r2 = ((xx - IMG_W / 2) ** 2 + (yy - IMG_H / 2) ** 2) / (IMG_W / 2) ** 2
    img = img * (1 - 0.15 * r2)[..., None]
    if rng is not None:
        img = img + rng.normal(0, 1.5 / 255, img.shape)
    out = np.clip(img * 255, 0, 255).astype(np.uint8)
    return out[..., ::-1].copy()
