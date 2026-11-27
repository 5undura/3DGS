"""합성 복도 장면 + numpy 광선 추적 렌더러 + 정답(GT) 카메라 경로.

복도: x 0..L(24m), y -W/2..W/2(2.4m), z 0..H(2.7m). 도면 좌표 = 이 좌표 (x 오른쪽, y 위, z 위, 바닥 z=0).
일부러 헷갈리게: 양 끝 구간(x<END_ZONE, x>L-END_ZONE)은 복도 중심축 기준 180도 회전 대칭
(벽 A(y=-W/2)의 x 위치 무늬 = 벽 B(y=+W/2)의 L-x 위치 무늬, 양 끝 벽 같음, 바닥·천장 주기 무늬).
=> 한쪽 끝에서 앞을 본 사진 ≈ 반대쪽 끝에서 뒤를 본 사진. 가운데 구간만 고유한(약한) 무늬.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

L, W, H = 24.0, 2.4, 2.7
PPM = 200                     # 텍스처 해상도 (px/m)
END_ZONE = 8.0                # 대칭 구간 길이 (양 끝)
IMG_W, IMG_H = 640, 360
HFOV_DEG = 80.0
FX = IMG_W / 2 / math.tan(math.radians(HFOV_DEG / 2))
K = np.array([[FX, 0, IMG_W / 2], [0, FX, IMG_H / 2], [0, 0, 1.0]])
SRC_FPS = 30.0
STEP = 6                      # 원본 6프레임마다 1장 (5장/초)

# 벽 평면 (도면 좌표): n·p = d, 안쪽을 향한 법선
PLANES = {
    "wall_A_y-": ([0, 1, 0], -W / 2),
    "wall_B_y+": ([0, -1, 0], -W / 2),
    "end_x0": ([1, 0, 0], 0.0),
    "end_xL": ([-1, 0, 0], -L),
    "floor": ([0, 0, 1], 0.0),
    "ceiling": ([0, 0, -1], -H),
}


def planes_json() -> dict:
    return {"version": 1, "note": "n·p = d (n 은 복도 안쪽), 도면 좌표 m",
            "box": {"x": [0, L], "y": [-W / 2, W / 2], "z": [0, H]},
            "planes": [{"name": k, "n": n, "d": d} for k, (n, d) in PLANES.items()]}


# ---------------------------------------------------------------- 텍스처
def _lowfreq(rng, h, w, scale_px, amp):
    small = rng.standard_normal((max(2, h // scale_px), max(2, w // scale_px))).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC) * amp


def _wall_base(rng, w_m, h_m=H):
    h, w = int(h_m * PPM), int(w_m * PPM)
    base = np.ones((h, w, 3), np.float32) * np.array([0.86, 0.85, 0.82], np.float32)
    shade = 1 + _lowfreq(rng, h, w, 60, 0.025) + _lowfreq(rng, h, w, 6, 0.006)
    base *= shade[..., None]
    base += rng.standard_normal((h, w, 1)).astype(np.float32) * 0.004
    # 걸레받이 (아래 0.1m) + 가로 몰딩 (1.0m)
    base[h - int(0.10 * PPM):] = np.array([0.35, 0.33, 0.30])
    y = h - int(1.0 * PPM)
    base[y:y + 3] *= 0.93
    return base


def _text(img, txt, org, scale, color, thick):
    """OpenCV 5 putText 는 uint8 만 -> 마스크로 그려 섞음."""
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.putText(m, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, 255, thick)
    a = (m.astype(np.float32) / 255)[..., None]
    img[:] = img * (1 - a) + np.asarray(color, np.float32) * a


def _poster(seed, w_m=0.6, h_m=0.8):
    """seed 로 정해지는 포스터 (같은 seed = 똑같은 포스터)."""
    r = np.random.default_rng(seed)
    h, w = int(h_m * PPM), int(w_m * PPM)
    img = np.ones((h, w, 3), np.float32) * r.uniform(0.6, 0.95, 3)
    for _ in range(10):
        c = tuple(float(v) for v in r.uniform(0.05, 0.9, 3))
        kind = r.integers(3)
        p = (int(r.integers(0, w)), int(r.integers(0, h)))
        if kind == 0:
            cv2.circle(img, p, int(r.integers(6, w // 4)), c, -1)
        elif kind == 1:
            q = (int(r.integers(0, w)), int(r.integers(0, h)))
            cv2.rectangle(img, p, q, c, -1)
        else:
            _text(img, "".join(chr(int(r.integers(65, 90))) for _ in range(4)), p, r.uniform(0.6, 1.4), c, 2)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0.1, 0.1, 0.1), 3)
    return img


def _door(seed, w_m=0.95, h_m=2.1):
    r = np.random.default_rng(seed)
    h, w = int(h_m * PPM), int(w_m * PPM)
    img = np.ones((h, w, 3), np.float32) * np.array([0.42, 0.30, 0.20])
    grain = _lowfreq(r, h, w, 4, 0.05)
    grain = cv2.resize(grain[:, ::8], (w, h))  # 세로 결
    img *= (1 + grain)[..., None]
    f = int(0.05 * PPM)
    img[:f], img[:, :f], img[:, -f:] = 0.25, 0.25, 0.25  # 문틀
    hy, hx = int(1.0 * PPM), w - int(0.12 * PPM)
    cv2.rectangle(img, (hx - 6, hy - 25), (hx + 6, hy + 25), (0.75, 0.75, 0.7), -1)  # 손잡이
    cv2.rectangle(img, (w // 2 - 30, int(0.35 * PPM)), (w // 2 + 30, int(0.35 * PPM) + 18), (0.9, 0.9, 0.85), -1)
    _text(img, f"B1{r.integers(10, 99)}", (w // 2 - 26, int(0.35 * PPM) + 15), 0.45, (0.1, 0.1, 0.1), 1)  # 호실 번호판 (작아서 거의 안 보임)
    return img


def _paste(canvas, img, x_m, z_m):
    """x_m: 왼쪽 끝 (텍스처 u, m), z_m: 아래 끝 높이(m)."""
    h, w = img.shape[:2]
    H_px = canvas.shape[0]
    x0, y1 = int(x_m * PPM), H_px - int(z_m * PPM)
    canvas[y1 - h:y1, x0:x0 + w] = img


def make_textures(seed: int = 0, ambiguity: str = "ends") -> dict[str, np.ndarray]:
    """면별 텍스처 (float32 RGB 0..1). 텍스처 u 는 그 면을 안에서 봤을 때 오른쪽 방향.
    ambiguity:
      "extreme": 바닥 타일·천장 텍스가 완전 주기 무늬 (0.6m, 조명 2.4m) -> 복도 어디서나 몇 m 씩 헷갈림 (v1)
      "ends":    바닥·천장에 위치마다 다른 옅은 얼룩을 더하고, 양 끝 END_ZONE 만 180도 대칭 복사
                 -> 헷갈리는 곳은 양 끝 + 타일 하나하나의 잔무늬는 여전히 같음 (v2)
      "ends_unique": "ends" + 타일 잔무늬·천장 텍스 잔무늬도 위치마다 다름 (테라조처럼). 양 끝만 180도 대칭 (v3)"""
    if ambiguity not in ("extreme", "ends", "ends_unique"):
        raise ValueError(ambiguity)
    rng = np.random.default_rng(seed)
    # 벽 A (y=-W/2, 안에서 보면 -y 방향을 봄 -> 오른쪽 = -x). 저장은 u=x 로 하고 렌더러가 방향 처리
    A = _wall_base(rng, L)
    # 끝 구간 (x<END_ZONE): 문 2개 + 포스터 3개 (A 벽)
    for x, s in [(1.2, 11), (5.6, 12)]:
        _paste(A, _door(s), x, 0.0)
    for x, s in [(3.0, 21), (3.9, 22), (7.0, 23)]:
        _paste(A, _poster(s), x, 1.2)
    # 같은 끝 구간 B 벽 쪽 무늬 (나중에 대칭 복사로 반대편 끝 A 벽에도 생김)
    Bend = _wall_base(np.random.default_rng(seed + 1), L)
    for x, s in [(2.2, 31)]:
        _paste(Bend, _door(s), x, 0.0)
    for x, s in [(4.5, 41), (6.2, 42)]:
        _paste(Bend, _poster(s), x, 1.3)
    # 가운데 구간 고유 무늬 (약하게: 대비 낮은 게시판·표지)
    mid_a = [(9.5, 51), (14.0, 52)]
    mid_b = [(11.0, 61), (15.0, 62)]

    def faint(img, k=0.45):
        return 0.85 + (img - 0.85) * k

    for x, s in mid_a:
        _paste(A, faint(_poster(s, 0.8, 0.6)), x, 1.4)
    _paste(A, _door(71), 12.0, 0.0)          # 가운데 문 (하나)
    for x, s in mid_b:
        _paste(Bend, faint(_poster(s, 0.8, 0.6)), x, 1.4)
    # 180도 대칭: B(u) = flip(A)(L-u) 끝 구간, A(u) = flip(Bend)(L-u) 반대 끝
    e = int(END_ZONE * PPM)
    B = Bend.copy()
    B[:, -e:] = A[:, :e][:, ::-1]               # B 벽 먼 끝 = A 벽 가까운 끝을 뒤집은 것
    A[:, -e:] = Bend[:, :e][:, ::-1]            # A 벽 먼 끝 = B 벽 가까운 끝을 뒤집은 것

    # 끝 벽 (둘 다 같음): 소화전 + 표지
    E = _wall_base(np.random.default_rng(seed + 2), W)
    _paste(E, _poster(81, 0.5, 0.35), 0.4, 1.8)
    box = np.ones((int(0.8 * PPM), int(0.6 * PPM), 3), np.float32) * np.array([0.75, 0.12, 0.10])
    cv2.rectangle(box, (8, 8), (box.shape[1] - 9, box.shape[0] - 9), (0.95, 0.9, 0.9), 3)
    _paste(E, box, 1.4, 0.6)

    # 바닥: 0.6m 타일, 타일 하나 무늬를 180도 대칭으로 만들어 반복 (=> 회전 대칭)
    t = int(0.6 * PPM)
    tr = np.random.default_rng(seed + 3)
    tile = 0.55 + _lowfreq(tr, t, t, 5, 0.05) + tr.standard_normal((t, t)).astype(np.float32) * 0.015
    tile = 0.5 * (tile + tile[::-1, ::-1])
    tile[:2], tile[-2:], tile[:, :2], tile[:, -2:] = 0.35, 0.35, 0.35, 0.35  # 줄눈
    F = np.tile(tile, (int(W / 0.6), int(L / 0.6)))
    if ambiguity == "ends_unique":   # 잔무늬를 바닥 전체에서 새로 (줄눈은 유지)
        grout = F <= 0.351
        F = 0.55 + _lowfreq(tr, *F.shape, 5, 0.05) + tr.standard_normal(F.shape).astype(np.float32) * 0.015
        F[grout] = 0.35
    F = F[..., None] * np.array([1.0, 0.97, 0.92], np.float32)
    # 천장: 0.6m 텍스 + 2.4m 마다 조명 패널 (가운데 맞춤 -> 대칭)
    cr = np.random.default_rng(seed + 4)
    ct = 0.80 + cr.standard_normal((t, t)).astype(np.float32) * 0.01
    ct = 0.5 * (ct + ct[::-1, ::-1])
    ct[:2], ct[:, :2], ct[-2:], ct[:, -2:] = 0.6, 0.6, 0.6, 0.6
    C = np.tile(ct, (int(W / 0.6), int(L / 0.6)))
    if ambiguity == "ends_unique":
        grout = C <= 0.601
        C = 0.80 + _lowfreq(cr, *C.shape, 3, 0.02) + cr.standard_normal(C.shape).astype(np.float32) * 0.01
        C[grout] = 0.6
    C = C[..., None] * np.ones(3, np.float32)
    for k in range(10):
        xc = 1.2 + 2.4 * k
        x0, x1 = int((xc - 0.6) * PPM), int((xc + 0.6) * PPM)
        y0, y1 = int((W / 2 - 0.3) * PPM), int((W / 2 + 0.3) * PPM)
        C[y0:y1, x0:x1] = 1.0
        C[y0:y0 + 3, x0:x1] = C[y1 - 3:y1, x0:x1] = 0.5
        C[y0:y1, x0:x0 + 3] = C[y0:y1, x1 - 3:x1] = 0.5
    if ambiguity in ("ends", "ends_unique"):
        sr = np.random.default_rng(seed + 5)
        for T, amp in ((F, 0.07), (C, 0.03)):
            h, w = T.shape[:2]
            blot = 1 + _lowfreq(sr, h, w, 40, amp) + _lowfreq(sr, h, w, 8, amp / 2)
            spots = (sr.random((h // 4, w // 4)) < 0.004).astype(np.float32)   # 작은 얼룩·흠집
            spots = cv2.resize(cv2.dilate(spots, np.ones((2, 2), np.uint8)), (w, h), interpolation=cv2.INTER_LINEAR)
            T *= (blot * (1 - 0.25 * spots))[..., None]
            T[:, -e:] = T[::-1, :e][:, ::-1]      # 양 끝은 180도 대칭 (행 = y+W/2, 열 = x)
    return {"A": np.clip(A, 0, 1), "B": np.clip(B, 0, 1), "E": np.clip(E, 0, 1),
            "F": np.clip(F, 0, 1), "C": np.clip(C, 0, 1)}


# ---------------------------------------------------------------- 렌더러
def _bilinear(tex, u_px, v_px):
    h, w = tex.shape[:2]
    u = np.clip(u_px, 0, w - 1.001)
    v = np.clip(v_px, 0, h - 1.001)
    u0, v0 = u.astype(np.int32), v.astype(np.int32)
    fu, fv = (u - u0)[:, None], (v - v0)[:, None]
    a = tex[v0, u0] * (1 - fu) + tex[v0, u0 + 1] * fu
    b = tex[v0 + 1, u0] * (1 - fu) + tex[v0 + 1, u0 + 1] * fu
    return a * (1 - fv) + b * fv


def render(tex, R_wc, C, ss: int = 2, rng=None):
    """R_wc: world_from_cam 회전 (열 = 카메라 x오른쪽, y아래, z앞), C: 카메라 중심. uint8 BGR 반환."""
    w, h = IMG_W * ss, IMG_H * ss
    xs = (np.arange(w) + 0.5) / ss
    ys = (np.arange(h) + 0.5) / ss
    u, v = np.meshgrid(xs, ys)
    d_cam = np.stack([(u - K[0, 2]) / FX, (v - K[1, 2]) / FX, np.ones_like(u)], -1).reshape(-1, 3)
    d = d_cam @ R_wc.T
    o = np.asarray(C, np.float64)
    lo, hi = np.array([0, -W / 2, 0]), np.array([L, W / 2, H])
    with np.errstate(divide="ignore", invalid="ignore"):
        tb = np.where(d > 0, (hi - o) / d, (lo - o) / d)
    tb[~np.isfinite(tb)] = np.inf
    ax = np.argmin(tb, 1)
    t = tb[np.arange(len(d)), ax]
    P = o + d * t[:, None]
    col = np.zeros((len(d), 3), np.float32)
    # x 면 (끝 벽): x=0 벽은 +x 를 봄 -> 오른쪽 = -y ... u = (W/2 - y) ; x=L 벽 -> 오른쪽 = +y, u = y + W/2
    m = ax == 0
    pos = d[:, 0] > 0
    for sel, uu in [(m & ~pos, W / 2 - P[:, 1]), (m & pos, P[:, 1] + W / 2)]:
        col[sel] = _bilinear(tex["E"], uu[sel] * PPM, (H - P[sel, 2]) * PPM)
    # y 면: A (y=-W/2) 텍스처 u=x, B (y=+W/2) u=x (B 는 make_textures 에서 이미 대칭 처리)
    m = ax == 1
    pos = d[:, 1] > 0
    for sel, key in [(m & ~pos, "A"), (m & pos, "B")]:
        col[sel] = _bilinear(tex[key], P[sel, 0] * PPM, (H - P[sel, 2]) * PPM)
    # z 면: 바닥/천장 텍스처 행 = y + W/2, 열 = x
    m = ax == 2
    pos = d[:, 2] > 0
    for sel, key in [(m & ~pos, "F"), (m & pos, "C")]:
        col[sel] = _bilinear(tex[key], P[sel, 0] * PPM, (P[sel, 1] + W / 2) * PPM)
    # 조명: 천장 패널(2.4m 주기)로부터 거리에 따른 약한 밝기 변화 + 거리 감쇠 (대칭 유지)
    phase = np.cos(2 * np.pi * (P[:, 0] - 1.2) / 2.4)
    light = 0.85 + 0.08 * phase - 0.012 * np.minimum(t, 15)
    col *= light[:, None].astype(np.float32)
    img = col.reshape(h, w, 3)
    if ss > 1:
        img = img.reshape(IMG_H, ss, IMG_W, ss, 3).mean((1, 3))
    # 비네팅 + 센서 잡음
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]
    r2 = ((xx - IMG_W / 2) ** 2 + (yy - IMG_H / 2) ** 2) / (IMG_W / 2) ** 2
    img = img * (1 - 0.15 * r2)[..., None]
    if rng is not None:
        img = img + rng.normal(0, 1.5 / 255, img.shape)
    out = np.clip(img * 255, 0, 255).astype(np.uint8)
    return out[..., ::-1].copy()  # RGB -> BGR


# ---------------------------------------------------------------- 카메라 경로
def _smooth_interp(tk, vk, t):
    from scipy.interpolate import PchipInterpolator
    return PchipInterpolator(tk, vk)(t)


def rot_from_ypr(yaw, pitch, roll):
    """world_from_cam. yaw: z축 기준(0 = +x 를 봄), pitch: 위로 +, roll."""
    f = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
    up = np.array([0, 0, 1.0])
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    # roll: r,u 를 f 축으로 회전
    cr, sr = math.cos(roll), math.sin(roll)
    r2, u2 = cr * r + sr * u, -sr * r + cr * u
    return np.stack([r2, -u2, f], 1)  # 열: x 오른쪽, y 아래, z 앞


# 정답 경로 키프레임 (t초, x, y). 오른쪽 통행: 갈 때 y<0, 올 때 y>0. 10~12초 멈춤.
GT_KEYS = [(0, 1.0, -0.30), (4, 4.6, -0.35), (8, 8.4, -0.20), (10, 10.0, -0.25), (12, 10.1, -0.28),
           (17, 15.2, -0.40), (21, 19.6, -0.30), (24.5, 22.4, -0.20), (26.5, 22.9, 0.05),
           (28.5, 22.4, 0.35), (32, 19.0, 0.40), (37, 14.0, 0.30), (42, 9.5, 0.45), (46, 6.0, 0.35),
           (50, 2.6, 0.30), (52, 1.2, 0.25)]
DURATION = 52.0


def gt_trajectory():
    """프레임별 (이름, 시간, R_wc, C). 원본 30fps 중 6프레임마다."""
    tk = np.array([k[0] for k in GT_KEYS])
    n_src = int(DURATION * SRC_FPS)
    src = np.arange(0, n_src + 1, STEP)
    t = src / SRC_FPS
    x = _smooth_interp(tk, [k[1] for k in GT_KEYS], t)
    y = _smooth_interp(tk, [k[2] for k in GT_KEYS], t)
    z = 1.45 + 0.03 * np.sin(2 * np.pi * t / 0.53) + 0.04 * np.sin(2 * np.pi * t / 9.0)
    base_yaw = _smooth_interp([0, 24.0, 29.0, DURATION], [0, 0, math.pi, math.pi], t)
    yaw = base_yaw + math.radians(35) * np.sin(2 * np.pi * t / 7.0) + math.radians(8) * np.sin(2 * np.pi * t / 2.3)
    pitch = math.radians(12) * np.sin(2 * np.pi * t / 5.3)
    for tc, amp, wdt in [(14.5, 45, 1.2), (34.0, -40, 1.2), (44.0, 35, 1.0)]:  # 천장·바닥 보기
        pitch += math.radians(amp) * np.exp(-((t - tc) / wdt) ** 2)
    roll = math.radians(3) * np.sin(2 * np.pi * t / 3.1)
    out = []
    for i in range(len(t)):
        name = f"f_{i + 1:05d}_{src[i]:06d}.jpg"
        out.append((name, float(t[i]), rot_from_ypr(yaw[i], pitch[i], roll[i]), np.array([x[i], y[i], z[i]])))
    return out


def drawn_path(time_shift: float = 0.0, offset_m: float = 0.0) -> dict:
    """사용자가 도면에 손으로 그린 동선 (정답과 0.2~0.3m 다름) + 기억에 의존한 대략적 고정점 (초 단위 반올림).
    일부러 틀린 동선:
      time_shift: 가운데 두 고정점 시간을 +time_shift 초 (시간을 잘못 기억)
      offset_m:   선과 고정점 전체를 x 로 offset_m 밀기 (도면에 그린 위치가 통째로 어긋남)"""
    stroke = [[1.0, -0.1], [12.0, -0.5], [22.5, -0.1], [23.0, 0.2], [22.5, 0.5], [12.0, 0.2], [1.0, 0.5]]
    anchors = [{"t": 0.0, "point": [1.0, -0.1]},
               {"t": 11.0 + time_shift, "point": [10.0, -0.4]},     # "포스터 앞에서 잠깐 섰음"
               {"t": 26.0 + time_shift, "point": [23.0, 0.2]},      # 끝에서 돔
               {"t": 52.0, "point": [1.0, 0.5]}]
    stroke = [[x + offset_m, y] for x, y in stroke]
    for a in anchors:
        a["point"] = [a["point"][0] + offset_m, a["point"][1]]
    return {"version": 1, "camera_height": 1.4, "sigma_xy": 0.5, "sigma_z": 0.3,
            "stroke": stroke, "anchors": anchors}
