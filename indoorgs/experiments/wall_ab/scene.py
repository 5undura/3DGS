"""wall_ab 장면: 6 x 2.4 x 2.7 m 복도 + 문으로 이어진 3.55 x 3.0 x 2.62 m 방 (실제 사용자 문제와 비슷한 흰 벽 실내).

- 흰/미색 벽 (아주 약한 얼룩, 무늬 거의 없음) + 걸레받이, 스위치, 닫힌 문(칠한 문, 평면), 비상구 표지.
- 복도 벽에 **2.5 cm 튀어나온 포스터** (앞면 + 옆면 4개, 아래 30% 는 흰 종이 여백 -> snap 색 검사로 못 지키는 경우),
  방 벽에 2 cm 튀어나온 액자 (어두운 테두리 + 그림).
- 바닥 30 cm 타일 + 줄눈, 천장 60 cm 텍스 + 조명 패널. 조명은 램프 거리 감쇠 (시점 무관 -> 3DGS 가 원리상 맞출 수 있음).
- 도면(plan.json) 은 줄자 오차 흉내: 방 동쪽 벽 실제 x=6.05 (도면 6.00), 방 천장 실제 2.62 (도면 2.60). 나머지는 같음.
- 카메라: 손에 들고 걷는 스캔 (복도 앞으로 걸으며 좌우 벽·천장·바닥 보기 -> 문으로 방 들어가 둘러보기 -> 나와서 뒤로).
- COLMAP 비슷한 초기 점: 밝기 기울기가 큰 곳(무늬·모서리)에 몰리고 흰 벽 가운데엔 거의 없음, 1.5 cm 잡음, 2% 허공 점.

좌표 = 도면 좌표 (m, z 위). 복도 x 0..6, y -1.2..1.2. 방 x 2.5..6.05, y 1.3..4.3 (벽 두께 0.1, 문 x 3.2..4.1, z 0..2.1).

공개 API
  QUADS                               면 조각 목록 (광선 추적용)
  raycast(o, d) -> (t, quad 번호)      가장 가까운 면
  render_gt(cam, ss=2) -> uint8 RGB;  render_gt_full(cam, ss) -> (rgb float, 입사각 deg, 깊이, quad 번호)
  path_cameras(n, w, h) / grazing_cameras(w, h) / novel_cameras(test_cams)
  make_scene(out_dir, ...) -> dict(model_dir, image_dir, plan_json, ...)
  TRUE_PLANES  실제 면 (평가용 Plane 목록, 도면 오차 없음)
  POSTER       포스터 정보 (벽 평면, 범위, 튀어나온 거리)
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
MINI = HERE.parent / "mini3dgs"
SRC = HERE.parents[1] / "src"
for p in (str(MINI), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from gs import Camera  # noqa: E402

PPM = 150                    # 텍스처 px/m
HFOV = 75.0
CW, CH = 1.2, 2.7            # 복도 반폭, 높이
CX1 = 6.0                    # 복도 길이
RX0, RX1, RY0, RY1, RH = 2.5, 6.05, 1.3, 4.3, 2.62   # 방 (실제)
DX0, DX1, DH = 3.2, 4.1, 2.1                          # 문 구멍
POSTER = {"wall_y": -CW, "x": (1.4, 2.0), "z": (1.05, 1.85), "depth": 0.025}
FRAME = {"wall_x": RX0, "y": (2.3, 3.1), "z": (1.2, 1.75), "depth": 0.02}
LAMPS = np.array([[1.0, 0, 2.69], [3.0, 0, 2.69], [5.0, 0, 2.69], [3.6, 2.8, 2.61], [5.0, 2.8, 2.61]])


# ---------------------------------------------------------------- 텍스처
def _lowfreq(rng, h, w, cells, amp):
    g = rng.normal(0, 1, (max(2, cells[0]), max(2, cells[1]), 3)).astype(np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC) * amp


def _white(rng, w_m, h_m, skirting=True, base=(0.87, 0.86, 0.83)):
    h, w = max(2, int(round(h_m * PPM))), max(2, int(round(w_m * PPM)))
    img = np.empty((h, w, 3), np.float32)
    img[:] = base
    img += _lowfreq(rng, h, w, (max(2, int(h_m)), max(2, int(w_m * 1.5))), 0.012)
    img += rng.normal(0, 0.003, (h, w, 1)).astype(np.float32)
    if skirting:
        img[h - int(0.10 * PPM):] = (0.33, 0.31, 0.29)
    return img


def _rect(img, u0, v0, u1, v1, col):
    """(u, v) m 좌표 (v 는 위에서 아래가 아니라 아래에서 위 = 높이). img 행 0 = 위."""
    h = img.shape[0]
    r0, r1 = h - int(v1 * PPM), h - int(v0 * PPM)
    c0, c1 = int(u0 * PPM), int(u1 * PPM)
    img[max(0, r0):max(0, r1), max(0, c0):max(0, c1)] = col
    return (r0, r1, c0, c1)


def _door_tex(rng, img, u0, w=0.9, hgt=2.05, col=(0.42, 0.30, 0.20)):
    r0, r1, c0, c1 = _rect(img, u0, 0.0, u0 + w, hgt, col)
    img[r0:r1, c0:c1] += _lowfreq(rng, r1 - r0, c1 - c0, (8, 2), 0.025)
    cv2.rectangle(img, (c0, r0), (c1 - 1, r1 - 1), (0.18, 0.15, 0.12), 3)
    cv2.circle(img, (c1 - int(0.1 * PPM), (r0 + r1) // 2), 5, (0.8, 0.78, 0.7), -1)


def poster_tex(rng, w_m, h_m):
    """포스터 앞면: 위 70% 색 그림 + 글자 줄, 아래 30% 흰 종이에 검은 글자 (흰 여백은 벽 색과 비슷)."""
    h, w = int(h_m * PPM), int(w_m * PPM)
    img = np.empty((h, w, 3), np.float32)
    img[:] = (0.90, 0.89, 0.86)
    top = int(0.70 * h)
    img[:top] = (0.15, 0.35, 0.65)
    img[:top] += _lowfreq(rng, top, w, (4, 3), 0.08)
    cv2.circle(img, (int(0.65 * w), int(0.35 * h)), int(0.15 * w), (0.95, 0.80, 0.15), -1)
    cv2.rectangle(img, (int(0.1 * w), int(0.45 * h)), (int(0.45 * w), int(0.6 * h)), (0.85, 0.2, 0.2), -1)
    for k in range(4):
        y = top + int((0.2 + 0.18 * k) * (h - top))
        cv2.line(img, (int(0.08 * w), y), (int((0.5 + 0.4 * rng.random()) * w), y), (0.1, 0.1, 0.1), 2)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0.1, 0.1, 0.1), 2)
    return np.clip(img, 0, 1)


def frame_tex(rng, w_m, h_m):
    h, w = int(h_m * PPM), int(w_m * PPM)
    img = np.empty((h, w, 3), np.float32)
    img[:] = (0.15, 0.12, 0.10)
    b = int(0.04 * PPM)
    inner = img[b:h - b, b:w - b]
    inner[:] = (0.55, 0.65, 0.40)
    inner += _lowfreq(rng, inner.shape[0], inner.shape[1], (3, 4), 0.12)
    cv2.circle(inner, (inner.shape[1] // 3, inner.shape[0] // 3), inner.shape[0] // 5, (0.9, 0.6, 0.3), -1)
    return np.clip(img, 0, 1)


def floor_tex(rng, w_m, h_m):
    h, w = int(h_m * PPM), int(w_m * PPM)
    img = np.full((h, w, 3), (0.56, 0.53, 0.49), np.float32) + _lowfreq(rng, h, w, (int(h_m * 4), int(w_m * 4)), 0.03)
    tile = rng.normal(0, 0.025, (int(h_m / 0.3) + 2, int(w_m / 0.3) + 2, 1)).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    img += tile[(yy / (0.3 * PPM)).astype(int), (xx / (0.3 * PPM)).astype(int)]
    step = int(0.3 * PPM)
    img[:, ::step] = 0.36
    img[:, 1::step] = 0.36
    img[::step] = 0.36
    img[1::step] = 0.36
    return np.clip(img, 0, 1)


def ceil_tex(rng, w_m, h_m, lamps_uv):
    h, w = int(h_m * PPM), int(w_m * PPM)
    img = np.full((h, w, 3), 0.90, np.float32) + _lowfreq(rng, h, w, (3, 4), 0.008)
    img += rng.normal(0, 0.006, (h, w, 1)).astype(np.float32)     # 텍스 점무늬 (아주 약함)
    step = int(0.6 * PPM)
    img[:, ::step] = 0.62
    img[::step] = 0.62
    for cu, cv in lamps_uv:
        _rect(img, cu - 0.3, cv - 0.3, cu + 0.3, cv + 0.3, (1.0, 1.0, 0.98))
    return np.clip(img, 0, 1)


# ---------------------------------------------------------------- 면 조각
@dataclass
class Quad:
    name: str
    axis: int            # 법선 축
    pos: float           # 그 축 좌표
    sign: int            # 보이는 쪽 법선 부호 (+1 = +축 방향을 봄)
    ua: int              # 텍스처 가로 축 (세계 축 번호)
    va: int              # 세로 축
    u: tuple             # (u0, u1) 세계 좌표 범위
    v: tuple
    tex: np.ndarray      # 행 0 = v1 쪽 (위), 열 0 = u0 쪽 (flip_u 면 u1 쪽)
    flip_u: bool = False
    tu0: float = 0.0     # 텍스처의 u0 이 해당하는 세계 좌표 (조각이 큰 텍스처의 일부일 때)
    tv0: float = 0.0


def _split_wall(name, axis, pos, sign, ua, u_rng, v_rng, tex, holes, flip_u=False, tu0=None):
    """벽 하나를 문 구멍(u0,u1,v0,v1) 피해 조각으로 (같은 텍스처 공유)."""
    tu0 = u_rng[0] if tu0 is None else tu0
    out = []
    if not holes:
        return [Quad(name, axis, pos, sign, ua, 2, u_rng, v_rng, tex, flip_u, tu0, v_rng[0])]
    (h0, h1, hv0, hv1), = holes
    out.append(Quad(name + "_a", axis, pos, sign, ua, 2, (u_rng[0], h0), v_rng, tex, flip_u, tu0, v_rng[0]))
    out.append(Quad(name + "_b", axis, pos, sign, ua, 2, (h1, u_rng[1]), v_rng, tex, flip_u, tu0, v_rng[0]))
    out.append(Quad(name + "_c", axis, pos, sign, ua, 2, (h0, h1), (hv1, v_rng[1]), tex, flip_u, tu0, v_rng[0]))
    return out


def build_quads(seed=0):
    rng = np.random.default_rng(seed)
    Q = []
    # 복도 남쪽 벽 y=-1.2 (북쪽을 봄, sign +1). u = x. 포스터 자리 + 스위치
    t = _white(rng, CX1, CH)
    _rect(t, 0.75, 1.15, 0.83, 1.27, (0.95, 0.95, 0.92))
    _rect(t, 4.6, 1.15, 4.68, 1.27, (0.95, 0.95, 0.92))
    Q += _split_wall("c_south", 1, -CW, +1, 0, (0, CX1), (0, CH), t, [])
    # 복도 북쪽 벽 y=+1.2 (남쪽을 봄), 문 구멍. u = x, 텍스처는 x 반대 방향으로 보이지만 그대로 둠 (시점 무관)
    t = _white(rng, CX1, CH)
    _rect(t, 1.2, 2.25, 1.6, 2.4, (0.15, 0.65, 0.25))                 # 비상구 표지 (초록)
    _rect(t, 5.0, 1.15, 5.08, 1.27, (0.95, 0.95, 0.92))
    Q += _split_wall("c_north", 1, CW, -1, 0, (0, CX1), (0, CH), t, [(DX0, DX1, 0, DH)])
    # 복도 서쪽 끝 x=0: 닫힌 문 (칠한 평면)
    t = _white(rng, 2 * CW, CH)
    _door_tex(rng, t, 0.75)
    Q += _split_wall("c_west", 0, 0.0, +1, 1, (-CW, CW), (0, CH), t, [])
    # 복도 동쪽 끝 x=6 (흰 벽만)
    Q += _split_wall("c_east", 0, CX1, -1, 1, (-CW, CW), (0, CH), _white(rng, 2 * CW, CH), [])
    # 바닥·천장 (복도)
    Q.append(Quad("c_floor", 2, 0.0, +1, 0, 1, (0, CX1), (-CW, CW), floor_tex(rng, CX1, 2 * CW)[::-1], tu0=0, tv0=-CW))
    Q.append(Quad("c_ceil", 2, CH, -1, 0, 1, (0, CX1), (-CW, CW),
                  ceil_tex(rng, CX1, 2 * CW, [(1.0, CW), (3.0, CW), (5.0, CW)])[::-1], tu0=0, tv0=-CW))
    # 방 남쪽 벽 y=1.3 (북쪽을 봄), 문 구멍
    t = _white(rng, RX1 - RX0, RH)
    Q += _split_wall("r_south", 1, RY0, +1, 0, (RX0, RX1), (0, RH), t, [(DX0, DX1, 0, DH)])
    # 방 북쪽 y=4.3, 서쪽 x=2.5 (액자), 동쪽 x=6.05
    t = _white(rng, RX1 - RX0, RH)
    _rect(t, 1.0, 1.15, 1.08, 1.27, (0.95, 0.95, 0.92))
    Q += _split_wall("r_north", 1, RY1, -1, 0, (RX0, RX1), (0, RH), t, [])
    Q += _split_wall("r_west", 0, RX0, +1, 1, (RY0, RY1), (0, RH), _white(rng, RY1 - RY0, RH), [])
    t = _white(rng, RY1 - RY0, RH)
    _door_tex(rng, t, 1.6, col=(0.30, 0.32, 0.38))                     # 방 동쪽 벽의 닫힌 문 (회청색)
    Q += _split_wall("r_east", 0, RX1, -1, 1, (RY0, RY1), (0, RH), t, [])
    Q.append(Quad("r_floor", 2, 0.0, +1, 0, 1, (RX0, RX1), (RY0, RY1),
                  floor_tex(rng, RX1 - RX0, RY1 - RY0)[::-1], tu0=RX0, tv0=RY0))
    Q.append(Quad("r_ceil", 2, RH, -1, 0, 1, (RX0, RX1), (RY0, RY1),
                  ceil_tex(rng, RX1 - RX0, RY1 - RY0, [(3.6 - RX0, 2.8 - RY0), (5.0 - RX0, 2.8 - RY0)])[::-1],
                  tu0=RX0, tv0=RY0))
    # 문틀 (벽 두께 0.1): 옆 두 개 + 위
    jamb = _white(rng, 0.1, DH, skirting=False, base=(0.80, 0.79, 0.76))
    Q.append(Quad("jamb_w", 0, DX0, +1, 1, 2, (CW, RY0), (0, DH), jamb, tu0=CW, tv0=0))
    Q.append(Quad("jamb_e", 0, DX1, -1, 1, 2, (CW, RY0), (0, DH), jamb, tu0=CW, tv0=0))
    head = _white(rng, DX1 - DX0, 0.1, skirting=False, base=(0.80, 0.79, 0.76))
    Q.append(Quad("jamb_top", 2, DH, -1, 0, 1, (DX0, DX1), (CW, RY0), head, tu0=DX0, tv0=CW))
    # 포스터 (복도 남쪽 벽에서 +y 로 2.5 cm): 앞면 + 옆면
    px0, px1 = POSTER["x"]
    pz0, pz1 = POSTER["z"]
    py = -CW + POSTER["depth"]
    Q.append(Quad("poster", 1, py, +1, 0, 2, (px0, px1), (pz0, pz1), poster_tex(rng, px1 - px0, pz1 - pz0),
                  tu0=px0, tv0=pz0))
    edge = np.full((max(2, int(POSTER["depth"] * PPM) + 1), int((pz1 - pz0) * PPM), 3), 0.25, np.float32)
    edge_h = np.full((int((pz1 - pz0) * PPM), max(2, int(POSTER["depth"] * PPM) + 1), 3), 0.25, np.float32)
    Q.append(Quad("poster_w", 0, px0, -1, 1, 2, (-CW, py), (pz0, pz1), edge_h, tu0=-CW, tv0=pz0))
    Q.append(Quad("poster_e", 0, px1, +1, 1, 2, (-CW, py), (pz0, pz1), edge_h, tu0=-CW, tv0=pz0))
    edge_t = np.full((max(2, int(POSTER["depth"] * PPM) + 1), int((px1 - px0) * PPM), 3), 0.25, np.float32)
    Q.append(Quad("poster_t", 2, pz1, +1, 0, 1, (px0, px1), (-CW, py), edge_t, tu0=px0, tv0=-CW))
    Q.append(Quad("poster_b", 2, pz0, -1, 0, 1, (px0, px1), (-CW, py), edge_t, tu0=px0, tv0=-CW))
    del edge
    # 액자 (방 서쪽 벽에서 +x 로 2 cm)
    fy0, fy1 = FRAME["y"]
    fz0, fz1 = FRAME["z"]
    fx = RX0 + FRAME["depth"]
    Q.append(Quad("frame", 0, fx, +1, 1, 2, (fy0, fy1), (fz0, fz1), frame_tex(rng, fy1 - fy0, fz1 - fz0),
                  tu0=fy0, tv0=fz0))
    side_v = np.full((int((fz1 - fz0) * PPM), 4, 3), 0.12, np.float32)
    side_h = np.full((4, int((fy1 - fy0) * PPM), 3), 0.12, np.float32)
    Q.append(Quad("frame_s", 1, fy0, -1, 0, 2, (RX0, fx), (fz0, fz1), side_v, tu0=RX0, tv0=fz0))
    Q.append(Quad("frame_n", 1, fy1, +1, 0, 2, (RX0, fx), (fz0, fz1), side_v, tu0=RX0, tv0=fz0))
    Q.append(Quad("frame_t", 2, fz1, +1, 0, 1, (RX0, fx), (fy0, fy1), side_h.transpose(1, 0, 2), tu0=RX0, tv0=fy0))
    Q.append(Quad("frame_b", 2, fz0, -1, 0, 1, (RX0, fx), (fy0, fy1), side_h.transpose(1, 0, 2), tu0=RX0, tv0=fy0))
    return Q


QUADS = build_quads(0)


def _bilinear(tex, u, v):
    h, w = tex.shape[:2]
    u = np.clip(u, 0, w - 1.001)
    v = np.clip(v, 0, h - 1.001)
    u0, v0 = u.astype(np.int32), v.astype(np.int32)
    fu, fv = (u - u0)[:, None], (v - v0)[:, None]
    a = tex[v0, u0] * (1 - fu) + tex[v0, u0 + 1] * fu
    b = tex[v0 + 1, u0] * (1 - fu) + tex[v0 + 1, u0 + 1] * fu
    return a * (1 - fv) + b * fv


def raycast(o, d):
    """반환 t (N,), quad 번호 (N,) (-1 = 없음)."""
    o = np.broadcast_to(o, d.shape)
    N = len(d)
    best = np.full(N, np.inf)
    qi = np.full(N, -1, np.int32)
    for k, q in enumerate(QUADS):
        da = d[:, q.axis]
        ok = da * q.sign < -1e-9
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (q.pos - o[:, q.axis]) / da
        ok &= (t > 1e-6) & (t < best)
        if not ok.any():
            continue
        idx = np.flatnonzero(ok)
        P = o[idx] + d[idx] * t[idx, None]
        pu, pv = P[:, q.ua], P[:, q.va]
        inside = (pu >= q.u[0]) & (pu <= q.u[1]) & (pv >= q.v[0]) & (pv <= q.v[1])
        idx = idx[inside]
        best[idx] = t[idx]
        qi[idx] = k
    return best, qi


def shade(P, qi):
    col = np.zeros((len(P), 3), np.float32)
    for k in np.unique(qi):
        if k < 0:
            continue
        q = QUADS[k]
        m = qi == k
        tex = q.tex
        h = tex.shape[0]
        u = (P[m, q.ua] - q.tu0) * PPM
        v = h - 1 - (P[m, q.va] - q.tv0) * PPM
        col[m] = _bilinear(tex, u, v)
    lamp = np.zeros(len(P))
    for L in LAMPS:
        lamp += 1.0 / (1.0 + 0.35 * np.sum((P - L) ** 2, 1))
    light = 0.50 + 0.30 * lamp / 1.3
    return np.clip(col * light[:, None].astype(np.float32), 0, 1)


def _rays(cam: Camera, ss: int):
    w, h = cam.width * ss, cam.height * ss
    u, v = np.meshgrid((np.arange(w) + 0.5) / ss, (np.arange(h) + 0.5) / ss)
    dc = np.stack([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, np.ones_like(u)], -1).reshape(-1, 3)
    d = dc @ np.asarray(cam.R)
    return d, dc[:, 2] / np.linalg.norm(dc, axis=1)


def render_gt_full(cam: Camera, ss: int = 2):
    """(rgb float (H,W,3), 입사각 deg (H,W), z 깊이 (H,W), 화면 중심 quad 번호 (H,W))."""
    d, cosz = _rays(cam, ss)
    dn = d / np.linalg.norm(d, axis=1, keepdims=True)
    t, qi = raycast(cam.center, d)
    P = cam.center + d * t[:, None]
    rgb = shade(P, qi)
    ax = np.array([QUADS[k].axis for k in range(len(QUADS))])[np.maximum(qi, 0)]
    inc = np.degrees(np.arccos(np.clip(np.abs(dn[np.arange(len(dn)), ax]), 0, 1)))
    depth = t * np.linalg.norm(d, axis=1) * cosz
    H, W = cam.height, cam.width
    rgb = rgb.reshape(H, ss, W, ss, 3).mean((1, 3))
    inc = inc.reshape(H, ss, W, ss).mean((1, 3))
    depth = depth.reshape(H, ss, W, ss).mean((1, 3))
    qi = qi.reshape(H, ss, W, ss)[:, ss // 2, :, ss // 2]
    return rgb, inc, depth, qi


def render_gt(cam: Camera, ss: int = 2, rng=None) -> np.ndarray:
    rgb = render_gt_full(cam, ss)[0]
    if rng is not None:
        rgb = rgb + rng.normal(0, 1.0 / 255, rgb.shape)
    return (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)


# ---------------------------------------------------------------- 카메라
def look(yaw, pitch, roll=0.0):
    f = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    cr, sr = math.cos(roll), math.sin(roll)
    r2, u2 = cr * r + sr * u, -sr * r + cr * u
    return np.stack([r2, -u2, f], 1)


def _fx(width):
    return width / 2 / math.tan(math.radians(HFOV / 2))


def _cam(Rwc, C, width, height, name):
    fx = _fx(width)
    return Camera.from_world(Rwc, np.asarray(C, float), fx=fx, fy=fx, cx=width / 2, cy=height / 2,
                             width=width, height=height, name=name)


def free_space(C, margin=0.25):
    """복도·방·문 안의 자유 공간 (벽에서 margin)."""
    x, y, z = C
    if z < margin:
        return False
    in_c = (margin <= x <= CX1 - margin) and (-CW + margin <= y <= CW - margin) and z <= CH - margin
    in_r = (RX0 + margin <= x <= RX1 - margin) and (RY0 + margin <= y <= RY1 - margin) and z <= RH - margin
    in_d = (DX0 + margin <= x <= DX1 - margin) and (CW - 0.3 <= y <= RY0 + 0.3) and z <= DH - margin
    return in_c or in_r or in_d


def path_cameras(n=128, width=384, height=240, seed=1):
    """손에 든 스캔. 구간: 복도 서->동 (좌우·위아래 둘러보기) / 문으로 방 / 방 둘러보기 / 나와서 동->서."""
    rng = np.random.default_rng(seed)
    plan = []                                     # (C, yaw, pitch)
    n1, n2, n3, n4 = int(0.32 * n), int(0.10 * n), int(0.26 * n), 0
    n4 = n - n1 - n2 - n3 - int(0.06 * n)
    n_out = n - n1 - n2 - n3 - n4
    for i in range(n1):                            # 복도 앞으로 (yaw 0) + 둘러보기
        s = i / max(1, n1 - 1)
        C = [0.5 + 4.9 * s, -0.15 + 0.25 * math.sin(2 * math.pi * 1.3 * s), 1.45]
        yaw = math.radians(70) * math.sin(2 * math.pi * 3.0 * s)
        pitch = 0.0
        k = i % 8
        if k == 3:
            pitch = math.radians(45)
        elif k == 6:
            pitch = math.radians(-40)
        plan.append((C, yaw, pitch))
    for i in range(n2):                            # 문 쪽으로 돌아 방으로
        s = i / max(1, n2 - 1)
        C = [5.2 - 1.55 * min(1.0, 2 * s), 0.1, 1.45] if s < 0.5 else [3.65, 0.1 + (2.6 - 0.1) * (s - 0.5) * 2, 1.45]
        yaw = math.radians(90) * min(1, 2 * s) + math.radians(25) * math.sin(5 * s)
        plan.append((C, yaw, math.radians(-10 if i % 3 == 0 else 5)))
    for i in range(n3):                            # 방 둘러보기 (두 자리)
        s = i / max(1, n3 - 1)
        C = [4.0 + 0.6 * math.cos(2 * math.pi * s), 2.85 + 0.5 * math.sin(2 * math.pi * s), 1.45]
        yaw = math.pi / 2 + 2 * math.pi * 1.5 * s
        pitch = [0.0, math.radians(40), 0.0, math.radians(-35)][i % 4]
        plan.append((C, yaw, pitch))
    for i in range(n_out):                         # 방에서 나옴
        s = i / max(1, n_out - 1)
        C = [3.65, 2.5 - 2.6 * s, 1.45]
        plan.append((C, -math.pi / 2 + math.radians(30) * math.sin(4 * s), 0.0))
    for i in range(n4):                            # 복도 동->서 (yaw 180), 벽 가까이 걷기도
        s = i / max(1, n4 - 1)
        C = [5.5 - 5.0 * s, 0.2 - 0.6 * s, 1.40 + 0.1 * math.sin(3 * s)]
        yaw = math.pi + math.radians(75) * math.sin(2 * math.pi * 2.5 * s + 1.0)
        pitch = [0.0, 0.0, math.radians(35), 0.0, math.radians(-35)][i % 5]
        plan.append((C, yaw, pitch))
    cams = []
    for i, (C, yaw, pitch) in enumerate(plan):
        C = np.array(C, float) + rng.normal(0, [0.03, 0.03, 0.02])
        yaw += rng.normal(0, math.radians(4))
        pitch += rng.normal(0, math.radians(3))
        roll = rng.normal(0, math.radians(2))
        assert free_space(C, 0.2), (i, C)
        cams.append(_cam(look(yaw, pitch, roll), C, width, height, f"img_{i:03d}.png"))
    return cams


def grazing_cameras(width=384, height=240):
    """벽을 비스듬히 보는 시점 (줄무늬 측정): 벽에서 0.35~0.6 m 떨어져 벽을 따라 10~25도로 봄 + 바닥·천장 스침."""
    out = []
    k = 0
    for x, y, yaw_deg, z in [(0.7, -0.75, 15, 1.5), (5.3, -0.8, 165, 1.4), (0.8, 0.75, -15, 1.6),
                             (5.2, 0.7, -165, 1.3), (2.8, 0.0, 20, 1.5), (3.2, 0.0, -160, 1.5),
                             (3.0, 3.8, -10, 1.5), (5.6, 2.0, 100, 1.5), (3.0, 1.8, 80, 1.6),
                             (5.5, 3.9, 190, 1.4)]:
        for pitch in (0.0, -12.0):
            C = np.array([x, y, z])
            assert free_space(C, 0.25), C
            out.append(_cam(look(math.radians(yaw_deg), math.radians(pitch)), C, width, height, f"graze_{k:02d}"))
            k += 1
    return out


def novel_cameras(test_cams, offset=0.3, grazing_deg=60.0):
    """vr_cameras (옆 ±, 위·아래 ±, 비스듬히 ±60도) 중 자유 공간 안 + grazing_cameras."""
    from train import vr_cameras
    out = [c for c in vr_cameras(test_cams, offset, grazing_deg, up=[0, 0, 1]) if free_space(c.center, 0.25)]
    w, h = test_cams[0].width, test_cams[0].height
    return out + grazing_cameras(w, h)


# ---------------------------------------------------------------- 점
def colmap_like_points(cams, images, n_target=6000, noise=0.015, outlier_frac=0.02, grad_min=25.0, seed=0):
    """밝기 기울기 > grad_min 인 픽셀에서만 (SfM 처럼 무늬 없는 흰 벽 가운데엔 점이 없음). 사진마다 점 수는
    그 사진의 무늬 픽셀 수에 비례 (흰 벽만 보는 사진은 점이 거의 없음). 기울기 크기로 가중."""
    rng = np.random.default_rng(seed)
    mags = []
    for img in images:
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        mags.append(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)).ravel())
    ncand = np.array([(m > grad_min).sum() for m in mags], float)
    alloc = np.floor(n_target * ncand / max(1.0, ncand.sum())).astype(int)
    X, Cl = [], []
    for cam, img, mag, k in zip(cams, images, mags, alloc):
        cand = np.flatnonzero(mag > grad_min)
        k = min(k, len(cand))
        if k == 0:
            continue
        pr = mag[cand] / mag[cand].sum()
        pix = rng.choice(cand, k, replace=False, p=pr)
        u = pix % cam.width + 0.5
        v = pix // cam.width + 0.5
        dc = np.stack([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, np.ones_like(u)], -1)
        d = dc @ np.asarray(cam.R)
        t, qi = raycast(cam.center, d)
        P = cam.center + d * t[:, None]
        ok = np.isfinite(P).all(1)                 # 면 모서리 틈으로 빠진 광선 (아주 드묾)
        X.append(P[ok] + rng.normal(0, noise, P[ok].shape))
        Cl.append(img.reshape(-1, 3)[pix[ok]])
    X, Cl = np.concatenate(X), np.concatenate(Cl)
    k = int(outlier_frac * len(X))
    o = rng.choice(len(X), k, replace=False)
    X[o] = rng.uniform([-0.2, -1.4, -0.2], [6.3, 4.5, 2.9], (k, 3))
    return X, Cl.astype(np.uint8)


# ---------------------------------------------------------------- 도면·면
def plan_dict():
    return {"version": 1, "units": "m",
            "rooms": [{"id": "corr", "box": {"center": [3.0, 0.0], "size": [6.0, 2.4], "yaw_deg": 0},
                       "floor_z": 0, "ceiling_z": 2.7},
                      {"id": "room", "box": {"center": [4.25, 2.8], "size": [3.5, 3.0], "yaw_deg": 0},
                       "floor_z": 0, "ceiling_z": 2.6}],
            "doors": [{"id": "d1", "rooms": ["corr", "room"], "center": [3.65, 1.25], "width": 0.9, "height": 2.1}],
            "source": {"kind": "boxes"}}


def true_plan():
    from indoorgs import schemas
    d = plan_dict()
    d["rooms"][1]["box"] = {"center": [(RX0 + RX1) / 2, 2.8], "size": [RX1 - RX0, 3.0], "yaw_deg": 0}
    d["rooms"][1]["ceiling_z"] = RH
    return schemas.parse_plan(d)


def true_planes():
    """실제 면 (평가용): 도면 오차 없는 plan 으로 planes_from_plan."""
    from indoorgs.walls import planes as wp
    return wp.planes_from_plan(true_plan())


def make_scene(out_dir, n_views=128, width=384, height=240, n_points=6000, seed=0) -> dict:
    from indoorgs import plyio
    from indoorgs.colmap import model as cm
    out = Path(out_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    cams = path_cameras(n_views, width, height)
    rng = np.random.default_rng(seed)
    imgs = []
    for c in cams:
        p = img_dir / c.name
        if p.exists():
            im = cv2.imread(str(p))[..., ::-1].copy()
        else:
            im = render_gt(c, rng=rng)
            cv2.imwrite(str(p), im[..., ::-1])
        imgs.append(im)
    tr = [i for i in range(len(cams)) if i % 8 != 0]
    xyz, rgb = colmap_like_points([cams[i] for i in tr], [imgs[i] for i in tr], n_points, seed=seed)
    M = cm.Model(cameras={1: cm.Camera(1, 1, width, height, np.array([cams[0].fx, cams[0].fy, cams[0].cx, cams[0].cy]))},
                 images={}, points={})
    for i, c in enumerate(cams):
        M.images[i + 1] = cm.Image(i + 1, plyio.mat_to_quat(c.R), np.asarray(c.t), 1, c.name)
    # 트랙 길이 흉내 (refine 가중치): 3~8
    for j in range(len(xyz)):
        L = int(rng.integers(3, 9))
        M.points[j + 1] = cm.Point3D(j + 1, xyz[j], rgb[j], 0.5, np.zeros((L, 2), np.uint32))
    md = out / "sparse" / "0"
    cm.write_model(M, md)
    (out / "plan.json").write_text(json.dumps(plan_dict(), indent=1), encoding="utf-8")
    return {"model_dir": str(md), "image_dir": str(img_dir), "plan_json": str(out / "plan.json"),
            "n_points": int(len(xyz)), "n_images": len(cams)}


if __name__ == "__main__":
    import time
    t0 = time.time()
    o = Path(sys.argv[1])
    sc = make_scene(o)
    print(sc, f"{time.time() - t0:.1f}s")
    cams = path_cameras()
    strip = np.concatenate([render_gt(cams[i]) for i in (0, 10, 45, 60, 100)], 1)
    g = grazing_cameras()
    strip2 = np.concatenate([render_gt(g[i]) for i in (0, 4, 8, 12, 16)], 1)
    cv2.imwrite(str(o / "preview.png"), np.concatenate([strip, strip2], 0)[..., ::-1])
