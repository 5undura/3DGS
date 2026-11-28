"""합성 비상대피도 (시험·정확도 측정용, 실내만): 정답 방·복도 마스크가 있는 포스터 그림과 그 '벽에 붙은 포스터 사진'.

배치 (벽 중심선 좌표 m, 건물 왼아래 = 0,0):
  long  긴 복도 + 양쪽 방, 빗금 빈 칸 하나
  L     ㄱ자 복도 (가로 + 오른쪽 세로)
  T     T자 복도 (가로 + 아래로 내려가는 줄기)
  hall  학생회관 지하 흉내: 가운데 넓은 가로 복도, 위·아래 방 줄, 위로 올라가는 계단실, 아래 빗금 칸

그림 방식 (style):
  thick   굵고 어두운 벽, 복도는 색으로 칠함, 방은 옅은 색·흰색, 초록 대피 화살표
  thin    실제 학생회관 대피도처럼: 크림색 바탕, 올리브색 가는 이중선 벽(두 줄 사이 = 벽 두께), 복도 색 없음,
          빨간 화살표·점선, 소화기 기호, 문 호(arc), X 빗금 칸, 계단 사선
공통 장식: 방 번호 글자, 현위치(빨간 점 + 사람 기호 대신 'HERE'), 20m/25m 거리 글자, 위 파란 제목 띠, 빨간 테두리,
  안쪽 테두리선(네 점은 이 선의 모서리), 테두리 안 오른쪽 아래 범례 상자, 아래 행동요령 상자.
OpenCV Hershey 글꼴이라 한글은 영문으로 대신 씀 (이 PC 에 CJK 글꼴 없음).

좌표: 정답과 불러온 결과 모두 '안쪽 테두리선 사각형' 왼아래가 (0, 0), y 위, m.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

Rect = tuple[float, float, float, float]   # x0, y0, x1, y1 (벽 중심선 m)


@dataclass
class Item:
    id: str
    kind: str                 # room | corridor | void (빗금 칸·계단: 정답에서 방 아님)
    rects: list[Rect]
    label: str = ""


@dataclass
class Door:
    x: float
    y: float
    width: float
    horizontal_wall: bool     # True = 가로벽에 난 문 (틈이 x 방향)
    swing: int = 1            # 호 방향 (+1 / -1)


@dataclass
class Layout:
    name: str
    items: list[Item]
    doors: list[Door]
    here: tuple[float, float] = (0.0, 0.0)          # 현위치
    arrows: list[list[tuple[float, float]]] = field(default_factory=list)   # 대피 경로 꺾은선


def _row(prefix: str, xs, y0, y1, start: int) -> list[Item]:
    return [Item(f"{prefix}{start + i}", "room", [(xs[i], y0, xs[i + 1], y1)], f"B1{start + i:02d}") for i in range(len(xs) - 1)]


def layout(name: str) -> Layout:
    if name == "long":
        below = _row("r", [0, 6, 12, 18, 24, 30], 0, 5.5, 1)
        above = _row("r", [0, 7, 14, 22, 29, 36], 8, 14, 6)
        items = [Item("corr", "corridor", [(0, 5.5, 36, 8)])] + below + above + [Item("v1", "void", [(30, 0, 36, 5.5)])]
        doors = [Door(x + 1.6, 5.5, 1.0, True, -1) for x in (0, 6, 12, 18, 24)]
        doors += [Door(x + 1.6, 8, 1.0, True, 1) for x in (0, 7, 14, 22, 29)]
        return Layout(name, items, doors, (17, 6.75), [[(14, 6.75), (2, 6.75)], [(20, 6.75), (34, 6.75)]])
    if name == "L":
        below = _row("r", [0, 7, 14, 21, 28], 0, 6, 1)
        mid = _row("r", [0, 8, 16, 25.5], 8.5, 15, 5)
        top = _row("r", [0, 12.75, 25.5], 15, 22, 8)
        items = [Item("corr", "corridor", [(0, 6, 28, 8.5), (25.5, 8.5, 28, 22)])] + below + mid + top
        doors = [Door(x + 1.5, 6, 1.0, True, -1) for x in (0, 7, 14, 21)]
        doors += [Door(x + 1.5, 8.5, 1.0, True, 1) for x in (0, 8, 16)]
        doors += [Door(25.5, 18.5, 1.0, False, -1), Door(6, 15, 1.0, True, 1)]
        return Layout(name, items, doors, (12, 7.25), [[(10, 7.25), (1, 7.25)], [(26.75, 20), (26.75, 9)]])
    if name == "T":
        above = _row("r", [0, 7.5, 15, 22.5, 30], 14.5, 20, 1)
        items = [Item("corr", "corridor", [(0, 12, 30, 14.5), (13.75, 0, 16.25, 12)])] + above
        items += [Item("r5", "room", [(0, 6, 13.75, 12)], "B105"), Item("r6", "room", [(0, 0, 13.75, 6)], "B106"),
                  Item("r7", "room", [(16.25, 6, 30, 12)], "B107"), Item("r8", "room", [(16.25, 0, 23, 6)], "B108"),
                  Item("r9", "room", [(23, 0, 30, 6)], "B109")]
        doors = [Door(x + 1.5, 14.5, 1.0, True, 1) for x in (0, 7.5, 15, 22.5)]
        doors += [Door(13.75, 9, 1.0, False, -1), Door(13.75, 3, 1.0, False, -1), Door(16.25, 9, 1.0, False, 1),
                  Door(16.25, 3, 1.0, False, 1), Door(26.5, 6, 1.0, True, 1)]
        return Layout(name, items, doors, (15, 8), [[(15, 10), (15, 1)], [(12, 13.25), (1, 13.25)]])
    if name == "hall":
        # 넓은 가로 복도 (x 4~50, y 9~14) + 오른쪽 위 계단실 (x 24~27, y 14~25)
        corr = Item("corr", "corridor", [(4, 9, 50, 14)])
        below = _row("r", [4, 9.5, 15, 20.5, 26, 31.5, 37, 42.5], 3, 9, 7)        # B107~B113 (작은 번호 오른쪽은 생략)
        upper_l = _row("r", [12, 18, 21, 24], 14, 18.5, 14)                       # B114~B116
        upper_r = _row("r", [27, 33, 41, 49], 14, 21, 1)                         # B101~B103
        items = [corr] + below + upper_l + upper_r
        items += [Item("r17", "room", [(4, 14, 12, 18.5)], "B117"), Item("r4", "room", [(50, 9, 56, 16)], "B104"),
                  Item("r6", "room", [(42.5, 5, 50, 9)], "B106"),
                  Item("stair", "void", [(24, 14, 27, 25)]), Item("v1", "void", [(4, 0, 26, 3)]),
                  Item("v2", "void", [(26, 0, 42.5, 3)]), Item("v3", "void", [(27, 21, 49, 23)])]
        doors = [Door(x + 1.2, 9, 1.0, True, -1) for x in (4, 9.5, 15, 20.5, 26, 31.5, 37, 42.5)]
        doors += [Door(x + 1.2, 14, 1.0, True, 1) for x in (4, 12, 18, 21, 27, 33, 41)]
        doors += [Door(50, 12.5, 1.0, False, 1), Door(25.5, 14, 1.2, True, 1)]
        return Layout(name, items, doors, (22, 11.5), [[(20, 11.5), (6, 11.5), (6, 13)], [(24, 11.5), (25.5, 11.5), (25.5, 23)]])
    raise ValueError(f"모르는 배치: {name}")


LAYOUTS = ("long", "L", "T", "hall")


@dataclass
class Poster:
    image: np.ndarray                     # BGR 포스터 그림
    corners: np.ndarray                   # 안쪽 테두리선 네 모서리 (왼위, 오른위, 오른아래, 왼아래) px
    size_m: tuple[float, float]           # 그 사각형 실제 크기
    ppm: float
    truth: dict[str, tuple[str, np.ndarray]]   # id -> (kind, bool 마스크, 테두리 좌표계, truth_res m/칸, 원점 왼아래)
    truth_res: float
    doors_m: list[tuple[float, float]]    # 문 중심 (테두리 좌표 m)
    layout: Layout
    style: str


THICK = {"thick": 0.3, "thin": 0.35}


def render(lay: Layout | str, style: str = "thin", ppm: float = 40.0, margin_m: float = 5.0,
           seed: int = 0, truth_res: float = 0.05) -> Poster:
    lay = layout(lay) if isinstance(lay, str) else lay
    rng = np.random.default_rng(seed)
    T = THICK[style]
    allr = [r for it in lay.items for r in it.rects]
    bx0, by0 = min(r[0] for r in allr), min(r[1] for r in allr)
    bx1, by1 = max(r[2] for r in allr), max(r[3] for r in allr)
    # 안쪽 테두리 = 건물 바깥 + margin (오른쪽은 범례 자리 더)
    fx0, fy0 = bx0 - T / 2 - margin_m, by0 - T / 2 - margin_m - 6.0
    fx1, fy1 = bx1 + T / 2 + margin_m + 10.0, by1 + T / 2 + margin_m
    FW, FH = fx1 - fx0, fy1 - fy0
    pad_l, pad_t, pad_r, pad_b = 60, 140, 60, 60       # 테두리 바깥 (제목 띠, 빨간 테두리)
    W, H = int(round(FW * ppm)) + pad_l + pad_r, int(round(FH * ppm)) + pad_t + pad_b
    paper = (200, 236, 244) if style == "thin" else (245, 245, 245)
    img = np.full((H, W, 3), paper, np.uint8)

    def P(x, y):    # 건물 m -> 그림 px (실수, 픽셀 경계 기준)
        return pad_l + (x - fx0) * ppm, pad_t + (fy1 - y) * ppm

    def Pi(x, y):
        p = P(x, y)
        return int(round(p[0])), int(round(p[1]))

    # ---- 마스크 (그림 해상도) ----
    def rect_mask(rects, grow=0.0):
        m = np.zeros((H, W), np.uint8)
        for x0, y0, x1, y1 in rects:
            a, b = P(x0 - grow, y1 + grow), P(x1 + grow, y0 - grow)
            m[int(round(a[1])):int(round(b[1])), int(round(a[0])):int(round(b[0]))] = 1
        return m

    k = max(1, int(round(T / 2 * ppm)))
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * k + 1, 2 * k + 1))
    inter = {}
    for it in lay.items:
        inter[it.id] = cv2.erode(rect_mask(it.rects), ker)
    building = rect_mask(allr, grow=T / 2)
    door_m = np.zeros((H, W), np.uint8)
    for d in lay.doors:
        if d.horizontal_wall:
            door_m |= rect_mask([(d.x - d.width / 2, d.y - T, d.x + d.width / 2, d.y + T)])
        else:
            door_m |= rect_mask([(d.x - T, d.y - d.width / 2, d.x + T, d.y + d.width / 2)])
    free = np.zeros((H, W), np.uint8)
    for m in inter.values():
        free |= m
    free |= door_m & building
    band = building & (1 - free)
    wall_col = (60, 85, 70) if style == "thin" else (35, 35, 35)

    # ---- 칠하기 ----
    corr_col = tuple(int(v) for v in rng.choice([(150, 225, 250), (190, 235, 190), (235, 215, 170)]))
    for it in lay.items:
        m = inter[it.id].astype(bool)
        if style == "thick":
            if it.kind == "corridor":
                img[m] = corr_col
            elif it.kind == "room":
                img[m] = (255, 255, 255) if rng.random() < 0.5 else tuple(int(v) for v in rng.integers(215, 250, 3))
    if style == "thick":
        img[(free & building & (1 - sum(inter.values()).clip(0, 1))).astype(bool)] = corr_col   # 문 틈
        img[band.astype(bool)] = wall_col
    else:
        cs, _ = cv2.findContours(band, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, wall_col, max(1, int(round(0.09 * ppm))), cv2.LINE_AA)

    line_t = max(1, int(round(0.06 * ppm)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    red = (40, 40, 210)
    # 빗금 칸 (X) / 계단 사선
    for it in lay.items:
        if it.kind != "void":
            continue
        for x0, y0, x1, y1 in it.rects:
            a, b = Pi(x0 + T / 2, y1 - T / 2), Pi(x1 - T / 2, y0 + T / 2)
            if it.id == "stair":
                for yy in np.arange(y0 + 1.0, y1 - 1.0, 0.5):
                    cv2.line(img, Pi(x0 + 0.4, yy), Pi(x1 - 1.2, yy), wall_col, line_t, cv2.LINE_AA)
                cv2.line(img, Pi(x0 + 0.3, y0 + 0.5), Pi(x1 - 0.3, y1 - 0.5), wall_col, line_t, cv2.LINE_AA)
                cv2.putText(img, "UP", Pi(x0 + 0.3, y1 - 1.5), font, ppm / 40 * 0.7, (40, 40, 40), 1, cv2.LINE_AA)
            else:
                cv2.line(img, a, b, wall_col, line_t, cv2.LINE_AA)
                cv2.line(img, (a[0], b[1]), (b[0], a[1]), wall_col, line_t, cv2.LINE_AA)
    # 문 호
    for d in lay.doors:
        r = int(d.width * 0.8 * ppm)
        if d.horizontal_wall:
            c = Pi(d.x - d.width / 2, d.y + d.swing * T / 2)
            e = Pi(d.x - d.width / 2, d.y + d.swing * (T / 2 + d.width * 0.8))
            st, en = (-90, 0) if d.swing > 0 else (0, 90)
        else:
            c = Pi(d.x + d.swing * T / 2, d.y - d.width / 2)
            e = Pi(d.x + d.swing * (T / 2 + d.width * 0.8), d.y - d.width / 2)
            st, en = (-180, -90) if d.swing < 0 else (-90, 0)
        cv2.ellipse(img, c, (r, r), 0, st, en, wall_col, line_t, cv2.LINE_AA)
        cv2.line(img, c, e, wall_col, line_t, cv2.LINE_AA)
    # 방 번호, 소화기
    for it in lay.items:
        if it.kind != "room":
            continue
        x0, y0, x1, y1 = it.rects[0]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        s = ppm / 40 * 1.1
        (tw, th), _ = cv2.getTextSize(it.label, font, s, 2)
        p = Pi(cx, cy)
        cv2.putText(img, it.label, (p[0] - tw // 2, p[1] + th // 2), font, s, (45, 45, 45), max(1, int(ppm / 25)), cv2.LINE_AA)
        if rng.random() < 0.6:   # 빨간 소화기 (작은 기둥 + T)
            q = Pi(x0 + 0.6, y1 - 0.7)
            cv2.rectangle(img, (q[0] - 3, q[1] - 2), (q[0] + 3, q[1] + int(0.5 * ppm)), red, -1)
            cv2.line(img, (q[0] - 6, q[1] - 4), (q[0] + 6, q[1] - 4), red, 2)
    # 대피 화살표 (빨강 / thick 이면 초록) + 거리 글자
    arr = red if style == "thin" else (60, 165, 60)
    aw = max(2, int(round(0.12 * ppm)))
    for path in lay.arrows:
        pts = [Pi(*p) for p in path]
        for a, b in zip(pts[:-2], pts[1:-1]):
            cv2.line(img, a, b, arr, aw, cv2.LINE_AA)
        cv2.arrowedLine(img, pts[-2], pts[-1], arr, aw, cv2.LINE_AA, tipLength=min(0.3, 1.2 * ppm / max(1, math.dist(pts[-2], pts[-1]))))
        mid = ((pts[0][0] + pts[1][0]) // 2, (pts[0][1] + pts[1][1]) // 2 - 6)
        cv2.putText(img, f"{int(round(math.dist(path[0], path[-1])))}m", mid, font, ppm / 40 * 0.6, arr, 2, cv2.LINE_AA)
    # 현위치
    h = Pi(*lay.here)
    cv2.circle(img, h, int(0.35 * ppm), red, -1, cv2.LINE_AA)
    cv2.putText(img, "HERE", (h[0] - int(0.6 * ppm), h[1] + int(0.9 * ppm)), font, ppm / 40 * 0.6, red, 2, cv2.LINE_AA)
    # 안쪽 테두리선 (둥근 모서리 흉내 없이 사각), 범례, 행동요령, 제목 띠, 빨간 테두리
    c0, c1 = Pi(fx0, fy1), Pi(fx1, fy0)
    cv2.rectangle(img, c0, (c1[0] - 1, c1[1] - 1), (50, 50, 50), max(1, int(0.05 * ppm)))
    lx0, ly0 = Pi(fx1 - 9.0, by0 + 6.0)
    lx1, ly1 = Pi(fx1 - 1.0, by0 - 0.5)
    cv2.rectangle(img, (lx0, ly0), (lx1, ly1), (40, 40, 40), 2)
    cv2.putText(img, "LEGEND", (lx0 + 10, ly0 + int(0.9 * ppm)), font, ppm / 40 * 0.7, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.line(img, (lx0, ly0 + int(1.3 * ppm)), (lx1, ly0 + int(1.3 * ppm)), (40, 40, 40), 2)
    for i, col in enumerate([red, red, arr, corr_col if style == "thick" else (40, 40, 210)]):
        q = (lx0 + int((1.0 + 1.9 * i) * ppm), ly0 + int(2.5 * ppm))
        cv2.rectangle(img, q, (q[0] + int(0.9 * ppm), q[1] + int(0.9 * ppm)), col, -1)
        cv2.putText(img, "abc", (q[0], q[1] + int(1.8 * ppm)), font, ppm / 40 * 0.45, col, 1, cv2.LINE_AA)
    ix0, iy0 = Pi(bx0 + 4, by0 - 1.5)
    ix1, iy1 = Pi(bx0 + 30, fy0 + 0.6)
    cv2.rectangle(img, (ix0, iy0), (ix1, iy1), (60, 60, 60), 2)
    for i in range(3):
        cv2.putText(img, f"{i + 1}. keep calm and follow the exit route", (ix0 + 12, iy0 + int((1.2 + 1.1 * i) * ppm)),
                    font, ppm / 40 * 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.rectangle(img, (pad_l - 20, 20), (W - pad_l + 20, pad_t - 25), (150, 60, 30), -1)
    cv2.putText(img, "EMERGENCY EVACUATION MAP B1", (pad_l + 40, pad_t - 50), font, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.rectangle(img, (6, 6), (W - 7, H - 7), (40, 40, 200), 12)

    # ---- 정답 (테두리 좌표, 왼아래 원점, truth_res m/칸) ----
    tw_, th_ = int(round(FW / truth_res)), int(round(FH / truth_res))
    truth = {}
    for it in lay.items:
        m = inter[it.id]
        a, b = c0, c1
        crop = m[a[1]:b[1], a[0]:b[0]]
        truth[it.id] = (it.kind, cv2.resize(crop, (tw_, th_), interpolation=cv2.INTER_NEAREST).astype(bool))
    doors_m = [(d.x - fx0, d.y - fy0) for d in lay.doors]
    corners = np.float32([[c0[0], c0[1]], [c1[0], c0[1]], [c1[0], c1[1]], [c0[0], c1[1]]]) - 0.5
    return Poster(img, corners, (FW, FH), ppm, truth, truth_res, doors_m, lay, style)


def photograph(poster: Poster, seed: int = 0, out_w: int = 2000, poster_frac: float = 0.45, tilt: float = 0.08,
               blur: float = 0.8, light: float = 0.35, noise: float = 3.0, jpeg: int = 75) -> tuple[np.ndarray, np.ndarray]:
    """벽에 붙은 포스터를 아래에서 비스듬히 찍은 사진 흉내. 돌려줌 (사진, 네 점 사진 좌표)."""
    rng = np.random.default_rng(seed)
    img = poster.image
    h, w = img.shape[:2]
    out_h = int(out_w * 9 / 16)
    pw = out_w * poster_frac
    ph = pw * h / w
    x0, y0 = out_w * 0.3, out_h * 0.38
    dst = np.float32([[x0, y0], [x0 + pw, y0], [x0 + pw, y0 + ph], [x0, y0 + ph]])
    # 아래에서 찍음: 아래 변이 조금 넓고, 왼쪽이 기울어짐 + 무작위
    dst[2, 0] += pw * tilt * 0.5
    dst[3, 0] -= pw * tilt * 0.5 + pw * 0.03
    dst += rng.uniform(-1, 1, (4, 2)).astype(np.float32) * np.float32([pw, ph]) * 0.015
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    Hm = cv2.getPerspectiveTransform(src, dst)
    wall = np.full((out_h, out_w, 3), (175, 222, 235), np.uint8)
    cv2.rectangle(wall, (0, int(out_h * 0.5)), (int(out_w * 0.1), int(out_h * 0.78)), (40, 40, 200), -1)   # 빨간 종이
    cv2.rectangle(wall, (int(out_w * 0.77), int(out_h * 0.4)), (int(out_w * 0.9), int(out_h * 0.73)), (235, 235, 235), -1)
    for i in range(12):                                                                 # 옆 안내문 글자
        cv2.line(wall, (int(out_w * 0.78), int(out_h * (0.43 + 0.022 * i))), (int(out_w * 0.88), int(out_h * (0.43 + 0.022 * i))),
                 (60, 60, 160) if i % 3 == 0 else (40, 40, 40), 2)
    cv2.rectangle(wall, (0, int(out_h * 0.05)), (out_w, int(out_h * 0.14)), (110, 150, 160), -1)                # 그림자 띠
    warped = cv2.warpPerspective(img, Hm, (out_w, out_h), flags=cv2.INTER_AREA)
    m = cv2.warpPerspective(np.full((h, w), 255, np.uint8), Hm, (out_w, out_h), flags=cv2.INTER_NEAREST) > 0
    ph_ = wall.copy()
    ph_[m] = warped[m]
    if blur:
        ph_ = cv2.GaussianBlur(ph_, (0, 0), blur)
    if light:
        yy, xx = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
        g = 1 - light * (xx / out_w) * 0.7 - light * ((yy / out_h - 0.35) ** 2) * 1.2
        ph_ = ph_.astype(np.float32) * g[..., None] * np.float32([0.96, 1.0, 1.02])   # 따뜻한 조명
    ph_ = np.clip(ph_ + rng.normal(0, noise, ph_.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", ph_, [cv2.IMWRITE_JPEG_QUALITY, jpeg])
    ph_ = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    c = cv2.perspectiveTransform(poster.corners[None] + 0.5, Hm)[0] - 0.5
    return ph_, c.astype(np.float32)
