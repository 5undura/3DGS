"""도면 사진 -> 복도·방 영역 -> 박스 (원근 보정 뒤 그림에서). image_import.import_image(method="segment") 가 씀.

순서 (모두 보정 이미지 px, ppm = px/m):
1. 조명 고르기: 채널 최댓값(V)을 크게 팽창·흐림 = 종이 밝기 -> 나눔 (사진 밝기 기울기·그림자 줄임)
2. 선(잉크) 마스크: 종이보다 어두운 곳. 채도 높고 밝은 색(빨강·초록 화살표, 소화기)은 벽이 아님.
   작은 덩어리(글자·기호, 최대 변 < small_ink_m)는 지움. 굵은 벽이든 가는 이중선 벽이든 같음.
3. 문 틈 닫기: 가로·세로 선 커널(door_m) 닫힘 -> 빈 공간 덩어리 (4-연결)
4. 건물 밖 버림: 잉크가 가장 많은 덩어리(벽 전체)의 바깥 윤곽 안만. 테두리·범례·행동요령 상자는 건물 밖
5. 덩어리 거름: 넓이 < min_room_m2, 폭(2*최대 거리변환) < min_width_m(이중선 사이 틈), 빗금(X)·계단 사선
   (윤곽 둘레 중 비스듬한 변 비율 > hatch_frac) -> 버림
6. 복도/방: 가늘고 긴 정도 thin = 넓이 / (4 * 최대거리²) 가 corridor_thin 이상이면 복도 후보.
   복도 후보의 색이 방들 색과 다르면(색칠한 복도) 그 색(Lab 거리 colour_tol 안) 비율로 복도를 고르고,
   벽 없이 이어진 복도색/다른색 부분은 나눔. 색이 같으면(크림색 바탕) 모양만으로.
7. 박스: 영역(작은 돌기 열림, 글자 구멍 메움)에서 가장 큰 축 정렬 직사각형을 차례로 뽑음 (ㄱ·T 자 복도 -> 여러 박스)
8. 문: 다른 영역 박스 둘이 벽(1m 안)을 사이에 두고 마주 보는 띠를 벽 방향으로 훑어, 잉크 없는 줄이
   0.5~2.5m 이어진 곳 (호·글자는 2단계에서 지워져 문을 막지 않음)
9. 믿음(conf, 0~1): 벽으로 둘러싸인 정도와 박스가 영역을 채우는 정도 (+ 색칠 복도면 색 일치)
마법봉 magic_wand: 점 하나 -> 색 허용 오차 안에서 벽(문 틈 닫은 것) 넘지 않게 채움 -> 같은 7·9 단계.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class SegOptions:
    door_m: float = 1.2           # 이보다 좁은 틈은 문으로 보고 닫음
    min_room_m2: float = 2.0
    min_width_m: float = 0.8      # 영역 폭이 이보다 작으면 벽 속 틈(이중선 사이)
    small_ink_m: float = 0.8      # 이보다 작은 잉크 덩어리 = 글자·기호
    ink_rel: float = 0.88         # 종이 밝기 대비 이보다 어두우면 잉크 (위 한계. 가는 선은 흐려서 0.8 이면 끊김)
    corridor_thin: float = 2.6
    colour_tol: float = 14.0      # Lab 거리
    hatch_frac: float = 0.25
    open_m: float = 0.3           # 영역 돌기 지우기
    min_box_m: float = 0.6        # 박스 최소 변
    cover: float = 0.96           # 영역을 이만큼 덮으면 박스 그만


@dataclass
class Item:
    kind: str                       # corridor | room
    mask: np.ndarray                # bool, 보정 이미지 크기
    boxes: list[tuple[int, int, int, int]]   # px (c0, r0, c1, r1) 끝 제외
    conf: float
    thin: float = 0.0
    info: dict = field(default_factory=dict)


@dataclass
class Prepared:
    norm: np.ndarray                # float32 BGR, 종이 = 1
    lab: np.ndarray                 # float32 Lab (0..100, -128..127)
    ink: np.ndarray                 # uint8 0/1, 작은 덩어리 뺀 잉크
    closed: np.ndarray              # uint8 0/1, 문 틈 닫은 잉크
    building: np.ndarray            # bool
    ppm: float


def normalise(img: np.ndarray, ppm: float) -> np.ndarray:
    f = img.astype(np.float32)
    v = f.max(axis=2)
    k = max(5, int(1.5 * ppm) | 1)
    bg = cv2.dilate(v, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 2)
    return np.clip(f / np.maximum(bg, 20)[..., None], 0, 1.2)


def ink_mask(norm: np.ndarray, rel: float = 0.88, long_px: int = 0) -> np.ndarray:
    """종이보다 어두운 곳 = 잉크. 단 빨강·초록 쪽으로 색이 뚜렷한 곳(대피 화살표·소화기·현위치)은 벽이 아님:
    채널 비율(rgb/합)이 종이 비율보다 R 또는 G 가 0.1 넘게 크면 뺌. 크림색 종이·올리브 벽은 비율이 비슷해서 남음."""
    v = norm.max(axis=2)
    c = norm / np.maximum(norm.sum(axis=2, keepdims=True), 1e-3)
    bright = v > np.percentile(v, 60)
    p0 = np.median(c[bright], axis=0) if bright.any() else np.full(3, 1 / 3, np.float32)
    colourful = ((c[..., 2] - p0[2] > 0.1) | (c[..., 1] - p0[1] > 0.1)) & (v > 0.3)
    # 긴 색 선(화살표, 최대 변 >= long_px)의 어두운 가장자리(흐림·JPEG)도 벽이 아님: 2px 넓혀서 뺌.
    # 작은 기호(소화기)는 넓히지 않음: 문 옆에 붙어 있어 넓히면 문 틈이 커져 방이 복도와 붙음
    if long_px:
        n, lab, st, _ = cv2.connectedComponentsWithStats(colourful.astype(np.uint8), connectivity=8)
        big = np.zeros(n, bool)
        big[1:] = np.maximum(st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]) >= long_px
        grow = cv2.dilate(big[lab].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & (v > 0.3)
        colourful = colourful | grow
    return ((v < rel) & ~colourful).astype(np.uint8)


def drop_small(mask: np.ndarray, min_px: float, ppm: float | None = None) -> np.ndarray:
    """작은 덩어리(최대 변 < min_px) 지움. ppm 을 주면 글자 줄처럼 꽉 찬 덩어리도 지움:
    최대 변 < 5m, 최소 변 > 0.5m, bbox 채움 > 0.22 (벽 선은 가늘거나, 그물이면 채움이 낮음)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    w, h, a = st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT], st[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(n, bool)
    keep[1:] = np.maximum(w, h) >= min_px
    if ppm:
        text = (np.maximum(w, h) < 5 * ppm) & (np.minimum(w, h) > 0.5 * ppm) & (a / np.maximum(w * h, 1) > 0.22)
        keep[1:] &= ~text
    return keep[lab].astype(np.uint8)


def long_lines(ink: np.ndarray, run_px: int) -> np.ndarray:
    """가로·세로로 run_px 이상 이어진 잉크 (벽 선)."""
    r = max(3, run_px)
    h = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (r, 1)))
    v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, r)))
    return h | v


def strip_text(ink: np.ndarray, ppm: float, run_m: float = 1.2) -> np.ndarray:
    """벽에 붙은 글자·호 지우기: 긴 가로·세로 선(run_m, 끊긴 점선은 3px 닫아 이음) 근처 잉크는 남기고,
    나머지 조각은 가는 선 모양(짧은 변 <= 0.3m, 긴 변 >= 0.3m: 문 옆 짧은 벽)과 긴 사선(1.5m 이상, 채움 < 0.15: 빗금 X)만
    남김. 글자는 짧은 변이 굵고 꽉 차서, 문 호는 짧아서 빠짐."""
    inks = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    L = cv2.dilate(long_lines(inks, int(run_m * ppm)), np.ones((3, 3), np.uint8))
    resid = (ink & (1 - L)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(resid, connectivity=8)
    w, h, a = st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT], st[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(n, bool)
    line = (np.minimum(w, h) <= 0.3 * ppm) & (np.maximum(w, h) >= 0.3 * ppm)
    diag = (np.maximum(w, h) >= 1.5 * ppm) & (a / np.maximum(w * h, 1) < 0.15)   # 긴 사선 (빗금 X·계단): 빗금 칸 알아보기에 필요
    keep[1:] = line | diag
    return (ink & L) | keep[lab].astype(np.uint8)


def close_gaps(ink: np.ndarray, gap_px: int) -> np.ndarray:
    g = max(3, gap_px)
    h = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (g, 1)))
    v = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, g)))
    return h | v


def building_mask(ink: np.ndarray) -> np.ndarray:
    """잉크가 가장 많은 덩어리(벽 전체)의 바깥 윤곽 안. 그림 거의 전체를 두르는 덩어리(포스터 테두리선)는 후보에서 뺌."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    H, W = ink.shape
    x, y, w, h, a = (st[1:, k] for k in range(5))
    inner = ~((w > 0.9 * W) & (h > 0.9 * H))          # 포스터 테두리선 (그림 거의 전체를 두름)
    if n <= 1 or not inner.any():
        return np.ones(ink.shape, bool)
    i = 1 + int(np.argmax(np.where(inner, a, -1)))
    comp = (lab == i).astype(np.uint8)
    cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(ink.shape, np.uint8)
    cv2.drawContours(out, cs, -1, 1, -1)
    return out.astype(bool)


def prepare(img: np.ndarray, ppm: float, opts: SegOptions | None = None) -> Prepared:
    o = opts or SegOptions()
    norm = normalise(img, ppm)
    lab = cv2.cvtColor(np.clip(norm, 0, 1).astype(np.float32), cv2.COLOR_BGR2Lab)
    # 문턱: Otsu + 0.11 (색칠한 복도·문 틈이 종이보다 조금 어두운 굵은 벽 그림) 과 ink_rel(흐린 가는 선) 중 작은 것
    v8 = (np.clip(norm.max(axis=2), 0, 1) * 255).astype(np.uint8)
    otsu = cv2.threshold(v8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0] / 255
    rel = min(o.ink_rel, otsu + 0.11)
    ink = strip_text(drop_small(ink_mask(norm, rel, int(1.5 * ppm)), o.small_ink_m * ppm, ppm), ppm)
    closed = close_gaps(ink, int(round(o.door_m * ppm)))
    k = max(3, int(0.3 * ppm) | 1)      # 건물 고르기는 점선 끊김만 이은 잉크로 (문 틈 닫기는 옆 상자까지 이어 붙일 수 있음)
    bld = building_mask(cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)))
    return Prepared(norm, lab, ink, closed, bld, ppm)


# ---------------- 박스 ----------------
def _max_rect(grid: np.ndarray) -> tuple[int, int, int, int, int]:
    """bool 격자에서 가장 큰 직사각형 (넓이, r0, c0, r1, c1) 끝 제외."""
    R, C = grid.shape
    h = np.zeros(C + 1, np.int32)
    best = (0, 0, 0, 0, 0)
    for r in range(R):
        h[:C] = np.where(grid[r], h[:C] + 1, 0)
        st: list[int] = []
        for c in range(C + 1):
            while st and h[st[-1]] >= h[c]:
                top = st.pop()
                hh = h[top]
                if hh == 0:
                    continue
                left = st[-1] + 1 if st else 0
                a = hh * (c - left)
                if a > best[0]:
                    best = (int(a), r - hh + 1, left, r + 1, c)
            st.append(c)
    return best


def _grow(rect, region: np.ndarray, taken: np.ndarray, frac: float = 0.6):
    """직사각형 변을 바깥 줄이 region 안에 frac 이상이고 이미 뽑은 박스(taken)와 안 겹치면 한 줄씩 넓힘."""
    c0, r0, c1, r1 = rect

    def ok(rs, cs):
        return region[rs, cs].mean() >= frac and not taken[rs, cs].any()

    H, W = region.shape
    changed = True
    while changed:
        changed = False
        if r0 > 0 and ok(r0 - 1, slice(c0, c1)):
            r0 -= 1; changed = True
        if r1 < H and ok(r1, slice(c0, c1)):
            r1 += 1; changed = True
        if c0 > 0 and ok(slice(r0, r1), c0 - 1):
            c0 -= 1; changed = True
        if c1 < W and ok(slice(r0, r1), c1):
            c1 += 1; changed = True
    return c0, r0, c1, r1


def boxes_from_mask(mask: np.ndarray, ppm: float, opts: SegOptions | None = None) -> list[tuple[int, int, int, int]]:
    """bool 영역 -> 겹치지 않는 축 정렬 박스들 (c0, r0, c1, r1 px, 끝 제외). 큰 것부터."""
    o = opts or SegOptions()
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    reg = mask[y0:y1, x0:x1].copy()
    total = reg.sum()
    cell = max(1, int(math.ceil(max(reg.shape) / 160)))
    min_side = max(2, int(o.min_box_m * ppm))
    out = []
    left = reg.copy()
    taken = np.zeros(reg.shape, bool)
    for _ in range(12):
        Hc, Wc = left.shape[0] // cell, left.shape[1] // cell
        if Hc == 0 or Wc == 0:
            break
        g = (left[:Hc * cell, :Wc * cell].reshape(Hc, cell, Wc, cell).mean(axis=(1, 3)) > 0.5) & \
            ~taken[:Hc * cell, :Wc * cell].reshape(Hc, cell, Wc, cell).any(axis=(1, 3))
        a, r0, c0, r1, c1 = _max_rect(g)
        if a == 0:
            break
        rect = (c0 * cell, r0 * cell, c1 * cell, r1 * cell)
        rect = _grow(rect, reg, taken)
        c0, r0, c1, r1 = rect
        if min(c1 - c0, r1 - r0) < min_side or (c1 - c0) * (r1 - r0) < max(ppm * ppm * 1.0, 0.03 * total):
            break
        out.append((c0 + x0, r0 + y0, c1 + x0, r1 + y0))
        left[r0:r1, c0:c1] = False
        taken[r0:r1, c0:c1] = True
        if 1 - left.sum() / total >= o.cover:
            break
    return out


def _fill_holes(m: np.ndarray) -> np.ndarray:
    u = m.astype(np.uint8)
    cs, _ = cv2.findContours(u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(u)
    cv2.drawContours(out, cs, -1, 1, -1)
    return out.astype(bool)


def clean_region(m: np.ndarray, ppm: float, opts: SegOptions) -> np.ndarray:
    m = _fill_holes(m)
    k = max(3, int(opts.open_m * ppm) | 1)
    op = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))).astype(bool)
    if op.sum() == 0:
        return m
    # 열림으로 떨어져 나간 작은 조각은 버리고 가장 큰 덩어리만
    n, lab, st, _ = cv2.connectedComponentsWithStats(op.astype(np.uint8), connectivity=4)
    if n > 2:
        op = lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return op


def thinness(m: np.ndarray) -> tuple[float, float]:
    """(thin = 넓이/(4 r²), 폭 2r px). r = 최대 거리변환."""
    d = cv2.distanceTransform(np.pad(m.astype(np.uint8), 1), cv2.DIST_L2, 5)
    r = float(d.max())
    return float(m.sum()) / max(4 * r * r, 1.0), 2 * r


def diagonal_fraction(m: np.ndarray, ppm: float) -> float:
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return 0.0
    c = max(cs, key=cv2.contourArea)
    p = cv2.approxPolyDP(c, max(1.0, 0.25 * ppm), True).reshape(-1, 2).astype(float)
    tot = diag = 0.0
    for a, b in zip(p, np.roll(p, -1, axis=0)):
        L = math.dist(a, b)
        ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 90
        tot += L
        if 6 < ang < 84:          # 납작한 X 빗금(22x3m 칸이면 7.8도)도 잡게
            diag += L
    return diag / max(tot, 1e-9)


def enclosure(m: np.ndarray, wall: np.ndarray) -> float:
    """영역 둘레 바로 바깥(2px) 중 벽(닫힌 잉크) 비율."""
    u = m.astype(np.uint8)
    ring = cv2.dilate(u, np.ones((5, 5), np.uint8)) & (1 - u)
    n = ring.sum()
    return float((ring & wall).sum() / n) if n else 0.0


def box_fill(m: np.ndarray, boxes) -> float:
    cov = np.zeros(m.shape, bool)
    for c0, r0, c1, r1 in boxes:
        cov[r0:r1, c0:c1] = True
    inter = (cov & m).sum()
    union = (cov | m).sum()
    return float(inter / union) if union else 0.0


def _median_lab(lab: np.ndarray, m: np.ndarray, norm: np.ndarray, rel: float) -> np.ndarray:
    sel = m & (norm.max(axis=2) >= rel)
    if sel.sum() < 10:
        sel = m
    return np.median(lab[sel], axis=0)


def make_item(m: np.ndarray, prep: Prepared, opts: SegOptions, kind: str | None = None,
              corr_lab: np.ndarray | None = None) -> Item | None:
    ppm = prep.ppm
    m = clean_region(m, ppm, opts)
    if m.sum() < opts.min_room_m2 * ppm * ppm:
        return None
    th, _ = thinness(m)
    boxes = boxes_from_mask(m, ppm, opts)
    if not boxes:
        return None
    enc = enclosure(m, prep.closed)
    fill = box_fill(m, boxes)
    if kind is None:
        kind = "corridor" if th >= opts.corridor_thin else "room"
    conf = 0.5 * enc + 0.5 * fill
    info = {"enclosure": round(enc, 3), "fill": round(fill, 3), "thin": round(th, 2)}
    if corr_lab is not None and kind == "corridor":
        d = np.linalg.norm(prep.lab[m] - corr_lab, axis=1)
        cf = float((d < opts.colour_tol).mean())
        info["colour"] = round(cf, 3)
        conf = (conf * 2 + cf) / 3
    return Item(kind, m, boxes, round(float(np.clip(conf, 0, 1)), 3), th, info)


# ---------------- 전체 분할 ----------------
def segment(img: np.ndarray, ppm: float, opts: SegOptions | None = None, prep: Prepared | None = None) -> dict:
    """돌려줌: {items, doors[(cx_px, cy_px, width_px, a_idx, b_idx)], dropped{reason: n}, prep, colour_mode}"""
    o = opts or SegOptions()
    prep = prep or prepare(img, ppm, o)
    H, W = prep.ink.shape
    free = (1 - prep.closed).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    dropped = {"border": 0, "outside": 0, "small": 0, "narrow": 0, "hatch": 0}
    min_a = o.min_room_m2 * ppm * ppm
    cand = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 0.5 * ppm * ppm:
            continue
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            dropped["border"] += 1
            continue
        m = lab == i
        if prep.building[m].mean() < 0.5:
            dropped["outside"] += 1
            continue
        if a >= min_a and diagonal_fraction(m, ppm) > o.hatch_frac:
            dropped["hatch"] += 1
            continue
        cand.append((i, m))
    comps = []
    for _, m in cand:
        if m.sum() < min_a:
            dropped["small"] += 1
            continue
        th, width = thinness(m)
        if width < o.min_width_m * ppm:
            dropped["narrow"] += 1
            continue
        comps.append((m, th, _median_lab(prep.lab, m, prep.norm, o.ink_rel)))
    # 색칠한 복도인지
    thin_c = [c for c in comps if c[1] >= o.corridor_thin]
    rest = [c for c in comps if c[1] < o.corridor_thin]
    corr_lab = None
    if thin_c:
        wts = np.array([c[0].sum() for c in thin_c], float)
        ref = np.average(np.array([c[2] for c in thin_c]), axis=0, weights=wts)
        room_ref = np.median(np.array([c[2] for c in rest]), axis=0) if rest else None
        chroma = math.hypot(ref[1], ref[2])
        if chroma > 8 and (room_ref is None or np.linalg.norm(ref - room_ref) > o.colour_tol):
            corr_lab = ref
    pieces: list[tuple[np.ndarray, str | None]] = []
    if corr_lab is not None:
        cm = (np.linalg.norm(prep.lab - corr_lab, axis=2) < o.colour_tol).astype(np.uint8)
        k = max(3, int(0.6 * ppm) | 1)
        cm = cv2.morphologyEx(cm, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))).astype(bool)
        for m, th, _ in comps:
            frac = cm[m].mean()
            cpart = m & cm
            rpart = m & ~cm
            if 0.15 < frac < 0.85 and cpart.sum() > min_a and rpart.sum() > min_a:   # 벽 없이 붙은 복도 + 방
                pieces += [(cpart, "corridor"), (rpart, "room")]
            else:
                pieces.append((m, "corridor" if frac >= 0.5 else "room"))
    else:
        pieces = [(m, None) for m, _, _ in comps]
    items: list[Item] = []
    for m, kind in pieces:
        it = make_item(m, prep, o, kind, corr_lab)
        if it is not None:
            items.append(it)
    doors = find_doors(prep, items, o)
    return {"items": items, "doors": doors, "dropped": dropped, "prep": prep, "colour_mode": corr_lab is not None}


def long_lines(ink: np.ndarray, run_px: int) -> np.ndarray:
    r = max(3, run_px)
    h = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (r, 1)))
    v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, r)))
    return (h | v).astype(bool)


def find_doors(prep: Prepared, items: list[Item], o: SegOptions,
               max_wall_m: float = 1.0, min_w_m: float = 0.5, max_w_m: float = 2.5) -> list[tuple[float, float, float, int, int]]:
    """서로 다른 영역의 박스 두 개가 벽 하나(max_wall_m 안)를 사이에 두고 마주 보면, 그 사이 띠를 벽 방향으로 훑어
    잉크가 하나도 없는 줄이 min_w_m~max_w_m 이어진 곳 = 문. 돌려줌 (cx, cy, 폭 px, 영역 a, 영역 b)."""
    ppm = prep.ppm
    gap = int(max_wall_m * ppm)
    out = []
    boxes = [(k, b) for k, it in enumerate(items) for b in it.boxes]
    for i, (ka, A) in enumerate(boxes):
        for kb, B in boxes[i + 1:]:
            if ka == kb:
                continue
            for horiz in (True, False):
                # horiz: A 위/아래에 B (가로벽). 좌표를 바꿔 한 가지 코드로
                if horiz:
                    (a0, a1, ar0, ar1), (b0, b1, br0, br1) = (A[0], A[2], A[1], A[3]), (B[0], B[2], B[1], B[3])
                else:
                    (a0, a1, ar0, ar1), (b0, b1, br0, br1) = (A[1], A[3], A[0], A[2]), (B[1], B[3], B[0], B[2])
                lo, hi = max(a0, b0), min(a1, b1)
                if hi - lo < min_w_m * ppm:
                    continue
                if 0 <= br0 - ar1 <= gap:
                    s0, s1 = ar1, br0
                elif 0 <= ar0 - br1 <= gap:
                    s0, s1 = br1, ar0
                else:
                    continue
                s0, s1 = max(0, s0 - 2), s1 + 2
                strip = prep.ink[s0:s1, lo:hi] if horiz else prep.ink[lo:hi, s0:s1].T
                free = strip.sum(axis=0) == 0
                # 이어진 빈 줄 구간
                run = 0
                for t in range(len(free) + 1):
                    if t < len(free) and free[t]:
                        run += 1
                        continue
                    if min_w_m * ppm <= run <= max_w_m * ppm:
                        c = lo + t - run / 2
                        m = (s0 + s1) / 2
                        cx, cy = (c, m) if horiz else (m, c)
                        if not any(math.dist((cx, cy), d[:2]) < 0.8 * ppm for d in out):
                            out.append((float(cx), float(cy), float(run), ka, kb))
                    run = 0
    return out


# ---------------- 마법봉 ----------------
def magic_wand(prep: Prepared, point_px: tuple[float, float], tol: float = 18.0, opts: SegOptions | None = None) -> Item:
    """점(보정 이미지 px) -> 영역 Item. 벽(문 틈 닫은 잉크)은 넘지 않고, 찍은 곳 바탕색과 Lab 거리 tol 넘게 다른
    밝은 큰 덩어리(1.5m 이상: 다른 색 방·복도)도 넘지 않음. 어두운 글자·작은 기호는 지나감(구멍은 메움)."""
    o = opts or SegOptions()
    ppm = prep.ppm
    H, W = prep.ink.shape
    x, y = int(round(point_px[0])), int(round(point_px[1]))
    if not (0 <= x < W and 0 <= y < H):
        raise ValueError("점이 도면 밖")
    barrier = prep.closed.astype(bool)
    if barrier[y, x]:
        # 벽·글자 위를 찍었으면 0.6m 안의 가장 가까운 빈 곳
        r = max(2, int(0.6 * ppm))
        y0, x0 = max(0, y - r), max(0, x - r)
        ys, xs = np.nonzero(~barrier[y0:y + r + 1, x0:x + r + 1])
        if len(xs) == 0:
            raise ValueError("벽 위를 찍음 (방 안쪽을 찍을 것)")
        k = int(np.argmin((ys + y0 - y) ** 2 + (xs + x0 - x) ** 2))
        y, x = int(ys[k] + y0), int(xs[k] + x0)
    rr = max(1, int(0.5 * ppm))            # 바탕색 = 둘레 0.5m 안 밝은 30% (글자 위를 찍어도 바탕)
    win = (slice(max(0, y - rr), y + rr + 1), slice(max(0, x - rr), x + rr + 1))
    bright = prep.norm[win].max(axis=2) >= np.percentile(prep.norm[win].max(axis=2), 70)
    seed = np.median(prep.lab[win][bright], axis=0)
    # 밝은 면의 색 차이만 경계로 봄 (어두운 글자·선은 지나감)
    differ = (np.linalg.norm(prep.lab - seed, axis=2) >= tol) & (prep.norm.max(axis=2) > 0.75) & ~barrier
    big = drop_small(differ.astype(np.uint8), 1.5 * ppm).astype(bool)
    free = (~barrier & ~big).astype(np.uint8)
    n, lab = cv2.connectedComponents(free, connectivity=4)
    if lab[y, x] == 0:
        raise ValueError("찍은 곳이 바탕색과 너무 다름 (허용 오차를 늘릴 것)")
    reg = lab == lab[y, x]
    if reg.sum() > 0.6 * H * W or reg[0].any() or reg[-1].any() or reg[:, 0].any() or reg[:, -1].any():
        raise ValueError("영역이 닫히지 않음 (벽 틈이 문보다 넓거나 도면 밖). 허용 오차를 줄이거나 직접 그릴 것")
    if prep.building[reg].mean() < 0.5:
        raise ValueError("건물 바깥 (건물 벽 안쪽을 찍을 것)")
    it = make_item(reg, prep, o)
    if it is None:
        raise ValueError(f"영역이 너무 작음 (< {o.min_room_m2} m²)")
    it.info["seed_px"] = [x, y]
    return it


def px_box_to_m(b, h_px: int, ppm: float) -> tuple[float, float, float, float]:
    c0, r0, c1, r1 = b
    return c0 / ppm, (h_px - r1) / ppm, c1 / ppm, (h_px - r0) / ppm
