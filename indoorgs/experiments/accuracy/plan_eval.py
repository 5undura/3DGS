"""도면 불러오기 정확도: 피난 안내도 사진(합성) / DXF / SVG -> 벽 위치 오차(m), 방 다각형 IoU.

정답 건물 (벽 중심선, m): 24 x 14, 벽 두께 0.2. 복도 y 4~6.4 (x 0~24), 아래 방 2개, 위 방 3개, 문 틈 0.9m.
사용자 입력: 사진 속 건물 바깥벽 바깥 모서리 네 점(클릭 오차 ±3px) + 실제 크기 24.2 x 14.2 m.
불러온 좌표 = 바깥 모서리 왼아래가 (0,0) -> 정답 중심선 좌표 + 0.1.

사진 흉내: 방마다 옅은 색, 방 이름 글자, 초록 대피 화살표, 빨간 현재 위치·소화기 기호, 문 열림 호(가는 선),
계단 빗금, 제목·범례(색 상자) 테두리, 약한 원근 + 흐림 + 고르지 않은 밝기 + 잡음 + JPEG 70.

python experiments/accuracy/plan_eval.py
"""
from __future__ import annotations

import json
import math

import cv2
import numpy as np

from common import OUT, fmt, stats
from indoorgs.space import DraftOptions, import_dxf, import_image, import_svg

T = 0.2                      # 벽 두께
W_M, H_M = 24.0, 14.0        # 바깥벽 중심선 사각형
OFF = T / 2                  # 불러온 좌표 = 중심선 + OFF
DOOR = 0.9


def walls_centerline(exit_gap: bool = False):
    """벽 중심선 선분 (문 틈 뺀 것). [(a, b)]"""
    H = []                    # (y, x0, x1) 가로
    V = []                    # (x, y0, y1) 세로

    def hgaps(y, x0, x1, gaps):
        cur = x0
        for g in sorted(gaps):
            H.append((y, cur, g)); cur = g + DOOR
        H.append((y, cur, x1))

    def vgaps(x, y0, y1, gaps):
        cur = y0
        for g in sorted(gaps):
            V.append((x, cur, g[0] if isinstance(g, tuple) else g))
            cur = (g[1] if isinstance(g, tuple) else g + DOOR)
        V.append((x, cur, y1))

    hgaps(0, 0, W_M, []); hgaps(H_M, 0, W_M, [])
    vgaps(0, 0, H_M, [(4.3, 6.1)] if exit_gap else []); vgaps(W_M, 0, H_M, [])
    hgaps(4.0, 0, W_M, [4.0, 16.0])
    hgaps(6.4, 0, W_M, [3.0, 11.0, 19.0])
    vgaps(10.0, 0, 4.0, []); vgaps(8.0, 6.4, H_M, []); vgaps(15.0, 6.4, H_M, [])
    segs = [((x0, y), (x1, y)) for y, x0, x1 in H] + [((x, y0), (x, y1)) for x, y0, y1 in V]
    return segs


ROOMS_C = {   # 벽 안쪽 다각형 (중심선 좌표)
    "L1": (0.1, 0.1, 9.9, 3.9), "L2": (10.1, 0.1, 23.9, 3.9), "corr": (0.1, 4.1, 23.9, 6.3),
    "U1": (0.1, 6.5, 7.9, 13.9), "U2": (8.1, 6.5, 14.9, 13.9), "U3": (15.1, 6.5, 23.9, 13.9)}


def truth_rooms():
    return {k: [(x0 + OFF, y0 + OFF), (x1 + OFF, y0 + OFF), (x1 + OFF, y1 + OFF), (x0 + OFF, y1 + OFF)]
            for k, (x0, y0, x1, y1) in ROOMS_C.items()}


def truth_walls(exit_gap=False):
    return [((a[0] + OFF, a[1] + OFF), (b[0] + OFF, b[1] + OFF)) for a, b in walls_centerline(exit_gap)]


# ---------------- 그림 ----------------
def draw_map(ppm=40, exit_gap=False, decor=True, thickness=T):
    mx, top, right, bot = 60, 150, 320, 60
    Wp, Hp = int(mx + (W_M + T) * ppm + right), int(top + (H_M + T) * ppm + bot)
    img = np.full((Hp, Wp, 3), 250, np.uint8)

    def P(x, y):   # 중심선 m -> 픽셀 (실수)
        return mx + (x + OFF) * ppm, top + (H_M + T - (y + OFF)) * ppm

    def Pi(x, y):
        p = P(x, y); return int(round(p[0])), int(round(p[1]))

    if decor:
        cols = {"L1": (200, 230, 255), "L2": (230, 210, 250), "corr": (205, 245, 205),
                "U1": (250, 235, 200), "U2": (210, 240, 250), "U3": (225, 225, 245)}
        for k, (x0, y0, x1, y1) in ROOMS_C.items():
            cv2.rectangle(img, Pi(x0, y1), Pi(x1, y0), cols[k], -1)
    half = thickness * ppm / 2
    for a, b in walls_centerline(exit_gap):
        (x0, y0), (x1, y1) = P(*a), P(*b)
        # 벽 = 두께 있는 사각형 (모서리에서 반 두께 늘림)
        if abs(y0 - y1) < 1e-9:
            r = (min(x0, x1) - half, y0 - half, max(x0, x1) + half, y0 + half)
        else:
            r = (x0 - half, min(y0, y1) - half, x0 + half, max(y0, y1) + half)
        cv2.rectangle(img, (int(round(r[0])), int(round(r[1]))), (int(round(r[2])) - 1, int(round(r[3])) - 1),
                      (30, 30, 30), -1)
    if decor:
        f = cv2.FONT_HERSHEY_SIMPLEX
        for k, (x0, y0, x1, y1) in ROOMS_C.items():
            if k != "corr":
                cv2.putText(img, f"B1{list(ROOMS_C).index(k):02d}", Pi(x0 + 1.0, (y0 + y1) / 2), f, 1.1, (20, 20, 20), 2)
                cv2.putText(img, "Lab", Pi(x0 + 1.0, (y0 + y1) / 2 - 1.0), f, 0.7, (40, 40, 40), 1)
        # 대피 화살표 (초록, 굵게) 복도 따라 + 문에서 복도로
        for xa in (20, 14, 8):
            cv2.arrowedLine(img, Pi(xa, 5.2), Pi(xa - 4, 5.2), (40, 160, 40), 7, tipLength=0.25)
        cv2.arrowedLine(img, Pi(3.45, 8), Pi(3.45, 5.6), (40, 160, 40), 6, tipLength=0.3)
        # 현재 위치 (빨간 원 + 글자), 소화기 (빨간 네모)
        cv2.circle(img, Pi(12, 5.2), 14, (30, 30, 220), -1)
        cv2.putText(img, "YOU ARE HERE", Pi(12.6, 4.6), f, 0.6, (30, 30, 220), 2)
        for xs, ys in ((6, 4.4), (17, 6.0), (2, 12.5), (22, 1.0)):
            p = Pi(xs, ys); cv2.rectangle(img, (p[0] - 7, p[1] - 9), (p[0] + 7, p[1] + 9), (30, 30, 200), -1)
        # 문 열림 호 (가는 검은 선)
        for gx, y, sgn in ((4.0, 4.0, -1), (16.0, 4.0, -1), (3.0, 6.4, 1), (11.0, 6.4, 1), (19.0, 6.4, 1)):
            c = Pi(gx, y)
            r = int(DOOR * ppm)
            cv2.ellipse(img, c, (r, r), 0, -90 if sgn > 0 else 0, 0 if sgn > 0 else 90, (40, 40, 40), 1)
            e = Pi(gx, y + sgn * DOOR); cv2.line(img, c, e, (40, 40, 40), 2)
        # 계단 빗금 (U3 오른위 2 x 3 m)
        for k in range(0, int(3 * ppm), 9):
            p0 = Pi(21.5, 13.8); cv2.line(img, (p0[0], p0[1] + k), (p0[0] + int(2.2 * ppm), p0[1] + k), (60, 60, 60), 2)
        # 제목, 범례, 종이 테두리
        cv2.putText(img, "EVACUATION MAP  B1F", (mx, 70), f, 1.6, (20, 20, 160), 3)
        lx = int(mx + (W_M + T) * ppm + 40)
        for i, (c, s) in enumerate([((40, 160, 40), "Exit route"), ((30, 30, 220), "You are here"),
                                    ((30, 30, 200), "Extinguisher"), ((250, 235, 200), "Office")]):
            cv2.rectangle(img, (lx, top + 30 + 50 * i), (lx + 30, top + 60 + 50 * i), c, -1)
            cv2.putText(img, s, (lx + 40, top + 52 + 50 * i), f, 0.6, (20, 20, 20), 1)
        cv2.rectangle(img, (8, 8), (Wp - 9, Hp - 9), (20, 20, 20), 4)
    corners = np.float32([P(-OFF, H_M + OFF), P(W_M + OFF, H_M + OFF), P(W_M + OFF, -OFF), P(-OFF, -OFF)])
    # 바깥 모서리 = 픽셀 경계 (중심 좌표로는 -0.5)
    corners -= 0.5
    return img, corners


def photo(img, corners, rng, persp=0.06, blur=1.5, jpeg=70, light=0.25, noise=4.0, out_w=2000):
    h, w = img.shape[:2]
    s = out_w / w
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jit = rng.uniform(-persp, persp, (4, 2)) * np.array([w, h]) * s
    dst = (np.float32([[60, 60], [out_w - 60, 60], [out_w - 60, h * s - 60], [60, h * s - 60]]) + jit).astype(np.float32)
    Hm = cv2.getPerspectiveTransform(src, dst)
    out_h = int(h * s + 2 * 60 * 0 + 40)
    ph = cv2.warpPerspective(img, Hm, (out_w, out_h), flags=cv2.INTER_AREA, borderValue=(120, 110, 100))
    if blur:
        ph = cv2.GaussianBlur(ph, (0, 0), blur)
    if light:
        yy, xx = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
        g = 1 - light * (xx / out_w) * 0.8 - light * ((yy / out_h - 0.3) ** 2)
        ph = np.clip(ph.astype(np.float32) * g[..., None], 0, 255)
    if noise:
        ph = np.clip(ph + rng.normal(0, noise, ph.shape), 0, 255)
    ph = ph.astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", ph, [cv2.IMWRITE_JPEG_QUALITY, jpeg])
    ph = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    c = cv2.perspectiveTransform(corners[None] + 0.5, Hm)[0] - 0.5
    return ph, c


# ---------------- 재기 ----------------
def wall_errors(det, truth, tol=0.4):
    """정답 벽마다 가장 길게 겹친 같은 방향 검출 선까지 수직 거리. 반환 (벽별 오차 m, 정답 덮인 비율,
    어느 벽에서도 tol 넘게 떨어진 헛선 길이 m)"""
    errs, false_len = [], 0.0
    per_wall = [(0.0, None)] * len(truth)
    covered = [np.zeros(int(math.dist(*t) / 0.05) + 1, bool) for t in truth]
    for a, b in det:
        hor = abs(a[1] - b[1]) < 1e-6
        ver = abs(a[0] - b[0]) < 1e-6
        L = math.dist(a, b)
        best = None
        for k, (p, q) in enumerate(truth):
            th = abs(p[1] - q[1]) < 1e-9
            if hor and th:
                d = a[1] - p[1]
                lo, hi = max(min(a[0], b[0]), min(p[0], q[0])), min(max(a[0], b[0]), max(p[0], q[0]))
                s0 = min(p[0], q[0])
            elif ver and not th:
                d = a[0] - p[0]
                lo, hi = max(min(a[1], b[1]), min(p[1], q[1])), min(max(a[1], b[1]), max(p[1], q[1]))
                s0 = min(p[1], q[1])
            else:
                continue
            if abs(d) <= tol and hi > lo - 0.1:
                if best is None or abs(d) < abs(best[0]):
                    best = (d, k, lo, hi, s0)
                if hi > lo:
                    i0, i1 = int((lo - s0) / 0.05), int((hi - s0) / 0.05)
                    covered[k][max(0, i0):i1 + 1] = True
                    if hi - lo > per_wall[k][0]:
                        per_wall[k] = (hi - lo, abs(d))
        if best is None:
            false_len += L
    # 벽마다: 가장 길게 겹친 검출 선의 수직 거리 (근처 빗금·글자 선은 헛선으로만 셈)
    errs = [e for o, e in per_wall if o > 0]
    tot = sum(math.dist(*t) for t in truth)
    cov = sum(c.mean() * math.dist(*t) for c, t in zip(covered, truth)) / tot
    return errs, cov, false_len


def iou(a, b, res=0.02):
    pts = np.array(list(a) + list(b))
    mn = pts.min(0) - 0.1
    sz = np.ceil((pts.max(0) + 0.1 - mn) / res).astype(int)
    A = np.zeros(sz[::-1], np.uint8); B = np.zeros_like(A)
    cv2.fillPoly(A, [np.round((np.array(a) - mn) / res).astype(np.int32)], 1)
    cv2.fillPoly(B, [np.round((np.array(b) - mn) / res).astype(np.int32)], 1)
    return float((A & B).sum() / max((A | B).sum(), 1))


def room_scores(model, truth):
    det = [r.floor for r in model.rooms]
    out = {}
    used = set()
    for k, t in truth.items():
        sc = [(iou(t, d), i) for i, d in enumerate(det)]
        best = max(sc, default=(0.0, -1))
        out[k] = round(best[0], 3)
        if best[0] > 0.5:
            used.add(best[1])
    extra = len(det) - len(used)
    return out, extra


def evaluate(name, res, truth_w, truth_r):
    errs, cov, false_len = wall_errors(res.segments_detected if hasattr(res, "segments_detected") else res.segments,
                                       truth_w)
    rs, extra = room_scores(res.model, truth_r)
    r = {"case": name, "wall_err_m": stats(errs), "wall_coverage": round(cov, 3), "false_wall_m": round(false_len, 2),
         "room_iou": rs, "mean_iou": round(float(np.mean(list(rs.values()))), 3), "rooms_found": len(res.model.rooms),
         "extra_rooms": extra}
    print(f"{name:38s} wall err [{fmt({k: v for k, v in r['wall_err_m'].items() if k in ('median', 'p95', 'max')})}] "
          f"cover={r['wall_coverage']} false={r['false_wall_m']}m | rooms {r['rooms_found']} (extra {extra}) "
          f"IoU mean={r['mean_iou']} {rs}")
    return r


def run_image_cases(seeds=(0, 1, 2)):
    out = []
    tw, tr = truth_walls(), truth_rooms()
    img, corners = draw_map()
    res = import_image(img, corners=corners, size_m=(W_M + T, H_M + T), px_per_m=50)
    out.append(evaluate("image: flat render, exact corners", res, tw, tr))
    img0, c0 = draw_map(decor=False)
    res = import_image(img0, corners=c0, size_m=(W_M + T, H_M + T), px_per_m=50)
    out.append(evaluate("image: walls only (no decor)", res, tw, tr))
    for sd in seeds:
        rng = np.random.default_rng(sd)
        ph, c = photo(img, corners, rng)
        click = c + rng.uniform(-3, 3, c.shape)
        if sd == 0:
            OUT.mkdir(exist_ok=True)
            cv2.imwrite(str(OUT / "plan_photo_seed0.jpg"), ph)
        res = import_image(ph, corners=click, size_m=(W_M + T, H_M + T), px_per_m=50)
        if sd == 0:
            cv2.imwrite(str(OUT / "plan_photo_seed0_mask.png"), res.extra["wall_mask"])
        out.append(evaluate(f"image: photo seed{sd} (Otsu)", res, tw, tr))
        res = import_image(ph, corners=click, size_m=(W_M + T, H_M + T), px_per_m=50, adaptive_block_m=1.0)
        out.append(evaluate(f"image: photo seed{sd} (adaptive 1m)", res, tw, tr))
    # 밝기 고름 (light=0): 남은 복도 실패가 조명 탓인지
    ph, c = photo(img, corners, np.random.default_rng(0), light=0.0)
    res = import_image(ph, corners=c, size_m=(W_M + T, H_M + T), px_per_m=50)
    out.append(evaluate("image: photo seed0, even lighting", res, tw, tr))
    # 출구(바깥벽 1.8m 틈) 있는 경우
    tw2 = truth_walls(exit_gap=True)
    img2, c2 = draw_map(exit_gap=True)
    ph, c = photo(img2, c2, np.random.default_rng(5))
    res = import_image(ph, corners=c, size_m=(W_M + T, H_M + T), px_per_m=50)
    out.append(evaluate("image: photo + 1.8m exit in outer wall", res, tw2, tr))
    return out


# ---------------- DXF / SVG ----------------
def make_dxf(kind: str):
    import ezdxf
    doc = ezdxf.new()
    doc.header["$INSUNITS"] = 4       # mm
    msp = doc.modelspace()
    mm = lambda p: (p[0] * 1000, p[1] * 1000)
    if kind == "room_polylines":
        for poly in truth_rooms().values():
            msp.add_lwpolyline([mm(p) for p in poly], close=True, dxfattribs={"layer": "ROOM"})
    elif kind == "wall_centerlines":
        for a, b in truth_walls():
            msp.add_line(mm(a), mm(b), dxfattribs={"layer": "WALL"})
    elif kind == "wall_centerlines_closed_doors":
        for a, b in truth_walls_closed():
            msp.add_line(mm(a), mm(b), dxfattribs={"layer": "WALL"})
    msp.add_text("B101", dxfattribs={"layer": "TEXT"}).set_placement((2000, 2000))
    msp.add_circle((12000, 5000), 300, dxfattribs={"layer": "SYM"})
    return doc


def truth_walls_closed():
    """문 틈 없이 (중심선, 불러온 좌표). 벽 중심선 방 = 정답 방보다 반 두께씩 큼."""
    segs = [((0, 0), (W_M, 0)), ((W_M, 0), (W_M, H_M)), ((W_M, H_M), (0, H_M)), ((0, H_M), (0, 0)),
            ((0, 4), (W_M, 4)), ((0, 6.4), (W_M, 6.4)), ((10, 0), (10, 4)), ((8, 6.4), (8, H_M)), ((15, 6.4), (15, H_M))]
    return [((a[0] + OFF, a[1] + OFF), (b[0] + OFF, b[1] + OFF)) for a, b in segs]


def centerline_rooms():
    out = {}
    for k, (x0, y0, x1, y1) in ROOMS_C.items():
        x0, y0, x1, y1 = x0 - 0.1 + OFF, y0 - 0.1 + OFF, x1 + 0.1 + OFF, y1 + 0.1 + OFF
        out[k] = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return out


def make_svg(ppm=50):
    """방 = <rect>, 벽 중심선 = <line>, 글자, 원. y 아래로, 그룹 translate·scale 포함."""
    Hpx = (H_M + T) * ppm
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {(W_M + T) * ppm} {Hpx}">',
             '<g transform="translate(10,0) scale(1)">', '<g transform="translate(-10,0)">']
    for k, (x0, y0, x1, y1) in ROOMS_C.items():
        X0, X1 = (x0 + OFF) * ppm, (x1 + OFF) * ppm
        Y0, Y1 = Hpx - (y1 + OFF) * ppm, Hpx - (y0 + OFF) * ppm
        parts.append(f'<rect x="{X0}" y="{Y0}" width="{X1 - X0}" height="{Y1 - Y0}" fill="#eef"/>')
    parts.append('<text x="100" y="100">B101</text><circle cx="600" cy="250" r="15"/>')
    parts.append('</g></g></svg>')
    return "\n".join(parts)


def run_cad_cases():
    out = []
    tw, tr = truth_walls(), truth_rooms()
    opts = DraftOptions(boxes=True)
    r = import_dxf_doc_safe(make_dxf("room_polylines"), opts)
    out.append(evaluate("dxf: closed room polylines (mm)", r, tw, tr) | {"note": "벽 오차 = 방 테두리 vs 벽 중심선 (0.1 이 정상)"})
    r = import_dxf_doc_safe(make_dxf("wall_centerlines"), opts)
    out.append(evaluate("dxf: wall centerlines with door gaps", r, tw, centerline_rooms()))
    r = import_dxf_doc_safe(make_dxf("wall_centerlines_closed_doors"), opts)
    out.append(evaluate("dxf: wall centerlines, no gaps", r, tw, centerline_rooms()))
    r = import_svg(make_svg(), opts, px_per_m=50)
    out.append(evaluate("svg: room rects (+groups transform)", r, tw, tr) | {"note": "벽 오차 = 방 테두리 vs 벽 중심선 (0.1 이 정상)"})
    return out


def import_dxf_doc_safe(doc, opts):
    from indoorgs.space.dxf_import import import_dxf_doc
    return import_dxf_doc(doc, opts)


def main():
    res = {"image": run_image_cases(), "cad": run_cad_cases()}
    OUT.mkdir(exist_ok=True)
    (OUT / "plan.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
