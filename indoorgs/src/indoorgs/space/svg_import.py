"""SVG 도면 -> 초안 공간.

읽는 요소: <line>, <polyline>, <polygon>, <rect>, <path>(M/L/H/V/Z, 대소문자). 곡선 명령(C/S/Q/T/A)은
끝점으로 직선 처리하고 경고. transform(translate/scale/rotate/matrix)은 그룹까지 적용.
축척: px_per_m 또는 scale_points=((x1, y1), (x2, y2), 실제 거리 m) (SVG 좌표). SVG 는 y 가 아래로 커지므로
도면 좌표(y 위)로 뒤집는다: y_m = (H - y) / px_per_m, H = viewBox 높이(없으면 0 -> 그냥 부호만 바꿈).
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from .draft import DraftOptions, DraftResult, build_draft
from .geom import Pt, Seg

Mat = tuple[float, float, float, float, float, float]   # a b c d e f (SVG matrix)
I: Mat = (1, 0, 0, 1, 0, 0)
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _mul(m: Mat, n: Mat) -> Mat:
    a, b, c, d, e, f = m
    A, B, C, D, E, F = n
    return (a * A + c * B, b * A + d * B, a * C + c * D, b * C + d * D, a * E + c * F + e, b * E + d * F + f)


def _apply(m: Mat, p: Pt) -> Pt:
    return m[0] * p[0] + m[2] * p[1] + m[4], m[1] * p[0] + m[3] * p[1] + m[5]


def parse_transform(s: str | None) -> Mat:
    m = I
    if not s:
        return m
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", s):
        v = [float(x) for x in re.findall(_NUM, args)]
        if name == "matrix" and len(v) == 6:
            t = tuple(v)
        elif name == "translate":
            t = (1, 0, 0, 1, v[0], v[1] if len(v) > 1 else 0)
        elif name == "scale":
            t = (v[0], 0, 0, v[1] if len(v) > 1 else v[0], 0, 0)
        elif name == "rotate":
            a = math.radians(v[0])
            r = (math.cos(a), math.sin(a), -math.sin(a), math.cos(a), 0, 0)
            if len(v) == 3:
                t = _mul(_mul((1, 0, 0, 1, v[1], v[2]), r), (1, 0, 0, 1, -v[1], -v[2]))
            else:
                t = r
        else:
            continue
        m = _mul(m, t)
    return m


def _length(s: str | None) -> float:
    if not s:
        return 0.0
    m = re.match(_NUM, s.strip())
    return float(m.group(0)) if m else 0.0


def _points(s: str) -> list[Pt]:
    v = [float(x) for x in re.findall(_NUM, s or "")]
    return list(zip(v[0::2], v[1::2]))


def parse_path_d(d: str) -> tuple[list[tuple[list[Pt], bool]], int]:
    """path d -> ([(점 목록, 닫힘)], 곡선 명령 수). 곡선은 끝점만 씀."""
    toks = re.findall(r"[MmLlHhVvZzCcSsQqTtAa]|" + _NUM, d or "")
    subs: list[tuple[list[Pt], bool]] = []
    cur: list[Pt] = []
    x = y = 0.0
    start = (0.0, 0.0)
    cmd = None
    curves = 0
    nargs = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7, "Z": 0}
    i = 0

    def flush(closed):
        nonlocal cur
        if len(cur) >= 2:
            subs.append((cur, closed))
        cur = []

    while i < len(toks):
        if re.fullmatch(r"[A-Za-z]", toks[i]):
            cmd = toks[i]
            i += 1
            if cmd in "Zz":
                if cur and cur[0] == cur[-1]:
                    cur.pop()
                flush(True)
                x, y = start
                cur = []
                cmd = None
                continue
        if cmd is None:      # Z 뒤 떠도는 숫자는 건너뜀
            i += 1
            continue
        n = nargs[cmd.upper()]
        v = [float(t) for t in toks[i:i + n]]
        if len(v) < n:
            break
        i += n
        rel = cmd.islower()
        C = cmd.upper()
        if C == "M":
            flush(False)
            x, y = (x + v[0], y + v[1]) if rel else (v[0], v[1])
            start = (x, y)
            cur = [(x, y)]
            cmd = "l" if rel else "L"     # M 뒤 숫자는 L
            continue
        if C == "H":
            x = x + v[0] if rel else v[0]
        elif C == "V":
            y = y + v[0] if rel else v[0]
        else:
            if C != "L":
                curves += 1
            ex, ey = v[-2], v[-1]
            x, y = (x + ex, y + ey) if rel else (ex, ey)
        if not cur:
            cur = [start]
        cur.append((x, y))
    flush(False)
    return subs, curves


def import_svg(path_or_text: str | Path, opts: DraftOptions | None = None, *, px_per_m: float | None = None,
               scale_points: tuple[Pt, Pt, float] | None = None, flip_y: bool = True) -> DraftResult:
    """SVG 파일 경로 또는 SVG 글자 -> DraftResult. px_per_m 나 scale_points 중 하나는 필요."""
    opts = opts or DraftOptions()
    text = str(path_or_text)
    name = ""
    if not text.lstrip().startswith("<"):
        name = Path(text).name
        text = Path(text).read_text(encoding="utf-8")
    root = ET.fromstring(text)
    warnings: list[str] = []
    if scale_points is not None:
        (x1, y1), (x2, y2), dist_m = scale_points
        if dist_m <= 0 or math.dist((x1, y1), (x2, y2)) <= 0:
            raise ValueError("scale_points: 두 점이 같거나 거리가 0")
        px_per_m = math.dist((x1, y1), (x2, y2)) / dist_m
    if not px_per_m or px_per_m <= 0:
        raise ValueError("축척 필요: px_per_m 또는 scale_points=((x1,y1),(x2,y2),거리m)")
    vb = [float(v) for v in re.findall(_NUM, root.get("viewBox", ""))]
    H = (vb[1] + vb[3]) if len(vb) == 4 else _length(root.get("height"))
    if not H:
        warnings.append("viewBox/height 없음: y 부호만 뒤집음")

    raw_polys: list[list[Pt]] = []
    raw_segs: list[Seg] = []
    curves = 0

    def tag(el):
        return el.tag.rsplit("}", 1)[-1]

    def walk(el, mat: Mat):
        nonlocal curves
        mat = _mul(mat, parse_transform(el.get("transform")))
        t = tag(el)
        if t in ("defs", "clipPath", "mask", "symbol", "text", "style", "title", "desc"):
            return
        if t == "line":
            a = (_length(el.get("x1")), _length(el.get("y1")))
            b = (_length(el.get("x2")), _length(el.get("y2")))
            raw_segs.append((_apply(mat, a), _apply(mat, b)))
        elif t in ("polyline", "polygon"):
            pts = [_apply(mat, p) for p in _points(el.get("points", ""))]
            if t == "polygon" or (len(pts) > 2 and pts[0] == pts[-1]):
                if len(pts) > 2 and pts[0] == pts[-1]:
                    pts = pts[:-1]
                if len(pts) >= 3:
                    raw_polys.append(pts)
            else:
                raw_segs.extend(zip(pts, pts[1:]))
        elif t == "rect":
            x, y = _length(el.get("x")), _length(el.get("y"))
            w, h = _length(el.get("width")), _length(el.get("height"))
            if w > 0 and h > 0:
                raw_polys.append([_apply(mat, p) for p in ((x, y), (x + w, y), (x + w, y + h), (x, y + h))])
        elif t == "path":
            subs, c = parse_path_d(el.get("d", ""))
            curves += c
            for pts, closed in subs:
                pts = [_apply(mat, p) for p in pts]
                if closed and len(pts) >= 3:
                    raw_polys.append(pts)
                else:
                    raw_segs.extend(zip(pts, pts[1:]))
        for ch in el:
            walk(ch, mat)

    walk(root, I)
    if curves:
        warnings.append(f"path 곡선 명령 {curves}개를 직선으로 폄")

    def m(p: Pt) -> Pt:
        y = (H - p[1]) if (flip_y and H) else (-p[1] if flip_y else p[1])
        return p[0] / px_per_m, y / px_per_m

    polys = [[m(p) for p in poly] for poly in raw_polys]
    segs = [(m(a), m(b)) for a, b in raw_segs]
    source = {"kind": "svg", "file": name, "px_per_m": px_per_m}
    res = build_draft(polys, segs, opts, source, warnings)
    res.extra["n_polygons"] = len(polys)
    res.extra["n_segments"] = len(segs)
    return res
