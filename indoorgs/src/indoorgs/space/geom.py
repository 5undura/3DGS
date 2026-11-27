"""2D 도형 도우미 (shapely 없이). 단위 m, 도면 좌표 (x 오른쪽, y 위).

점 = (x, y), 다각형 = 점 목록(닫는 점 반복 없음), 선분 = ((x1, y1), (x2, y2)).
"""
from __future__ import annotations

import math
from collections import defaultdict

Pt = tuple[float, float]
Seg = tuple[Pt, Pt]


def signed_area(poly: list[Pt]) -> float:
    """반시계면 양수."""
    n = len(poly)
    return 0.5 * sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1] for i in range(n))


def area(poly: list[Pt]) -> float:
    return abs(signed_area(poly))


def perimeter(poly: list[Pt]) -> float:
    n = len(poly)
    return sum(math.dist(poly[i], poly[(i + 1) % n]) for i in range(n))


def ccw(poly: list[Pt]) -> list[Pt]:
    return list(poly) if signed_area(poly) >= 0 else list(poly)[::-1]


def centroid(poly: list[Pt]) -> Pt:
    a = signed_area(poly)
    n = len(poly)
    if abs(a) < 1e-12:
        return sum(p[0] for p in poly) / n, sum(p[1] for p in poly) / n
    cx = cy = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        c = x1 * y2 - x2 * y1
        cx += (x1 + x2) * c
        cy += (y1 + y2) * c
    return cx / (6 * a), cy / (6 * a)


def bbox(poly: list[Pt]) -> tuple[float, float, float, float]:
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def point_in_poly(p: Pt, poly: list[Pt]) -> bool:
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            if x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
    return inside


def closest_on_segment(p: Pt, a: Pt, b: Pt) -> tuple[Pt, float, float]:
    """(가장 가까운 점, 거리, 선분 위 비율 u)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    u = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2))
    q = (a[0] + u * dx, a[1] + u * dy)
    return q, math.dist(p, q), u


def edges(poly: list[Pt]) -> list[Seg]:
    return [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]


def nearest_edge(p: Pt, poly: list[Pt]) -> tuple[int, Pt, float]:
    """(변 번호, 변 위 가장 가까운 점, 거리)."""
    best = (-1, p, math.inf)
    for i, (a, b) in enumerate(edges(poly)):
        q, d, _ = closest_on_segment(p, a, b)
        if d < best[2]:
            best = (i, q, d)
    return best


def dist_to_boundary(p: Pt, poly: list[Pt]) -> float:
    return nearest_edge(p, poly)[2]


def is_convex(poly: list[Pt]) -> bool:
    n = len(poly)
    sign = 0
    for i in range(n):
        (x1, y1), (x2, y2), (x3, y3) = poly[i], poly[(i + 1) % n], poly[(i + 2) % n]
        c = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
        if abs(c) > 1e-12:
            s = 1 if c > 0 else -1
            if sign and s != sign:
                return False
            sign = s
    return True


def clip_convex(subject: list[Pt], clip: list[Pt]) -> list[Pt]:
    """Sutherland-Hodgman. clip 은 볼록 다각형."""
    out = list(subject)
    clip = ccw(clip)
    n = len(clip)
    for i in range(n):
        a, b = clip[i], clip[(i + 1) % n]
        inp, out = out, []
        if not inp:
            break

        def inside(p):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-12

        def cross(p, q):
            x1, y1, x2, y2 = p[0], p[1], q[0], q[1]
            x3, y3, x4, y4 = a[0], a[1], b[0], b[1]
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            return x1 + t * (x2 - x1), y1 + t * (y2 - y1)

        for j in range(len(inp)):
            p, q = inp[j], inp[(j + 1) % len(inp)]
            if inside(q):
                if not inside(p):
                    out.append(cross(p, q))
                out.append(q)
            elif inside(p):
                out.append(cross(p, q))
    return out


def overlap_area(a: list[Pt], b: list[Pt], grid: float = 0.1) -> float:
    """두 다각형 겹친 넓이. 한쪽이 볼록이면 정확, 아니면 격자 표본(grid m)으로 어림."""
    ax0, ay0, ax1, ay1 = bbox(a)
    bx0, by0, bx1, by1 = bbox(b)
    x0, y0, x1, y1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    if is_convex(b):
        r = clip_convex(a, b)
        return area(r) if len(r) >= 3 else 0.0
    if is_convex(a):
        r = clip_convex(b, a)
        return area(r) if len(r) >= 3 else 0.0
    cnt = 0
    nx, ny = max(1, int((x1 - x0) / grid)), max(1, int((y1 - y0) / grid))
    dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
    for i in range(nx):
        for j in range(ny):
            p = (x0 + (i + 0.5) * dx, y0 + (j + 0.5) * dy)
            if point_in_poly(p, a) and point_in_poly(p, b):
                cnt += 1
    return cnt * dx * dy


def shared_walls(a: list[Pt], b: list[Pt], tol: float = 0.3, min_len: float = 0.3) -> list[Seg]:
    """두 다각형 변이 맞닿은(평행, 거리 tol 이하, 겹침 min_len 이상) 구간. a 의 변 위 좌표로 돌려줌."""
    out = []
    for p1, p2 in edges(a):
        L = math.dist(p1, p2)
        if L < 1e-9:
            continue
        ux, uy = (p2[0] - p1[0]) / L, (p2[1] - p1[1]) / L
        for q1, q2 in edges(b):
            M = math.dist(q1, q2)
            if M < 1e-9:
                continue
            vx, vy = (q2[0] - q1[0]) / M, (q2[1] - q1[1]) / M
            if abs(ux * vy - uy * vx) > 0.05:      # 약 3도 넘게 기울면 평행 아님
                continue
            # q 끝점들의 a 변 방향 거리와 수직 거리
            off = [((q[0] - p1[0]) * ux + (q[1] - p1[1]) * uy, (q[0] - p1[0]) * -uy + (q[1] - p1[1]) * ux) for q in (q1, q2)]
            if max(abs(off[0][1]), abs(off[1][1])) > tol:
                continue
            lo, hi = max(0.0, min(off[0][0], off[1][0])), min(L, max(off[0][0], off[1][0]))
            if hi - lo >= min_len:
                out.append(((p1[0] + ux * lo, p1[1] + uy * lo), (p1[0] + ux * hi, p1[1] + uy * hi)))
    return out


# ---------------- 선분 정리·닫힌 다각형 찾기 ----------------
def _snap_points(segs: list[Seg], tol: float) -> tuple[list[Pt], list[tuple[int, int]]]:
    """끝점을 tol 안에서 하나로 합침. (점 목록, 선분 = 점 번호 쌍)."""
    pts: list[Pt] = []
    cnt: list[int] = []
    cell: dict[tuple[int, int], list[int]] = defaultdict(list)
    size = max(tol, 1e-9)

    def find(p: Pt) -> int:
        gx, gy = int(math.floor(p[0] / size)), int(math.floor(p[1] / size))
        best, bd = -1, tol
        for ix in (gx - 1, gx, gx + 1):
            for iy in (gy - 1, gy, gy + 1):
                for k in cell[(ix, iy)]:
                    d = math.dist(pts[k], p)
                    if d <= bd:
                        best, bd = k, d
        if best >= 0:
            # 평균 위치로 이동 (격자 칸은 처음 자리 유지, tol 안이라 문제 없음)
            n = cnt[best]
            pts[best] = ((pts[best][0] * n + p[0]) / (n + 1), (pts[best][1] * n + p[1]) / (n + 1))
            cnt[best] += 1
            return best
        pts.append(p)
        cnt.append(1)
        cell[(gx, gy)].append(len(pts) - 1)
        return len(pts) - 1

    idx = []
    for a, b in segs:
        i, j = find(a), find(b)
        if i != j:
            idx.append((min(i, j), max(i, j)))
    return pts, sorted(set(idx))


def split_at_touches(segs: list[Seg], tol: float) -> list[Seg]:
    """다른 선분의 끝점이 선분 중간에 닿으면(T자) 그 자리에서 나눔. 교차(X자)도 나눔."""
    out = []
    for i, (a, b) in enumerate(segs):
        L = math.dist(a, b)
        if L < 1e-9:
            continue
        cuts = [0.0, 1.0]
        for j, (c, d) in enumerate(segs):
            if i == j:
                continue
            for p in (c, d):
                _, dist, u = closest_on_segment(p, a, b)
                if dist <= tol and tol / L < u < 1 - tol / L:
                    cuts.append(u)
            x = _intersect(a, b, c, d)
            if x is not None and tol / L < x < 1 - tol / L:
                cuts.append(x)
        cuts = sorted(set(round(u, 9) for u in cuts))
        for u0, u1 in zip(cuts, cuts[1:]):
            if (u1 - u0) * L > tol:
                out.append(((a[0] + u0 * (b[0] - a[0]), a[1] + u0 * (b[1] - a[1])),
                            (a[0] + u1 * (b[0] - a[0]), a[1] + u1 * (b[1] - a[1]))))
    return out


def _intersect(a: Pt, b: Pt, c: Pt, d: Pt) -> float | None:
    """선분 ab 와 cd 가 엄밀히 교차하면 ab 위 비율 u."""
    rx, ry = b[0] - a[0], b[1] - a[1]
    sx, sy = d[0] - c[0], d[1] - c[1]
    den = rx * sy - ry * sx
    if abs(den) < 1e-12:
        return None
    u = ((c[0] - a[0]) * sy - (c[1] - a[1]) * sx) / den
    v = ((c[0] - a[0]) * ry - (c[1] - a[1]) * rx) / den
    if 0 < u < 1 and 0 < v < 1:
        return u
    return None


def faces_from_segments(segs: list[Seg], tol: float = 0.02) -> list[list[Pt]]:
    """선분 그물에서 닫힌 면(가장 작은 고리)들을 반시계 다각형으로. 바깥 면은 뺌.

    끝점은 tol 안에서 합치고, T자로 닿는 곳은 나눈다. 막다른 선(끝이 한 곳만 연결)은 버린다.
    """
    segs = split_at_touches(segs, tol)
    pts, idx = _snap_points(segs, tol)
    adj: dict[int, set[int]] = defaultdict(set)
    for i, j in idx:
        adj[i].add(j)
        adj[j].add(i)
    # 막다른 가지 반복 제거
    stack = [k for k in adj if len(adj[k]) < 2]
    while stack:
        k = stack.pop()
        for m in list(adj[k]):
            adj[m].discard(k)
            if len(adj[m]) < 2:
                stack.append(m)
        adj[k].clear()
    order = {k: sorted(adj[k], key=lambda m: math.atan2(pts[m][1] - pts[k][1], pts[m][0] - pts[k][0]))
             for k in adj if adj[k]}
    used = set()
    faces = []
    for u in order:
        for v in order[u]:
            if (u, v) in used:
                continue
            face, a, b = [], u, v
            for _ in range(len(idx) * 2 + 2):
                if (a, b) in used:
                    break
                used.add((a, b))
                face.append(a)
                nb = order[b]
                k = nb.index(a)
                a, b = b, nb[(k - 1) % len(nb)]   # b 에서 시계방향으로 다음 이웃 = 왼쪽 면 따라가기
            poly = [pts[k] for k in face]
            if len(poly) >= 3 and signed_area(poly) > 1e-9:
                faces.append(simplify_collinear(poly))
    return faces


def simplify_collinear(poly: list[Pt], tol: float = 1e-3) -> list[Pt]:
    """한 줄 위의 가운데 점 없앰."""
    pts = list(poly)
    changed = True
    while changed and len(pts) > 3:
        changed = False
        for i in range(len(pts)):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
            if closest_on_segment(b, a, c)[1] < tol:
                pts.pop(i)
                changed = True
                break
    return pts


def rect_fit(poly: list[Pt], ratio: float = 0.9) -> tuple[Pt, Pt, float] | None:
    """다각형이 거의 직사각형(넓이 비 ratio 이상)이면 (중심, (가로, 세로), 회전 도)."""
    pts = ccw(poly)
    if len(pts) < 4:
        return None
    best = None
    for a, b in edges(pts):
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        c, s = math.cos(-ang), math.sin(-ang)
        rx = [c * p[0] - s * p[1] for p in pts]
        ry = [s * p[0] + c * p[1] for p in pts]
        w, h = max(rx) - min(rx), max(ry) - min(ry)
        if best is None or w * h < best[0]:
            lx, ly = (max(rx) + min(rx)) / 2, (max(ry) + min(ry)) / 2
            cx, cy = math.cos(ang) * lx - math.sin(ang) * ly, math.sin(ang) * lx + math.cos(ang) * ly
            best = (w * h, (cx, cy), (w, h), math.degrees(ang))
    if best is None or best[0] <= 0 or area(pts) / best[0] < ratio:
        return None
    _, center, (w, h), yaw = best
    # 회전은 -45~45 도로 정리 (가로·세로 바꿈)
    while yaw > 45:
        yaw -= 90
        w, h = h, w
    while yaw <= -45:
        yaw += 90
        w, h = h, w
    return center, (w, h), yaw
