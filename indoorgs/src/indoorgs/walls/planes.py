"""plan.json -> 실내 면(벽·바닥·천장) 평면 목록. 벽 평탄화 기능(walls/)의 공통 자료형.

좌표는 도면 좌표 (CONTRACT 3절): x 오른쪽, y 위, z 위, 미터.

Plane
- 무한 평면: n·x = d (n 은 단위 법선, **방 안쪽**을 향함). 점의 부호 거리 = n·x - d (방 안이면 +).
- 면의 범위: 평면 위 2D 좌표 (u, v) = ((x - origin)·U, (x - origin)·V) 로 직사각형 [0, size_u] x [0, size_v].
  벽: origin = 벽 시작 모서리의 바닥점, U = 벽 방향(수평), V = +z. 바닥·천장은 방 다각형 전체를 감싸는 사각형이고
  실제 범위는 polygon(도면 xy) 으로 다시 거름.
- holes: 벽 위 문 구멍 [(u0, u1, v0, v1)] (면 2D 좌표). 문 구멍 안의 점은 벽에 속하지 않음.

공개 API
  planes_from_plan(plan, rooms=None)       -> list[Plane]
  rooms_for_cameras(plan, centers)         카메라 위치가 들어 있는 방 id (구간이 찍은 방)
  Plane.signed_dist(P) / .uv(P) / .contains_uv(uv, margin) / .point(u, v)
  plane_to_json / plane_from_json          walls/planes.json 저장용
  better_face(...)                         점 배정 규칙 (가까운 면, 등을 맞댄 공유 벽은 어느 쪽인지로)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..schemas import Plan, Room

WALL, FLOOR, CEILING = "wall", "floor", "ceiling"


@dataclass
class Plane:
    id: str                      # "<room>:w<변 번호>" / "<room>:floor" / "<room>:ceiling"
    kind: str                    # wall | floor | ceiling
    room: str
    n: np.ndarray                # (3,) 단위 법선, 방 안쪽
    d: float                     # n·x = d
    origin: np.ndarray           # (3,) 면 2D 좌표의 원점 (평면 위)
    U: np.ndarray                # (3,) 면 가로축 (단위)
    V: np.ndarray                # (3,) 면 세로축 (단위)
    size: tuple[float, float]    # (size_u, size_v) m
    holes: list[tuple[float, float, float, float]] = field(default_factory=list)  # (u0, u1, v0, v1)
    polygon: list[tuple[float, float]] | None = None  # 바닥·천장: 방 다각형 (도면 xy)

    def signed_dist(self, P: np.ndarray) -> np.ndarray:
        return np.asarray(P, float) @ self.n - self.d

    def uv(self, P: np.ndarray) -> np.ndarray:
        Q = np.asarray(P, float) - self.origin
        return np.stack([Q @ self.U, Q @ self.V], axis=-1)

    def point(self, u, v) -> np.ndarray:
        u = np.asarray(u, float)[..., None]
        v = np.asarray(v, float)[..., None]
        return self.origin + u * self.U + v * self.V

    def contains_uv(self, uv: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """면 범위 안(문 구멍 제외)인지. margin > 0 이면 가장자리 바깥으로 margin 까지 허용."""
        uv = np.atleast_2d(np.asarray(uv, float))
        u, v = uv[:, 0], uv[:, 1]
        ok = (u >= -margin) & (u <= self.size[0] + margin) & (v >= -margin) & (v <= self.size[1] + margin)
        for (u0, u1, v0, v1) in self.holes:
            ok &= ~((u > u0) & (u < u1) & (v > v0) & (v < v1))
        if self.polygon is not None:
            P = self.point(u, v)
            ok &= _in_polygon(P[:, 0], P[:, 1], self.polygon, margin)
        return ok

    def area(self) -> float:
        if self.polygon is not None:
            return abs(_poly_area(self.polygon))
        a = self.size[0] * self.size[1]
        return a - sum((u1 - u0) * (v1 - v0) for u0, u1, v0, v1 in self.holes)

    def shifted(self, delta: float, n: np.ndarray | None = None) -> "Plane":
        """법선 방향으로 delta 만큼 옮긴(그리고 법선을 n 으로 바꾼) 사본. 면 2D 좌표계는 새 평면에 다시 투영."""
        nn = self.n if n is None else np.asarray(n, float) / np.linalg.norm(n)
        o = self.origin + delta * self.n
        d = float(nn @ o)
        U = self.U - (self.U @ nn) * nn
        U /= np.linalg.norm(U)
        V = np.cross(nn, U) if self.kind == WALL else self.V - (self.V @ nn) * nn - (self.V @ U) * U
        V /= np.linalg.norm(V)
        if V @ self.V < 0:
            V = -V
        return Plane(self.id, self.kind, self.room, nn, d, o, U, V, self.size, list(self.holes), self.polygon)


def _poly_area(poly) -> float:
    return 0.5 * sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
                     for i in range(len(poly)))


def _in_polygon(x: np.ndarray, y: np.ndarray, poly, margin: float = 0.0) -> np.ndarray:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    inside = np.zeros(x.shape, bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cond & (x < xc)
    if margin > 0:
        for i in range(n):
            a = np.array(poly[i], float)
            b = np.array(poly[(i + 1) % n], float)
            ab = b - a
            t = np.clip(((x - a[0]) * ab[0] + (y - a[1]) * ab[1]) / max(ab @ ab, 1e-12), 0, 1)
            dx = x - (a[0] + t * ab[0])
            dy = y - (a[1] + t * ab[1])
            inside |= dx * dx + dy * dy <= margin * margin
    return inside


DOOR_TOL = 0.15        # 문 중심이 벽선에서 이 거리(m) 안이면 그 벽의 문 (문의 rooms 가 비었을 때)
DOOR_TOL_ROOM = 0.30   # 문의 rooms 에 이 방이 있으면 벽 두께로 그린 도면(문 중심이 벽 가운데)까지 허용


def _room_wall_holes(room: Room, poly, doors) -> list[list[tuple[float, float, float, float]]]:
    """방 다각형 변마다 문 구멍 [(u0, u1, 0, 높이)].

    - 구간 겹침: 문 [s - w/2, s + w/2] 가 변 [0, L] 과 겹치면 (s + w/2 > 0 and s - w/2 < L) 그 변에 잘라서 넣음.
      그래서 한 직선 위에서 꺾임점(정점)으로 나뉜 두 변에 걸친 문은 두 변 모두에 뚫림.
    - 수직 허용: 문 중심의 벽선까지 거리 < DOOR_TOL (문의 rooms 에 이 방이 있으면 DOOR_TOL_ROOM: 벽 두께 도면).
    - 같은 방의 여러 변이 후보면 수직 거리가 가장 작은 변들(+2 cm, 같은 직선 위 조각)만. 모서리 옆 문이 옆
      (수직) 벽에 구멍을 내지 않게."""
    n = len(poly)
    out: list[list] = [[] for _ in range(n)]
    for dr in doors:
        if dr.rooms and room.id not in dr.rooms:
            continue
        tol = DOOR_TOL_ROOM if (dr.rooms and room.id in dr.rooms) else DOOR_TOL
        c0 = np.asarray(dr.center, float)
        hw = dr.width / 2
        cands = []
        for i in range(n):
            a = np.array(poly[i], float)
            b = np.array(poly[(i + 1) % n], float)
            ab = b - a
            L = float(np.linalg.norm(ab))
            if L < 1e-6:
                continue
            U2 = ab / L
            c = c0 - a
            perp = abs(float(c @ np.array([-U2[1], U2[0]])))
            s = float(c @ U2)
            if perp < tol and s + hw > 0 and s - hw < L:
                cands.append((perp, i, max(s - hw, 0.0), min(s + hw, L)))
        if not cands:
            continue
        pmin = min(c[0] for c in cands)
        for perp, i, u0, u1 in cands:
            if perp <= pmin + 0.02 and u1 - u0 > 1e-6:
                out[i].append((u0, u1, 0.0, dr.height))
    return out


def planes_from_plan(plan: Plan, rooms: list[str] | None = None, default_ceiling: float | None = None) -> list[Plane]:
    """방 다각형의 각 변 = 벽 하나, 바닥, 천장(ceiling_z 가 있을 때). rooms 를 주면 그 방만.

    방 다각형은 반시계(parse_plan 이 맞춤)라서 변 a->b 의 왼쪽 법선 (-dy, dx) 이 방 안쪽."""
    out: list[Plane] = []
    for r in plan.rooms:
        if rooms is not None and r.id not in rooms:
            continue
        fz = r.floor_z
        cz = r.ceiling_z if r.ceiling_z is not None else default_ceiling
        poly = [tuple(map(float, p)) for p in r.floor]
        if _poly_area(poly) < 0:
            poly = poly[::-1]
        height = (cz - fz) if cz is not None else 3.0
        holes = _room_wall_holes(r, poly, plan.doors)
        for i in range(len(poly)):
            a = np.array(poly[i], float)
            b = np.array(poly[(i + 1) % len(poly)], float)
            ab = b - a
            L = float(np.linalg.norm(ab))
            if L < 1e-6:
                continue
            U2 = ab / L
            n = np.array([-U2[1], U2[0], 0.0])
            origin = np.array([a[0], a[1], fz])
            U = np.array([U2[0], U2[1], 0.0])
            V = np.array([0.0, 0.0, 1.0])
            out.append(Plane(f"{r.id}:w{i}", WALL, r.id, n, float(n @ origin), origin, U, V, (L, height),
                             holes[i]))
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        o = np.array([x0, y0, fz])
        out.append(Plane(f"{r.id}:floor", FLOOR, r.id, np.array([0.0, 0.0, 1.0]), float(fz), o,
                         np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), (x1 - x0, y1 - y0), [], poly))
        if cz is not None:
            o = np.array([x0, y0, cz])
            out.append(Plane(f"{r.id}:ceiling", CEILING, r.id, np.array([0.0, 0.0, -1.0]), float(-cz), o,
                             np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), (x1 - x0, y1 - y0), [], poly))
    return out


def rooms_for_cameras(plan: Plan, centers: np.ndarray, min_frac: float = 0.02) -> list[str]:
    """카메라 위치(도면 좌표)가 min_frac 이상 들어 있는 방 id. 하나도 없으면 빈 목록."""
    centers = np.asarray(centers, float).reshape(-1, 3)
    if len(centers) == 0:
        return []
    out = []
    for r in plan.rooms:
        inside = _in_polygon(centers[:, 0], centers[:, 1], r.floor)
        if inside.mean() >= min_frac:
            out.append(r.id)
    return out


def plane_to_json(p: Plane) -> dict:
    d = {"id": p.id, "kind": p.kind, "room": p.room, "n": [float(v) for v in p.n], "d": float(p.d),
         "origin": [float(v) for v in p.origin], "U": [float(v) for v in p.U], "V": [float(v) for v in p.V],
         "size": [float(p.size[0]), float(p.size[1])], "holes": [list(map(float, h)) for h in p.holes]}
    if p.polygon is not None:
        d["polygon"] = [list(map(float, q)) for q in p.polygon]
    return d


def plane_from_json(d: dict) -> Plane:
    return Plane(d["id"], d["kind"], d["room"], np.asarray(d["n"], float), float(d["d"]),
                 np.asarray(d["origin"], float), np.asarray(d["U"], float), np.asarray(d["V"], float),
                 (float(d["size"][0]), float(d["size"][1])), [tuple(h) for h in d.get("holes", [])],
                 [tuple(q) for q in d["polygon"]] if d.get("polygon") else None)


BACK_TO_BACK_COS = -0.9    # 법선이 거의 반대 (약 154도 넘게)
BACK_TO_BACK_GAP = 0.02    # 두 면 부호 거리 합 <= 이 값이면 등을 맞댄 면 (공유 벽, 도면 벽 두께 0 포함)


def better_face(sd_new: np.ndarray, n_new: np.ndarray, sd_cur: np.ndarray, n_cur: np.ndarray,
                has_cur: np.ndarray) -> np.ndarray:
    """점 배정에서 새 면이 지금 면보다 나은지 (refine.assign_points, snap.assign 공통).

    기본은 |부호 거리| 가 작은 쪽. 단 두 면이 **등을 맞댄** 경우(법선 거의 반대이고 sd_new + sd_cur <= 작은 값:
    두 방이 공유하는 벽, 도면 벽 두께 0 이면 두 면이 겹침)는 거리가 아니라 **어느 쪽에 있는지**로 나눔:
    부호 거리가 큰 면 (= 두 면 가운데 평면을 기준으로 그 면의 방 쪽). 그래서 x = 4.0 에서 B 쪽으로 3.5 cm 인 점은
    B 면 (A 면에서는 -3.5 cm). 같은 방의 마주 보는 벽(합이 +폭)은 그대로 가까운 쪽."""
    sd_new = np.asarray(sd_new, float)
    sd_cur = np.asarray(sd_cur, float)
    has_cur = np.asarray(has_cur, bool)
    cosn = np.asarray(n_cur, float).reshape(-1, 3) @ np.asarray(n_new, float)
    b2b = has_cur & (cosn < BACK_TO_BACK_COS) & (sd_new + np.where(has_cur, sd_cur, 0.0) <= BACK_TO_BACK_GAP)
    near = np.abs(sd_new) < np.where(has_cur, np.abs(sd_cur), np.inf)
    return np.where(b2b, sd_new > sd_cur, near)


def angle_deg(n1: np.ndarray, n2: np.ndarray) -> float:
    c = float(np.clip(np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2)), -1, 1))
    return math.degrees(math.acos(c))
