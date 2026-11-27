"""박스 편집 데이터 모델 (담당 B). GUI·도면 불러오기가 함께 쓰는 공간 모델.

- 방: 박스(중심·크기·회전) 또는 다각형. 박스 방은 plan.json 에 box 로 저장돼 다시 읽어도 박스로 남음.
- 문: 한 점을 찍으면 가장 가까운 벽에 붙이고, 그 벽에 닿은 방(1~2개)을 자동으로 연결.
- 검사: 방 겹침, 문이 벽 위에 없음, 문 폭이 벽보다 넓음 등 -> Issue 목록.

모든 편집 함수는 바뀐 방·문 id 를 돌려주고, 없는 id 면 KeyError.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import schemas
from . import geom
from .geom import Pt

DEFAULT_CEILING = 2.7
DEFAULT_DOOR_W = 0.9
DEFAULT_DOOR_H = 2.1


@dataclass
class Box:
    center: Pt
    size: Pt                 # (가로 x, 세로 y) 회전 전 기준
    yaw_deg: float = 0.0

    def floor(self) -> list[Pt]:
        return schemas.box_to_floor(self.center, self.size, self.yaw_deg)

    def to_json(self) -> dict:
        return {"center": list(self.center), "size": list(self.size), "yaw_deg": self.yaw_deg}


@dataclass
class SpaceRoom:
    id: str
    name: str = ""
    box: Box | None = None
    polygon: list[Pt] | None = None      # box 가 없을 때만 (반시계)
    floor_z: float = 0.0
    ceiling_z: float | None = DEFAULT_CEILING

    @property
    def floor(self) -> list[Pt]:
        return self.box.floor() if self.box is not None else list(self.polygon or [])

    @property
    def is_box(self) -> bool:
        return self.box is not None

    @property
    def area(self) -> float:
        return geom.area(self.floor)

    @property
    def center(self) -> Pt:
        return self.box.center if self.box else geom.centroid(self.floor)


@dataclass
class SpaceDoor:
    id: str
    rooms: list[str]
    center: Pt
    width: float = DEFAULT_DOOR_W
    height: float = DEFAULT_DOOR_H


@dataclass
class Issue:
    level: str             # "error" | "warn"
    kind: str              # overlap | door_off_wall | door_rooms | door_too_wide | small_room | no_ceiling
    message: str
    ids: list[str] = field(default_factory=list)


class SpaceModel:
    def __init__(self, source: dict[str, Any] | None = None):
        self.rooms: list[SpaceRoom] = []
        self.doors: list[SpaceDoor] = []
        self.source: dict[str, Any] = dict(source or {"kind": "boxes"})

    # ---------- 찾기 ----------
    def room(self, rid: str) -> SpaceRoom:
        for r in self.rooms:
            if r.id == rid:
                return r
        raise KeyError(f"방 없음: {rid}")

    def door(self, did: str) -> SpaceDoor:
        for d in self.doors:
            if d.id == did:
                return d
        raise KeyError(f"문 없음: {did}")

    def room_at(self, x: float, y: float) -> SpaceRoom | None:
        for r in self.rooms:
            if geom.point_in_poly((x, y), r.floor):
                return r
        return None

    def _new_id(self, prefix: str, taken: set[str]) -> str:
        n = 1
        while f"{prefix}{n}" in taken:
            n += 1
        return f"{prefix}{n}"

    def _check_id(self, rid: str, taken: set[str]):
        if not rid or rid in taken:
            raise ValueError(f"id 가 비었거나 중복: {rid!r}")
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", rid):
            raise ValueError(f"id 는 영문·숫자·_- 만: {rid!r}")

    # ---------- 방 편집 ----------
    def add_box_room(self, center: Pt, size: Pt, yaw_deg: float = 0.0, *, id: str | None = None, name: str = "",
                     floor_z: float = 0.0, ceiling_z: float | None = DEFAULT_CEILING) -> str:
        if size[0] <= 0 or size[1] <= 0:
            raise ValueError(f"크기는 양수: {size}")
        taken = {r.id for r in self.rooms}
        rid = id or self._new_id("room", taken)
        self._check_id(rid, taken)
        self._check_z(floor_z, ceiling_z)
        self.rooms.append(SpaceRoom(rid, name, Box((float(center[0]), float(center[1])),
                                                   (float(size[0]), float(size[1])), float(yaw_deg)),
                                    None, float(floor_z), ceiling_z))
        return rid

    def add_polygon_room(self, points: list[Pt], *, id: str | None = None, name: str = "",
                         floor_z: float = 0.0, ceiling_z: float | None = DEFAULT_CEILING) -> str:
        pts = [(float(x), float(y)) for x, y in points]
        if len(pts) >= 2 and pts[0] == pts[-1]:
            pts = pts[:-1]
        if len(pts) < 3 or geom.area(pts) <= 1e-9:
            raise ValueError("다각형은 점 3개 이상, 넓이 > 0")
        taken = {r.id for r in self.rooms}
        rid = id or self._new_id("room", taken)
        self._check_id(rid, taken)
        self._check_z(floor_z, ceiling_z)
        self.rooms.append(SpaceRoom(rid, name, None, geom.ccw(pts), float(floor_z), ceiling_z))
        return rid

    @staticmethod
    def _check_z(floor_z, ceiling_z):
        if ceiling_z is not None and ceiling_z <= floor_z:
            raise ValueError("ceiling_z 가 floor_z 보다 커야 함")

    def move_room(self, rid: str, dx: float, dy: float, move_doors: bool = True) -> str:
        """방을 (dx, dy) 만큼 옮김. move_doors 면 이 방에만 붙은 문도 같이 옮김."""
        r = self.room(rid)
        if r.box:
            r.box.center = (r.box.center[0] + dx, r.box.center[1] + dy)
        else:
            r.polygon = [(x + dx, y + dy) for x, y in r.polygon]
        if move_doors:
            for d in self.doors:
                if d.rooms == [rid]:
                    d.center = (d.center[0] + dx, d.center[1] + dy)
        return rid

    def set_box(self, rid: str, center: Pt | None = None, size: Pt | None = None, yaw_deg: float | None = None) -> str:
        """박스 방의 중심·크기·회전 바꾸기 (None 이면 그대로)."""
        r = self.room(rid)
        if r.box is None:
            raise ValueError(f"{rid}: 박스 방이 아님 (다각형)")
        if center is not None:
            r.box.center = (float(center[0]), float(center[1]))
        if size is not None:
            if size[0] <= 0 or size[1] <= 0:
                raise ValueError(f"크기는 양수: {size}")
            r.box.size = (float(size[0]), float(size[1]))
        if yaw_deg is not None:
            r.box.yaw_deg = float(yaw_deg)
        return rid

    def resize_room(self, rid: str, size: Pt) -> str:
        return self.set_box(rid, size=size)

    def rotate_room(self, rid: str, yaw_deg: float) -> str:
        """박스는 회전값을 정함. 다각형은 중심 기준으로 yaw_deg 만큼 돌림."""
        r = self.room(rid)
        if r.box:
            return self.set_box(rid, yaw_deg=yaw_deg)
        cx, cy = geom.centroid(r.polygon)
        c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
        r.polygon = [(cx + c * (x - cx) - s * (y - cy), cy + s * (x - cx) + c * (y - cy)) for x, y in r.polygon]
        return rid

    def set_polygon(self, rid: str, points: list[Pt]) -> str:
        """방 모양을 다각형으로 바꿈 (박스 정보는 사라짐)."""
        r = self.room(rid)
        pts = [(float(x), float(y)) for x, y in points]
        if len(pts) < 3 or geom.area(pts) <= 1e-9:
            raise ValueError("다각형은 점 3개 이상, 넓이 > 0")
        r.box, r.polygon = None, geom.ccw(pts)
        return rid

    def box_from_polygon(self, rid: str, ratio: float = 0.9) -> bool:
        """거의 직사각형인 다각형 방을 박스로 바꿈. 바꿨으면 True."""
        r = self.room(rid)
        if r.box:
            return True
        fit = geom.rect_fit(r.polygon, ratio)
        if fit is None:
            return False
        center, size, yaw = fit
        r.box, r.polygon = Box(center, size, round(yaw, 6)), None
        return True

    def set_heights(self, rid: str, ceiling_z: float | None = None, floor_z: float | None = None) -> str:
        r = self.room(rid)
        fz = r.floor_z if floor_z is None else float(floor_z)
        cz = r.ceiling_z if ceiling_z is None else float(ceiling_z)
        self._check_z(fz, cz)
        r.floor_z, r.ceiling_z = fz, cz
        return rid

    def set_ceiling_all(self, ceiling_z: float) -> None:
        for r in self.rooms:
            self.set_heights(r.id, ceiling_z=ceiling_z)

    def rename_room(self, rid: str, name: str) -> str:
        self.room(rid).name = name
        return rid

    def delete_room(self, rid: str) -> list[str]:
        """방을 지움. 그 방만 쓰던 문은 지우고, 두 방을 잇던 문은 남은 방만 가리키게. 지운 문 id 목록."""
        self.rooms.remove(self.room(rid))
        removed = []
        for d in list(self.doors):
            if rid in d.rooms:
                d.rooms = [x for x in d.rooms if x != rid]
                if not d.rooms:
                    self.doors.remove(d)
                    removed.append(d.id)
        return removed

    # ---------- 문 ----------
    def walls_near(self, p: Pt, tol: float = 0.3) -> list[tuple[str, int, Pt, float]]:
        """점에서 tol 안의 벽: (방 id, 변 번호, 벽 위 점, 거리), 가까운 순."""
        out = []
        for r in self.rooms:
            i, q, d = geom.nearest_edge(p, r.floor)
            if d <= tol:
                out.append((r.id, i, q, d))
        return sorted(out, key=lambda t: t[3])

    def snap_to_wall(self, p: Pt) -> tuple[str, int, Pt, float]:
        """가장 가까운 벽 (방 id, 변 번호, 벽 위 점, 거리)."""
        best = None
        for r in self.rooms:
            i, q, d = geom.nearest_edge(p, r.floor)
            if best is None or d < best[3]:
                best = (r.id, i, q, d)
        if best is None:
            raise ValueError("방이 없음")
        return best

    def add_door(self, point: Pt, width: float = DEFAULT_DOOR_W, height: float = DEFAULT_DOOR_H, *,
                 id: str | None = None, rooms: list[str] | None = None, snap: bool = True,
                 wall_tol: float = 0.3) -> str:
        """문 추가. snap 이면 가장 가까운 벽 위로 옮기고, rooms 를 안 주면 그 자리 벽에 닿은 방(최대 2개)을 연결."""
        if width <= 0 or height <= 0:
            raise ValueError("문 폭·높이는 양수")
        p = (float(point[0]), float(point[1]))
        if snap:
            rid, _, p, _ = self.snap_to_wall(p)
        if rooms is None:
            rooms = [w[0] for w in self.walls_near(p, wall_tol)][:2]
        for x in rooms:
            self.room(x)
        taken = {d.id for d in self.doors}
        did = id or self._new_id("door", taken)
        self._check_id(did, taken)
        self.doors.append(SpaceDoor(did, list(rooms), p, float(width), float(height)))
        return did

    def move_door(self, did: str, point: Pt, snap: bool = True, relink: bool = True, wall_tol: float = 0.3) -> str:
        d = self.door(did)
        p = (float(point[0]), float(point[1]))
        if snap:
            p = self.snap_to_wall(p)[2]
        d.center = p
        if relink:
            d.rooms = [w[0] for w in self.walls_near(p, wall_tol)][:2]
        return did

    def set_door(self, did: str, width: float | None = None, height: float | None = None,
                 rooms: list[str] | None = None) -> str:
        d = self.door(did)
        if width is not None:
            d.width = float(width)
        if height is not None:
            d.height = float(height)
        if rooms is not None:
            for x in rooms:
                self.room(x)
            d.rooms = list(rooms)
        return did

    def delete_door(self, did: str) -> None:
        self.doors.remove(self.door(did))

    def shared_walls(self, a: str, b: str, tol: float = 0.3, min_len: float = 0.5) -> list[tuple[Pt, Pt]]:
        """두 방 벽이 맞닿은 구간 (a 방 벽 위 좌표)."""
        return geom.shared_walls(self.room(a).floor, self.room(b).floor, tol, min_len)

    def adjacent_pairs(self, tol: float = 0.3, min_len: float = 0.5) -> list[tuple[str, str, float]]:
        """벽이 맞닿은 방 쌍과 맞닿은 길이."""
        out = []
        for i, a in enumerate(self.rooms):
            for b in self.rooms[i + 1:]:
                segs = geom.shared_walls(a.floor, b.floor, tol, min_len)
                if segs:
                    out.append((a.id, b.id, sum(math.dist(*s) for s in segs)))
        return out

    def connect(self, a: str, b: str, width: float = DEFAULT_DOOR_W, height: float = DEFAULT_DOOR_H,
                tol: float = 0.3, id: str | None = None) -> str:
        """두 방이 맞닿은 가장 긴 벽 가운데에 문을 만듦."""
        segs = self.shared_walls(a, b, tol, min_len=min(width, 0.5))
        if not segs:
            raise ValueError(f"{a} 와 {b} 는 맞닿은 벽이 없음 (허용 {tol}m)")
        p, q = max(segs, key=lambda s: math.dist(*s))
        if math.dist(p, q) < width:
            width = math.dist(p, q)
        mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        return self.add_door(mid, width, height, id=id or self._new_id(f"{a}_{b}_", {d.id for d in self.doors}),
                             rooms=[a, b], snap=False)

    def connect_all(self, width: float = DEFAULT_DOOR_W, tol: float = 0.3, skip_linked: bool = True) -> list[str]:
        """맞닿은 모든 방 쌍에 문을 만듦 (이미 문으로 이어진 쌍은 건너뜀). 만든 문 id."""
        linked = {frozenset(d.rooms) for d in self.doors if len(d.rooms) == 2}
        made = []
        for a, b, _ in self.adjacent_pairs(tol, min_len=min(width, 0.5)):
            if skip_linked and frozenset((a, b)) in linked:
                continue
            made.append(self.connect(a, b, width, tol=tol))
        return made

    # ---------- 검사 ----------
    def validate(self, overlap_tol_m2: float = 0.05, wall_tol: float = 0.15, min_room_m2: float = 0.5) -> list[Issue]:
        issues: list[Issue] = []
        for r in self.rooms:
            if r.area < min_room_m2:
                issues.append(Issue("warn", "small_room", f"{r.id}: 넓이 {r.area:.2f}m² 로 너무 작음", [r.id]))
            if r.ceiling_z is None:
                issues.append(Issue("warn", "no_ceiling", f"{r.id}: 천장 높이 없음", [r.id]))
        for i, a in enumerate(self.rooms):
            for b in self.rooms[i + 1:]:
                ov = geom.overlap_area(a.floor, b.floor)
                if ov > overlap_tol_m2:
                    issues.append(Issue("error", "overlap", f"{a.id} 와 {b.id} 가 {ov:.2f}m² 겹침", [a.id, b.id]))
        for d in self.doors:
            if not d.rooms:
                issues.append(Issue("warn", "door_rooms", f"{d.id}: 연결된 방이 없음", [d.id]))
            for rid in d.rooms:
                try:
                    r = self.room(rid)
                except KeyError:
                    issues.append(Issue("error", "door_rooms", f"{d.id}: 없는 방 {rid}", [d.id]))
                    continue
                i, _, dist = geom.nearest_edge(d.center, r.floor)
                if dist > wall_tol:
                    issues.append(Issue("error", "door_off_wall",
                                        f"{d.id}: {rid} 벽에서 {dist:.2f}m 떨어짐", [d.id, rid]))
                else:
                    a, b = geom.edges(r.floor)[i]
                    if d.width > math.dist(a, b) + 1e-6:
                        issues.append(Issue("warn", "door_too_wide",
                                            f"{d.id}: 폭 {d.width}m 가 벽 길이 {math.dist(a, b):.2f}m 보다 넓음", [d.id, rid]))
        return issues

    # ---------- 변환·저장 ----------
    def to_plan(self) -> schemas.Plan:
        rooms = [schemas.Room(r.id, r.floor, r.floor_z, r.ceiling_z, r.name,
                              {"center": r.box.center, "size": r.box.size, "yaw_deg": r.box.yaw_deg} if r.box else None)
                 for r in self.rooms]
        doors = [schemas.Door(d.id, list(d.rooms), d.center, d.width, d.height) for d in self.doors]
        return schemas.Plan(rooms, doors, dict(self.source))

    @classmethod
    def from_plan(cls, plan: schemas.Plan) -> "SpaceModel":
        m = cls(plan.source)
        for r in plan.rooms:
            box = None
            if getattr(r, "box", None):
                b = r.box
                box = Box(tuple(map(float, b["center"])), tuple(map(float, b["size"])), float(b.get("yaw_deg", 0)))
            m.rooms.append(SpaceRoom(r.id, r.name, box, None if box else geom.ccw(list(r.floor)),
                                     r.floor_z, r.ceiling_z))
        for d in plan.doors:
            m.doors.append(SpaceDoor(d.id, list(d.rooms), tuple(d.center), d.width, d.height))
        return m

    def to_json(self) -> dict:
        return self.to_plan().to_json()

    @classmethod
    def from_json(cls, d: dict) -> "SpaceModel":
        return cls.from_plan(schemas.parse_plan(d))

    def save(self, path: str | Path) -> Path:
        """plan.json 으로 저장 (방이 없으면 schemas 검사에서 막히므로 ValueError)."""
        if not self.rooms:
            raise ValueError("방이 하나도 없음")
        data = self.to_json()
        schemas.parse_plan(data)          # 저장 전에 형식 검사
        schemas.write_json(path, data)
        return Path(path)

    @classmethod
    def load(cls, path: str | Path) -> "SpaceModel":
        return cls.from_plan(schemas.load_plan(path))

    def bounds(self) -> tuple[float, float, float, float]:
        pts = [p for r in self.rooms for p in r.floor]
        return geom.bbox(pts) if pts else (0.0, 0.0, 0.0, 0.0)

    def translate(self, dx: float, dy: float) -> None:
        """방·문 전체를 옮김 (도면 원점 맞추기)."""
        for r in self.rooms:
            self.move_room(r.id, dx, dy, move_doors=False)
        for d in self.doors:
            d.center = (d.center[0] + dx, d.center[1] + dy)
