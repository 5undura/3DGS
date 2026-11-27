"""plan.json · path.json · priors.json · diagnose.json 형식 (CONTRACT.md 4절).

읽을 때 검사하고, 방의 box 는 floor 다각형으로 바꿔 둔다.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

VERSION = 1


class SchemaError(ValueError):
    pass


def _xy(v, where) -> tuple[float, float]:
    if not (isinstance(v, (list, tuple)) and len(v) == 2):
        raise SchemaError(f"{where}: [x, y] 두 숫자여야 함")
    return float(v[0]), float(v[1])


def _check_version(d: dict, name: str):
    if d.get("version") != VERSION:
        raise SchemaError(f"{name}: version {d.get('version')!r} (지원: {VERSION})")


# ---------------- plan.json ----------------
@dataclass
class Room:
    id: str
    floor: list[tuple[float, float]]   # 바닥 다각형, 반시계
    floor_z: float = 0.0
    ceiling_z: float | None = None
    name: str = ""

    def contains(self, x: float, y: float) -> bool:
        """점이 바닥 다각형 안에 있는지 (경계 포함 아님, 짝홀 규칙)."""
        inside = False
        pts = self.floor
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            if (y1 > y) != (y2 > y):
                xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x < xc:
                    inside = not inside
        return inside


@dataclass
class Door:
    id: str
    rooms: list[str]
    center: tuple[float, float]
    width: float
    height: float = 2.1


@dataclass
class Plan:
    rooms: list[Room]
    doors: list[Door] = field(default_factory=list)
    source: dict[str, Any] = field(default_factory=lambda: {"kind": "boxes"})

    def room_at(self, x: float, y: float) -> Room | None:
        for r in self.rooms:
            if r.contains(x, y):
                return r
        return None

    def to_json(self) -> dict:
        return {"version": VERSION, "units": "m",
                "rooms": [{"id": r.id, "name": r.name, "floor": [list(p) for p in r.floor],
                           "floor_z": r.floor_z, "ceiling_z": r.ceiling_z} for r in self.rooms],
                "doors": [asdict(d) | {"center": list(d.center)} for d in self.doors],
                "source": self.source}


def box_to_floor(center, size, yaw_deg=0.0) -> list[tuple[float, float]]:
    cx, cy = center
    w, d = size
    if w <= 0 or d <= 0:
        raise SchemaError(f"box size {size}: 양수여야 함")
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    pts = []
    for lx, ly in ((-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2)):  # 반시계
        pts.append((cx + c * lx - s * ly, cy + s * lx + c * ly))
    return pts


def parse_plan(d: dict) -> Plan:
    _check_version(d, "plan.json")
    if d.get("units", "m") != "m":
        raise SchemaError("plan.json: units 는 m 만 지원")
    rooms, ids = [], set()
    for i, r in enumerate(d.get("rooms") or []):
        where = f"plan.json rooms[{i}]"
        rid = r.get("id")
        if not rid or rid in ids:
            raise SchemaError(f"{where}: id 가 없거나 중복")
        ids.add(rid)
        if ("box" in r) == ("floor" in r):
            raise SchemaError(f"{where}: box 와 floor 중 하나만")
        if "box" in r:
            b = r["box"]
            floor = box_to_floor(_xy(b.get("center"), where + ".box.center"),
                                 _xy(b.get("size"), where + ".box.size"), float(b.get("yaw_deg", 0)))
        else:
            floor = [_xy(p, f"{where}.floor") for p in r["floor"]]
            if len(floor) < 3:
                raise SchemaError(f"{where}: floor 는 점 3개 이상")
            area2 = sum(floor[k][0] * floor[(k + 1) % len(floor)][1] - floor[(k + 1) % len(floor)][0] * floor[k][1]
                        for k in range(len(floor)))
            if area2 < 0:
                floor = floor[::-1]  # 시계 방향이면 반시계로
        fz = float(r.get("floor_z", 0.0))
        cz = r.get("ceiling_z")
        if cz is not None and float(cz) <= fz:
            raise SchemaError(f"{where}: ceiling_z 가 floor_z 보다 커야 함")
        rooms.append(Room(rid, floor, fz, None if cz is None else float(cz), r.get("name", "")))
    if not rooms:
        raise SchemaError("plan.json: rooms 가 비어 있음")
    doors = []
    for i, dr in enumerate(d.get("doors") or []):
        where = f"plan.json doors[{i}]"
        for rid in dr.get("rooms", []):
            if rid not in ids:
                raise SchemaError(f"{where}: 없는 방 {rid}")
        doors.append(Door(dr["id"], list(dr.get("rooms", [])), _xy(dr.get("center"), where + ".center"),
                          float(dr["width"]), float(dr.get("height", 2.1))))
    return Plan(rooms, doors, d.get("source") or {"kind": "boxes"})


# ---------------- path.json ----------------
@dataclass
class Anchor:
    t: float
    point: tuple[float, float] | None = None
    s: float | None = None


@dataclass
class CameraPath:
    anchors: list[Anchor]
    stroke: list[tuple[float, float]] | None = None
    camera_height: float = 1.4
    sigma_xy: float = 0.5
    sigma_z: float = 0.3


def parse_path(d: dict) -> CameraPath:
    _check_version(d, "path.json")
    anchors = []
    for i, a in enumerate(d.get("anchors") or []):
        where = f"path.json anchors[{i}]"
        if ("point" in a) == ("s" in a):
            raise SchemaError(f"{where}: point 와 s 중 하나만")
        anchors.append(Anchor(float(a["t"]),
                              _xy(a["point"], where + ".point") if "point" in a else None,
                              float(a["s"]) if "s" in a else None))
    if len(anchors) < 2:
        raise SchemaError("path.json: 고정점(anchors)은 최소 2개 (시작·끝)")
    stroke = d.get("stroke")
    if stroke is not None:
        stroke = [_xy(p, "path.json stroke") for p in stroke]
        if len(stroke) < 2:
            raise SchemaError("path.json: stroke 는 점 2개 이상")
    elif any(a.point is None for a in anchors):
        raise SchemaError("path.json: stroke 가 없으면 모든 고정점에 point 가 있어야 함")
    return CameraPath(anchors, stroke, float(d.get("camera_height", 1.4)),
                      float(d.get("sigma_xy", 0.5)), float(d.get("sigma_z", 0.3)))


# ---------------- 공통 읽기·쓰기 ----------------
def read_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    tmp.replace(path)


def load_plan(path) -> Plan:
    return parse_plan(read_json(path))


def load_path(path) -> CameraPath:
    return parse_path(read_json(path))
