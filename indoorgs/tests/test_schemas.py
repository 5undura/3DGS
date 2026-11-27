from pathlib import Path

import pytest

from indoorgs import schemas

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo"


def test_demo_plan():
    plan = schemas.load_plan(DEMO / "floorplan" / "plan.json")
    assert {r.id for r in plan.rooms} == {"corridor1", "b117"}
    assert plan.room_at(-2.9, 5).id == "corridor1"
    assert plan.room_at(3, 6).id == "b117"
    assert plan.room_at(20, 20) is None


def test_box_rotation_and_ccw():
    pts = schemas.box_to_floor((0, 0), (4, 2), 90)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    assert max(xs) == pytest.approx(1) and max(ys) == pytest.approx(2)
    cw = {"version": 1, "rooms": [{"id": "r", "floor": [[0, 0], [0, 1], [1, 1], [1, 0]]}]}
    room = schemas.parse_plan(cw).rooms[0]
    assert room.contains(0.5, 0.5)


@pytest.mark.parametrize("bad, msg", [
    ({"version": 2, "rooms": []}, "version"),
    ({"version": 1, "rooms": []}, "비어"),
    ({"version": 1, "rooms": [{"id": "r", "box": {"center": [0, 0], "size": [1, 1]}, "floor": [[0, 0]]}]}, "하나만"),
    ({"version": 1, "rooms": [{"id": "r", "box": {"center": [0, 0], "size": [1, 1]}, "floor_z": 1, "ceiling_z": 0.5}]}, "ceiling_z"),
    ({"version": 1, "rooms": [{"id": "r", "box": {"center": [0, 0], "size": [1, 1]}}],
      "doors": [{"id": "d", "rooms": ["x"], "center": [0, 0], "width": 1}]}, "없는 방"),
])
def test_bad_plans(bad, msg):
    with pytest.raises(schemas.SchemaError, match=msg):
        schemas.parse_plan(bad)


def test_path_needs_two_anchors():
    with pytest.raises(schemas.SchemaError, match="최소 2개"):
        schemas.parse_path({"version": 1, "anchors": [{"t": 0, "point": [0, 0]}]})
