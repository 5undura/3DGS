import json
from pathlib import Path

import pytest

from indoorgs import schemas
from indoorgs.pathing import PathModel, build_priors, frame_time

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo"


def mk(stroke, anchors):
    return schemas.parse_path({"version": 1, "stroke": stroke, "anchors": anchors})


def test_follows_stroke_not_straight_line():
    # L 자 선: (0,0) -> (10,0) -> (10,10), 0초에 시작, 20초에 끝 -> 10초에는 모퉁이
    m = PathModel(mk([[0, 0], [10, 0], [10, 10]], [{"t": 0, "point": [0, 0]}, {"t": 20, "point": [10, 10]}]))
    x, y, z = m.position(10)
    assert (x, y) == pytest.approx((10, 0))
    assert z == 1.4


def test_stop_and_clamp():
    m = PathModel(mk([[0, 0], [10, 0]], [{"t": 5, "s": 0}, {"t": 10, "s": 5}, {"t": 15, "s": 5}, {"t": 20, "s": 10}]))
    assert m.position(0)[:2] == pytest.approx((0, 0))      # 첫 고정점 전: 멈춤
    assert m.position(12)[:2] == pytest.approx((5, 0))     # 멈춘 구간
    assert m.position(17.5)[:2] == pytest.approx((7.5, 0))
    assert m.position(99)[:2] == pytest.approx((10, 0))    # 끝 뒤: 멈춤


def test_no_stroke_uses_anchor_points():
    m = PathModel(schemas.parse_path({"version": 1, "anchors": [{"t": 0, "point": [0, 0]}, {"t": 4, "point": [4, 3]}]}))
    assert m.position(2)[:2] == pytest.approx((2, 1.5))


def test_warnings_for_far_anchor_and_backwards():
    m = PathModel(mk([[0, 0], [10, 0]], [{"t": 0, "point": [0, 3]}, {"t": 5, "point": [8, 0]}, {"t": 9, "point": [2, 0]}]))
    assert any("떨어져" in w for w in m.warnings)
    assert any("거꾸로" in w for w in m.warnings)


def test_frame_time_and_priors():
    assert frame_time("f_00012_000240.jpg", 24.0) == pytest.approx(10.0)
    with pytest.raises(ValueError):
        frame_time("img_1.jpg", 24)
    cam = schemas.load_path(DEMO / "sections" / "room_b117" / "path.json")
    names = [f"f_{i + 1:05d}_{i * 24:06d}.jpg" for i in range(90)]  # 1초마다, 0~89초
    priors, warnings = build_priors(cam, names, 24.0)
    assert priors["coordinate_system"] == "CARTESIAN" and len(priors["frames"]) == 90
    assert priors["frames"][0]["xyz"] == pytest.approx([-2.9, 3.97, 1.4])
    assert priors["frames"][7]["xyz"][:2] == pytest.approx([0.5, 3.97])   # 6~8초 멈춤
    assert any("마지막 고정점" in w for w in warnings)                      # 80초 뒤 프레임 있음
    json.dumps(priors)


def test_demo_path_stays_inside_rooms():
    plan = schemas.load_plan(DEMO / "floorplan" / "plan.json")
    m = PathModel(schemas.load_path(DEMO / "sections" / "room_b117" / "path.json"))
    for t in range(0, 81):
        x, y, _ = m.position(t)
        assert plan.room_at(x, y) is not None, (t, x, y)
