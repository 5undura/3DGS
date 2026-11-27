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


def test_loop_back_to_door_anchor_snaps_forward():
    # 방을 한 바퀴 돌아 문(시작점) 0.3m 옆으로 나옴. 끝 고정점이 선 시작 쪽(s=0.3)에 붙으면 마지막 구간이
    # 거꾸로 걸은 것이 되어 5m 가까이 틀렸음 (experiments/accuracy/pathing_eval.py)
    stroke = [[0, 0], [0, 5], [5, 5], [5, 0.3], [0, 0.3]]
    m = PathModel(mk(stroke, [{"t": 0, "point": [0, 0]}, {"t": 15, "point": [5, 5]},
                              {"t": 22.5, "point": [2.5, 0.3]}, {"t": 25, "point": [0, 0.3]}]))
    assert m.ss[-1] == pytest.approx(m.line.length)
    assert m.position(24)[:2] == pytest.approx((1.0, 0.3))
    assert not any("거꾸로" in w for w in m.warnings)


def test_real_backwards_anchor_still_warns():
    # 앞쪽에 가까운 후보가 없으면 예전처럼 가장 가까운 곳 + 경고
    m = PathModel(mk([[0, 0], [10, 0]], [{"t": 0, "point": [0, 0]}, {"t": 5, "point": [8, 0]}, {"t": 9, "point": [2, 0]}]))
    assert m.ss == pytest.approx([0, 8, 2])
    assert any("거꾸로" in w for w in m.warnings)
