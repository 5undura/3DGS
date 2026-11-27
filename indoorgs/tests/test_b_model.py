import pytest

from indoorgs import schemas
from indoorgs.space import SpaceModel


def two_rooms():
    m = SpaceModel()
    m.add_box_room((0, 0), (10, 2.4), id="corr", ceiling_z=2.7)       # x -5..5, y -1.2..1.2
    m.add_box_room((-2, 3.2), (4, 4), id="r1", ceiling_z=2.6)          # y 1.2..5.2 : 복도와 맞닿음
    return m


def test_add_edit_validate():
    m = two_rooms()
    assert m.room_at(0, 0).id == "corr" and m.room_at(-2, 3).id == "r1"
    assert m.validate() == []
    m.move_room("r1", 0, -1)                                              # 복도와 겹침
    assert any(i.kind == "overlap" and set(i.ids) == {"corr", "r1"} for i in m.validate())
    m.move_room("r1", 0, 1)
    m.set_box("r1", size=(4, 4), yaw_deg=90)
    assert m.room("r1").box.yaw_deg == 90
    m.rotate_room("r1", 0)
    m.set_heights("corr", ceiling_z=3.0)
    with pytest.raises(ValueError):
        m.set_heights("corr", ceiling_z=-1)
    with pytest.raises(ValueError):
        m.add_box_room((0, 0), (1, 1), id="corr")


def test_door_snap_and_auto_link():
    m = two_rooms()
    d = m.add_door((-2.3, 1.0), width=1.0)                                # 벽 근처 찍음
    door = m.door(d)
    assert door.center == pytest.approx((-2.3, 1.2))
    assert set(door.rooms) == {"corr", "r1"}
    assert m.validate() == []
    m.door(d).center = (-2.3, 0.0)                                        # 벽에서 떨어뜨림
    assert any(i.kind == "door_off_wall" for i in m.validate())
    m.delete_door(d)


def test_connect_shared_wall():
    m = two_rooms()
    assert m.adjacent_pairs()[0][:2] == ("corr", "r1")
    did = m.connect("corr", "r1", width=0.9)
    assert m.door(did).center == pytest.approx((-2, 1.2))
    assert m.connect_all() == []                                          # 이미 이어짐
    m.add_box_room((10, 10), (2, 2), id="far")
    with pytest.raises(ValueError):
        m.connect("corr", "far")
    removed = m.delete_room("r1")
    assert removed == [] and m.door(did).rooms == ["corr"]


def test_save_load_roundtrip_keeps_box(tmp_path):
    m = two_rooms()
    m.add_polygon_room([(5, -1.2), (8, -1.2), (8, 3), (6, 3), (6, 1.2), (5, 1.2)], id="L", name="ㄱ자 방")
    m.connect_all()
    f = m.save(tmp_path / "floorplan" / "plan.json")
    raw = schemas.read_json(f)
    assert "box" in raw["rooms"][0] and "floor" not in raw["rooms"][0]
    assert "floor" in raw["rooms"][2]
    m2 = SpaceModel.load(f)
    assert m2.room("corr").is_box and m2.room("corr").box.size == (10, 2.4)
    assert not m2.room("L").is_box and m2.room("L").name == "ㄱ자 방"
    assert len(m2.doors) == len(m.doors) >= 2
    assert m2.to_json() == m.to_json()
    # schemas 로 읽어도 floor 가 있고, 다시 써도 box 유지
    plan = schemas.load_plan(f)
    assert plan.room_at(0, 0).id == "corr" and plan.rooms[0].box is not None
    assert "box" in plan.to_json()["rooms"][0]


def test_box_from_polygon():
    m = SpaceModel()
    rid = m.add_polygon_room([(0, 0), (4, 0), (4, 3), (0, 3)])
    assert m.box_from_polygon(rid)
    b = m.room(rid).box
    assert b.center == pytest.approx((2, 1.5)) and sorted(b.size) == pytest.approx([3, 4])
