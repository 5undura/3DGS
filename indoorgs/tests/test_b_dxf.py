import pytest

ezdxf = pytest.importorskip("ezdxf")

from indoorgs.space import DraftOptions, import_dxf


def make_dxf(path):
    doc = ezdxf.new()
    doc.header["$INSUNITS"] = 4        # mm
    msp = doc.modelspace()
    # 방 A 6x4m, 방 B 5x4m: 닫힌 폴리라인 (mm)
    msp.add_lwpolyline([(0, 0), (6000, 0), (6000, 4000), (0, 4000)], close=True)
    msp.add_lwpolyline([(7000, 0), (12000, 0), (12000, 4000), (7000, 4000)], close=True)
    # 방 C 4x3m: 흩어진 선, 끝점이 조금씩(10mm) 어긋나고 한 변은 두 조각
    msp.add_line((0, 5000), (2000, 5000))
    msp.add_line((2010, 5005), (4000, 5000))
    msp.add_line((4000, 5000), (4005, 8000))
    msp.add_line((4000, 8010), (0, 8000))
    msp.add_line((0, 8000), (0, 5008))
    # 작은 기둥(0.3x0.3) 과 건물 외곽 (모든 방을 품음)
    msp.add_lwpolyline([(3000, 1000), (3300, 1000), (3300, 1300), (3000, 1300)], close=True)
    msp.add_lwpolyline([(-500, -500), (12500, -500), (12500, 8500), (-500, 8500)], close=True)
    doc.saveas(path)


def test_dxf_rooms(tmp_path):
    f = tmp_path / "plan.dxf"
    make_dxf(f)
    res = import_dxf(f, DraftOptions(snap_tol=0.03))
    rooms = res.model.rooms
    assert len(rooms) == 3, res.warnings
    areas = sorted(r.area for r in rooms)
    assert areas == pytest.approx([12, 20, 24], abs=0.2)
    assert all(r.is_box for r in rooms)
    sizes = sorted(tuple(sorted(r.box.size)) for r in rooms)
    assert sizes[0] == pytest.approx((3, 4), abs=0.05)
    assert res.model.source["kind"] == "dxf" and res.model.source["scale_m"] == 0.001
    assert any("외곽" in w for w in res.warnings)
    assert any(r == "small" for r, _ in res.dropped)
    res.model.save(tmp_path / "out.json")           # 형식 검사 통과


def test_dxf_tight_tolerance_fails_to_close(tmp_path):
    f = tmp_path / "plan.dxf"
    make_dxf(f)
    res = import_dxf(f, DraftOptions(snap_tol=0.001, drop_outer=False))
    assert len(res.model.rooms) == 3                 # A, B, 외곽 (C 는 안 닫힘)
