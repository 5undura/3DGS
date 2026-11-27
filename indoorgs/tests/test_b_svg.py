import pytest

from indoorgs.space import DraftOptions, import_svg
from indoorgs.space.svg_import import parse_path_d

# 100 px = 1 m. viewBox 높이 600 -> y 뒤집기
SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 600" width="1200" height="600">
  <rect x="0" y="0" width="500" height="300" fill="none" stroke="black"/>
  <g transform="translate(600,0)">
    <polygon points="0,0 400,0 400,300 0,300"/>
  </g>
  <path d="M 0 350 H 300 V 550 H 0 Z"/>
  <polyline points="600,350 1000,350 1000,550"/>
  <path d="M1000 550 L 600 550 v -200"/>
  <text x="10" y="10">B117</text>
</svg>"""


def test_svg_rooms():
    res = import_svg(SVG, DraftOptions(), px_per_m=100)
    rooms = sorted(res.model.rooms, key=lambda r: (r.center[1], r.center[0]))
    assert len(rooms) == 4, res.warnings
    assert [round(r.area, 2) for r in rooms] == [6.0, 8.0, 15.0, 12.0]
    # 위쪽 사각형(svg y 0~300) 은 도면에서 y 3~6
    top_left = rooms[2]
    assert top_left.center == pytest.approx((2.5, 4.5))
    assert res.model.source["kind"] == "svg"


def test_svg_scale_points():
    res = import_svg(SVG, px_per_m=None, scale_points=((0, 0), (500, 0), 10.0))   # 500px = 10m
    assert res.model.source["px_per_m"] == pytest.approx(50)
    assert max(r.area for r in res.model.rooms) == pytest.approx(60.0)
    with pytest.raises(ValueError):
        import_svg(SVG)


def test_path_parser():
    subs, curves = parse_path_d("m0 0 l10 0 0 10 h-10 z M 20 20 C 1 1 2 2 30 30")
    assert subs[0] == ([(0, 0), (10, 0), (10, 10), (0, 10)], True)
    assert subs[1] == ([(20, 20), (30, 30)], False) and curves == 1
