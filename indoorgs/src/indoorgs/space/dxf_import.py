"""DXF 도면 -> 초안 공간 (ezdxf 1.4.5 로 확인). `pip install indoorgs[space]`.

LINE, LWPOLYLINE, POLYLINE 만 읽는다. 닫힌 폴리라인은 방 후보, 흩어진 선은 끝점을 맞춰 닫힌 다각형으로.
단위는 $INSUNITS 를 m 로 바꾼다 (0=단위 없음이면 unitless_scale 사용, 기본 mm 로 봄).
호(bulge)·곡선은 직선으로 펴고 경고한다. 블록(INSERT) 안 선은 읽지 않는다.
"""
from __future__ import annotations

from pathlib import Path

from .draft import DraftOptions, DraftResult, build_draft
from .geom import Pt, Seg

# $INSUNITS 코드 -> m (ezdxf.units 와 같은 값)
INSUNITS_M = {1: 0.0254, 2: 0.3048, 3: 1609.344, 4: 0.001, 5: 0.01, 6: 1.0, 7: 1000.0,
              8: 2.54e-8, 9: 2.54e-5, 10: 0.9144, 14: 0.1, 15: 10.0}


def import_dxf(path: str | Path, opts: DraftOptions | None = None, *, layers: list[str] | None = None,
               unitless_scale: float = 0.001, scale: float | None = None) -> DraftResult:
    """DXF 파일 -> DraftResult. layers 를 주면 그 레이어만. scale 을 주면 단위 무시하고 그 배율(m/도면단위)."""
    try:
        import ezdxf
    except ImportError as e:
        raise ImportError("DXF 를 읽으려면 ezdxf 가 필요함: pip install indoorgs[space]") from e
    doc = ezdxf.readfile(str(path))
    return import_dxf_doc(doc, opts, layers=layers, unitless_scale=unitless_scale, scale=scale, name=str(path))


def import_dxf_doc(doc, opts: DraftOptions | None = None, *, layers: list[str] | None = None,
                   unitless_scale: float = 0.001, scale: float | None = None, name: str = "") -> DraftResult:
    opts = opts or DraftOptions()
    warnings: list[str] = []
    code = int(doc.header.get("$INSUNITS", 0) or 0)
    if scale is None:
        if code in INSUNITS_M:
            scale = INSUNITS_M[code]
        else:
            scale = unitless_scale
            warnings.append(f"$INSUNITS={code} (단위 없음/모름): 1 도면단위 = {unitless_scale}m 로 봄. 크기가 이상하면 scale 을 줄 것")
    layer_set = {l.lower() for l in layers} if layers else None

    def m(p) -> Pt:
        return float(p[0]) * scale, float(p[1]) * scale

    polygons: list[list[Pt]] = []
    segments: list[Seg] = []
    bulges = skipped = 0
    for e in doc.modelspace():
        t = e.dxftype()
        if layer_set is not None and e.dxf.get("layer", "0").lower() not in layer_set:
            continue
        if t == "LINE":
            segments.append((m(e.dxf.start), m(e.dxf.end)))
        elif t in ("LWPOLYLINE", "POLYLINE"):
            if t == "LWPOLYLINE":
                raw = list(e.get_points("xyb"))
                pts = [m(p) for p in raw]
                bulges += sum(1 for p in raw if abs(p[2]) > 1e-9)
                closed = e.closed
            else:
                if e.is_poly_face_mesh or e.is_polygon_mesh:
                    skipped += 1
                    continue
                pts = [m(v.dxf.location) for v in e.vertices]
                bulges += sum(1 for v in e.vertices if abs(v.dxf.get("bulge", 0)) > 1e-9)
                closed = e.is_closed
            if len(pts) >= 2 and pts[0] == pts[-1]:
                pts, closed = pts[:-1], True
            if closed and len(pts) >= 3:
                polygons.append(pts)
            else:
                segments += list(zip(pts, pts[1:]))
        else:
            skipped += 1
    if bulges:
        warnings.append(f"호(bulge) {bulges}곳을 직선으로 폄")
    if skipped:
        warnings.append(f"LINE/LWPOLYLINE/POLYLINE 아닌 요소 {skipped}개 무시 (블록·호·글자 등)")
    source = {"kind": "dxf", "file": Path(name).name if name else "", "insunits": code, "scale_m": scale}
    res = build_draft(polygons, segments, opts, source, warnings)
    res.extra["n_polylines_closed"] = len(polygons)
    res.extra["n_segments"] = len(segments)
    return res
