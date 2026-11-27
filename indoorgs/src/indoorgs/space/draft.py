"""도면 불러오기 공통: 닫힌 다각형 + 흩어진 선분 -> 초안 SpaceModel + 경고.

DXF·SVG·사진 불러오기가 모두 이걸 쓴다. 결과는 사람이 박스 편집기에서 고칠 초안이다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from . import geom
from .geom import Pt, Seg
from .model import DEFAULT_CEILING, SpaceModel


@dataclass
class DraftOptions:
    snap_tol: float = 0.05          # 끝점 맞추기 허용 오차 (m)
    min_area: float = 1.0           # 이보다 작은 다각형 버림 (m²): 기둥·글자·기호
    max_area: float | None = None   # 이보다 큰 다각형 버림 (m²)
    min_thickness: float = 0.4      # 넓이/(둘레/2) 가 이보다 작으면 벽 두께로 보고 버림 (m)
    drop_outer: bool = True         # 다른 후보 2개 이상을 품은 다각형(건물 외곽) 버림
    boxes: bool = True              # 거의 직사각형이면 박스 방으로
    box_ratio: float = 0.92
    ceiling_z: float = DEFAULT_CEILING
    recenter: bool = False          # 전체 왼쪽 아래를 (0, 0) 으로 옮김
    id_prefix: str = "room"


@dataclass
class DraftResult:
    model: SpaceModel
    warnings: list[str] = field(default_factory=list)
    segments: list[Seg] = field(default_factory=list)   # 읽은 선분 (m, 도면 좌표). GUI 밑그림용
    dropped: list[tuple[str, list[Pt]]] = field(default_factory=list)  # (이유, 다각형)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def plan(self):
        return self.model.to_plan()


def _same_poly(a: list[Pt], b: list[Pt], tol: float) -> bool:
    if abs(geom.area(a) - geom.area(b)) > max(tol, 0.02 * geom.area(a)):
        return False
    return math.dist(geom.centroid(a), geom.centroid(b)) < max(tol * 4, 0.05)


def build_draft(polygons: list[list[Pt]], segments: list[Seg], opts: DraftOptions,
                source: dict[str, Any], warnings: list[str] | None = None) -> DraftResult:
    warnings = list(warnings or [])
    cands: list[list[Pt]] = []
    for p in polygons:
        p = geom.simplify_collinear(geom.ccw(p))
        if len(p) >= 3 and geom.area(p) > 1e-9:
            cands.append(p)
    if segments:
        faces = geom.faces_from_segments(segments, opts.snap_tol)
        for f in faces:
            if not any(_same_poly(f, c, opts.snap_tol) for c in cands):
                cands.append(f)
        if not faces:
            warnings.append(f"흩어진 선 {len(segments)}개에서 닫힌 방을 못 만듦 (끝점 허용 오차 {opts.snap_tol}m 를 늘려 볼 것)")
    dropped: list[tuple[str, list[Pt]]] = []
    keep = []
    for p in cands:
        a = geom.area(p)
        if a < opts.min_area:
            dropped.append(("small", p))
        elif opts.max_area is not None and a > opts.max_area:
            dropped.append(("large", p))
        elif a / (geom.perimeter(p) / 2) < opts.min_thickness:
            dropped.append(("thin", p))
        else:
            keep.append(p)
    if opts.drop_outer and len(keep) > 2:
        cents = [geom.centroid(p) for p in keep]
        outer = [i for i, p in enumerate(keep)
                 if sum(1 for j, c in enumerate(cents) if j != i and geom.point_in_poly(c, p)) >= 2]
        for i in outer:
            dropped.append(("outer", keep[i]))
        keep = [p for i, p in enumerate(keep) if i not in outer]
    # 남은 것끼리 서로 품으면(안쪽 기둥 등) 큰 쪽 경고
    for i, p in enumerate(keep):
        for j, q in enumerate(keep):
            if i != j and geom.point_in_poly(geom.centroid(q), p) and geom.area(q) < geom.area(p):
                warnings.append(f"방 후보가 다른 후보 안에 있음 (넓이 {geom.area(q):.1f}m² 가 {geom.area(p):.1f}m² 안). 확인 필요")
    keep.sort(key=lambda p: (-round(geom.centroid(p)[1] * 2), geom.centroid(p)[0]))   # 위에서 아래, 왼쪽에서 오른쪽
    m = SpaceModel(source)
    for k, p in enumerate(keep, 1):
        rid = m.add_polygon_room(p, id=f"{opts.id_prefix}{k}", ceiling_z=opts.ceiling_z)
        if opts.boxes:
            m.box_from_polygon(rid, opts.box_ratio)
    if not keep:
        warnings.append("방 후보가 없음. 단위·축척·최소 넓이를 확인할 것")
    n_small = sum(1 for r, _ in dropped if r == "small")
    if n_small:
        warnings.append(f"최소 넓이 {opts.min_area}m² 보다 작은 다각형 {n_small}개 뺌")
    if any(r == "outer" for r, _ in dropped):
        warnings.append("다른 방을 품은 외곽 다각형을 뺌 (건물 외곽선으로 봄)")
    segs = list(segments) + [e for p in polygons for e in geom.edges(list(p))]   # 밑그림용 전체 선
    if opts.recenter and m.rooms:
        x0, y0, _, _ = m.bounds()
        m.translate(-x0, -y0)
        segs = [((a[0] - x0, a[1] - y0), (b[0] - x0, b[1] - y0)) for a, b in segs]
        m.source["offset"] = [-x0, -y0]
    return DraftResult(m, warnings, segs, dropped)
