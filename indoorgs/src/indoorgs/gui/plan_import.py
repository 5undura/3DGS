"""도면 불러오기 (GUI 용): 업로드 파일 보관 -> space.import_* -> 초안 JSON. plan.json 은 안 씀 (미리보기).

- 원본은 floorplan/source/<원래 이름> 에 보관. 같은 이름이 있으면 내용이 같으면 그 파일을 다시 쓰고,
  다르면 이름_2.확장자 처럼 번호를 붙임 (덮어쓰기 없음).
- 사진이면 원근 보정 결과를 floorplan/plan.png 로 저장할 수 있음 (기존 plan.png 는 plan_<시각>.png 로 옮김).
"""
from __future__ import annotations

import base64
import datetime as dt
import math
import re
from pathlib import Path
from typing import Any

from .. import schemas
from ..space import DraftOptions, SpaceModel

DXF_EXT = {".dxf"}
SVG_EXT = {".svg"}
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
MAX_SEGMENTS = 20000


class ImportError_(ValueError):
    """사용자에게 보일 불러오기 오류 (400)."""


def kind_of(name: str) -> str:
    ext = Path(name).suffix.lower()
    if ext in DXF_EXT:
        return "dxf"
    if ext in SVG_EXT:
        return "svg"
    if ext in IMG_EXT:
        return "image"
    raise ImportError_(f"모르는 파일 형식 {ext or '(확장자 없음)'}: .dxf .svg 또는 사진(.png .jpg) 만")


def safe_name(name: str) -> str:
    n = Path(str(name).replace("\\", "/")).name
    n = re.sub(r'[\x00-\x1f<>:"/\\|?*]', "_", n).strip(" .")
    return n or "plan"


def keep_source(src_dir: Path, name: str, data: bytes) -> tuple[Path, bool]:
    """원본 보관. (경로, 새로 만들었는지). 같은 내용 파일이 있으면 그걸 씀."""
    src_dir.mkdir(parents=True, exist_ok=True)
    base = safe_name(name)
    stem, ext = Path(base).stem, Path(base).suffix
    n = 1
    while True:
        f = src_dir / (base if n == 1 else f"{stem}_{n}{ext}")
        if not f.exists():
            f.write_bytes(data)
            return f, True
        if f.read_bytes() == data:
            return f, False
        n += 1


def move_aside(png: Path) -> Path | None:
    """기존 plan.png -> plan_<시각>.png (겹치면 번호)."""
    if not png.exists():
        return None
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = png.with_name(f"{png.stem}_{ts}{png.suffix}")
    n = 2
    while dst.exists():
        dst = png.with_name(f"{png.stem}_{ts}_{n}{png.suffix}")
        n += 1
    png.rename(dst)
    return dst


def _seg_json(segs) -> list:
    return [[[round(a[0], 4), round(a[1], 4)], [round(b[0], 4), round(b[1], 4)]] for a, b in segs[:MAX_SEGMENTS]]


def _pos(v: float | None, name: str) -> float | None:
    if v is None:
        return None
    if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
        raise ImportError_(f"{name} 는 양수여야 함")
    return float(v)


def run_import(floorplan: Path, filename: str, data: bytes, opt: dict[str, Any]) -> dict:
    """opt: px_per_m, scale_m(DXF m/단위), scale_points [[x1,y1],[x2,y2],m], corners [[x,y]x4], size_m [w,h],
    save_png, recenter, min_area, snap_tol. 돌려줌: plan, warnings, segments, kind, source_file, ..."""
    kind = kind_of(filename)
    if not data:
        raise ImportError_("빈 파일")
    dopt = DraftOptions(recenter=bool(opt.get("recenter", kind != "image")))
    if opt.get("min_area") is not None:
        dopt.min_area = float(opt["min_area"])
    if opt.get("snap_tol") is not None:
        dopt.snap_tol = float(opt["snap_tol"])
    src, created = keep_source(floorplan / "source", filename, data)
    out: dict[str, Any] = {"kind": kind, "source_file": f"floorplan/source/{src.name}"}
    try:
        res = _import(kind, src, dopt, opt, out)
    except ImportError_:
        if created:
            src.unlink(missing_ok=True)       # 방금 만든 것만 치움
        raise
    except Exception as e:                   # 파서 오류 -> 사용자 메시지
        if created:
            src.unlink(missing_ok=True)
        raise ImportError_(f"{kind.upper()} 를 못 읽음: {type(e).__name__}: {e}") from e
    res.model.source["file"] = src.name
    plan = res.model.to_json()
    if "image_align" in out:
        plan["source"]["image_align"] = out["image_align"]
    out.update({"plan": plan, "warnings": list(res.warnings), "segments": _seg_json(res.segments),
                "n_rooms": len(res.model.rooms), "bounds": list(res.model.bounds()) if res.model.rooms else None})
    if len(res.segments) > MAX_SEGMENTS:
        out["warnings"].append(f"선분 {len(res.segments)}개 중 {MAX_SEGMENTS}개만 보냄 (밑그림)")
    return out


def _import(kind: str, src: Path, dopt: DraftOptions, opt: dict, out: dict):
    from .. import space
    sp = opt.get("scale_points")
    if kind == "svg":
        ppm = _pos(opt.get("px_per_m"), "px_per_m")
        pts = None
        if sp:
            (x1, y1), (x2, y2), m = sp
            pts = ((float(x1), float(y1)), (float(x2), float(y2)), float(m))
        if ppm is None and pts is None:
            raise ImportError_("SVG 는 축척이 필요: px_per_m 또는 두 점 실측")
        try:
            return space.import_svg(src, dopt, px_per_m=ppm, scale_points=pts)
        except ValueError as e:
            raise ImportError_(str(e)) from e
    if kind == "dxf":
        scale = _pos(opt.get("scale_m"), "scale_m")
        if sp:
            (x1, y1), (x2, y2), m = sp
            d = math.dist((float(x1), float(y1)), (float(x2), float(y2)))
            if d <= 0 or float(m) <= 0:
                raise ImportError_("두 점 실측: 두 점이 같거나 거리가 0")
            scale = float(m) / d
        return space.import_dxf(src, dopt, scale=scale)
    # 사진
    import cv2
    ppm = _pos(opt.get("px_per_m"), "px_per_m") or 50.0
    corners, size_m = opt.get("corners"), opt.get("size_m")
    if corners is not None:
        if len(corners) != 4 or size_m is None or len(size_m) != 2:
            raise ImportError_("사진: 네 모서리 점과 실제 크기(가로, 세로 m)가 필요")
        size_m = (_pos(float(size_m[0]), "가로"), _pos(float(size_m[1]), "세로"))
        if size_m[0] * ppm * size_m[1] * ppm > 40e6:
            raise ImportError_("보정 이미지가 너무 큼 (px_per_m 를 줄일 것)")
    try:
        res = space.import_image(src, dopt, corners=corners, size_m=size_m, px_per_m=ppm)
    except ValueError as e:
        raise ImportError_(str(e)) from e
    img = res.extra["rectified"]
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ImportError_("보정 이미지를 PNG 로 못 만듦")
    h = int(img.shape[0])
    # 보정 이미지 픽셀 -> 도면: x = (px - ox)/ppm, y = (oy - py)/ppm. image_import 는 왼쪽 아래가 (0, 0)
    out["image_align"] = {"px_per_m": ppm, "origin_px": [0.0, float(h)], "flip_y": True}
    out["image_size"] = [int(img.shape[1]), h]
    if opt.get("save_png"):
        png = src.parent.parent / "plan.png"
        moved = move_aside(png)
        png.write_bytes(buf.tobytes())
        out["plan_png"] = "floorplan/plan.png"
        out["moved_png"] = f"floorplan/{moved.name}" if moved else None
    else:
        out["rectified_png"] = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()
    return res


def validate_plan(data: dict) -> list[dict]:
    """plan JSON -> SpaceModel.validate() 결과. 형식 오류면 그 하나만."""
    if not data.get("rooms"):
        return []
    try:
        m = SpaceModel.from_json({"version": schemas.VERSION, **data})
    except (schemas.SchemaError, KeyError, TypeError, ValueError) as e:
        return [{"level": "error", "kind": "schema", "message": str(e), "ids": []}]
    return [{"level": i.level, "kind": i.kind, "message": i.message, "ids": list(i.ids)} for i in m.validate()]
