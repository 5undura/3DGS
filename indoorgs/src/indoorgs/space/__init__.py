"""공간 도구 (담당 B): 박스 편집 모델, 도면(DXF·SVG·사진) 불러오기.

GUI 는 SpaceModel 하나만 들고 편집하고, save()/load() 로 plan.json 을 읽고 쓴다.
불러오기 함수는 모두 DraftResult(model, warnings, segments, dropped, extra) 를 돌려준다.
DXF 는 ezdxf, 사진은 OpenCV 가 필요해서 각 모듈 안에서 늦게 불러온다.
"""
from .draft import DraftOptions, DraftResult
from .model import Box, Issue, SpaceDoor, SpaceModel, SpaceRoom


def import_dxf(*args, **kw):
    from .dxf_import import import_dxf as f
    return f(*args, **kw)


def import_svg(*args, **kw):
    from .svg_import import import_svg as f
    return f(*args, **kw)


def import_image(*args, **kw):
    from .image_import import import_image as f
    return f(*args, **kw)


__all__ = ["Box", "Issue", "SpaceDoor", "SpaceModel", "SpaceRoom", "DraftOptions", "DraftResult",
           "import_dxf", "import_svg", "import_image"]
