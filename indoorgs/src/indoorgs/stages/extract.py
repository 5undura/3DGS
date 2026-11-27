"""1. 추출 (담당 A). 영상 -> frames/*.jpg + frames/done.json

할 일
- 저장소 1_extract/pick_best.py 를 함수로 옮기거나 subprocess 로 부름 (옵션: cfg.extract.min_diff, sharp_ratio)
- 저장은 cv2.imencode + tofile (한글 경로)
- done.json 에 fps, total_frames, count, args 를 적음 (prior 가 fps 로 시간을 계산)
"""
from pathlib import Path

from .base import Stage, StageContext, StageResult


class Extract(Stage):
    name = "extract"

    def outputs(self, ctx: StageContext) -> list[Path]:
        return [ctx.paths.frames_done]

    def run(self, ctx: StageContext) -> StageResult:
        raise NotImplementedError
