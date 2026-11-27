"""7. 보고서 (담당 D, 프로젝트 범위). 모든 json -> report.html (항상 다시 만듦)

할 일
- 구간별 표: 등록 장수, 제외 장수, PSNR, 반투명 비율(경고: 0.8 이상이면 VR 비침), 시간
- 공간이 있으면 도면 위에 카메라 궤적(제외 사진 빨간색)
"""
from pathlib import Path

from .base import Stage, StageContext, StageResult


class Report(Stage):
    name = "report"
    scope = "project"

    def outputs(self, ctx: StageContext) -> list[Path]:
        return [ctx.project.report]

    def is_done(self, ctx: StageContext) -> bool:
        return False

    def run(self, ctx: StageContext) -> StageResult:
        raise NotImplementedError
