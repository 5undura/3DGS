"""3. 진단·정리 (담당 A, 공간 검사는 B). data/ -> diagnose.json + data_clean/

할 일
- 저장소 tools/diagnose.py 의 순간이동 검사(촬영 순서대로 속도, 중앙값 대비)를 함수로
- 공간이 있으면: 카메라가 어느 방에도 없음(outside_space), priors 에서 1m 이상(far_from_prior)
- excluded = 위 목록 합집합 -> COLMAP image_deleter 로 data_clean/ (사진은 복사하지 않고 원본 연결)
- 복도 1 기준 결과: 상위 15곳 25장 제외, PSNR +0.25
"""
from pathlib import Path

from .base import Stage, StageContext, StageResult


class Diagnose(Stage):
    name = "diagnose"

    def outputs(self, ctx: StageContext) -> list[Path]:
        return [ctx.paths.diagnose, ctx.paths.data_clean / "sparse" / "0"]

    def run(self, ctx: StageContext) -> StageResult:
        raise NotImplementedError
