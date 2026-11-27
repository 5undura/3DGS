"""5. 보강 (담당 C). splat.ply + data_clean/ (+ plan.json) -> fill/holes_backdrop.ply

할 일
- 저장소 tools/fill/fill_section.py(바탕층 생성) + merge_holes.py(바닥·천장 구멍만 합침) 를 함수로
- 공간이 있으면 바닥·천장 높이를 plan.json 의 floor_z / ceiling_z 로 (지금은 점에서 추정해 벽 바탕층 80% 어긋남)
- cfg.fill.holes_only = True 가 기본 (벽 바탕층 전체는 흐려짐)
- PLY 는 속성 이름으로 읽기 (Brush 는 xyz 가 맨 뒤, LichtFeld 는 맨 앞)
"""
from pathlib import Path

from .base import Stage, StageContext, StageResult


class Fill(Stage):
    name = "fill"

    def outputs(self, ctx: StageContext) -> list[Path]:
        return [ctx.paths.fill_ply]

    def run(self, ctx: StageContext) -> StageResult:
        raise NotImplementedError
