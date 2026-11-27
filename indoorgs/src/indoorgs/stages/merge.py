"""6. 합치기 (담당 C, 프로젝트 범위). 구간별 최종 PLY -> merged/building.ply + layout.json

할 일
- 공간 정합(pose_prior_mapper)을 쓴 구간은 이미 도면 좌표: 변환 = 단위 행렬. 문에서 겹치는 구간끼리 미세 정렬(ICP 등)만
- 공간이 없으면 사용자가 준 layout.json(4x4) 또는 저장소 tools/realign_layout.py 방식
- 구간마다 fill 결과가 있으면 그것, 없으면 train 결과
"""
from pathlib import Path

from .base import Stage, StageContext, StageResult


class Merge(Stage):
    name = "merge"
    scope = "project"

    def outputs(self, ctx: StageContext) -> list[Path]:
        return [ctx.project.building_ply, ctx.project.layout_json]

    def run(self, ctx: StageContext) -> StageResult:
        raise NotImplementedError
