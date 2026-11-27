"""2. 정합 (담당 A, prior 연결은 B와 함께). frames/ (+ priors.json) -> colmap/, data/

할 일
- 저장소 2_register/register_all.ps1 를 파이썬으로 옮김: feature_extractor(single_camera, cfg.register.camera_model)
  -> {sequential|exhaustive}_matcher -> mapper -> 가장 큰 모델 -> image_undistorter -> data/
- priors.json 이 있으면: DB pose_priors 표에 기록 -> pose_prior_mapper. 실패하면 mapper 로 자동 전환하고 로그에 남김
- COLMAP 4.2.1 에서 -h 로 확인한 옵션 이름·pose_priors 스키마를 이 주석에 적을 것 (확인 전 추측 금지)
- 복도 1 실험: global_mapper 는 사진 9장이 수천 단위 밖으로 튀어 못 씀 -> 기본 incremental
"""
from pathlib import Path

from .base import Stage, StageContext, StageResult


class Register(Stage):
    name = "register"

    def outputs(self, ctx: StageContext) -> list[Path]:
        return [ctx.paths.data_model]

    def run(self, ctx: StageContext) -> StageResult:
        raise NotImplementedError
