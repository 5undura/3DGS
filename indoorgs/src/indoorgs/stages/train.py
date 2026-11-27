"""4. 학습 (담당 A). data_clean/ -> train/<trainer>_<preset>/splat.ply, eval.json, stats.json

할 일
- trainers/ 의 어댑터로 명령을 만들어 실행, 출력은 logs/ 와 run 폴더의 train_log.txt 로
- 결과 PLY 를 splat.ply 로 복사(또는 링크), 평가는 tools/eval_renders.py(Brush) / metrics.csv(LichtFeld) -> eval.json
- 통계는 tools/ply_stats.py 정의 -> stats.json
- 실패하면 run 폴더를 <이름>_failed_<시각> 으로 옆으로
"""
from pathlib import Path

from .base import Stage, StageContext, StageResult


class Train(Stage):
    name = "train"
    uses_gpu = True

    def outputs(self, ctx: StageContext) -> list[Path]:
        t = ctx.project.cfg.train
        run = ctx.paths.train_run(t.trainer, t.preset)
        return [run / "splat.ply", run / "eval.json", run / "stats.json"]

    def run(self, ctx: StageContext) -> StageResult:
        raise NotImplementedError
