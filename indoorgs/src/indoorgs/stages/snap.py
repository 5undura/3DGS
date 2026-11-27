"""4½. 벽에 붙이기 (train 뒤, fill 앞). train/<run>/splat.ply + walls/planes.json -> train/<run>/splat_walls.ply, walls.json

- 벽 뒤(문 구멍이 아닌 곳)와 바닥 아래·천장 위 부유물을 지우고
- 면에서 band 안의 스플랫을 면 위로 옮겨 법선 방향으로 납작하게 (walls/snap.py)
- 원본 splat.ply 는 그대로. fill·merge 는 splat_walls.ply 가 있으면 그것을 씀 (stages/fill.py train_ply)
"""
from __future__ import annotations

from pathlib import Path

from .. import schemas
from ..walls import planes as wp
from .base import Stage, StageContext, StageResult
from .walls import walls_apply

SNAP_PLY = "splat_walls.ply"
SNAP_JSON = "walls.json"


def latest_run(ctx: StageContext) -> Path:
    t = ctx.project.cfg.train
    return ctx.paths.latest_train_run(t.trainer, t.preset)


def train_ply(ctx: StageContext) -> Path:
    """fill·merge 가 쓸 학습 결과: 벽에 붙인 것이 있으면 그것, 없으면 splat.ply."""
    run = latest_run(ctx)
    return run / SNAP_PLY if (run / SNAP_PLY).exists() else run / "splat.ply"


class Snap(Stage):
    name = "snap"

    def applies(self, ctx: StageContext) -> bool:
        return walls_apply(ctx) and ctx.project.cfg.walls.snap

    def outputs(self, ctx: StageContext) -> list[Path]:
        run = latest_run(ctx)
        return [run / SNAP_PLY, run / SNAP_JSON]

    def run(self, ctx: StageContext) -> StageResult:
        from ..walls import snap

        p, wc = ctx.paths, ctx.project.cfg.walls
        run = latest_run(ctx)
        src = run / "splat.ply"
        if not src.exists():
            return StageResult(False, f"{src} 없음 (train 먼저)")
        if not p.planes_json.exists():
            return StageResult(False, "walls/planes.json 없음 (walls 단계 먼저)")
        planes = [wp.plane_from_json(d) for d in schemas.read_json(p.planes_json)["planes"]]
        plan = schemas.load_plan(ctx.project.plan_json)
        out = run / SNAP_PLY
        info = snap.snap_file(src, out, planes, plan, band=wc.band, remove_behind=wc.remove_behind)
        schemas.write_json(run / SNAP_JSON, {"version": schemas.VERSION, **info})
        fb, fa = info.get("flatness_before", {}), info.get("flatness_after", {})
        # thick_m = 면까지 거리 + 법선 방향 두께 (rms_m 은 붙이면 당연히 0 이 되므로 두께를 같이 봄)
        msg = (f"벽 평탄도(두께 포함 RMS) {fb.get('thick_m', float('nan')) * 100:.1f} -> "
               f"{fa.get('thick_m', float('nan')) * 100:.1f}cm, "
               f"붙임 {info.get('n_moved', 0):,}, 지움 {info.get('n_removed', 0):,}")
        return StageResult(True, msg, self.outputs(ctx))
