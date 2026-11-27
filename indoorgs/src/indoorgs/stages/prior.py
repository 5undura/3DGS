"""1.5 위치 제약 (담당 B). plan.json + path.json + frames -> colmap/priors.json

공간(floorplan/plan.json)과 동선(sections/<name>/path.json)이 둘 다 있을 때만 돈다.
"""
from pathlib import Path

from .. import schemas
from ..pathing import build_priors, check_duration, outside_segments
from .base import Stage, StageContext, StageResult


class Prior(Stage):
    name = "prior"

    def applies(self, ctx: StageContext) -> bool:
        return (ctx.project.cfg.register.use_floorplan_prior
                and ctx.project.has_space and ctx.paths.path_json.exists())

    def outputs(self, ctx: StageContext) -> list[Path]:
        return [ctx.paths.priors]

    def run(self, ctx: StageContext) -> StageResult:
        p = ctx.paths
        if not p.frames_done.exists():
            return StageResult(False, "frames/done.json 없음 (extract 먼저)")
        fps = float(schemas.read_json(p.frames_done)["fps"])
        frames = sorted(f.name for f in p.frames.glob("f_*.jpg"))
        if not frames:
            return StageResult(False, "frames/ 에 사진 없음")
        plan = schemas.load_plan(ctx.project.plan_json)
        cam = schemas.load_path(p.path_json)
        priors, warnings = build_priors(cam, frames, fps)
        outside = [f["image"] for f in priors["frames"] if plan.room_at(f["xyz"][0], f["xyz"][1]) is None]
        if outside:
            warnings.append(f"동선 위치가 어느 방에도 안 들어가는 프레임 {len(outside)}장 (예: {outside[0]}). 동선이나 방 크기 확인")
        segs = outside_segments(cam, plan, min_len=0.3)
        for seg in segs:
            warnings.append(f"동선이 방 밖으로 나감: s={seg['s0']}~{seg['s1']}m ({seg['start']} -> {seg['end']})")
        done = schemas.read_json(p.frames_done)
        duration = None
        if done.get("total_frames"):
            duration = float(done["total_frames"]) / fps
            warnings += check_duration(cam, duration)
        for w in warnings:
            ctx.log("경고: " + w)
        schemas.write_json(p.priors, priors)
        # GUI·report 가 읽는 경고 파일 (priors.json 옆)
        schemas.write_json(p.colmap / "priors_warnings.json",
                           {"version": schemas.VERSION, "warnings": warnings, "outside_frames": outside,
                            "outside_segments": segs, "video_duration": duration})
        return StageResult(True, f"{len(frames)}장 위치 생성, 경고 {len(warnings)}개", [p.priors])
