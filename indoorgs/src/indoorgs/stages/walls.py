"""3½. 벽 면 맞추기·초기 점 심기 (diagnose 뒤, train 앞). data_clean/ + plan.json -> walls/planes.json, data_seed/

도면을 정합 전에 잡은 이유의 두 번째 절반: 정합 결과가 도면 좌표(pose_prior_mapper)이면
1) 구간 카메라가 들어 있는 방의 벽·바닥·천장 면을 plan.json 에서 만들고 (walls/planes.py)
2) 그 면 근처의 COLMAP 점으로 실제 면 위치를 다시 맞춘 뒤 (walls/refine.py, 줄자 도면 오차·축척 오차 흡수)
3) 흰 벽처럼 점이 없는 면에 사진 색으로 초기 점을 심어 data_seed/ 를 만든다 (walls/seed.py).
   학습기는 data_seed/ 가 있으면 그걸 읽음 (stages/train.py). 원본 data/, data_clean/ 은 그대로.
학습 뒤 벽에 붙이기는 stages/snap.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import schemas
from ..colmap import model as cmodel
from ..colmap.runner import move_aside
from ..walls import planes as wp
from .base import Stage, StageContext, StageResult
from .diagnose import link_images, model_in_plan_coords


def source_dir(ctx: StageContext) -> Path:
    p = ctx.paths
    return p.data_clean if (p.data_clean / "sparse" / "0").exists() else p.data


def walls_apply(ctx: StageContext) -> bool:
    return ctx.project.cfg.walls.enabled and ctx.project.has_space and model_in_plan_coords(ctx)


class Walls(Stage):
    name = "walls"

    def applies(self, ctx: StageContext) -> bool:
        return walls_apply(ctx)

    def outputs(self, ctx: StageContext) -> list[Path]:
        p = ctx.paths
        outs = [p.planes_json]
        if ctx.project.cfg.walls.seed:
            outs.append(p.data_seed / "sparse" / "0")
        return outs

    def prev_targets(self, ctx: StageContext) -> list[Path]:
        return [ctx.paths.walls, ctx.paths.data_seed]

    def run(self, ctx: StageContext) -> StageResult:
        from ..walls import refine, seed

        p, wc = ctx.paths, ctx.project.cfg.walls
        src = source_dir(ctx)
        if not (src / "sparse" / "0").exists():
            return StageResult(False, "data_clean/ 또는 data/ 에 COLMAP 모델 없음 (diagnose 먼저)")
        m = cmodel.read_model(src / "sparse" / "0")
        plan = schemas.load_plan(ctx.project.plan_json)
        centers = np.array([im.center for im in m.images.values()]) if m.images else np.zeros((0, 3))
        rooms = wp.rooms_for_cameras(plan, centers)
        if not rooms:
            return StageResult(False, "카메라가 들어 있는 방이 없음 (도면 좌표가 맞는지 diagnose.json outside_space 확인)")
        planes = wp.planes_from_plan(plan, rooms)
        xyz = np.array([pt.xyz for pt in m.points.values()]) if m.points else np.zeros((0, 3))
        w = np.array([len(pt.track) for pt in m.points.values()], float) if m.points else None
        fitted, info = refine.refine_planes(planes, xyz, weights=w, search=wc.search)
        before = refine.flatness(xyz, planes)
        after = refine.flatness(xyz, fitted)
        n_fit = sum(1 for i in info if i.get("source") == "fit")
        ctx.log(f"방 {rooms}: 면 {len(planes)}개 중 {n_fit}개를 정합 점으로 다시 맞춤")
        for i in info:
            if i.get("source") == "fit" and abs(i.get("shift_m", 0)) >= 0.05:
                ctx.log(f"  {i['id']}: 도면보다 {i['shift_m'] * 100:+.0f}cm, 기울기 {i.get('tilt_deg', 0):.1f}도")
        p.walls.mkdir(parents=True, exist_ok=True)
        out = {"version": schemas.VERSION, "rooms": rooms, "source_model": src.name,
               "planes": [wp.plane_to_json(q) for q in fitted],
               "plan_planes": [wp.plane_to_json(q) for q in planes],
               "fit": info, "room_widths": refine.room_widths(planes, fitted),
               "points_flatness_before": before, "points_flatness_after": after}

        msg = f"면 {len(planes)}개 (맞춤 {n_fit})"
        if wc.seed:
            seed_dir = p.data_seed
            if seed_dir.exists():
                ctx.log(f"끝나지 않은 data_seed -> {move_aside(seed_dir).name}")
            try:
                sxyz, srgb, sinfo = seed.seed_points(fitted, m, src / "images", spacing=wc.seed_spacing)
                seed.write_seeded_model(m, sxyz, srgb, seed_dir / "sparse" / "0_tmp")
                link_images(src / "images", seed_dir / "images")
                (seed_dir / "sparse" / "0_tmp").rename(seed_dir / "sparse" / "0")
            except Exception as e:  # noqa: BLE001
                old = move_aside(seed_dir)
                return StageResult(False, f"초기 점 심기 실패: {e} (중간 결과 {old.name if old else '-'})")
            out["seed"] = sinfo
            msg += f", 초기 점 {len(sxyz):,}개 추가 (기존 {len(m.points):,})"
        schemas.write_json(p.planes_json, out)
        return StageResult(True, msg, self.outputs(ctx))
