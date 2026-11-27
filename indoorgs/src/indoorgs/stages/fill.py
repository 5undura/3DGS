"""5. 보강 (담당 C). splat.ply + data_clean/ (+ plan.json) -> fill/holes_backdrop.ply

tools/fill/fill_section.py(바탕층) + merge_holes.py(바탕층 중 학습 스플랫이 안 덮은 곳만 합침) 이식.
- 면: 구간이 도면 좌표로 정합됐으면(plan.json + register.json mapper = pose_prior_mapper, diagnose.model_in_plan_coords) 카메라가 든 방의 floor_z·ceiling_z·바닥 다각형,
  아니면 예전처럼 점에서 추정.
- cfg.fill.holes_only = True(기본): 벽 바탕층은 빼고 바닥·천장 구멍만. False 면 벽도 구멍이면 넣음.
- 결과 = 학습 스플랫 전부 + 고른 바탕층 (학습 PLY 의 속성 순서·주석 그대로).
- 시작점 보강 sparse(fill_section 6단계, 재학습용)는 이 단계에서 안 만듦.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import plyio, schemas
from ..fillcore import backdrop as bd
from ..fillcore import colmap_io
from .base import Stage, StageContext, StageResult

VIEW_STEP = 6          # 색 입힐 때 사진 간격 (fill_section 기본)
COVER_THR = 1.0        # 덮임 기준 (merge_holes 기본)
RADIUS_MUL = 3.0       # 반경 배수 (PROGRESS: L7_fill 에서 3 사용)


def train_ply(ctx: StageContext) -> Path:
    """최신 학습 (_2, _3 ...) 의 결과. 벽에 붙인 splat_walls.ply(stages/snap.py) 가 있으면 그것."""
    from .snap import train_ply as snapped
    return snapped(ctx)


def model_dirs(ctx: StageContext) -> tuple[Path | None, Path]:
    """(COLMAP 모델 폴더, 사진 폴더). data_clean 우선, 없으면 data."""
    p = ctx.paths
    for root in (p.data_clean, p.data):
        m = colmap_io.find_model(root)
        if m is not None:
            return m, root / "images"
    return None, p.data_clean / "images"


def in_plan_coords(ctx: StageContext) -> bool:
    from .diagnose import model_in_plan_coords
    return ctx.project.has_space and model_in_plan_coords(ctx)


def make_fill(train: plyio.GaussianPly, model: Path, image_dir: Path, plan: schemas.Plan | None,
              holes_only: bool = True, log=lambda m: None, step: int = VIEW_STEP,
              thr: float = COVER_THR, rmul: float = RADIUS_MUL) -> tuple[plyio.GaussianPly, dict]:
    """학습 PLY + COLMAP 모델 (+ 도면) -> (합친 PLY, 정보)."""
    cams = colmap_io.read_cameras(model / "cameras.bin")
    ims = colmap_io.read_images(model / "images.bin")
    pts = colmap_io.read_points3D(model / "points3D.bin")
    if not ims:
        raise ValueError("정합된 사진이 없음")
    C = np.array([-im["R"].T @ im["t"] for im in ims])
    log(f"사진 {len(ims)}, 점 {len(pts['xyz'])}")
    if plan is not None:
        rooms = bd.rooms_of_cameras(plan.rooms, C)
        sm = bd.samples_from_plan(rooms, C, pts["xyz"], log)
    else:
        down = np.array([im["R"].T @ [0, 1, 0] for im in ims])
        sm = bd.samples_from_points(C, down, pts["xyz"], pts["err"], log=log)
    log(f"바탕층 샘플 {len(sm.P):,} (방식 {sm.info['mode']})")
    col = bd.colorize(sm, ims, cams[ims[0]["cam"]], image_dir, step, log)
    back = bd.to_splats(sm, col, train.names)
    hole = bd.hole_mask(train, back.xyz, sm.kind, thr, rmul, keep_walls=not holes_only, log=log)
    out = plyio.GaussianPly(np.concatenate([train.data, back.data[hole]]), list(train.comments))
    info = dict(sm.info, samples=int(len(sm.P)), added=int(hole.sum()), train=int(train.n), holes_only=holes_only)
    return out, info


class Fill(Stage):
    name = "fill"

    def outputs(self, ctx: StageContext) -> list[Path]:
        return [ctx.paths.fill_ply]

    def prev_targets(self, ctx: StageContext) -> list[Path]:
        return [ctx.paths.fill]

    def run(self, ctx: StageContext) -> StageResult:
        src = train_ply(ctx)
        if not src.exists():
            return StageResult(False, f"학습 결과 없음: {src} (train 먼저)")
        model, image_dir = model_dirs(ctx)
        if model is None:
            return StageResult(False, "data_clean/ 또는 data/ 에 COLMAP 모델(sparse/0) 없음")
        plan = None
        if in_plan_coords(ctx):
            plan = schemas.load_plan(ctx.project.plan_json)
            ctx.log("도면 좌표 구간: 바닥·천장을 plan.json 방에서 정함")
        elif ctx.project.has_space:
            ctx.log("plan.json 은 있지만 이 구간은 도면 좌표로 정합 안 됨(register.json mapper 가 pose_prior_mapper 아님)"
                    " -> 점에서 추정")
        try:
            train = plyio.read(src)
            out, info = make_fill(train, model, image_dir, plan, ctx.project.cfg.fill.holes_only, ctx.log)
        except (ValueError, plyio.PlyError) as e:
            return StageResult(False, f"보강 실패: {e}")
        plyio.write(ctx.paths.fill_ply, out)
        schemas.write_json(ctx.paths.fill / "fill.json", {"version": 1, "source": str(src), "model": str(model), **{
            k: v for k, v in info.items()}})
        return StageResult(True, f"바탕층 {info['added']:,}개 추가 (학습 {info['train']:,})", [ctx.paths.fill_ply])
