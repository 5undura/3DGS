"""3½. 벽 면 맞추기·초기 점 심기 (diagnose 뒤, train 앞). data_clean/ + plan.json -> walls/planes.json, data_seed/

도면을 정합 전에 잡은 이유의 두 번째 절반: 정합 결과가 도면 좌표(pose_prior_mapper)이면
1) 구간 카메라가 들어 있는 방의 벽·바닥·천장 면을 plan.json 에서 만들고 (walls/planes.py)
2) 그 면 근처의 COLMAP 점으로 실제 면 위치를 다시 맞춘 뒤 (walls/refine.py, 줄자 도면 오차·축척 오차 흡수)
3) 흰 벽처럼 점이 없는 면에 사진 색으로 초기 점을 심어 data_seed/ 를 만든다 (walls/seed.py).
   맞춘 면(fit source "fit"·"infer")에만 심음: 도면 그대로인 면은 실제 위치를 몰라 실제 벽 앞·뒤에 가짜 면을 만듦.
   "fit_scaled"(축척으로 옮겨 넓힌 탐색으로 맞춘 먼 끝 벽, walls.scale_search)는 walls.seed_scaled 일 때만 (seed_sources).
   planes.json scale_check: 마주 보는 맞춘 면 쌍의 맞춘 폭/도면 폭 (2% 넘게 다르면 경고, 동선 축척 오차).
   planes.json source_identity (사진 수 + images.bin sha1) 로 train 이 data_seed 가 지금 모델에서 나왔는지 확인.
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


def model_identity(model_dir: Path) -> dict | None:
    """COLMAP 모델 폴더의 정체: 사진 수 + images.bin sha1 (data_seed 가 지금 data_clean 에서 만든 것인지 확인용)."""
    import hashlib
    ib = Path(model_dir) / "images.bin"
    if not ib.exists():
        return None
    h = hashlib.sha1()
    with open(ib, "rb") as f:
        n_images = int.from_bytes(f.read(8), "little")
        f.seek(0)
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"n_images": n_images, "images_bin_sha1": h.hexdigest()}


def seed_sources(wc) -> tuple[str, ...]:
    """초기 점을 심을 면의 fit source. "fit_scaled"(축척으로 옮겨 넓힌 탐색으로 맞춘 먼 면)는 walls.seed_scaled 일 때만:
    점 기준(inlier·덮임·기울기)은 같지만 탐색 폭이 최대 1.5 m 라 유리문·가구 앞면·문 너머 벽을 잡았을 위험이 남고,
    잘못 심은 면은 학습에 가짜 벽으로 굳음 (안 심으면 지금과 같음 = 손해 없음). snap·repair 도 이 면은 안 씀."""
    return ("fit", "infer", "fit_scaled") if getattr(wc, "seed_scaled", False) else ("fit", "infer")


def seed_usable(ctx: StageContext) -> tuple[bool, str]:
    """train 이 data_seed 를 써도 되는지와 이유. 조건: 벽 기능이 이 구간에 적용(도면 좌표·walls.enabled)되고
    walls.seed 가 켜져 있고, planes.json 에 적힌 원본 모델 정체가 지금 data_clean(없으면 data) 모델과 같음."""
    p = ctx.paths
    if not (p.data_seed / "sparse" / "0").exists():
        return False, "data_seed 없음"
    if not walls_apply(ctx):
        return False, "벽 기능이 이 구간에 적용 안 됨 (walls.enabled 꺼짐 또는 도면 좌표 아님)"
    if not ctx.project.cfg.walls.seed:
        return False, "walls.seed 꺼짐"
    if not p.planes_json.exists():
        return False, "walls/planes.json 없음"
    try:
        rec = schemas.read_json(p.planes_json).get("source_identity")
    except Exception as e:  # noqa: BLE001
        return False, f"planes.json 읽기 실패 {e}"
    now = model_identity(source_dir(ctx) / "sparse" / "0")
    if not rec or rec != now:
        return False, "data_seed 를 만든 원본 모델이 지금 data_clean 과 다름 (diagnose·register 를 다시 돌림?)"
    return True, "ok"


class Walls(Stage):
    name = "walls"
    invalidate_when_skipped = True

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
        rep: dict = {}
        fitted, info = refine.refine_planes(planes, xyz, weights=w, search=wc.search,
                                            trust_plan_dims=wc.trust_plan_dims, scale_search=wc.scale_search,
                                            centers=centers, report=rep)
        trusted = [q for q, i in zip(fitted, info) if i.get("source") in seed_sources(wc)]   # 초기 점은 맞춘 면에만
        before = refine.flatness(xyz, planes)
        after = refine.flatness(xyz, fitted)
        n_fit = sum(1 for i in info if i.get("source") in ("fit", "infer", "fit_scaled"))
        ctx.log(f"방 {rooms}: 면 {len(planes)}개 중 {n_fit}개를 정합 점으로 다시 맞춤")
        for i in info:
            if i.get("source") in ("fit", "fit_scaled") and abs(i.get("shift_m", 0)) >= 0.05:
                tag = " (축척으로 옮겨 넓힌 탐색)" if i["source"] == "fit_scaled" else ""
                ctx.log(f"  {i['id']}: 도면보다 {i['shift_m'] * 100:+.0f}cm, 기울기 {i.get('tilt_deg', 0):.1f}도{tag}")
        # 축척 확인: 2차 탐색에 쓴 것(fit 만)과 최종(fit + fit_scaled, 먼 끝 벽 쌍까지)
        sc = refine.scale_check(planes, fitted, info, sources=("fit", "fit_scaled"))
        sc["used_for_search"] = rep.get("scale_check", {}).get("scale")
        if sc["warn"]:
            ctx.log("경고: " + sc["message"])
        p.walls.mkdir(parents=True, exist_ok=True)
        out = {"version": schemas.VERSION, "rooms": rooms, "source_model": src.name,
               "source_identity": model_identity(src / "sparse" / "0"),
               "planes": [wp.plane_to_json(q) for q in fitted],
               "plan_planes": [wp.plane_to_json(q) for q in planes],
               "fit": info, "room_widths": refine.room_widths(planes, fitted), "scale_check": sc,
               "scale_search": rep.get("scale_search"),
               "points_flatness_before": before, "points_flatness_after": after}

        msg = f"면 {len(planes)}개 (맞춤 {n_fit})"
        if sc["warn"]:
            msg += f", 축척 경고 {sc['scale']:.3f}"
        if wc.seed:
            seed_dir = p.data_seed
            if seed_dir.exists():
                ctx.log(f"끝나지 않은 data_seed -> {move_aside(seed_dir).name}")
            try:
                if trusted:
                    sxyz, srgb, sinfo = seed.seed_points(trusted, m, src / "images", spacing=wc.seed_spacing)
                else:
                    sxyz, srgb = np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
                    sinfo = {"n_added": 0, "reason": "맞춘 면 없음"}
                sinfo["planes"] = [q.id for q in trusted]
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
