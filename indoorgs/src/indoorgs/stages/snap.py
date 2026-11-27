"""4½. 벽에 붙이기 (train 뒤, fill 앞). train/<run>/splat.ply + walls/planes.json -> train/<run>/splat_walls.ply, walls.json

- 벽 뒤(문 구멍이 아닌 곳)와 바닥 아래·천장 위 부유물을 지우고
- 면에서 band 안의 스플랫을 면 위로 옮겨 법선 방향으로 납작하게 (walls/snap.py)
- 원본 splat.ply 는 그대로. fill·merge 는 splat_walls.ply 가 있으면 그것을 씀 (stages/fill.py train_ply)
- 맞춘 면(planes.json fit source "fit")만 씀. 바닥·천장은 스플랫 밀도 봉우리로 한 번 더 확인 (walls/snap.py).
  불투명 스플랫 절반 넘게 도면 방 밖이면 도면 좌표 PLY 가 아닌 것으로 보고 건너뜀 (walls.json skipped).
- 낡은 결과 고르지 않기: best_ply 는 설정(walls.snap/repair)과 시각(repaired >= walls >= splat.ply)을 확인
"""
from __future__ import annotations

from pathlib import Path

from .. import schemas
from ..walls import planes as wp
from .base import Stage, StageContext, StageResult
from .walls import walls_apply

SNAP_PLY = "splat_walls.ply"
SNAP_JSON = "walls.json"
REPAIR_PLY = "splat_repaired.ply"
REPAIR_JSON = "repair.json"


def latest_run(ctx: StageContext) -> Path:
    t = ctx.project.cfg.train
    return ctx.paths.latest_train_run(t.trainer, t.preset)


def walls_ply_flags(project, paths) -> tuple[bool, bool]:
    """(splat_walls.ply 를 써도 되나, splat_repaired.ply 를 써도 되나) - 지금 설정·좌표 기준.
    설정을 끄거나 구간이 도면 좌표가 아니게 되면 예전에 만든 파일은 안 씀."""
    from .diagnose import paths_in_plan_coords
    wc = project.cfg.walls
    snap_on = bool(wc.enabled and wc.snap and project.has_space and paths_in_plan_coords(paths))
    return snap_on, snap_on and bool(wc.repair)


def train_ply(ctx: StageContext) -> Path:
    """fill·merge 가 쓸 학습 결과: 메운 것 > 벽에 붙인 것 > splat.ply (설정이 켜져 있고 입력보다 새것일 때만)."""
    return best_ply(latest_run(ctx), *walls_ply_flags(ctx.project, ctx.paths))


def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else -1.0


def best_ply(run: Path, use_snap: bool = True, use_repair: bool = True) -> Path:
    """splat_repaired.ply (use_repair, splat_walls.ply 보다 새것) > splat_walls.ply (use_snap, splat.ply 보다 새것)
    > splat.ply. 시각이 거꾸로면 (splat.ply 를 다시 만들었는데 snap 은 안 돎) 낡은 결과로 보고 안 씀."""
    base = run / "splat.ply"
    walls, rep = run / SNAP_PLY, run / REPAIR_PLY
    walls_ok = use_snap and walls.exists() and _mtime(walls) >= _mtime(base)
    if walls_ok and use_repair and rep.exists() and _mtime(rep) >= _mtime(walls):
        return rep
    if walls_ok:
        return walls
    return base


def load_planes(paths) -> tuple[list, dict, dict]:
    """walls/planes.json -> (면 목록, 맞춤 정보 {id: fit 항목}, 원본 dict)."""
    d = schemas.read_json(paths.planes_json)
    planes = [wp.plane_from_json(q) for q in d["planes"]]
    fit = {f["id"]: f for f in d.get("fit", [])}
    return planes, fit, d


def fitted_planes(planes: list, fit: dict) -> list:
    """정합 점으로 맞춘 면만 (fit source == "fit"). 도면 그대로인 면(source "plan", 점 부족 등)은 실제 위치를 모르므로
    벽 뒤 지우기·붙이기·메우기·초기 점 심기에 안 씀 (11-28 검토: 도면 천장 2.6 m / 실제 2.8 m -> 실제 천장 100% 삭제,
    1.5도 기운 실제 벽 앞에 두 번째 벽). fit 정보가 없는 면도 안 씀."""
    return [p for p in planes if fit.get(p.id, {}).get("source") == "fit"]


class Snap(Stage):
    name = "snap"
    invalidate_when_skipped = True

    def applies(self, ctx: StageContext) -> bool:
        return walls_apply(ctx) and ctx.project.cfg.walls.snap

    def outputs(self, ctx: StageContext) -> list[Path]:
        run = latest_run(ctx)
        return [run / SNAP_PLY, run / SNAP_JSON]

    def prev_targets(self, ctx: StageContext) -> list[Path]:
        run = latest_run(ctx)
        return [run / SNAP_PLY, run / SNAP_JSON, run / REPAIR_PLY, run / REPAIR_JSON]

    def is_done(self, ctx: StageContext) -> bool:
        """출력이 있고 입력(splat.ply, planes.json)보다 새것. 건너뛴 기록(walls.json skipped)도 입력보다 새것이면 끝."""
        run = latest_run(ctx)
        inputs = max(_mtime(run / "splat.ply"), _mtime(ctx.paths.planes_json))
        js = run / SNAP_JSON
        if js.exists() and not (run / SNAP_PLY).exists():
            try:
                skipped = bool(schemas.read_json(js).get("skipped"))
            except Exception:  # noqa: BLE001
                skipped = False
            return skipped and _mtime(js) >= inputs
        return super().is_done(ctx) and _mtime(run / SNAP_PLY) >= inputs

    def run(self, ctx: StageContext) -> StageResult:
        from ..walls import snap

        p, wc = ctx.paths, ctx.project.cfg.walls
        run = latest_run(ctx)
        src = run / "splat.ply"
        if not src.exists():
            return StageResult(False, f"{src} 없음 (train 먼저)")
        if not p.planes_json.exists():
            return StageResult(False, "walls/planes.json 없음 (walls 단계 먼저)")
        planes, fit, _ = load_planes(p)
        used = fitted_planes(planes, fit)
        unfit = [q.id for q in planes if q not in used]
        if unfit:
            ctx.log(f"맞추지 못한 면 {len(unfit)}개는 안 씀 (도면 그대로라 실제 위치 모름): {', '.join(unfit)}")
        plan = schemas.load_plan(ctx.project.plan_json)
        out = run / SNAP_PLY
        for old in (out, run / REPAIR_PLY):           # 이번 실행보다 낡은 결과가 남지 않게 (fill 이 잘못 고름)
            if old.exists():
                from ..colmap.runner import move_aside
                ctx.log(f"낡은 {old.name} -> {move_aside(old, 'prev').name}")
        info = snap.snap_file(src, out, used, plan, band=wc.band, color_tol=wc.color_tol,
                              remove_behind=wc.remove_behind)
        info["unfitted_planes"] = unfit
        for w in info.get("warnings", []):
            ctx.log(w)
        schemas.write_json(run / SNAP_JSON, {"version": schemas.VERSION, **_jsonable(info)})
        if info.get("skipped"):
            return StageResult(True, f"건너뜀: {info['skipped']}", [run / SNAP_JSON], skipped=True)
        fb, fa = info.get("flatness_before", {}), info.get("flatness_after", {})
        # thick_m = 면까지 거리 + 법선 방향 두께 (rms_m 은 붙이면 당연히 0 이 되므로 두께를 같이 봄)
        msg = (f"벽 평탄도(두께 포함 RMS) {fb.get('thick_m', float('nan')) * 100:.1f} -> "
               f"{fa.get('thick_m', float('nan')) * 100:.1f}cm, "
               f"붙임 {info.get('n_moved', 0):,}, 지움 {info.get('n_removed', 0):,}, 면 {len(used)}/{len(planes)}")
        return StageResult(True, msg, self.outputs(ctx))


class Repair(Stage):
    """4¾. 면 메우기 (snap 뒤). splat_walls.ply -> splat_repaired.ply, repair.json (walls/repair.py)

    면을 칸으로 나눠 불투명하게 덮인 정도를 재고, 덜 덮인 칸(비치거나 빈 곳)을 둘레의 멀쩡한 스플랫을 통째로 복사해 메움
    (splat-repair Fast Fill 의 rim 방식을 도면 면에 맞게 고친 것, indoorgs/NOTICE). 학습 카메라가 볼 수 있던 칸만 메움."""
    name = "repair"
    invalidate_when_skipped = True

    def applies(self, ctx: StageContext) -> bool:
        wc = ctx.project.cfg.walls
        return walls_apply(ctx) and wc.snap and wc.repair

    def outputs(self, ctx: StageContext) -> list[Path]:
        run = latest_run(ctx)
        return [run / REPAIR_PLY, run / REPAIR_JSON]

    def is_done(self, ctx: StageContext) -> bool:
        run = latest_run(ctx)
        if _snap_skipped(run):
            return True                               # snap 이 건너뜀 -> 메울 입력 없음
        return super().is_done(ctx) and _mtime(run / REPAIR_PLY) >= _mtime(run / SNAP_PLY)

    def run(self, ctx: StageContext) -> StageResult:
        import numpy as np
        from ..colmap import model as cmodel
        from ..walls import repair
        from .walls import source_dir

        p, wc = ctx.paths, ctx.project.cfg.walls
        run = latest_run(ctx)
        src = run / SNAP_PLY
        if _snap_skipped(run):
            return StageResult(True, "건너뜀: snap 이 건너뜀 (walls.json skipped)", [], skipped=True)
        if not src.exists():
            return StageResult(False, f"{src} 없음 (snap 먼저)")
        if not p.planes_json.exists():
            return StageResult(False, "walls/planes.json 없음 (walls 단계 먼저)")
        all_planes, fit, _ = load_planes(p)
        planes = fitted_planes(all_planes, fit)
        try:                                          # snap 이 실제로 쓴 면 (바닥·천장 밀도 확인을 통과한 것)
            used = schemas.read_json(run / SNAP_JSON).get("planes_used")
        except Exception:  # noqa: BLE001
            used = None
        if used is not None:
            planes = [q for q in planes if q.id in set(used)]
        if len(planes) < len(all_planes):
            ctx.log(f"메우기에 쓰는 면 {len(planes)}/{len(all_planes)} (맞춘 면·확인된 면만)")
        cams, m = None, None
        mdir = source_dir(ctx) / "sparse" / "0"
        if mdir.exists():
            m = cmodel.read_model(mdir)
            cams = np.array([im.center for im in m.images.values()]) if m.images else None
        texture = None
        if wc.texture:
            texture = self._textures(ctx, planes, m)
        info = repair.repair_file(src, run / REPAIR_PLY, planes, cameras=cams, texture=texture,
                                  min_cover=wc.repair_min_cover, max_hole_m2=wc.repair_max_hole_m2)
        schemas.write_json(run / REPAIR_JSON, {"version": schemas.VERSION, **_jsonable(info)})
        return StageResult(True, f"메운 스플랫 {info.get('n_added', 0):,}개"
                           + (f" (면 텍스처 {len(texture)}개)" if texture else ""), self.outputs(ctx))

    def _textures(self, ctx: StageContext, planes, model):
        """walls/tex/ 에 면 텍스처가 있으면 읽고(사용자가 고친 PNG 포함), 없으면 사진으로 만들어 저장.
        실패하면 None (repair 는 둘레 기증자 색 + 색장 보정으로 메움)."""
        from ..walls import facetex
        from .walls import source_dir
        p = ctx.paths
        try:
            if p.textures.is_dir() and any(p.textures.glob("*.json")):
                tex = facetex.load_textures(p.textures, planes)
                ctx.log(f"면 텍스처 {len(tex)}개 읽음 ({p.textures})")
                return tex or None
            if model is None:
                return None
            tex = facetex.build_face_textures(planes, model, source_dir(ctx) / "images")
            facetex.save_textures(tex, p.textures, planes)
            ctx.log(f"면 텍스처 {len(tex)}개 만듦 -> {p.textures} (PNG 를 고치면 다음 repair 에 반영)")
            return tex or None
        except Exception as e:  # noqa: BLE001
            ctx.log(f"경고: 면 텍스처 실패 {e} -> 둘레 색으로 메움")
            return None


def _snap_skipped(run: Path) -> bool:
    js = run / SNAP_JSON
    if not js.exists() or (run / SNAP_PLY).exists():
        return False
    try:
        return bool(schemas.read_json(js).get("skipped"))
    except Exception:  # noqa: BLE001
        return False


def _jsonable(v):
    import numpy as np
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v
