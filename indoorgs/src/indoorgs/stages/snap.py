"""4½. 벽에 붙이기 (train 뒤, fill 앞). train/<run>/splat.ply + walls/planes.json -> train/<run>/splat_walls.ply, walls.json

- 벽 뒤(문 구멍이 아닌 곳)와 바닥 아래·천장 위 부유물을 지우고
- 면에서 band 안의 스플랫을 면 위로 옮겨 법선 방향으로 납작하게 (walls/snap.py)
- 원본 splat.ply 는 그대로. fill·merge 는 splat_walls.ply 가 있으면 그것을 씀 (stages/fill.py train_ply)
- 맞춘 면(planes.json fit source "fit")만 씀. 바닥·천장은 스플랫 밀도 봉우리로 한 번 더 확인 (walls/snap.py).
  불투명 스플랫 절반 넘게 도면 방 밖이면 도면 좌표 PLY 가 아닌 것으로 보고 건너뜀 (walls.json skipped).
- 낡은 결과 고르지 않기: best_ply 는 설정(walls.snap/repair, cleanup.enabled)과 시각을 확인
  (repaired >= 그 입력, clean >= 그 입력, walls >= splat.ply. 입력 이름은 clean.json·repair.json "input" 과도 대조)

정리(Cleanup, "cleanup"): snap 뒤·repair 앞. splat_walls.ply(유효할 때) 또는 splat.ply -> splat_clean.ply, clean.json,
removed_clean.ply (지운 스플랫만, 사람이 열어 확인용). 도면 좌표가 아닌 구간에서도 돎 (walls/cleanup.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import schemas
from ..walls import planes as wp
from .base import Stage, StageContext, StageResult
from .walls import walls_apply

SNAP_PLY = "splat_walls.ply"
SNAP_JSON = "walls.json"
REPAIR_PLY = "splat_repaired.ply"
REPAIR_JSON = "repair.json"
CLEAN_PLY = "splat_clean.ply"
CLEAN_JSON = "clean.json"
REMOVED_PLY = "removed_clean.ply"


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


def chain_flags(project, paths) -> tuple[bool, bool, bool]:
    """(snap, repair, cleanup) 결과를 써도 되나 - 지금 설정·좌표 기준. best_ply(run, *chain_flags(..)) 로 씀."""
    snap_on, rep_on = walls_ply_flags(project, paths)
    return snap_on, rep_on, bool(project.cfg.cleanup.enabled)


def train_ply(ctx: StageContext) -> Path:
    """fill·merge 가 쓸 학습 결과: 메운 것 > 정리한 것 > 벽에 붙인 것 > splat.ply (설정이 켜져 있고 입력보다 새것일 때만)."""
    return best_ply(latest_run(ctx), *chain_flags(ctx.project, ctx.paths))


def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else -1.0


def _made_from(out: Path, js: Path, src: Path) -> bool:
    """out 이 있고 src 보다 새것이고, js 에 적힌 입력 이름("input")이 있으면 src 와 같음 (설정을 바꿔 입력이 달라진 경우)."""
    if not out.exists() or _mtime(out) < _mtime(src):
        return False
    if js.exists():
        try:
            name = schemas.read_json(js).get("input")
        except Exception:  # noqa: BLE001
            name = None
        if name and name != src.name:
            return False
    return True


def best_ply(run: Path, use_snap: bool = True, use_repair: bool = True, use_clean: bool = True) -> Path:
    """splat_repaired.ply > splat_clean.ply > splat_walls.ply > splat.ply. 각자 설정이 켜져 있고 바로 앞 입력보다 새것일 때만
    (시각이 거꾸로면 - 앞 단계를 다시 돌렸는데 뒤 단계는 안 돎 - 낡은 결과로 보고 안 씀).
    입력: walls <- splat.ply, clean <- (유효한 walls, 없으면 splat.ply), repaired <- (유효한 clean, 없으면 walls; snap 필요)."""
    cur = run / "splat.ply"
    walls = run / SNAP_PLY
    walls_ok = use_snap and walls.exists() and _mtime(walls) >= _mtime(cur)
    if walls_ok:
        cur = walls
    if use_clean and _made_from(run / CLEAN_PLY, run / CLEAN_JSON, cur):
        cur = run / CLEAN_PLY
    if walls_ok and use_repair and _made_from(run / REPAIR_PLY, run / REPAIR_JSON, cur):
        cur = run / REPAIR_PLY
    return cur


def cleanup_input(ctx: StageContext) -> Path:
    """cleanup 의 입력: 유효한 splat_walls.ply, 아니면 splat.ply."""
    return best_ply(latest_run(ctx), walls_ply_flags(ctx.project, ctx.paths)[0], False, False)


def repair_input(ctx: StageContext) -> Path:
    """repair 의 입력: 유효한 splat_clean.ply (walls 에서 만든 것), 아니면 splat_walls.ply."""
    run = latest_run(ctx)
    snap_on, _, clean_on = chain_flags(ctx.project, ctx.paths)
    cur = best_ply(run, snap_on, False, clean_on)
    return cur if cur.name == CLEAN_PLY and _made_from(cur, run / CLEAN_JSON, run / SNAP_PLY) else run / SNAP_PLY


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
        return [run / SNAP_PLY, run / SNAP_JSON, run / CLEAN_PLY, run / CLEAN_JSON, run / REMOVED_PLY,
                run / REPAIR_PLY, run / REPAIR_JSON]

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
        for old in (out, run / CLEAN_PLY, run / REMOVED_PLY, run / REPAIR_PLY):   # 낡은 결과가 남지 않게 (fill 이 잘못 고름)
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
    """4¾. 면 메우기 (snap·cleanup 뒤). splat_clean.ply(유효할 때) 또는 splat_walls.ply -> splat_repaired.ply, repair.json

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
        return super().is_done(ctx) and _made_from(run / REPAIR_PLY, run / REPAIR_JSON, repair_input(ctx))

    def run(self, ctx: StageContext) -> StageResult:
        import numpy as np
        from ..colmap import model as cmodel
        from ..walls import repair
        from .walls import source_dir

        p, wc = ctx.paths, ctx.project.cfg.walls
        run = latest_run(ctx)
        src = repair_input(ctx)
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
        ctx.log(f"입력 {src.name}")
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
        schemas.write_json(run / REPAIR_JSON, {"version": schemas.VERSION, "input": src.name, **_jsonable(info)})
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


class Cleanup(Stage):
    """4⅝. 학습 후 자동 정리 (snap 뒤, repair 앞). 모든 구간 (도면 좌표가 아니어도: 부유물·바늘·거대·SH 는 스플랫만 봄).
    입력 cleanup_input (splat_walls.ply 유효하면 그것, 아니면 splat.ply) -> splat_clean.ply, clean.json, removed_clean.ply.
    - 도면 좌표면 planes.json 의 면 + plan 을 넘겨 면 근처는 보호하고 방 밖은 남기며 "air" 기준을 씀.
      면은 맞춘 면(fit)과, 맞추지 못한 면은 도면 위치 그대로 모두 넘김: cleanup 에서 면은 지우지 않게 '보호'만 하므로
      더 넓게 주는 쪽이 안전 (맞춘 면만 주면 점이 없어 못 맞춘 흰 천장이 air 기준으로 통째로 지워질 수 있음).
    - COLMAP 모델(data_clean/sparse/0, 없으면 data)과 카메라 중심 -> "through"·"camera" 기준.
    - 바늘 쪼개기로 스플랫 수가 늘어남 (기록만: clean.json quality_before/after count, 보고서)."""
    name = "cleanup"
    invalidate_when_skipped = True

    def applies(self, ctx: StageContext) -> bool:
        return bool(ctx.project.cfg.cleanup.enabled)

    def outputs(self, ctx: StageContext) -> list[Path]:
        run = latest_run(ctx)
        return [run / CLEAN_PLY, run / CLEAN_JSON]

    def prev_targets(self, ctx: StageContext) -> list[Path]:
        run = latest_run(ctx)
        return [run / CLEAN_PLY, run / CLEAN_JSON, run / REMOVED_PLY, run / REPAIR_PLY, run / REPAIR_JSON]

    def is_done(self, ctx: StageContext) -> bool:
        run = latest_run(ctx)
        src = cleanup_input(ctx)
        return super().is_done(ctx) and src.exists() and _made_from(run / CLEAN_PLY, run / CLEAN_JSON, src)

    def run(self, ctx: StageContext) -> StageResult:
        from ..colmap import model as cmodel
        from ..colmap.runner import move_aside
        from ..walls import cleanup
        from .diagnose import model_in_plan_coords
        from .walls import source_dir

        p, cc = ctx.paths, ctx.project.cfg.cleanup
        run = latest_run(ctx)
        src = cleanup_input(ctx)
        if not src.exists():
            return StageResult(False, f"{src} 없음 (train 먼저)")
        for old in (run / CLEAN_PLY, run / REMOVED_PLY, run / REPAIR_PLY):   # 낡은 결과는 덮지 않고 옆으로
            if old.exists():
                ctx.log(f"낡은 {old.name} -> {move_aside(old, 'prev').name}")
        in_plan = bool(ctx.project.has_space and model_in_plan_coords(ctx))
        plan, planes = None, None
        if in_plan:
            plan = schemas.load_plan(ctx.project.plan_json)
            if walls_apply(ctx) and p.planes_json.exists():
                planes, fit, _ = load_planes(p)
                nf = len(fitted_planes(planes, fit))
                ctx.log(f"보호할 면 {len(planes)}개 (맞춘 면 {nf}, 나머지는 도면 위치)")
            else:
                ctx.log("walls/planes.json 없음 -> 면 보호·air 는 도면 면으로")
        else:
            ctx.log("도면 좌표가 아님: 면·방 기준 없이 일반 정리만 (미터 단위 문턱은 정합 축척에 따라 근사)")
        model, cams = None, None
        mdir = source_dir(ctx) / "sparse" / "0"
        if mdir.exists():
            model = cmodel.read_model(mdir)
            cams = np.array([im.center for im in model.images.values()]) if model.images else None
        steps = tuple(cc.steps)
        ctx.log(f"입력 {src.name}, 단계 {', '.join(steps)}")
        info = cleanup.cleanup_file(src, run / CLEAN_PLY, planes=planes, plan=plan, cameras=cams, model=model,
                                    removed_ply=run / REMOVED_PLY, steps=steps)
        info.update(input=src.name, plan_coords=in_plan, n_planes=len(planes) if planes else 0,
                    model=str(mdir) if model is not None else None)
        schemas.write_json(run / CLEAN_JSON, {"version": schemas.VERSION, **_jsonable(info)})
        return StageResult(True, clean_summary(info), self.outputs(ctx))


def clean_summary(info: dict) -> str:
    """clean.json -> 한 줄 (단계 로그·보고서)."""
    fl = info.get("floaters") or {}
    hg = info.get("huge") or {}
    nd = info.get("needles") or {}
    sh = info.get("sh") or {}
    rb = {k: v for k, v in (fl.get("removed_by") or {}).items() if v}
    if hg.get("n_removed"):
        rb["huge"] = hg["n_removed"]
    n0 = (info.get("quality_before") or {}).get("count")
    n1 = (info.get("quality_after") or {}).get("count")
    parts = ["지움 " + (", ".join(f"{k} {v:,}" for k, v in rb.items()) if rb else "지움 0")]
    if nd:
        parts.append(f"바늘 {nd.get('n_needles', 0):,}개 -> 조각 {nd.get('n_pieces', 0):,}")
    if hg.get("n_split"):
        parts.append(f"거대 쪼갬 {hg['n_split']:,}")
    if sh:
        parts.append(f"SH 줄임 {sh.get('n_damped', 0):,}")
    if n0 is not None and n1 is not None:
        parts.append(f"스플랫 {n0:,} -> {n1:,} ({(n1 / n0 - 1) if n0 else 0:+.1%})")
    return ", ".join(parts)


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
