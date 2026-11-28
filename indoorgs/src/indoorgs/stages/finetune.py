"""5½. 후처리 뒤 짧은 재학습 (fill 뒤, merge 앞). 후처리한 PLY -> train/<run>/finetune/splat_ft.ply, finetune.json,
eval.json, stats.json, train_log.txt

wall_ab RESULTS2 E: snap·floaters 등으로 고친 PLY 에서 1000 반복 더 학습하면 raw + 같은 재학습을 이김 (고친 자리를 장면이 다시 맞춤).
- 입력 = 지금 merge 가 쓸 PLY (finetune_input): fill/holes_backdrop.ply 가 있으면 그것, 없으면
  최신 학습 폴더의 best_ply (repaired > clean > walls > splat.ply, 설정·시각 확인. stages/snap.best_ply)
- Brush: train/<run>/finetune/data/ 에 학습 데이터(data_clean, 없으면 data) 를 링크한 보기 폴더를 만들고 입력 PLY 를
  init.ply 로 (하드 링크, 안 되면 복사). 원본 data_clean·data_seed 는 안 건드림. 원래 반복 번호(export_<N>.ply 의 N,
  없으면 프리셋 반복 수)에서 이어 finetune.iters 반복: --start-iter N --total-steps N+iters, 위치·크기 학습률 시작=끝
  (Brush 는 --start-iter 로 시작해도 학습률 일정을 처음 값부터 다시 시작하므로), --growth-stop-iter N (늘리기 없음),
  refine=False 면 --refine-every 를 total 보다 크게 (지우고 다시 뽑기도 없음). 옵션 이름: trainers/brush.py OPT_*.
- LichtFeld: 밀도 조절을 끄는 확인된 방법이 없어 건너뜀 (applies 에서 로그). trainers/lichtfeld.py
- 입력 PLY 는 덮어쓰지 않음. 다시 돌리면 이전 finetune/ 은 finetune_prev_<시각> 로, 실패하면 finetune_failed_<시각> 로.
- merge 는 finetune_valid 일 때만 splat_ft.ply 를 씀: 설정이 켜져 있고(학습기 지원), finetune.json 의 입력 경로·입력 시각·
  설정이 지금과 같고, splat_ft.ply 가 입력보다 새것. 앞 단계를 다시 돌려 입력이 바뀌면 낡은 결과로 보고 안 씀.
"""
from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

from .. import schemas
from ..colmap import runner
from ..colmap.runner import move_aside
from ..config import FINETUNE_TRAINERS
from ..trainers import REGISTRY
from .base import Stage, StageContext, StageResult

FT_DIR = "finetune"
FT_PLY = "splat_ft.ply"
FT_JSON = "finetune.json"
DATA_VIEW = "data"


def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else -1.0


def _latest_run(project, paths) -> Path:
    t = project.cfg.train
    return paths.latest_train_run(t.trainer, t.preset)


def ft_dir(project, paths) -> Path:
    return _latest_run(project, paths) / FT_DIR


def ft_supported(project) -> tuple[bool, str]:
    """(이 학습기로 finetune 가능, 아니면 이유)."""
    t = project.cfg.train.trainer
    ad = REGISTRY.get(t)
    if ad is None or t not in FINETUNE_TRAINERS or ad.finetune_unsupported:
        return False, (ad.finetune_unsupported if ad is not None and ad.finetune_unsupported
                       else f"{t}: finetune 지원 안 함")
    return True, ""


def ft_on(project) -> bool:
    return bool(project.cfg.finetune.enabled) and ft_supported(project)[0]


def finetune_input(project, paths) -> Path:
    """finetune 의 입력 = finetune 이 없을 때 merge 가 쓰는 PLY (fill > best_ply)."""
    if paths.fill_ply.exists():
        return paths.fill_ply
    from .snap import best_ply, chain_flags
    return best_ply(_latest_run(project, paths), *chain_flags(project, paths))


def _rel(p: Path, root: Path) -> str:
    try:
        return p.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(p)


def fingerprint(project) -> dict:
    """결과를 다시 써도 되는지 가르는 설정 (바뀌면 다시 재학습)."""
    f, t = project.cfg.finetune, project.cfg.train
    return {"trainer": t.trainer, "preset": t.preset, "iters": int(f.iters), "lr_mean": float(f.lr_mean),
            "lr_scale": float(f.lr_scale), "refine": bool(f.refine)}


def finetune_valid(project, paths) -> Path | None:
    """써도 되는 splat_ft.ply, 아니면 None (설정 꺼짐·학습기 미지원·입력이 바뀜·설정이 바뀜·낡음)."""
    if not ft_on(project):
        return None
    d = ft_dir(project, paths)
    out, js = d / FT_PLY, d / FT_JSON
    src = finetune_input(project, paths)
    if not (out.exists() and js.exists() and src.exists()):
        return None
    try:
        rec = schemas.read_json(js)
    except Exception:  # noqa: BLE001
        return None
    if rec.get("input") != _rel(src, paths.root) or rec.get("config") != fingerprint(project):
        return None
    if _mtime(out) < _mtime(src):
        return None
    im = rec.get("input_mtime")
    if im is None or abs(float(im) - _mtime(src)) > 1e-3 or rec.get("input_size") != src.stat().st_size:
        return None                                   # 입력이 다른 파일로 바뀜 (예: 예전 것을 되돌려 놓음)
    return out


def section_ply(project, paths) -> Path:
    """merge 가 쓸 구간 PLY: 유효한 splat_ft.ply > fill > best_ply."""
    return finetune_valid(project, paths) or finetune_input(project, paths)


def start_iter(run: Path, preset_iters: int) -> int:
    """원래 학습의 마지막 반복 번호: run 의 export_<N>.ply 중 가장 큰 N, 없으면 프리셋 반복 수."""
    ns = [int(p.stem.split("_")[-1]) for p in run.glob("export_*.ply") if p.stem.split("_")[-1].isdigit()]
    return max(ns) if ns else int(preset_iters)


# ---------------- 데이터 보기 폴더 (원본 폴더에 init.ply 를 넣지 않으려고) ----------------
def _is_dir_link(p: Path) -> bool:
    """심볼릭 링크 또는 Windows 정션 (지울 때 대상이 아니라 링크만 지우려고)."""
    if p.is_symlink():
        return True
    isj = getattr(os.path, "isjunction", None)          # 3.12+
    if isj is not None:
        return bool(isj(p))
    try:
        attrs = getattr(os.lstat(p), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _link_dir(src: Path, dst: Path) -> str:
    """폴더 링크: 심볼릭 링크 > (Windows) 정션 > 파일마다 하드 링크(안 되면 복사). 돌려줌 방식 이름."""
    try:
        os.symlink(src, dst, target_is_directory=True)
        return "symlink"
    except (OSError, NotImplementedError):
        pass
    if os.name == "nt":
        try:
            import _winapi
            _winapi.CreateJunction(str(src), str(dst))
            return "junction"
        except (ImportError, OSError, AttributeError):
            pass
    from .train import link_or_copy
    for f in src.rglob("*"):
        t = dst / f.relative_to(src)
        if f.is_dir():
            t.mkdir(parents=True, exist_ok=True)
        else:
            t.parent.mkdir(parents=True, exist_ok=True)
            link_or_copy(f, t)
    return "copy"


def build_data_view(src_data: Path, view: Path, init_ply: Path) -> dict:
    """view/ 에 src_data 의 맨 위 항목을 링크하고 init_ply 를 view/init.ply 로 (하드 링크, 안 되면 복사)."""
    from ..trainers.brush import INIT_PLY
    from .train import link_or_copy
    view.mkdir(parents=True)
    modes = {}
    for e in sorted(src_data.iterdir()):
        if e.name == INIT_PLY:
            continue
        if e.is_dir():
            modes[e.name] = _link_dir(e, view / e.name)
        else:
            link_or_copy(e, view / e.name)
            modes[e.name] = "file"
    link_or_copy(init_ply, view / INIT_PLY)
    return modes


def remove_data_view(view: Path) -> None:
    """보기 폴더의 폴더 링크만 지움 (링크 대상 = 원본은 그대로). 하드 링크·복사한 파일은 그대로 둠."""
    if not view.is_dir():
        return
    for e in view.iterdir():
        if _is_dir_link(e):
            if e.is_symlink():
                e.unlink()
            else:
                os.rmdir(e)                            # 정션: 링크만 지워짐


def _has_init_ply(d: Path) -> list[Path]:
    from ..trainers.brush import INIT_PLY
    return [p for p in d.rglob(INIT_PLY)]


class Finetune(Stage):
    name = "finetune"
    uses_gpu = True

    def applies(self, ctx: StageContext) -> bool:
        if not ctx.project.cfg.finetune.enabled:
            return False
        ok, why = ft_supported(ctx.project)
        if not ok:
            ctx.log(f"finetune: {why}")
        return ok

    def outputs(self, ctx: StageContext) -> list[Path]:
        d = ft_dir(ctx.project, ctx.paths)
        return [d / FT_PLY, d / FT_JSON, d / "eval.json", d / "stats.json"]

    def is_done(self, ctx: StageContext) -> bool:
        return super().is_done(ctx) and finetune_valid(ctx.project, ctx.paths) is not None

    def prev_targets(self, ctx: StageContext) -> list[Path]:
        return [ft_dir(ctx.project, ctx.paths)]

    def run(self, ctx: StageContext) -> StageResult:
        from .train import ply_stats, link_or_copy
        pr, p = ctx.project, ctx.paths
        cfg, fc = pr.cfg, pr.cfg.finetune
        ad = REGISTRY[cfg.train.trainer]
        run = _latest_run(pr, p)
        if not (run / "splat.ply").exists():
            return StageResult(False, f"{run / 'splat.ply'} 없음 (train 먼저)")
        src = finetune_input(pr, p)
        if not src.exists():
            return StageResult(False, f"입력 PLY 없음: {src}")
        exe = getattr(cfg.tools, cfg.train.trainer)
        info = ad.detect(exe)
        if not info.found:
            return StageResult(False, info.message)
        if info.missing_options:
            return StageResult(False, f"{info.message} - 학습기 버전이 다름 ({exe})")
        try:
            help_text = ad.help_text(exe)
        except Exception as e:  # noqa: BLE001
            return StageResult(False, f"{ad.name}: --help 실행 실패 {e}")
        miss = [o for o in ad.finetune_options if o not in help_text]
        if miss:
            return StageResult(False, f"{ad.name}: finetune 옵션 없음 {miss} (버전 확인, 끄려면 finetune: {{enabled: false}})")
        presets = ad.presets()
        if cfg.train.preset not in presets:
            return StageResult(False, f"프리셋 없음: {cfg.train.preset} (가능: {list(presets)})")
        preset = presets[cfg.train.preset]
        # 사진·카메라만 쓰므로 data_clean (data_seed 는 같은 사진·카메라에 점만 더한 것; init.ply 가 시작 스플랫을 대신함)
        data_src = next((d for d in (p.data_clean, p.data) if (d / "sparse" / "0").exists()), None)
        if data_src is None:
            return StageResult(False, "학습 데이터 없음: data_clean/sparse/0, data/sparse/0 (register 먼저)")
        dup = _has_init_ply(data_src)
        if dup:
            return StageResult(False, f"학습 데이터 폴더에 이미 init.ply 가 있음 ({dup[0]}): Brush 가 어느 것을 쓸지 모름")

        d = run / FT_DIR
        if d.exists():
            ctx.log(f"이전 {d.name} -> {move_aside(d, 'prev').name}")
        d.mkdir(parents=True)
        view = d / DATA_VIEW
        try:
            modes = build_data_view(data_src, view, src)
        except OSError as e:
            remove_data_view(view)
            old = move_aside(d)
            return StageResult(False, f"데이터 보기 폴더 만들기 실패: {e} -> {old.name if old else '-'}")
        s0 = start_iter(run, preset.iters)
        cmd = ad.finetune_command(exe, view, d, preset, s0, fc.iters, fc.lr_mean, fc.lr_scale, fc.refine)
        ctx.log(f"입력 {_rel(src, p.root)} -> 반복 {s0} -> {s0 + fc.iters} (데이터 {data_src.name}, "
                f"위치 학습률 {fc.lr_mean:g}, 크기 {fc.lr_scale:g} 고정, 늘리기 없음"
                f"{'' if fc.refine else ', 지우고 다시 뽑기 없음'})")
        r = runner.run(cmd, log_file=d / "train_log.txt")
        remove_data_view(view)
        result = ad.find_result(d)
        if not r.ok or result is None:
            tail = " / ".join(r.tail[-3:])
            old = move_aside(d)
            return StageResult(False, f"재학습 실패 (코드 {r.code}, PLY {'있음' if result else '없음'}): {tail} "
                                      f"-> {old.name if old else '-'}")
        out = d / FT_PLY
        link_or_copy(result, out)
        try:
            stats = ply_stats(out)
            stats_in = ply_stats(src)
        except Exception as e:  # noqa: BLE001
            old = move_aside(d)
            return StageResult(False, f"PLY 읽기 실패: {e} -> {old.name if old else '-'}")
        schemas.write_json(d / "stats.json", stats)
        ev = ad.evaluate(d, data_src)
        if ev is None:
            ctx.log("경고: 평가 결과 없음 (eval.json 에 psnr=null)")
            ev = {"psnr": None, "ssim": None, "n_eval": 0}
        schemas.write_json(d / "eval.json", {"version": schemas.VERSION, **ev})
        ev0 = None
        try:
            ev0 = schemas.read_json(run / "eval.json").get("psnr")
        except Exception:  # noqa: BLE001
            pass
        schemas.write_json(d / FT_JSON, {
            "version": schemas.VERSION, "input": _rel(src, p.root), "input_mtime": _mtime(src),
            "input_size": src.stat().st_size, "config": fingerprint(pr), "data": _rel(data_src, p.root),
            "data_view": modes, "start_iter": s0, "end_iter": s0 + fc.iters, "result": result.name,
            "count_before": stats_in["count"], "translucent_before": stats_in["translucent_ratio"],
            "psnr_train": ev0, "command": cmd})
        msg = (f"{stats_in['count']:,} -> {stats['count']:,}개, 반투명 {stats_in['translucent_ratio'] * 100:.1f} -> "
               f"{stats['translucent_ratio'] * 100:.1f}%, PSNR {ev0 if ev0 is not None else '-'} (학습) -> "
               f"{ev['psnr'] if ev['psnr'] is not None else '-'}")
        return StageResult(True, msg, self.outputs(ctx))
