"""4. 학습 (담당 A). data_clean/ (없으면 data/) -> train/<trainer>_<preset>/splat.ply, eval.json, stats.json

- trainers/ 어댑터로 detect(실행 파일·--help 옵션 확인) -> 명령 -> runner.run (출력은 run 폴더 train_log.txt)
- 결과 PLY -> splat.ply (같은 드라이브면 하드 링크, 아니면 복사. 원본 export_*.ply 는 그대로)
- eval.json: 어댑터 evaluate (Brush: 평가 렌더 비교, LichtFeld: metrics.csv)
- stats.json: 저장소 tools/ply_stats.py 정의. translucent_ratio = 불투명도 0.5 미만 비율,
  needle_ratio = 가장 긴 축 / 중간 축 > 10 비율 (ply_stats.py 의 needle10. CONTRACT 문구는 '가장 짧은 축'이지만
  같은 정의라고 했으므로 실제 도구를 따름)
- 실패하면 run 폴더를 <이름>_failed_<시각> 으로 옆으로
- 다시 돌리기(--rerun train): 최신 폴더가 끝났으면 덮어쓰지 않고 <trainer>_<preset>_2, _3 ... 새 폴더
  (SectionPaths.latest_train_run / new_train_run). fill·merge·report 는 최신 폴더를 씀
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

from .. import schemas
from ..colmap import runner
from ..colmap.runner import move_aside
from ..trainers import REGISTRY
from .base import Stage, StageContext, StageResult

_PLY_TYPES = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8", "uchar": "u1", "uint8": "u1",
              "char": "i1", "int8": "i1", "ushort": "u2", "uint16": "u2", "short": "i2", "int16": "i2",
              "uint": "u4", "uint32": "u4", "int": "i4", "int32": "i4"}


def read_ply_fields(path: Path, names: list[str]) -> dict[str, np.ndarray]:
    """binary_little_endian PLY 의 vertex 속성을 이름으로 읽음 (Brush 는 xyz 가 맨 뒤, LichtFeld 는 맨 앞).
    memmap 이라 300만 개도 필요한 열만 메모리에 올림.
    plyio.stats 와 같은 정의지만, plyio.read 는 전체를 올리므로 통계용으로 이 가벼운 읽기를 따로 둠"""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"PLY 아님: {path}")
        fmt, n, props, in_vertex = None, 0, [], False
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"PLY 헤더 끝이 없음: {path}")
            s = line.decode("ascii", "replace").strip()
            if s.startswith("format"):
                fmt = s.split()[1]
            elif s.startswith("element"):
                _, el, cnt = s.split()
                in_vertex = el == "vertex"
                if in_vertex:
                    n = int(cnt)
                elif not props:
                    raise ValueError("vertex 가 첫 element 가 아님")
            elif s.startswith("property") and in_vertex:
                parts = s.split()
                if parts[1] == "list":
                    raise ValueError("vertex 에 list 속성은 지원 안 함")
                props.append((parts[2], "<" + _PLY_TYPES[parts[1]]))
            elif s == "end_header":
                break
        off = f.tell()
    if fmt != "binary_little_endian":
        raise ValueError(f"지원 안 하는 PLY 형식 {fmt}")
    arr = np.memmap(path, dtype=np.dtype(props), mode="r", offset=off, shape=(n,))
    missing = [k for k in names if k not in arr.dtype.names]
    if missing:
        raise ValueError(f"PLY 에 속성 없음: {missing}")
    out = {k: np.asarray(arr[k], np.float64) for k in names}
    del arr
    return out


def ply_stats(path: Path) -> dict:
    d = read_ply_fields(path, ["opacity", "scale_0", "scale_1", "scale_2"])
    s = np.exp(np.stack([d["scale_0"], d["scale_1"], d["scale_2"]], 1))
    s.sort(1)
    op = 1 / (1 + np.exp(-d["opacity"]))
    ratio = s[:, 2] / np.maximum(s[:, 1], 1e-12)
    n = len(op)
    return {"version": schemas.VERSION, "count": int(n),
            "translucent_ratio": round(float(np.mean(op < 0.5)), 4) if n else 0.0,
            "needle_ratio": round(float(np.mean(ratio > 10)), 4) if n else 0.0}


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


class Train(Stage):
    name = "train"
    uses_gpu = True

    def _run_dir(self, ctx: StageContext) -> Path:
        t = ctx.project.cfg.train
        return ctx.paths.latest_train_run(t.trainer, t.preset)

    @staticmethod
    def _files(run: Path) -> list[Path]:
        return [run / "splat.ply", run / "eval.json", run / "stats.json"]

    def outputs(self, ctx: StageContext) -> list[Path]:
        return self._files(self._run_dir(ctx))

    def prev_targets(self, ctx: StageContext) -> list[Path]:
        return []          # 이전 학습은 그 자리에 두고 새 번호 폴더에 학습

    def _target(self, ctx: StageContext) -> Path:
        """이번에 학습할 폴더. 최신이 끝났고 다시 돌리는 중이면 새 번호, 아니면 최신(끝나지 않은 것은 run 에서 옆으로)."""
        run = self._run_dir(ctx)
        if ctx.rerun and all(p.exists() for p in self._files(run)):
            t = ctx.project.cfg.train
            return ctx.paths.new_train_run(t.trainer, t.preset)
        return run

    def run(self, ctx: StageContext) -> StageResult:
        cfg, p = ctx.project.cfg, ctx.paths
        ad = REGISTRY[cfg.train.trainer]
        exe = getattr(cfg.tools, cfg.train.trainer)
        info = ad.detect(exe)
        if not info.found:
            return StageResult(False, info.message)
        if info.missing_options:
            return StageResult(False, f"{info.message} - 학습기 버전이 다름 ({exe})")
        presets = ad.presets()
        if cfg.train.preset not in presets:
            return StageResult(False, f"프리셋 없음: {cfg.train.preset} (가능: {list(presets)})")
        preset = presets[cfg.train.preset]
        # data_seed(벽 면에 초기 점을 심은 것, stages/walls.py) > data_clean > data
        data_dir = next((d for d in (p.data_seed, p.data_clean) if (d / "sparse" / "0").exists()), p.data)
        if data_dir == p.data_seed:
            ctx.log("data_seed (벽 면 초기 점 포함) 로 학습")
        if data_dir == p.data:
            ctx.log("참고: data_clean 없음 -> data 로 학습")
        if not (data_dir / "sparse" / "0").exists():
            return StageResult(False, f"학습 입력 없음: {data_dir}/sparse/0 (register 먼저)")

        run = self._target(ctx)
        if run.exists():
            ctx.log(f"끝나지 않은 {run.name} -> {move_aside(run).name}")
        run.mkdir(parents=True)
        cmd = ad.command(exe, data_dir, run, preset, cfg.gpu_profile)
        ctx.log(f"{ad.name} {info.version or ''} 프리셋 {cfg.train.preset}: {preset.note}")
        r = runner.run(cmd, log_file=run / "train_log.txt")
        result = ad.find_result(run)
        if not r.ok or result is None:
            tail = " / ".join(r.tail[-3:])
            old = move_aside(run)
            return StageResult(False, f"학습 실패 (코드 {r.code}, PLY {'있음' if result else '없음'}): {tail} "
                                      f"-> {old.name if old else '-'}")
        splat = run / "splat.ply"
        link_or_copy(result, splat)
        try:
            stats = ply_stats(splat)
        except Exception as e:  # noqa: BLE001
            old = move_aside(run)
            return StageResult(False, f"PLY 읽기 실패: {e} -> {old.name if old else '-'}")
        schemas.write_json(run / "stats.json", stats)
        ev = ad.evaluate(run, data_dir)
        if ev is None:
            ctx.log("경고: 평가 결과 없음 (eval.json 에 psnr=null)")
            ev = {"psnr": None, "ssim": None, "n_eval": 0}
        schemas.write_json(run / "eval.json", {"version": schemas.VERSION, **ev})
        msg = (f"{stats['count']:,}개, 반투명 {stats['translucent_ratio'] * 100:.1f}%, "
               f"PSNR {ev['psnr'] if ev['psnr'] is not None else '-'}")
        return StageResult(True, f"{run.name}: {msg}", self._files(run))
