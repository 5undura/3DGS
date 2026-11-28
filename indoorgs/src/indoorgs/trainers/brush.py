"""Brush 0.3.0 (Apache 2.0 로 알려짐, 배포 전 확인). 기본 학습기.

옵션 이름은 brush_app.exe 0.3.0 --help 로 확인 (2026-11-27, 3_train/brush_eval.ps1).
main 버전은 --total-steps 가 --total-train-iters 로 바뀜 -> detect 에서 걸러짐.
자세 보정 없음. 복도 1: PSNR 25.75, 반투명 86.9% (VR 에서 비침 경고 대상).
"""
from __future__ import annotations

from pathlib import Path

from .base import Preset, TrainerAdapter

EVAL_EVERY = 8   # 8장마다 1장 평가 (복도 1 실험들과 같음)

# ---- 후처리 뒤 재학습(finetune) 옵션 ----
# Brush v0.3.0 소스로 확인 (clap 이 필드 이름을 --kebab-case 로 만듦, 2026-11-28):
#   crates/brush-process/src/config.rs ProcessConfig.start_iter ("Iteration to resume from", 기본 0)
#   crates/brush-train/src/config.rs TrainConfig.lr_mean(2e-5)/lr_mean_end(1e-6)/lr_scale(1e-2)/lr_scale_end(6e-3)/
#     growth_stop_iter(15000)/refine_every(200)
#   crates/brush-dataset/src/formats/mod.rs: 데이터 폴더(하위 포함)에 .ply 가 하나뿐이거나 이름이 init.ply 인 것을 초기 스플랫으로 읽음
#   crates/brush-train/src/train.rs: 학습률은 지수 감소 decay = (end/start)^(1/total_steps) 를 start_iter 부터 처음 값으로 다시 시작
#     -> 시작 = 끝 으로 주면 고정. refine_if_needed 는 iter % refine_every == 0 마다 죽은 스플랫 지우고 다시 뽑음
#     (growth_stop_iter 뒤에도) -> refine_every 를 total_steps 보다 크게 주면 개수 그대로.
# TODO(PC에서 brush_app.exe --help 로 확인): 아래 옵션 이름 (소스로만 확인. detect 가 --help 에 없으면 멈춤)
OPT_START_ITER = "--start-iter"
OPT_LR_MEAN = "--lr-mean"
OPT_LR_MEAN_END = "--lr-mean-end"
OPT_LR_SCALE = "--lr-scale"
OPT_LR_SCALE_END = "--lr-scale-end"
OPT_GROWTH_STOP = "--growth-stop-iter"
OPT_REFINE_EVERY = "--refine-every"
INIT_PLY = "init.ply"            # 데이터 폴더 안 이 이름의 PLY 를 초기 스플랫으로 (formats/mod.rs)


class Brush(TrainerAdapter):
    name = "brush"
    required_options = ["--max-splats", "--sh-degree", "--max-resolution", "--total-steps",
                        "--export-every", "--export-path", "--eval-split-every"]
    finetune_options = [OPT_START_ITER, OPT_LR_MEAN, OPT_LR_MEAN_END, OPT_LR_SCALE, OPT_LR_SCALE_END,
                        OPT_GROWTH_STOP, OPT_REFINE_EVERY]
    finetune_unsupported = ""

    def presets(self) -> dict[str, Preset]:
        return {
            "default": Preset(1920, 3_000_000, 2, 30_000, note="기준선 (복도 1: 25.75 / 반투명 86.9%, 36분)"),
            "fast": Preset(1920, 1_000_000, 2, 5_000, note="파이프라인 점검용"),
        }

    def command(self, exe, data_dir: Path, out_dir: Path, preset: Preset, gpu: str) -> list[str]:
        return [exe, str(data_dir),
                "--max-splats", str(self.clamp_cap(preset, gpu)), "--sh-degree", str(preset.sh),
                "--max-resolution", str(preset.width), "--total-steps", str(preset.iters),
                "--export-every", str(preset.iters), "--export-path", str(out_dir),
                "--eval-split-every", str(EVAL_EVERY), "--eval-every", str(preset.iters), "--eval-save-to-disk",
                *preset.extra]

    def finetune_command(self, exe, data_dir: Path, out_dir: Path, preset: Preset, start_iter: int,
                         iters: int, lr_mean: float, lr_scale: float, refine: bool) -> list[str]:
        """data_dir/init.ply 에서 시작. 반복 번호를 start_iter 부터 이어 export_<start+iters>.ply.
        밀도 조절(늘리기)은 growth_stop_iter = start_iter 로 끄고, refine=False 면 지우고 다시 뽑기도 끔
        (refine_every > total_steps). 위치·크기 학습률은 시작 = 끝 (고정). --max-splats 는 안 줌 (늘리지 않으므로;
        후처리로 프리셋 상한보다 많아졌어도 그대로). 해상도·SH·평가 나누기는 원래 학습과 같게."""
        total = start_iter + iters
        cmd = [exe, str(data_dir),
               "--sh-degree", str(preset.sh), "--max-resolution", str(preset.width),
               "--total-steps", str(total), OPT_START_ITER, str(start_iter),
               OPT_LR_MEAN, f"{lr_mean:g}", OPT_LR_MEAN_END, f"{lr_mean:g}",
               OPT_LR_SCALE, f"{lr_scale:g}", OPT_LR_SCALE_END, f"{lr_scale:g}",
               OPT_GROWTH_STOP, str(start_iter)]
        if not refine:
            cmd += [OPT_REFINE_EVERY, str(total + 1)]
        cmd += ["--export-every", str(total), "--export-path", str(out_dir),
                "--eval-split-every", str(EVAL_EVERY), "--eval-every", str(total), "--eval-save-to-disk"]
        return cmd

    def find_result(self, out_dir: Path) -> Path | None:
        plys = sorted((p for p in out_dir.glob("export_*.ply") if p.stem.split("_")[-1].isdigit()),
                      key=lambda p: int(p.stem.split("_")[-1]))
        return plys[-1] if plys else None

    def evaluate(self, run_dir: Path, data_dir: Path) -> dict | None:
        """Brush CLI 는 점수를 안 냄 -> --eval-save-to-disk 가 남긴 eval_<iter>/*.png 를 원본과 비교
        (저장소 tools/eval_renders.py 방식). 폴더 이름은 3_train/brush_eval.ps1 기준.
        TODO(PC에서 확인): Brush 0.3.0 의 평가 렌더 폴더·파일 이름 (eval_<iter>/<사진이름>.png)"""
        from .metrics import compare_dir
        dirs = [d for d in run_dir.glob("eval_*") if d.is_dir() and d.name.split("_")[-1].isdigit()]
        if not dirs:
            return None
        last = max(dirs, key=lambda d: int(d.name.split("_")[-1]))
        rows = compare_dir(last, data_dir / "images")
        if not rows:
            return None
        return {"psnr": round(sum(r[1] for r in rows) / len(rows), 3),
                "ssim": round(sum(r[2] for r in rows) / len(rows), 4),
                "n_eval": len(rows), "eval_every": EVAL_EVERY, "iteration": int(last.name.split("_")[-1]),
                "worst": [r[0] for r in sorted(rows, key=lambda r: r[1])[:8]]}
