"""LichtFeld Studio v0.5.4.3 (GPLv3). 고화질 옵션: 사용자가 따로 빌드한 exe 를 프로세스로만 실행 (코드 포함 안 함).

옵션 이름은 v0.5.4.3 소스 argument_parser.cpp / --help 로 확인 (2026-11-27, 3_train/lfs_eval.ps1, PROGRESS.md).
- --bg-modulation 은 v0.5.3 부터 없음 -> --bg-mode=modulation
- v0.5.4 부터 --export=ply 를 줘야 <out>/splat_<iter>.ply 가 생김
- MRNF 는 기본 --exposure-correction 과 함께 씀 (--ppisp 와는 같이 못 씀)
- mip 필터, --gut 는 끔 (Unity 셰이더와 어긋남)
복도 1 최고(L7): MRNF + --pose-opt + 80,000회 -> PSNR 26.79, 반투명 49.4%, 91분 (3060 Ti).
"""
from __future__ import annotations

from pathlib import Path

from .base import Preset, TrainerAdapter

EVAL_EVERY = 8


class LichtFeld(TrainerAdapter):
    name = "lichtfeld"
    # v0.5.4.3 소스(src/core/argument_parser.cpp): --init <ply> 로 고친 PLY 에서 시작은 되지만 일정(학습률·밀도 조절)이
    # 0 부터 다시이고 밀도 조절을 끄는 CLI 옵션이 없음 (start_refine/stop_refine 는 --config json 에만, 전략별 동작 미확인).
    # --resume 은 자기 체크포인트만. -> 안전한 '짧은 마무리 학습' 방법이 없어 finetune 은 건너뜀
    finetune_unsupported = ("LichtFeld 는 밀도 조절을 끄고 고정 학습률로 이어 학습하는 확인된 옵션이 없음 "
                            "(--init 은 일정을 0 부터 다시 시작) -> finetune 건너뜀")
    required_options = ["--strategy", "--max-cap", "--iter", "--sh-degree", "--max-width",
                        "--bg-mode", "--export", "--pose-opt", "--test-every", "--headless"]

    def presets(self) -> dict[str, Preset]:
        return {
            "default": Preset(1920, 3_000_000, 2, 80_000, ["--strategy=mrnf", "--pose-opt"],
                              note="복도 1 최고 L7: 26.79 / 반투명 49.4%, 91분"),
            "quick": Preset(1920, 3_000_000, 2, 30_000, ["--strategy=mrnf", "--pose-opt"],
                            note="L3 수준: 26.52 / 반투명 60.9%, 42분"),
            "fast": Preset(1920, 1_000_000, 2, 5_000, ["--strategy=mrnf"], note="파이프라인 점검용"),
        }

    def command(self, exe, data_dir: Path, out_dir: Path, preset: Preset, gpu: str) -> list[str]:
        return [exe, "-d", str(data_dir), "-o", str(out_dir),
                f"--max-cap={self.clamp_cap(preset, gpu)}", f"--iter={preset.iters}",
                f"--sh-degree={preset.sh}", f"--max-width={preset.width}",
                "--bg-mode=modulation", "--export=ply", "--eval", f"--test-every={EVAL_EVERY}", "--headless",
                *preset.extra]

    def find_result(self, out_dir: Path) -> Path | None:
        plys = sorted((p for p in out_dir.rglob("splat_*.ply") if p.stem.split("_")[-1].isdigit()),
                      key=lambda p: int(p.stem.split("_")[-1]))
        return plys[-1] if plys else None

    def evaluate(self, run_dir: Path, data_dir: Path) -> dict | None:
        """<out>/metrics.csv 의 마지막(가장 큰 iteration) 줄.
        열 이름은 v0.5.4.3 소스 src/training/metrics/metrics.hpp EvalMetrics::to_csv_header 로 확인:
        iteration,psnr,ssim,lpips,time_per_image,num_gaussians,... (평가 장수는 csv 에 없음)"""
        import csv
        files = sorted(run_dir.rglob("metrics.csv"))
        best = None
        for f in files:
            with f.open(encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    try:
                        it, ps, ss = int(float(row["iteration"])), float(row["psnr"]), float(row["ssim"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if best is None or it >= best[0]:
                        best = (it, ps, ss)
        if best is None:
            return None
        return {"psnr": round(best[1], 3), "ssim": round(best[2], 4), "n_eval": None,
                "eval_every": EVAL_EVERY, "iteration": best[0]}
