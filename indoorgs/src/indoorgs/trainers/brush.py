"""Brush 0.3.0 (Apache 2.0 로 알려짐, 배포 전 확인). 기본 학습기.

옵션 이름은 brush_app.exe 0.3.0 --help 로 확인 (2026-11-27, 3_train/brush_eval.ps1).
main 버전은 --total-steps 가 --total-train-iters 로 바뀜 -> detect 에서 걸러짐.
자세 보정 없음. 복도 1: PSNR 25.75, 반투명 86.9% (VR 에서 비침 경고 대상).
"""
from __future__ import annotations

from pathlib import Path

from .base import Preset, TrainerAdapter

EVAL_EVERY = 8   # 8장마다 1장 평가 (복도 1 실험들과 같음)


class Brush(TrainerAdapter):
    name = "brush"
    required_options = ["--max-splats", "--sh-degree", "--max-resolution", "--total-steps",
                        "--export-every", "--export-path", "--eval-split-every"]

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
