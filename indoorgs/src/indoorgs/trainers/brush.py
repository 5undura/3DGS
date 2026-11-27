"""Brush 0.3.0 (Apache 2.0 로 알려짐, 배포 전 확인). 기본 학습기.

옵션 이름은 brush_app.exe 0.3.0 --help 로 확인 (2026-11-27, 3_train/brush_eval.ps1).
main 버전은 --total-steps 가 --total-train-iters 로 바뀜 -> detect 에서 걸러짐.
자세 보정 없음. 복도 1: PSNR 25.75, 반투명 86.9% (VR 에서 비침 경고 대상).
"""
from __future__ import annotations

from pathlib import Path

from .base import Preset, TrainerAdapter


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
                "--eval-split-every", "8", "--eval-every", str(preset.iters), "--eval-save-to-disk",
                *preset.extra]

    def find_result(self, out_dir: Path) -> Path | None:
        plys = sorted(out_dir.glob("export_*.ply"), key=lambda p: int(p.stem.split("_")[-1]))
        return plys[-1] if plys else None
