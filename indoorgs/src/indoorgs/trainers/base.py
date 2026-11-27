"""학습기 어댑터 규격 (CONTRACT.md 5절)."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Preset:
    width: int = 1920
    cap: int = 3_000_000
    sh: int = 2
    iters: int = 30_000
    extra: list[str] = field(default_factory=list)
    note: str = ""


# GPU 메모리별 상한. 12gb 도 1920 을 기본으로 둠: 복도 1 실험에서 2560 은 이득 없음 (Brush·LichtFeld 둘 다)
GPU_LIMITS = {"8gb": {"max_cap": 3_000_000}, "12gb": {"max_cap": 4_000_000}}


@dataclass
class TrainerInfo:
    found: bool
    version: str = ""
    missing_options: list[str] = field(default_factory=list)
    message: str = ""


class TrainerAdapter:
    name = "?"
    required_options: list[str] = []   # --help 출력에 있어야 하는 옵션

    def presets(self) -> dict[str, Preset]:
        raise NotImplementedError

    def command(self, exe: str, data_dir: Path, out_dir: Path, preset: Preset, gpu: str) -> list[str]:
        raise NotImplementedError

    def find_result(self, out_dir: Path) -> Path | None:
        raise NotImplementedError

    def help_text(self, exe: str) -> str:
        r = subprocess.run([exe, "--help"], capture_output=True, text=True, timeout=60,
                           encoding="utf-8", errors="replace")
        return (r.stdout or "") + (r.stderr or "")

    def detect(self, exe: str | None) -> TrainerInfo:
        if not exe or not Path(exe).exists():
            return TrainerInfo(False, message=f"{self.name}: 실행 파일 없음 ({exe})")
        try:
            text = self.help_text(exe)
        except Exception as e:  # noqa: BLE001
            return TrainerInfo(False, message=f"{self.name}: --help 실행 실패 {e}")
        missing = [o for o in self.required_options if o not in text]
        return TrainerInfo(True, "", missing,
                           f"{self.name}: 옵션 없음 {missing} (버전 확인)" if missing else "")

    @staticmethod
    def clamp_cap(preset: Preset, gpu: str) -> int:
        return min(preset.cap, GPU_LIMITS.get(gpu, GPU_LIMITS["8gb"])["max_cap"])
