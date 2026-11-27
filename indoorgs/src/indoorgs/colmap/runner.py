"""외부 프로그램(COLMAP, Brush, LichtFeld) 실행은 전부 이 run() 하나로.

테스트는 이 모듈의 run 을 가짜로 바꿔(monkeypatch) 명령 순서·인자·실패 처리를 확인한다.
호출하는 쪽은 `from ..colmap import runner` 후 `runner.run(...)` 처럼 모듈 이름으로 부를 것
(그래야 monkeypatch 가 먹음).
"""
from __future__ import annotations

import collections
import datetime as dt
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class ToolResult:
    code: int
    tail: list[str] = field(default_factory=list)   # 출력 마지막 줄들 (stdout+stderr 합침)

    @property
    def ok(self) -> bool:
        return self.code == 0

    @property
    def text(self) -> str:
        return "\n".join(self.tail)


def run(cmd: list[str], log_file: Path | None = None, echo: Callable[[str], None] | None = None,
        timeout: float | None = None, keep: int = 400) -> ToolResult:
    """명령 실행. 출력(stdout+stderr)은 log_file 에 이어 쓰고 마지막 keep 줄을 돌려줌.
    실행 파일이 없으면 code=127. COLMAP 은 결과를 stderr(glog)로 쓰므로 합쳐서 받음."""
    cmd = [str(c) for c in cmd]
    tail: collections.deque[str] = collections.deque(maxlen=keep)
    f = None
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        f = log_file.open("a", encoding="utf-8")
        f.write(f"\n[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] $ {subprocess.list2cmdline(cmd)}\n")
        f.flush()
    try:
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except (FileNotFoundError, PermissionError, OSError) as e:
            msg = f"실행 못 함: {cmd[0]} ({e})"
            if f:
                f.write(msg + "\n")
            return ToolResult(127, [msg])
        assert p.stdout is not None
        for raw in p.stdout:
            # Windows 콘솔 프로그램은 CP949 로 쓸 수도 있음 -> utf-8 실패하면 바꿔 읽음
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            tail.append(line)
            if f:
                f.write(line + "\n")
            if echo:
                echo(line)
        try:
            code = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            code = -9
            tail.append("시간 초과로 멈춤")
        if f:
            f.write(f"[종료 코드 {code}]\n")
        return ToolResult(code, list(tail))
    finally:
        if f:
            f.close()


def stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def move_aside(path: Path, tag: str = "failed") -> Path | None:
    """실패한 출력은 지우지 않고 <이름>_<tag>_<시각> 으로 옆으로 (CONTRACT 2절). 파일이면 확장자 앞에 붙임."""
    path = Path(path)
    if not path.exists():
        return None
    s = stamp()
    if path.is_file():
        new = path.with_name(f"{path.stem}_{tag}_{s}{path.suffix}")
    else:
        new = path.with_name(f"{path.name}_{tag}_{s}")
    i = 1
    while new.exists():
        new = new.with_name(new.name + f"_{i}")
        i += 1
    path.rename(new)
    return new
