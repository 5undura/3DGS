"""외부 프로그램(COLMAP, Brush, LichtFeld) 실행은 전부 이 run() 하나로.

테스트는 이 모듈의 run 을 가짜로 바꿔(monkeypatch) 명령 순서·인자·실패 처리를 확인한다.
호출하는 쪽은 `from ..colmap import runner` 후 `runner.run(...)` 처럼 모듈 이름으로 부를 것
(그래야 monkeypatch 가 먹음).
"""
from __future__ import annotations

import collections
import contextlib
import datetime as dt
import os
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class ToolResult:
    code: int
    tail: list[str] = field(default_factory=list)   # 출력 마지막 줄들 (stdout+stderr 합침)
    cancelled: bool = False                          # 취소(cancel_scope 의 Event)로 멈춤

    @property
    def ok(self) -> bool:
        return self.code == 0

    @property
    def text(self) -> str:
        return "\n".join(self.tail)


# 취소: run_all 이 cancel_scope 로 Event 를 걸어 두면, 실행 중인 run() 이 Event 가 켜질 때 프로세스를 끝냄.
# (단계 코드가 runner.run 에 인자를 넘기지 않아도 되게 모듈 전역. 실행은 한 번에 하나라 충분)
_cancel: threading.Event | None = None


@contextlib.contextmanager
def cancel_scope(ev: threading.Event | None):
    global _cancel
    old, _cancel = _cancel, ev
    try:
        yield
    finally:
        _cancel = old


def _kill_tree(p: subprocess.Popen) -> None:
    """프로세스 끝내기. Windows 는 COLMAP.bat -> colmap.exe 처럼 자식까지 (taskkill /T)."""
    if p.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
        else:
            p.terminate()
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()
    except OSError:
        pass


def _watch(p: subprocess.Popen, ev: threading.Event, hit: list[bool]) -> None:
    while p.poll() is None:
        if ev.wait(0.2):
            hit[0] = True
            _kill_tree(p)
            return


def run(cmd: list[str], log_file: Path | None = None, echo: Callable[[str], None] | None = None,
        timeout: float | None = None, keep: int = 400) -> ToolResult:
    """명령 실행. 출력(stdout+stderr)은 log_file 에 이어 쓰고 마지막 keep 줄을 돌려줌.
    실행 파일이 없으면 code=127. COLMAP 은 결과를 stderr(glog)로 쓰므로 합쳐서 받음.
    cancel_scope 의 Event 가 켜지면 프로세스를 끝내고 cancelled=True."""
    cmd = [str(c) for c in cmd]
    ev = _cancel
    if ev is not None and ev.is_set():
        return ToolResult(-15, ["취소로 실행 안 함"], cancelled=True)
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
        hit = [False]
        if ev is not None:
            threading.Thread(target=_watch, args=(p, ev, hit), daemon=True).start()
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
        if hit[0]:
            tail.append("취소로 멈춤")
            if f:
                f.write("취소로 멈춤\n")
        if f:
            f.write(f"[종료 코드 {code}]\n")
        return ToolResult(code, list(tail), cancelled=hit[0])
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
