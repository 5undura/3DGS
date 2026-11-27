"""Stage 규격과 실행 순서 (CONTRACT.md 5절)."""
from __future__ import annotations

import datetime as dt
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from ..project import Project, SectionPaths
from ..config import Section


@dataclass
class StageResult:
    ok: bool
    message: str = ""
    outputs: list[Path] = field(default_factory=list)
    skipped: bool = False


@dataclass
class StageContext:
    project: Project
    section: Section | None          # project 범위 단계는 None
    paths: SectionPaths | None
    log: Callable[[str], None]
    rerun: bool = False              # --rerun 으로 끝난 단계를 다시 돌리는 중 (train 은 새 번호 폴더)


class Stage:
    name: str = "?"
    scope: Literal["section", "project"] = "section"
    uses_gpu: bool = False

    def applies(self, ctx: StageContext) -> bool:
        """이 구간에서 돌아야 하는지 (예: prior 는 공간·동선이 있을 때만)."""
        return True

    def outputs(self, ctx: StageContext) -> list[Path]:
        raise NotImplementedError

    def is_done(self, ctx: StageContext) -> bool:
        outs = self.outputs(ctx)
        return bool(outs) and all(p.exists() for p in outs)

    def run(self, ctx: StageContext) -> StageResult:
        raise NotImplementedError

    def prev_targets(self, ctx: StageContext) -> list[Path]:
        """--rerun 때 _prev_<시각> 으로 옮겨 둘 이전 결과 (기본: outputs). 새 폴더에 쓰는 단계는 [] ."""
        return self.outputs(ctx)


class Cancelled(Exception):
    """사용자가 실행을 취소함 (run_all 안에서만 씀)."""


def move_prev(st: Stage, ctx: StageContext) -> None:
    """다시 돌리기 전에 이전 결과를 <이름>_prev_<시각> 으로 옮김 (삭제 안 함)."""
    from ..colmap.runner import move_aside
    for p in st.prev_targets(ctx):
        new = move_aside(p, "prev")
        if new is not None:
            ctx.log(f"{st.name}: 이전 결과 {p.name} -> {new.name}")


def file_logger(path: Path, echo: Callable[[str], None] | None = print) -> Callable[[str], None]:
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str):
        line = f"[{dt.datetime.now():%H:%M:%S}] {msg}"
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        if echo:
            echo(line)
    return log


def all_stages() -> list[Stage]:
    from . import extract, prior, register, diagnose, walls, train, snap, fill, merge, report
    return [extract.Extract(), prior.Prior(), register.Register(), diagnose.Diagnose(), walls.Walls(),
            train.Train(), snap.Snap(), snap.Repair(), fill.Fill(), merge.Merge(), report.Report()]


def status(project: Project) -> list[tuple[str, str, str]]:
    """(구간, 단계, 상태) 목록. 상태: done | todo | skip"""
    rows = []
    for st in all_stages():
        targets = ([(s, p) for s, p in project.sections()] if st.scope == "section" else [(None, None)])
        for sec, paths in targets:
            ctx = StageContext(project, sec, paths, lambda m: None)
            state = "skip" if not st.applies(ctx) else ("done" if st.is_done(ctx) else "todo")
            rows.append((sec.name if sec else "*", st.name, state))
    return rows


def run_all(project: Project, only_section: str | None = None, until: str | None = None,
            echo: Callable[[str], None] | None = print, rerun: str | None = None,
            cancel: threading.Event | None = None) -> bool:
    """구간별로 단계를 순서대로. 끝난 단계는 건너뜀. GPU 단계는 이 함수가 하나씩만 돌리므로 순서 실행으로 충분.
    실패하면 그 구간의 다음 단계는 멈추고 다른 구간은 계속. 전부 성공하면 True.
    rerun: 이 단계부터 끝난 것도 다시 (train 은 새 번호 폴더, 나머지는 이전 결과를 _prev_<시각> 으로 옮김).
    cancel: 켜지면 단계 사이에서 멈추고, 실행 중인 외부 프로그램(runner.run)도 끝냄. 취소면 False."""
    from ..colmap import runner
    stages = all_stages()
    names = [s.name for s in stages]
    for nm in (until, rerun):
        if nm and nm not in names:
            raise ValueError(f"모르는 단계 {nm} (가능: {names})")
    if until:
        stages = stages[: names.index(until) + 1]
    first_rerun = names.index(rerun) if rerun else len(names)
    ok_all = True
    last_log: list[Callable[[str], None]] = [file_logger(project.logs / "run.log", echo)]

    def check_cancel():
        if cancel is not None and cancel.is_set():
            raise Cancelled

    def run_one(st: Stage, ctx: StageContext) -> StageResult | None:
        """None = 건너뜀"""
        last_log[0] = ctx.log
        check_cancel()
        if not st.applies(ctx):
            return None
        ctx.rerun = names.index(st.name) >= first_rerun
        if not ctx.rerun and st.is_done(ctx):
            if st.scope == "section":
                ctx.log(f"{st.name}: 이미 끝남 (건너뜀)")
            return None
        if ctx.rerun:
            ctx.log(f"{st.name}: 다시 돌림 (--rerun)")
            if st.is_done(ctx):
                move_prev(st, ctx)
        if st.scope == "section":
            ctx.log(f"{st.name}: 시작")       # cli.progress_echo 가 이 줄 끝을 셈
        try:
            res = st.run(ctx)
        except NotImplementedError:
            res = StageResult(False, f"{st.name}: 아직 구현 안 됨")
        ctx.log(f"{st.name}: {'끝' if res.ok else '실패'} {res.message}")
        check_cancel()                      # 외부 프로그램이 취소로 멈춘 경우
        return res

    try:
        with runner.cancel_scope(cancel):
            for sec, paths in project.sections():
                if only_section and sec.name != only_section:
                    continue
                for st in [s for s in stages if s.scope == "section"]:
                    log = file_logger(project.logs / f"{sec.name}_{st.name}.log", echo)
                    res = run_one(st, StageContext(project, sec, paths, log))
                    if res is not None and not res.ok:
                        ok_all = False
                        break
            if only_section is None and ok_all:
                for st in [s for s in stages if s.scope == "project"]:
                    log = file_logger(project.logs / f"project_{st.name}.log", echo)
                    res = run_one(st, StageContext(project, None, None, log))
                    if res is not None:
                        ok_all &= res.ok
    except Cancelled:
        last_log[0]("취소: 사용자 요청으로 멈춤 (남은 단계는 다음 실행에서 이어서)")
        return False
    return ok_all
