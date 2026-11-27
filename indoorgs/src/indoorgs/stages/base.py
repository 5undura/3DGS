"""Stage 규격과 실행 순서 (CONTRACT.md 5절)."""
from __future__ import annotations

import datetime as dt
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
    from . import extract, prior, register, diagnose, train, fill, merge, report
    return [extract.Extract(), prior.Prior(), register.Register(), diagnose.Diagnose(),
            train.Train(), fill.Fill(), merge.Merge(), report.Report()]


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
            echo: Callable[[str], None] | None = print) -> bool:
    """구간별로 단계를 순서대로. 끝난 단계는 건너뜀. GPU 단계는 이 함수가 하나씩만 돌리므로 순서 실행으로 충분.
    실패하면 그 구간의 다음 단계는 멈추고 다른 구간은 계속. 전부 성공하면 True."""
    stages = all_stages()
    names = [s.name for s in stages]
    if until and until not in names:
        raise ValueError(f"모르는 단계 {until} (가능: {names})")
    if until:
        stages = stages[: names.index(until) + 1]
    ok_all = True
    for sec, paths in project.sections():
        if only_section and sec.name != only_section:
            continue
        for st in [s for s in stages if s.scope == "section"]:
            log = file_logger(project.logs / f"{sec.name}_{st.name}.log", echo)
            ctx = StageContext(project, sec, paths, log)
            if not st.applies(ctx):
                continue
            if st.is_done(ctx):
                log(f"{st.name}: 이미 끝남 (건너뜀)")
                continue
            log(f"{st.name}: 시작")
            try:
                res = st.run(ctx)
            except NotImplementedError:
                res = StageResult(False, f"{st.name}: 아직 구현 안 됨")
            log(f"{st.name}: {'끝' if res.ok else '실패'} {res.message}")
            if not res.ok:
                ok_all = False
                break
    if only_section is None and ok_all:
        for st in [s for s in stages if s.scope == "project"]:
            log = file_logger(project.logs / f"project_{st.name}.log", echo)
            ctx = StageContext(project, None, None, log)
            if not st.applies(ctx) or st.is_done(ctx):
                continue
            try:
                res = st.run(ctx)
            except NotImplementedError:
                res = StageResult(False, f"{st.name}: 아직 구현 안 됨")
            log(f"{st.name}: {'끝' if res.ok else '실패'} {res.message}")
            ok_all &= res.ok
    return ok_all
