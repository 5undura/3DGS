"""indoorgs init / check / status / run"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from . import config as config_mod
from .project import Project
from .stages.base import all_stages, run_all, status
from .colmap import runner
from .trainers import REGISTRY

COLMAP_WANT = "4.2.1"   # pose_priors 스키마·pose_prior_mapper 옵션을 확인한 버전


def colmap_version(exe: str) -> tuple[str, bool]:
    """(버전, pose_prior_mapper 있음). `colmap help` 첫 줄 'COLMAP 4.2.1 (Commit ...)' 형식.
    TODO(PC에서 확인): Windows COLMAP.bat help 출력 형식"""
    r = runner.run([exe, "help"], timeout=60)
    m = re.search(r"COLMAP\s+(\d+\.\d+(?:\.\d+)?)", r.text)
    return (m.group(1) if m else ""), ("pose_prior_mapper" in r.text)

TEMPLATE = """\
# IndoorGS 프로젝트 설정. 형식은 indoorgs/CONTRACT.md
tools:
  colmap: C:/tools/colmap/COLMAP.bat
  brush: C:/tools/brush/brush_app.exe
  lichtfeld: null            # 빌드했으면 경로. 있으면 train.trainer: lichtfeld 가능
gpu_profile: 8gb             # 8gb / 12gb
sections:
  - name: section1           # 영문·숫자·_·- (폴더 이름)
    video: D:/capture/1.MP4
extract: {min_diff: 4, sharp_ratio: 0.8}
register: {matcher: sequential, mapper: incremental, use_floorplan_prior: true}
train: {trainer: brush, preset: default}
fill: {holes_only: true}
"""


def cmd_init(a) -> int:
    root = Path(a.folder)
    cfg = root / "project.yaml"
    if cfg.exists():
        print(f"이미 있음: {cfg}")
        return 1
    for d in ("floorplan", "sections", "merged", "logs"):
        (root / d).mkdir(parents=True, exist_ok=True)
    from . import tools_detect as td   # 도구 자동 찾기 (PATH, C:/tools, --search 폴더)
    found = td.best(td.find_tools(getattr(a, "search", None) or []))
    text = TEMPLATE
    for name, line in (("colmap", "  colmap: C:/tools/colmap/COLMAP.bat"), ("brush", "  brush: C:/tools/brush/brush_app.exe"),
                       ("lichtfeld", "  lichtfeld: null")):
        if found[name]:
            text = text.replace(line, f"  {name}: {td.yaml_path(found[name])}", 1)
        print(f"{name:9}: {'찾음 ' + found[name] if found[name] else '못 찾음 (project.yaml 에 직접 적기)'}")
    cfg.write_text(text, encoding="utf-8")
    print(f"만듦: {cfg}  -> 도구 경로와 sections 를 고친 뒤 `indoorgs check {root}`")
    return 0


def cmd_check(a) -> int:
    p = Project(a.folder)
    bad = 0
    for w in p.warnings():
        print("경고:", w)
    t = p.cfg.tools
    has_colmap = bool(t.colmap and Path(t.colmap).exists())
    extra = ""
    if has_colmap:
        ver, ppm = colmap_version(t.colmap)
        extra = f" 버전 {ver or '?'}"
        if ver and ver != COLMAP_WANT:
            extra += f" (확인한 버전은 {COLMAP_WANT}: pose_priors 형식이 다를 수 있음)"
        if not ppm:
            extra += " / pose_prior_mapper 못 찾음 (도면 위치 제약은 mapper 로 대신)"
    print(f"COLMAP   : {'있음' if has_colmap else '없음'}  {t.colmap}{extra}")
    bad += not has_colmap
    for name, ad in REGISTRY.items():
        info = ad.detect(getattr(t, name))
        mark = "있음" if info.found and not info.missing_options else ("옵션 부족" if info.found else "없음")
        ver = f" (버전 {info.version})" if info.version else ""
        print(f"{name:9}: {mark}{ver}  {getattr(t, name)}  {info.message}")
        if name == p.cfg.train.trainer and mark != "있음":
            bad += 1
    if p.cfg.train.preset not in REGISTRY[p.cfg.train.trainer].presets():
        print(f"프리셋 없음: {p.cfg.train.preset} (가능: {list(REGISTRY[p.cfg.train.trainer].presets())})")
        bad += 1
    for sec, _ in p.sections():
        ok = Path(sec.video).exists()
        bad += not ok
        print(f"영상 {sec.name:12}: {'있음' if ok else '없음'}  {sec.video}")
    print(f"공간(plan.json): {'있음' if p.has_space else '없음 (공간 기능 꺼짐)'}")
    from . import tools_detect as td
    conf = {n: getattr(t, n) for n in td.TOOL_FILES}
    roots = (getattr(a, "search", None) or []) + td.roots_from_config(conf.values())
    for n, others in td.other_candidates(td.find_tools(roots), conf).items():
        for o in others:
            print(f"자동으로 찾은 다른 후보 {n:9}: {o}")
    return 1 if bad else 0


def cmd_status(a) -> int:
    p = Project(a.folder)
    rows = status(p)
    secs = list(dict.fromkeys(r[0] for r in rows))
    names = [s.name for s in all_stages()]
    table = {(r[0], r[1]): r[2] for r in rows}
    mark = {"done": "끝", "todo": "-", "skip": "건너뜀"}
    print("구간".ljust(14) + "".join(n.ljust(10) for n in names))
    for s in secs:
        print(s.ljust(14) + "".join(mark.get(table.get((s, n), ""), "").ljust(10) for n in names))
    return 0


def _todo(p: Project, section: str | None, until: str | None) -> list[tuple[str, str]]:
    names = [s.name for s in all_stages()]
    last = names.index(until) if until else len(names) - 1
    return [(sec, st) for sec, st, state in status(p)
            if state == "todo" and names.index(st) <= last and (section is None or sec in (section,))]


def progress_echo(total: int, out=print):
    """run_all 의 로그 줄을 받아 '[2/7 +00:12:34]' 를 붙여 출력. '<단계>: 시작' 줄마다 번호가 하나씩 오름."""
    t0 = time.time()
    n = [0]
    pat = re.compile(r"\] (\w+): 시작$")

    def echo(line: str):
        if pat.search(line):
            n[0] += 1
        el = int(time.time() - t0)
        out(f"[{min(n[0], total) if total else n[0]}/{total} +{el // 3600:02d}:{el % 3600 // 60:02d}:{el % 60:02d}] {line}")
    return echo


def cmd_run(a) -> int:
    p = Project(a.folder)
    for w in p.warnings():
        print("경고:", w)
    todo = _todo(p, a.section, a.until)
    print(f"할 단계 {len(todo)}개: " + (", ".join(f"{s}/{st}" for s, st in todo) or "없음"))
    ok = run_all(p, only_section=a.section, until=a.until, echo=progress_echo(len(todo)))
    print()
    cmd_status(a)
    print("결과:", "성공" if ok else "실패 (logs/ 확인)")
    return 0 if ok else 1


def cmd_gui(a) -> int:
    from .gui.server import serve      # fastapi 는 gui 선택 설치 ([gui])
    serve(a.folder, port=a.port, open_browser=a.open)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="indoorgs", description="실내 공간 원클릭 3DGS")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init", help="새 프로젝트 폴더와 project.yaml 만들기"); s.add_argument("folder"); s.set_defaults(f=cmd_init)
    s.add_argument("--search", action="extend", nargs="+", default=[], help="도구를 찾을 폴더 (예: F:/Sample/tools), 깊이 4")
    s = sub.add_parser("check", help="도구·영상·설정 점검"); s.add_argument("folder"); s.set_defaults(f=cmd_check)
    s.add_argument("--search", action="extend", nargs="+", default=[], help="다른 후보를 찾을 폴더")
    s = sub.add_parser("status", help="구간 × 단계 진행 상황"); s.add_argument("folder"); s.set_defaults(f=cmd_status)
    s = sub.add_parser("run", help="끝나지 않은 단계를 순서대로 실행")
    s.add_argument("folder"); s.add_argument("--section"); s.add_argument("--until", help="이 단계까지만")
    s.set_defaults(f=cmd_run)
    s = sub.add_parser("gui", help="로컬 웹 화면 (127.0.0.1)"); s.add_argument("folder"); s.add_argument("--port", type=int, default=8765); s.add_argument("--open", action="store_true", help="브라우저 열기"); s.set_defaults(f=cmd_gui)
    a = ap.parse_args(argv)
    try:
        return a.f(a)
    except (config_mod.ConfigError, FileNotFoundError, KeyError, ValueError) as e:
        print("오류:", e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
