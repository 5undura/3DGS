"""indoorgs init / check / status / run"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config as config_mod
from .project import Project
from .stages.base import all_stages, run_all, status
from .trainers import REGISTRY

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
    cfg.write_text(TEMPLATE, encoding="utf-8")
    print(f"만듦: {cfg}  -> 도구 경로와 sections 를 고친 뒤 `indoorgs check {root}`")
    return 0


def cmd_check(a) -> int:
    p = Project(a.folder)
    bad = 0
    for w in p.warnings():
        print("경고:", w)
    t = p.cfg.tools
    print(f"COLMAP   : {'있음' if t.colmap and Path(t.colmap).exists() else '없음'}  {t.colmap}")
    bad += not (t.colmap and Path(t.colmap).exists())
    for name, ad in REGISTRY.items():
        info = ad.detect(getattr(t, name))
        mark = "있음" if info.found and not info.missing_options else ("옵션 부족" if info.found else "없음")
        print(f"{name:9}: {mark}  {getattr(t, name)}  {info.message}")
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


def cmd_run(a) -> int:
    p = Project(a.folder)
    for w in p.warnings():
        print("경고:", w)
    return 0 if run_all(p, only_section=a.section, until=a.until) else 1


def cmd_gui(a) -> int:
    from .gui.server import serve      # fastapi 는 gui 선택 설치 ([gui])
    serve(a.folder, port=a.port, open_browser=a.open)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="indoorgs", description="실내 공간 원클릭 3DGS")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init", help="새 프로젝트 폴더와 project.yaml 만들기"); s.add_argument("folder"); s.set_defaults(f=cmd_init)
    s = sub.add_parser("check", help="도구·영상·설정 점검"); s.add_argument("folder"); s.set_defaults(f=cmd_check)
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
