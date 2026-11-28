"""합성 복도 v4 로 실제 단계(prior -> register -> diagnose -> walls)를 끝까지 돌림. COLMAP 대신 colmap_shim.py.

  python run_e2e.py <v4 장면 폴더> <작업 폴더> --venv <pycolmap venv> [--pair-verify 1|0] [--seed 0]

작업 폴더에 프로젝트를 만듦: project.yaml, floorplan/plan.json(줄자 오차 흉내), sections/v4/{frames, path.json}.
frames 는 장면 images/ 를 가리키는 심볼릭 링크 (원본 그대로).
"""
from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from indoorgs import schemas  # noqa: E402
from indoorgs.project import Project  # noqa: E402
from indoorgs.stages.base import run_all  # noqa: E402

# 줄자 도면 오차 흉내 (정답: x 0~24, y -1.2~1.2, z 0~2.7)
#   끝 벽 x0 은 10 cm 바깥(-0.10), xL 은 20 cm 바깥(24.20), 옆 벽은 5 cm 씩 바깥(±1.25), 천장 10 cm 낮게(2.6)
PLAN = {"version": 1, "units": "m",
        "rooms": [{"id": "corr", "box": {"center": [12.05, 0.0], "size": [24.3, 2.5], "yaw_deg": 0},
                   "floor_z": 0.0, "ceiling_z": 2.6}],
        "doors": [], "source": {"kind": "boxes"}}


def make(scene: Path, work: Path, venv: Path, pair_verify: bool, seed: int, matcher: str) -> Project:
    work.mkdir(parents=True, exist_ok=True)
    shim = work / "colmap"
    shim.write_text(f"#!/bin/sh\nSHIM_SEED={seed} exec {venv}/bin/python {HERE / 'colmap_shim.py'} \"$@\"\n")
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
    (work / "project.yaml").write_text(json.dumps({
        "sections": [{"name": "v4", "video": str(work / "none.mp4")}],
        "tools": {"colmap": str(shim)},
        "register": {"pair_verify": pair_verify, "matcher": matcher},
        "walls": {"enabled": True, "seed": True},
    }, indent=1))
    (work / "floorplan").mkdir(exist_ok=True)
    (work / "floorplan" / "plan.json").write_text(json.dumps(PLAN, indent=1))
    sec = work / "sections" / "v4"
    fr = sec / "frames"
    if not fr.exists():
        fr.mkdir(parents=True)
        for f in sorted((scene / "images").glob("f_*.jpg")):
            os.symlink(f.resolve(), fr / f.name)
        n = len(list(fr.glob("f_*.jpg")))
        schemas.write_json(fr / "done.json", {"version": 1, "video": "synthetic v4", "fps": 30.0,
                                              "total_frames": 1561, "count": n})
    (sec / "path.json").write_text((scene / "priors" / "path_good.json").read_text())
    return Project(work)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", type=Path)
    ap.add_argument("work", type=Path)
    ap.add_argument("--venv", type=Path, required=True)
    ap.add_argument("--pair-verify", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--matcher", default="sequential")
    ap.add_argument("--until", default="walls")
    a = ap.parse_args()
    pj = make(a.scene.resolve(), a.work.resolve(), a.venv.resolve(), bool(a.pair_verify), a.seed, a.matcher)
    t0 = time.time()
    ok = run_all(pj, until=a.until)
    print(f"run_all ok={ok} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
