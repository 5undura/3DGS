import json
import shutil
from pathlib import Path

import pytest

from indoorgs import schemas
from indoorgs.pathing import (PathModel, add_anchor, check_duration, clean_stroke, move_anchor, outside_segments,
                              path_to_json, path_warnings, remove_anchor, resample, save_path, simplify_rdp)
from indoorgs.project import Project
from indoorgs.stages.base import run_all

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo"


def test_rdp_and_resample():
    noisy = [(x * 0.1, 0.01 * ((-1) ** x)) for x in range(101)] + [(10, 0.1 * k) for k in range(1, 51)]
    s = simplify_rdp(noisy, 0.05)
    assert len(s) == 3 and s[1] == pytest.approx((10, 0), abs=0.05)
    r = resample([(0, 0), (10, 0), (10, 5)], 1.0)
    assert len(r) == 16 and r[-1] == (10, 5)
    c = clean_stroke(noisy, eps=0.05, step=0.5)
    assert c[0] == (0, 0.01) and len(c) == 31


def test_anchor_edit_and_json_roundtrip(tmp_path):
    p = schemas.parse_path({"version": 1, "stroke": [[0, 0], [10, 0]], "anchors": [{"t": 0, "s": 0}, {"t": 20, "s": 10}]})
    p = add_anchor(p, 10, point=(4, 0.7))                 # 선 위로 붙음
    assert [a.t for a in p.anchors] == [0, 10, 20] and p.anchors[1].point == pytest.approx((4, 0))
    p = move_anchor(p, 1, t=8, s=5)
    assert p.anchors[1].s == 5 and p.anchors[1].point is None
    assert PathModel(p).position(8)[:2] == pytest.approx((5, 0))
    p = remove_anchor(p, 1)
    assert len(p.anchors) == 2
    f = save_path(p, tmp_path / "path.json")
    assert schemas.load_path(f).anchors[1].s == 10
    assert json.loads(f.read_text(encoding="utf-8")) == path_to_json(p)


def test_outside_and_duration():
    plan = schemas.load_plan(DEMO / "floorplan" / "plan.json")
    cam = schemas.load_path(DEMO / "sections" / "room_b117" / "path.json")
    assert outside_segments(cam, plan) == []
    bad = schemas.parse_path({"version": 1, "stroke": [[-2.37, 5], [20, 5]],
                              "anchors": [{"t": 0, "s": 0}, {"t": 30, "s": 22.37}]})
    segs = outside_segments(bad, plan)
    assert len(segs) == 1 and segs[0]["start"][0] == pytest.approx(6.5, abs=0.15)
    w = path_warnings(bad, plan, video_duration=60)
    assert any("방 밖" in x for x in w) and any("영상 길이" in x for x in w)
    assert check_duration(cam, 80.5) == []


def test_prior_writes_warnings(tmp_path):
    root = tmp_path / "demo"
    shutil.copytree(DEMO, root)
    frames = root / "sections" / "room_b117" / "frames"
    frames.mkdir(parents=True)
    for i in range(5):
        (frames / f"f_{i + 1:05d}_{i * 240:06d}.jpg").write_bytes(b"")
    schemas.write_json(frames / "done.json", {"version": 1, "fps": 24.0, "count": 5, "total_frames": 24 * 120})
    run_all(Project(root), only_section="room_b117", until="prior", echo=None)
    w = schemas.read_json(root / "sections" / "room_b117" / "colmap" / "priors_warnings.json")
    assert w["video_duration"] == 120 and any("영상 길이" in x for x in w["warnings"])
