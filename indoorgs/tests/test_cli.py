import json
import shutil
from pathlib import Path

from indoorgs import cli, schemas
from indoorgs.project import Project
from indoorgs.stages.base import run_all, status

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo"


def test_init_and_status(tmp_path, capsys):
    assert cli.main(["init", str(tmp_path / "p")]) == 0
    assert cli.main(["status", str(tmp_path / "p")]) == 0
    assert "section1" in capsys.readouterr().out


def test_prior_stage_runs_on_demo(tmp_path):
    root = tmp_path / "demo"
    shutil.copytree(DEMO, root)
    frames = root / "sections" / "room_b117" / "frames"
    frames.mkdir(parents=True)
    for i in range(10):
        (frames / f"f_{i + 1:05d}_{i * 240:06d}.jpg").write_bytes(b"")
    schemas.write_json(frames / "done.json", {"version": 1, "fps": 24.0, "count": 10})
    p = Project(root)
    st = {(s, n): v for s, n, v in status(p)}
    assert st[("room_b117", "extract")] == "done"
    assert st[("room_b117", "prior")] == "todo"
    assert st[("corridor1", "prior")] == "skip"          # 동선 없음
    ok = run_all(p, only_section="room_b117", echo=None)
    assert not ok                                          # register 는 아직 구현 안 됨
    pri = json.loads((root / "sections" / "room_b117" / "colmap" / "priors.json").read_text(encoding="utf-8"))
    assert len(pri["frames"]) == 10
    log = (root / "logs" / "room_b117_register.log").read_text(encoding="utf-8")
    assert "아직 구현 안 됨" in log
