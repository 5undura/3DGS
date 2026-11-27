"""F: 재실행 규칙(--rerun, 새 번호 폴더, _prev_), 최신 학습 폴더, 실행 취소."""
import json
import sys
import threading
import time
from pathlib import Path

import pytest

from indoorgs import cli
from indoorgs.colmap import model as cm, runner
from indoorgs.colmap.runner import ToolResult
from indoorgs.project import Project
from indoorgs.stages import base
from indoorgs.stages.base import Stage, StageContext, StageResult, run_all
from indoorgs.stages import fill, merge
from test_a_train import BRUSH_HELP, setup
from a_fixtures import FakeRunner, arg, write_brush_ply


# ---------------- 최신 학습 폴더 ----------------
def test_latest_and_new_train_run(tmp_path):
    from indoorgs.project import SectionPaths
    sp = SectionPaths(tmp_path / "s")
    assert sp.latest_train_run("brush", "default") == sp.train / "brush_default"
    assert sp.new_train_run("brush", "default") == sp.train / "brush_default"
    for name, done in [("brush_default", True), ("brush_default_2", True), ("brush_default_10", False),
                       ("brush_default_3_failed_20260101_000000", True), ("brush_default_prev_x", True),
                       ("brush_fast_5", True), ("lichtfeld_default", True)]:
        (sp.train / name).mkdir(parents=True)
        if done:
            (sp.train / name / "splat.ply").write_bytes(b"x")
    assert [n for n, _ in sp.train_runs("brush", "default")] == [1, 2, 10]
    assert sp.latest_train_run("brush", "default").name == "brush_default_2"     # 끝난 것 중 가장 큰 번호
    assert sp.new_train_run("brush", "default").name == "brush_default_11"
    (sp.train / "brush_default_10" / "splat.ply").write_bytes(b"x")
    assert sp.latest_train_run("brush", "default").name == "brush_default_10"


def test_fill_merge_use_latest(tmp_path):
    p, ctx, _ = setup(tmp_path)
    for n in ("brush_fast", "brush_fast_2"):
        write_brush_ply(ctx.paths.train / n / "splat.ply", 5)
    assert fill.train_ply(ctx).parent.name == "brush_fast_2"
    assert merge.section_ply(p, ctx.section, ctx.paths).parent.name == "brush_fast_2"


# ---------------- --rerun train (실제 Train 단계, 가짜 Brush) ----------------
def _done_until_train(ctx):
    ps = ctx.paths
    cm.write_model(cm.read_model(ps.data_clean / "sparse" / "0"), ps.data / "sparse" / "0")
    ps.diagnose.write_text(json.dumps({"version": 1, "excluded": []}), encoding="utf-8")


def _fake_brush(monkeypatch, steps):
    def brush(cmd):
        if "--help" in cmd:
            return ToolResult(0, [BRUSH_HELP])
        out = Path(arg(cmd, "--export-path"))
        steps.append(out.name)
        write_brush_ply(out / "export_5000.ply", 10 + len(steps), opacity_logit=2.0)
        return 0
    fake = FakeRunner()
    fake.key = lambda cmd: "brush"
    fake.handlers = {"brush": brush}
    monkeypatch.setattr(runner, "run", fake)


def test_rerun_train_new_folder(tmp_path, monkeypatch):
    p, ctx, _ = setup(tmp_path)
    _done_until_train(ctx)
    steps = []
    _fake_brush(monkeypatch, steps)
    run_all(p, until="train", echo=None)
    assert steps == ["brush_fast"]
    first = (ctx.paths.train / "brush_fast" / "splat.ply").read_bytes()
    run_all(p, until="train", echo=None)                      # 끝남 -> 다시 안 돌림
    assert steps == ["brush_fast"]
    assert run_all(p, until="train", echo=None, rerun="train")
    assert run_all(p, until="train", echo=None, rerun="train")
    assert steps == ["brush_fast", "brush_fast_2", "brush_fast_3"]
    assert (ctx.paths.train / "brush_fast" / "splat.ply").read_bytes() == first    # 이전 결과 그대로
    assert ctx.paths.latest_train_run("brush", "fast").name == "brush_fast_3"
    # 앞 단계는 다시 안 돌고 이전 결과도 안 옮김
    assert not list(ctx.paths.root.glob("*_prev_*"))
    log = (p.logs / "s1_train.log").read_text(encoding="utf-8")
    assert "다시 돌림" in log


def test_rerun_cli_option(tmp_path, monkeypatch):
    p, ctx, _ = setup(tmp_path)
    _done_until_train(ctx)
    steps = []
    _fake_brush(monkeypatch, steps)
    assert cli.main(["run", str(p.root), "--until", "train"]) == 0
    assert cli.main(["run", str(p.root), "--until", "train", "--rerun", "train"]) == 0
    assert steps == ["brush_fast", "brush_fast_2"]
    assert cli.main(["run", str(p.root), "--rerun", "zzz"]) == 2           # 모르는 단계


# ---------------- 다른 단계: _prev_<시각> ----------------
class Toy(Stage):
    def __init__(self, name, scope="section", sleep=None):
        self.name, self.scope, self.sleep = name, scope, sleep
        self.runs = 0

    def _dir(self, ctx):
        return (ctx.paths.root if ctx.paths else ctx.project.root) / f"out_{self.name}"

    def outputs(self, ctx):
        return [self._dir(ctx) / "r.txt"]

    def prev_targets(self, ctx):
        return [self._dir(ctx)]

    def run(self, ctx):
        self.runs += 1
        if self.sleep:
            r = runner.run([sys.executable, "-c", f"import time; time.sleep({self.sleep})"],
                           log_file=self._dir(ctx).parent / "tool.log")
            if not r.ok:
                return StageResult(False, f"코드 {r.code}")
        self._dir(ctx).mkdir(parents=True, exist_ok=True)
        (self._dir(ctx) / "r.txt").write_text(f"run {self.runs}")
        return StageResult(True)


def toy_project(tmp_path):
    root = tmp_path / "p"
    root.mkdir()
    (root / "project.yaml").write_text("sections:\n  - {name: s1, video: v.mp4}\n", encoding="utf-8")
    return Project(root)


def test_rerun_moves_prev(tmp_path, monkeypatch):
    st = [Toy("a"), Toy("b"), Toy("c", "project")]
    monkeypatch.setattr(base, "all_stages", lambda: st)
    p = toy_project(tmp_path)
    assert run_all(p, echo=None)
    assert run_all(p, echo=None, rerun="b")
    assert [s.runs for s in st] == [1, 2, 2]
    sec = p.section("s1").root
    prev = list(sec.glob("out_b_prev_*"))
    assert len(prev) == 1 and (prev[0] / "r.txt").read_text() == "run 1"      # 이전 결과 보존
    assert (sec / "out_b" / "r.txt").read_text() == "run 2"
    assert not list(sec.glob("out_a_prev_*"))
    assert len(list(p.root.glob("out_c_prev_*"))) == 1
    with pytest.raises(ValueError):
        run_all(p, echo=None, rerun="zzz")


# ---------------- 취소 ----------------
def test_cancel_kills_process(tmp_path, monkeypatch):
    st = [Toy("slow", sleep=60), Toy("after")]
    monkeypatch.setattr(base, "all_stages", lambda: st)
    p = toy_project(tmp_path)
    ev = threading.Event()
    threading.Timer(0.5, ev.set).start()
    t0 = time.time()
    assert run_all(p, echo=None, cancel=ev) is False
    assert time.time() - t0 < 8
    assert st[1].runs == 0                                     # 다음 단계는 안 돎
    assert "취소" in (p.logs / "s1_slow.log").read_text(encoding="utf-8")
    assert "취소로 멈춤" in (p.section("s1").root / "tool.log").read_text(encoding="utf-8")
    # 다음 실행은 이어서 (취소된 단계부터)
    st[0].sleep = None
    assert run_all(p, echo=None)
    assert [x.runs for x in st] == [2, 1]


def test_cancel_before_start(tmp_path, monkeypatch):
    st = [Toy("a")]
    monkeypatch.setattr(base, "all_stages", lambda: st)
    p = toy_project(tmp_path)
    ev = threading.Event()
    ev.set()
    assert run_all(p, echo=None, cancel=ev) is False and st[0].runs == 0


def test_gui_cancel(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from indoorgs.gui.server import create_app
    st = [Toy("slow", sleep=60)]
    monkeypatch.setattr(base, "all_stages", lambda: st)
    p = toy_project(tmp_path)
    c = TestClient(create_app(p.root))
    assert c.post("/api/run/cancel").status_code == 409        # 실행 중 아님
    assert c.post("/api/run", json={}).status_code == 200
    time.sleep(0.5)
    t0 = time.time()
    assert c.post("/api/run/cancel").status_code == 200
    for _ in range(200):
        s = c.get("/api/run/state").json()
        if not s["running"]:
            break
        time.sleep(0.05)
    assert not s["running"] and s["ok"] is False and time.time() - t0 < 8
    assert any("취소" in line for line in s["lines"])
