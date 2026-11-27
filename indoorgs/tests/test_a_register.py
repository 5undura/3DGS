import json

import pytest

from indoorgs.colmap import database, model as cm, runner
from indoorgs.project import Project
from indoorgs.stages.base import StageContext
from indoorgs.stages.register import Register
from a_fixtures import FakeRunner, arg, colmap_handlers, make_project


def ctx_of(root):
    p = Project(root)
    sec, paths = p.sections()[0]
    logs = []
    return StageContext(p, sec, paths, logs.append), logs


def test_register_plain(tmp_path, monkeypatch):
    root = make_project(tmp_path / "p")
    fake = FakeRunner(colmap_handlers())
    monkeypatch.setattr(runner, "run", fake)
    ctx, logs = ctx_of(root)
    res = Register().run(ctx)
    assert res.ok, res.message
    assert fake.names() == ["feature_extractor", "sequential_matcher", "mapper", "image_undistorter"]
    fe = fake.calls[0]
    assert arg(fe, "--ImageReader.single_camera") == "1" and arg(fe, "--ImageReader.camera_model") == "OPENCV"
    # 가장 큰 모델(0) 을 썼는지
    assert arg(fake.calls[3], "--input_path").endswith("0")
    assert cm.count_images(ctx.paths.data_model) == 30
    assert json.loads((ctx.paths.colmap / "register.json").read_text())["mapper"] == "mapper"
    assert Register().is_done(ctx)


def test_register_prior_fallback(tmp_path, monkeypatch):
    root = make_project(tmp_path / "p", extra_yaml="register: {matcher: exhaustive}\n")
    ctx, logs = ctx_of(root)
    names = sorted(f.name for f in ctx.paths.frames.glob("*.jpg"))
    ctx.paths.colmap.mkdir(parents=True)
    ctx.paths.priors.write_text(json.dumps({"version": 1, "coordinate_system": "CARTESIAN", "frames": [
        {"image": n, "xyz": [0.1 * i, 0, 1.4], "sigma": [0.5, 0.5, 0.3]} for i, n in enumerate(names)]}))
    h = colmap_handlers()
    seen = {}

    def ppm(cmd):
        seen["priors"] = database.count_pose_priors(arg(cmd, "--database_path"))
        return 1                                          # pose_prior_mapper 실패
    h["pose_prior_mapper"] = ppm
    fake = FakeRunner(h)
    monkeypatch.setattr(runner, "run", fake)
    res = Register().run(ctx)
    assert res.ok, res.message
    assert fake.names() == ["feature_extractor", "exhaustive_matcher", "pose_prior_mapper", "mapper",
                            "image_undistorter"]
    assert seen["priors"] == 30
    assert any("pose_prior_mapper 실패" in m for m in logs)
    assert json.loads((ctx.paths.colmap / "register.json").read_text())["mapper"] == "mapper"


def test_register_prior_ok(tmp_path, monkeypatch):
    root = make_project(tmp_path / "p")
    ctx, _ = ctx_of(root)
    ctx.paths.colmap.mkdir(parents=True)
    ctx.paths.priors.write_text(json.dumps({"version": 1, "coordinate_system": "CARTESIAN", "frames": [
        {"image": "f_00001_000000.jpg", "xyz": [0, 0, 1.4], "sigma": [0.5, 0.5, 0.3]}]}))
    fake = FakeRunner(colmap_handlers())
    monkeypatch.setattr(runner, "run", fake)
    assert Register().run(ctx).ok
    assert "mapper" not in fake.names()
    assert json.loads((ctx.paths.colmap / "register.json").read_text())["mapper"] == "pose_prior_mapper"


def test_register_failure_moves_aside_and_resumes(tmp_path, monkeypatch):
    root = make_project(tmp_path / "p")
    ctx, _ = ctx_of(root)
    h = colmap_handlers()
    h["mapper"] = lambda cmd: 1
    fake = FakeRunner(h)
    monkeypatch.setattr(runner, "run", fake)
    res = Register().run(ctx)
    assert not res.ok and "정합 실패" in res.message
    assert any(p.name.startswith("sparse_failed_") for p in ctx.paths.colmap.iterdir())
    assert (ctx.paths.colmap / "matching.done").exists()
    # 다시 돌리면 매칭은 건너뜀
    fake2 = FakeRunner(colmap_handlers())
    monkeypatch.setattr(runner, "run", fake2)
    assert Register().run(ctx).ok
    assert fake2.names() == ["mapper", "image_undistorter"]


def test_register_feature_fail(tmp_path, monkeypatch):
    root = make_project(tmp_path / "p")
    ctx, _ = ctx_of(root)
    fake = FakeRunner({"feature_extractor": lambda c: 3})
    monkeypatch.setattr(runner, "run", fake)
    res = Register().run(ctx)
    assert not res.ok and "feature_extractor" in res.message
    assert fake.names() == ["feature_extractor"]


def test_register_no_colmap(tmp_path):
    root = make_project(tmp_path / "p", colmap=False)
    ctx, _ = ctx_of(root)
    assert "COLMAP 없음" in Register().run(ctx).message


def test_register_loop_detection_offline_fallback(tmp_path, monkeypatch):
    """어휘 트리를 못 받으면(오프라인) loop_detection 0 으로 한 번 더."""
    root = make_project(tmp_path / "p")
    ctx, logs = ctx_of(root)
    h = colmap_handlers()
    h["sequential_matcher"] = lambda cmd: 1 if arg(cmd, "--SequentialMatching.loop_detection") == "1" else 0
    fake = FakeRunner(h)
    monkeypatch.setattr(runner, "run", fake)
    assert Register().run(ctx).ok
    assert fake.names()[:3] == ["feature_extractor", "sequential_matcher", "sequential_matcher"]
    assert arg(fake.calls[2], "--SequentialMatching.loop_detection") == "0"
    assert "--SequentialMatching.vocab_tree_path" not in fake.calls[1]
    assert any("loop_detection 0" in m for m in logs)


def test_register_vocab_tree_path(tmp_path, monkeypatch):
    root = make_project(tmp_path / "p", extra_yaml="register: {vocab_tree: C:/tools/vocab.bin}\n")
    ctx, _ = ctx_of(root)
    fake = FakeRunner(colmap_handlers())
    monkeypatch.setattr(runner, "run", fake)
    assert Register().run(ctx).ok
    assert arg(fake.calls[1], "--SequentialMatching.vocab_tree_path") == "C:/tools/vocab.bin"
