import json

import numpy as np

from indoorgs.colmap import model as cm
from indoorgs.project import Project
from indoorgs.stages.base import StageContext
from indoorgs.stages.diagnose import Diagnose, find_teleports
from a_fixtures import make_model, make_project


def setup(tmp_path, teleport_at=None, start=(0.0, 0.0, 1.4), mapper="mapper"):
    root = make_project(tmp_path / "p")
    p = Project(root)
    sec, paths = p.sections()[0]
    cm.write_model(make_model(30, teleport_at, start), paths.data_model)
    (paths.data / "images").mkdir()
    for f in paths.frames.glob("*.jpg"):
        (paths.data / "images" / f.name).write_bytes(b"jpg")
    paths.colmap.mkdir(parents=True, exist_ok=True)
    (paths.colmap / "register.json").write_text(json.dumps({"version": 1, "mapper": mapper}))
    return p, StageContext(p, sec, paths, lambda m: None)


def test_find_teleports():
    m = make_model(30, teleport_at=12)
    ims = list(m.images.values())
    t = find_teleports([i.name for i in ims], np.array([i.center for i in ims]))
    assert {x["from"] for x in t} | {x["to"] for x in t} >= {ims[12].name}
    assert all(x["speed_x_median"] > 5 for x in t)
    m = make_model(30)
    ims = list(m.images.values())
    assert find_teleports([i.name for i in ims], np.array([i.center for i in ims])) == []


def test_diagnose_excludes_teleport(tmp_path):
    p, ctx = setup(tmp_path, teleport_at=12)
    res = Diagnose().run(ctx)
    assert res.ok, res.message
    d = json.loads(ctx.paths.diagnose.read_text(encoding="utf-8"))
    bad = "f_00013_000120.jpg"
    assert bad in d["excluded"] and d["registered"] == 30 and d["total"] == 30
    clean = cm.read_model(ctx.paths.data_clean / "sparse" / "0")
    assert bad not in {i.name for i in clean.images.values()}
    assert len(clean.images) == 30 - len(d["excluded"])
    assert (ctx.paths.data_clean / "images" / bad).exists()          # 사진은 원본 연결
    assert (ctx.paths.data_clean / "images").is_symlink()
    assert len(list(ctx.paths.data.glob("sparse/0/*.bin"))) >= 3       # 원본 그대로
    assert Diagnose().is_done(ctx)


def test_diagnose_space_and_prior(tmp_path):
    # 방: x 0~2 (사진 0~20 만 안에), priors 는 사진 5 만 3m 떨어짐
    p, ctx = setup(tmp_path, mapper="pose_prior_mapper")
    p.plan_json.parent.mkdir(parents=True, exist_ok=True)
    p.plan_json.write_text(json.dumps({"version": 1, "rooms": [
        {"id": "r", "box": {"center": [1.0, 0.0], "size": [2.05, 2.0]}}]}))
    m = cm.read_model(ctx.paths.data_model)
    frames = []
    for im in m.images.values():
        xyz = im.center.tolist()
        if im.id == 6:
            xyz[1] += 3.0
        frames.append({"image": im.name, "xyz": xyz, "sigma": [0.5, 0.5, 0.3]})
    ctx.paths.priors.write_text(json.dumps({"version": 1, "coordinate_system": "CARTESIAN", "frames": frames}))
    assert Diagnose().run(ctx).ok
    d = json.loads(ctx.paths.diagnose.read_text(encoding="utf-8"))
    assert d["far_from_prior"] == ["f_00006_000050.jpg"]
    assert len(d["outside_space"]) == 9                                 # x > 2.025 인 사진 21~29
    assert set(d["outside_space"]) | set(d["far_from_prior"]) <= set(d["excluded"])


def test_diagnose_space_ignored_without_prior_mapper(tmp_path):
    p, ctx = setup(tmp_path, start=(100.0, 100.0, 0.0))
    p.plan_json.parent.mkdir(parents=True, exist_ok=True)
    p.plan_json.write_text(json.dumps({"version": 1, "rooms": [
        {"id": "r", "box": {"center": [0.0, 0.0], "size": [2.0, 2.0]}}]}))
    assert Diagnose().run(ctx).ok
    d = json.loads(ctx.paths.diagnose.read_text(encoding="utf-8"))
    assert d["outside_space"] == [] and d["excluded"] == []
