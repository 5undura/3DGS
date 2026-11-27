"""walls·snap 단계를 프로젝트 폴더에서 끝까지: 도면 좌표 모델 -> planes.json·data_seed -> splat_walls.ply."""
import json

import numpy as np

from indoorgs import plyio, schemas
from indoorgs.colmap import model as cm
from indoorgs.project import Project
from indoorgs.stages import base, snap as snap_stage, walls as walls_stage
from indoorgs.stages.base import StageContext
from indoorgs.walls import seed as seed_mod
from a_fixtures import make_model, make_project

PLAN = {"version": 1, "units": "m", "rooms": [
    {"id": "A", "floor": [[0, 0], [6, 0], [6, 3], [0, 3]], "floor_z": 0.0, "ceiling_z": 2.6}],
    "doors": [{"id": "d", "rooms": ["A"], "center": [3.0, 0.0], "width": 1.0, "height": 2.1}]}
TRUE_NORTH_Y = 3.05          # 실제 북쪽 벽은 도면보다 5 cm 바깥


def wall_points(rng):
    pts = []
    n = 3000
    u = rng.uniform(0.3, 5.7, n)
    v = rng.uniform(0.2, 2.4, n)
    pts.append(np.column_stack([u, np.full(n, TRUE_NORTH_Y), v]))       # 북 (y=3.05)
    pts.append(np.column_stack([u, np.zeros(n), v])[np.abs(u - 3.0) > 0.7])  # 남 (문 구멍 빼고)
    w = rng.uniform(0.3, 2.7, n)
    pts.append(np.column_stack([np.zeros(n), w, v]))
    pts.append(np.column_stack([np.full(n, 6.0), w, v]))
    pts.append(np.column_stack([u, w, np.zeros(n)]))
    P = np.vstack(pts)
    return P + rng.normal(0, 0.005, P.shape)


# 11-28 RESULTS2 로 기본값이 바뀜 (snap·repair·remove_behind·flatten 끔, cleanup 은 floaters 만). 이 단계 테스트들은
# 전체 사슬(snap -> cleanup -> repair)을 확인하므로 예전 값을 명시적으로 켬. extra 의 같은 항목이 이것보다 우선.
FULL_CHAIN = {"walls": {"snap": True, "repair": True, "remove_behind": True, "flatten": True},
              "cleanup": {"steps": ["floaters", "huge", "needles", "sh"]}}


def _merge_yaml(extra: str) -> str:
    import yaml
    d = {k: dict(v) for k, v in FULL_CHAIN.items()}
    for k, v in (yaml.safe_load(extra) or {}).items():
        if isinstance(v, dict) and k in d:
            d[k].update(v)
        else:
            d[k] = v
    return yaml.safe_dump(d, allow_unicode=True)


def setup(tmp_path, extra="", full_chain=True):
    root = make_project(tmp_path / "p", extra_yaml=_merge_yaml(extra) if full_chain else extra)
    p = Project(root)
    (root / "floorplan").mkdir()
    (root / "floorplan" / "plan.json").write_text(json.dumps(PLAN), encoding="utf-8")
    sec, paths = p.sections()[0]
    paths.colmap.mkdir(parents=True, exist_ok=True)
    (paths.colmap / "register.json").write_text(json.dumps({"version": 1, "mapper": "pose_prior_mapper"}))
    m = make_model(30, start=(1.0, 1.5, 1.4))
    rng = np.random.default_rng(0)
    P = wall_points(rng)
    pts = {}
    for i, x in enumerate(P, start=1000):
        pts[i] = cm.Point3D(i, x, np.array([200, 200, 200], np.uint8), 0.5,
                            np.array([[1, 0], [2, 0], [3, 0]], np.uint32))
    m = cm.Model(m.cameras, m.images, pts, m.frames, m.rigs_bytes)
    cm.write_model(m, paths.data_clean / "sparse" / "0")
    (paths.data_clean / "images").mkdir(parents=True)
    logs = []
    return p, StageContext(p, sec, paths, logs.append), logs


def test_walls_stage_refits_and_seeds(tmp_path, monkeypatch):
    p, ctx, logs = setup(tmp_path)
    st = walls_stage.Walls()
    assert st.applies(ctx)

    def fake_seed(planes, model, image_dir, **kw):
        n = {q.id: q for q in planes}["A:w2"]          # 북쪽 벽 (다시 맞춘 면)
        xyz = n.point(np.linspace(0.5, 5.5, 20), np.full(20, 1.0))
        return xyz, np.full((20, 3), 180, np.uint8), {"n_added": 20}
    monkeypatch.setattr(seed_mod, "seed_points", fake_seed)

    res = st.run(ctx)
    assert res.ok, res.message
    d = schemas.read_json(ctx.paths.planes_json)
    fit = {i["id"]: i for i in d["fit"]}
    assert fit["A:w2"]["source"] == "fit" and abs(fit["A:w2"]["shift_m"] + 0.05) < 0.01   # 바깥으로 5 cm
    north = [q for q in d["planes"] if q["id"] == "A:w2"][0]
    assert abs(north["n"][1] * TRUE_NORTH_Y - north["d"]) < 0.01
    seeded = cm.read_model(ctx.paths.data_seed / "sparse" / "0")
    orig = cm.read_model(ctx.paths.data_clean / "sparse" / "0")
    assert len(seeded.points) == len(orig.points) + 20
    assert (ctx.paths.data_seed / "images").exists()
    assert st.is_done(ctx)
    # 학습 입력은 data_seed 우선
    assert "walls" in [s.name for s in base.all_stages()]


def test_walls_stage_skips_without_plan_coords(tmp_path):
    p, ctx, _ = setup(tmp_path)
    (ctx.paths.colmap / "register.json").write_text(json.dumps({"version": 1, "mapper": "mapper"}))
    assert not walls_stage.Walls().applies(ctx)
    assert not snap_stage.Snap().applies(ctx)


def test_walls_disabled_by_config(tmp_path):
    p, ctx, _ = setup(tmp_path, extra="walls: {enabled: false}\n")
    assert not walls_stage.Walls().applies(ctx)


def test_snap_stage(tmp_path, monkeypatch):
    p, ctx, _ = setup(tmp_path, extra="walls: {seed: false}\n")
    assert walls_stage.Walls().run(ctx).ok
    assert not (ctx.paths.data_seed).exists()
    run = ctx.paths.train_run("brush", "default")
    rng = np.random.default_rng(3)
    n = 500
    ply = plyio.make(n, plyio.brush_names(2))
    P = np.column_stack([rng.uniform(0.5, 5.5, n), TRUE_NORTH_Y + rng.uniform(-0.02, 0.02, n), rng.uniform(0.3, 2.3, n)])
    P[-5:] = [[3, 3.6, 1], [3, 3.7, 1], [2, 3.5, 1.5], [1, 3.8, 1], [4, 4, 1]]     # 벽 뒤 부유물
    for k, c in enumerate("xyz"):
        ply.data[c] = P[:, k]
    ply.data["rot_0"] = 1.0
    for k in range(3):
        ply.data[f"scale_{k}"] = np.log(0.02)
    ply.data["opacity"] = 3.0
    plyio.write(run / "splat.ply", ply)
    st = snap_stage.Snap()
    assert st.applies(ctx)
    res = st.run(ctx)
    assert res.ok, res.message
    out = plyio.read(run / snap_stage.SNAP_PLY)
    assert out.n == n - 5
    assert np.abs(out.data["y"] - TRUE_NORTH_Y).max() < 0.012     # 다시 맞춘 면 위
    info = json.loads((run / snap_stage.SNAP_JSON).read_text(encoding="utf-8"))
    assert info["n_removed"] == 5
    assert info["flatness_after"]["thick_m"] < info["flatness_before"]["thick_m"]
    assert plyio.read(run / "splat.ply").n == n                     # 원본 그대로
    # fill·merge 는 붙인 것을 씀
    from indoorgs.stages import fill, merge
    assert fill.train_ply(ctx).name == snap_stage.SNAP_PLY
    assert merge.section_ply(p, ctx.section, ctx.paths).name == snap_stage.SNAP_PLY
    # 보고서에 벽 평탄화 줄
    from indoorgs.stages import report
    html = report.build_html(p)
    assert "벽 평탄화" in html and "다시 맞춤" in html and "부유물 지움 5" in html


def test_repair_stage_fills_hole(tmp_path):
    p, ctx, _ = setup(tmp_path, extra="walls: {seed: false}\n")
    assert walls_stage.Walls().run(ctx).ok
    run = ctx.paths.train_run("brush", "default")
    # 북쪽 벽을 촘촘히 덮되 가운데 0.6 m 구멍
    g = np.mgrid[0.3:5.7:0.04, 0.2:2.4:0.04].reshape(2, -1).T
    g = g[np.hypot(g[:, 0] - 3.0, g[:, 1] - 1.3) > 0.3]
    n = len(g)
    ply = plyio.make(n, plyio.brush_names(2))
    ply.data["x"], ply.data["y"], ply.data["z"] = g[:, 0], TRUE_NORTH_Y, g[:, 1]
    ply.data["rot_0"] = 1.0
    ply.data["scale_0"] = ply.data["scale_2"] = np.log(0.03)
    ply.data["scale_1"] = np.log(0.003)
    ply.data["opacity"] = 4.0
    plyio.write(run / "splat.ply", ply)
    assert snap_stage.Snap().run(ctx).ok
    st = snap_stage.Repair()
    assert st.applies(ctx)
    res = st.run(ctx)
    assert res.ok, res.message
    out = plyio.read(run / snap_stage.REPAIR_PLY)
    assert out.n > plyio.read(run / snap_stage.SNAP_PLY).n
    info = json.loads((run / snap_stage.REPAIR_JSON).read_text(encoding="utf-8"))
    north = info["faces"]["A:w2"]
    assert north["hole_m2_before"] > 0.1 and north["hole_m2_after"] < 0.02
    from indoorgs.stages import fill
    assert fill.train_ply(ctx).name == snap_stage.REPAIR_PLY


# ---------------- 11-28 검토 회귀 ----------------
import os


def _wall_ply(extra=None):
    rng = np.random.default_rng(5)
    n = 2000
    P = np.column_stack([rng.uniform(0.5, 5.5, n), TRUE_NORTH_Y + rng.uniform(-0.01, 0.01, n), rng.uniform(0.3, 2.3, n)])
    if extra is not None:
        P = np.vstack([P, extra])
    ply = plyio.make(len(P), plyio.brush_names(2))
    for k, c in enumerate("xyz"):
        ply.data[c] = P[:, k]
    ply.data["rot_0"] = 1.0
    for k in range(3):
        ply.data[f"scale_{k}"] = np.log(0.02)
    ply.data["opacity"] = 3.0
    return ply


def test_snap_stage_uses_only_fitted_faces(tmp_path):
    """천장은 정합 점이 없어 도면 그대로(2.6 m). 실제 천장 스플랫(2.8 m)을 지우거나 2.6 m 로 끌어내리지 않음."""
    p, ctx, logs = setup(tmp_path, extra="walls: {seed: false}\n")
    assert walls_stage.Walls().run(ctx).ok
    fit = {i["id"]: i["source"] for i in schemas.read_json(ctx.paths.planes_json)["fit"]}
    assert fit["A:ceiling"] == "plan" and fit["A:w2"] == "fit"
    rng = np.random.default_rng(6)
    ceil = np.column_stack([rng.uniform(0.3, 5.7, 3000), rng.uniform(0.3, 2.7, 3000), np.full(3000, 2.8)])
    run = ctx.paths.train_run("brush", "default")
    plyio.write(run / "splat.ply", _wall_ply(ceil))
    res = snap_stage.Snap().run(ctx)
    assert res.ok, res.message
    out = plyio.read(run / snap_stage.SNAP_PLY)
    assert np.sum(np.abs(out.data["z"] - 2.8) < 1e-6) == 3000           # 실제 천장 그대로
    info = schemas.read_json(run / snap_stage.SNAP_JSON)
    assert "A:ceiling" not in info["planes_used"] and "A:w2" in info["planes_used"]
    assert "A:ceiling" in info["unfitted_planes"]
    assert "A:floor" not in info["planes_used"]                            # 바닥 스플랫 없음 -> 밀도 확인 실패
    assert any("맞추지 못한 면" in m for m in logs)


def test_snap_stage_skips_ply_not_in_plan_coords(tmp_path):
    p, ctx, logs = setup(tmp_path, extra="walls: {seed: false}\n")
    assert walls_stage.Walls().run(ctx).ok
    run = ctx.paths.train_run("brush", "default")
    ply = _wall_ply()
    ply.data["x"] += 40.0
    plyio.write(run / "splat.ply", ply)
    st = snap_stage.Snap()
    res = st.run(ctx)
    assert res.ok and res.skipped and not (run / snap_stage.SNAP_PLY).exists()
    assert st.is_done(ctx)
    rep = snap_stage.Repair()
    assert rep.is_done(ctx)
    r2 = rep.run(ctx)
    assert r2.ok and r2.skipped
    from indoorgs.stages import fill
    assert fill.train_ply(ctx).name == "splat.ply"


def test_repair_without_planes_json(tmp_path):
    p, ctx, _ = setup(tmp_path, extra="walls: {seed: false}\n")
    run = ctx.paths.train_run("brush", "default")
    plyio.write(run / snap_stage.SNAP_PLY, _wall_ply())
    res = snap_stage.Repair().run(ctx)
    assert not res.ok and "planes.json" in res.message


def _touch(path, t):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(b"x")
    os.utime(path, (t, t))


def test_best_ply_respects_mtime_and_config(tmp_path):
    p, ctx, _ = setup(tmp_path)
    run = ctx.paths.train_run("brush", "default")
    base_t = 1_700_000_000
    _touch(run / "splat.ply", base_t)
    _touch(run / snap_stage.SNAP_PLY, base_t + 10)
    _touch(run / snap_stage.REPAIR_PLY, base_t + 20)
    from indoorgs.stages import fill, merge
    assert fill.train_ply(ctx).name == snap_stage.REPAIR_PLY
    _touch(run / snap_stage.SNAP_PLY, base_t + 30)                     # snap 을 다시 돌림 -> repaired 는 낡음
    assert fill.train_ply(ctx).name == snap_stage.SNAP_PLY
    _touch(run / "splat.ply", base_t + 40)                              # 학습 결과를 바꿈 -> 둘 다 낡음
    assert fill.train_ply(ctx).name == "splat.ply"
    assert merge.section_ply(p, ctx.section, ctx.paths).name == "splat.ply"
    _touch(run / snap_stage.SNAP_PLY, base_t + 50)
    _touch(run / snap_stage.REPAIR_PLY, base_t + 60)
    assert snap_stage.best_ply(run, True, False).name == snap_stage.SNAP_PLY    # repair 꺼짐
    p2, ctx2, _ = setup(tmp_path / "b", extra="walls: {snap: false}\n")
    run2 = ctx2.paths.train_run("brush", "default")
    for i, nm in enumerate(("splat.ply", snap_stage.SNAP_PLY, snap_stage.REPAIR_PLY)):
        _touch(run2 / nm, base_t + i)
    assert fill.train_ply(ctx2).name == "splat.ply"                    # 설정이 꺼지면 예전 결과 안 씀
    (ctx2.paths.colmap / "register.json").write_text(json.dumps({"version": 1, "mapper": "mapper"}))
    assert fill.train_ply(ctx2).name == "splat.ply"


def test_snap_repair_is_done_and_prev_targets(tmp_path):
    p, ctx, _ = setup(tmp_path, extra="walls: {seed: false}\n")
    assert walls_stage.Walls().run(ctx).ok
    run = ctx.paths.train_run("brush", "default")
    t0 = ctx.paths.planes_json.stat().st_mtime + 10
    for i, nm in enumerate(("splat.ply", snap_stage.SNAP_PLY, snap_stage.SNAP_JSON,
                            snap_stage.REPAIR_PLY, snap_stage.REPAIR_JSON)):
        _touch(run / nm, t0 + i)
    sn, rp = snap_stage.Snap(), snap_stage.Repair()
    assert sn.is_done(ctx) and rp.is_done(ctx)
    names = {q.name for q in sn.prev_targets(ctx)}
    assert {snap_stage.REPAIR_PLY, snap_stage.REPAIR_JSON, snap_stage.SNAP_PLY} <= names
    _touch(run / snap_stage.SNAP_PLY, t0 + 100)
    assert not rp.is_done(ctx)
    _touch(run / "splat.ply", t0 + 200)
    assert not sn.is_done(ctx)


def test_train_uses_data_seed_only_when_valid(tmp_path, monkeypatch):
    p, ctx, logs = setup(tmp_path)
    monkeypatch.setattr(seed_mod, "seed_points",
                        lambda planes, model, image_dir, **kw: (np.zeros((0, 3)), np.zeros((0, 3), np.uint8), {}))
    assert walls_stage.Walls().run(ctx).ok
    assert (ctx.paths.data_seed / "sparse" / "0").exists()
    ok, why = walls_stage.seed_usable(ctx)
    assert ok, why
    # 설정을 끔 -> 안 씀
    p2 = Project(p.root)
    p2.cfg.walls.seed = False
    ok, why = walls_stage.seed_usable(StageContext(p2, ctx.section, ctx.paths, logs.append))
    assert not ok and "seed" in why
    # 도면 좌표가 아니게 됨 -> 안 씀
    reg = ctx.paths.colmap / "register.json"
    reg.write_text(json.dumps({"version": 1, "mapper": "mapper"}))
    assert not walls_stage.seed_usable(ctx)[0]
    reg.write_text(json.dumps({"version": 1, "mapper": "pose_prior_mapper"}))
    # data_clean 이 바뀜 (diagnose 다시) -> 안 씀
    m = cm.read_model(ctx.paths.data_clean / "sparse" / "0")
    new, _ = cm.remove_images(m, {next(iter(m.images.values())).name})
    import shutil
    shutil.rmtree(ctx.paths.data_clean / "sparse" / "0")
    cm.write_model(new, ctx.paths.data_clean / "sparse" / "0")
    ok, why = walls_stage.seed_usable(ctx)
    assert not ok and "다름" in why


def test_train_stage_falls_back_to_data_clean(tmp_path, monkeypatch):
    """Train.run 이 쓸모없는 data_seed 대신 data_clean 을 학습기에 넘기는지 (학습기 명령만 가로챔)."""
    from indoorgs.stages import train as train_stage
    from indoorgs.trainers import REGISTRY
    p, ctx, logs = setup(tmp_path, extra="walls: {seed: false}\n")
    (ctx.paths.data_seed / "sparse" / "0").mkdir(parents=True)        # 예전 실행이 남긴 data_seed
    seen = {}
    ad = REGISTRY[p.cfg.train.trainer]

    class Info:
        found, missing_options, version, message = True, [], "x", ""
    monkeypatch.setattr(ad, "detect", lambda exe: Info())

    def fake_cmd(exe, data_dir, out_dir, preset, gpu):
        seen["data"] = data_dir
        raise RuntimeError("stop")
    monkeypatch.setattr(ad, "command", fake_cmd)
    try:
        train_stage.Train().run(ctx)
    except RuntimeError:
        pass
    assert seen["data"] == ctx.paths.data_clean
    assert any("data_seed 안 씀" in m for m in logs)


def test_rerun_moves_walls_outputs_when_walls_no_longer_applies(tmp_path, monkeypatch):
    """register 를 다시 돌려 도면 좌표가 아니게 되면 walls/·data_seed 는 Walls 가 적용 안 돼도 _prev_ 로 옮겨짐."""
    p, ctx, _ = setup(tmp_path)
    ctx.paths.walls.mkdir(parents=True)
    ctx.paths.planes_json.write_text("{}")
    (ctx.paths.data_seed / "sparse" / "0").mkdir(parents=True)
    (ctx.paths.colmap / "register.json").write_text(json.dumps({"version": 1, "mapper": "mapper"}))
    monkeypatch.setattr(base, "all_stages", lambda: [walls_stage.Walls()])
    base.run_all(p, rerun="walls", echo=None)
    assert not ctx.paths.walls.exists() and not ctx.paths.data_seed.exists()
    left = [q.name for q in ctx.paths.root.iterdir()]
    assert any(n.startswith("walls_prev_") for n in left) and any(n.startswith("data_seed_prev_") for n in left)


def test_plan_coords_same_rule_for_all_stages(tmp_path):
    """fill·merge 도 diagnose 와 같은 판정 (register.json mapper). priors.json 이 남아 있어도 mapper 면 도면 좌표 아님."""
    from indoorgs.stages import fill, diagnose
    p, ctx, _ = setup(tmp_path)
    schemas.write_json(ctx.paths.priors, {"version": 1, "coordinate_system": "CARTESIAN", "frames": []})
    (ctx.paths.colmap / "register.json").write_text(json.dumps({"version": 1, "mapper": "mapper"}))
    assert not diagnose.model_in_plan_coords(ctx) and not fill.in_plan_coords(ctx)
    ctx.paths.priors.unlink()
    (ctx.paths.colmap / "register.json").write_text(json.dumps({"version": 1, "mapper": "pose_prior_mapper"}))
    assert fill.in_plan_coords(ctx)
    # merge: 도면 좌표가 아닌 구간은 floorplan 배치로 치지 않음 (구간 하나면 single)
    run = ctx.paths.train_run("brush", "default")
    plyio.write(run / "splat.ply", _wall_ply())
    (ctx.paths.colmap / "register.json").write_text(json.dumps({"version": 1, "mapper": "mapper"}))
    schemas.write_json(ctx.paths.priors, {"version": 1, "coordinate_system": "CARTESIAN", "frames": []})
    from indoorgs.stages import merge
    res = merge.Merge().run(StageContext(p, None, None, lambda m: None))
    assert res.ok, res.message
    assert schemas.read_json(p.layout_json)["sections"]["s1"]["source"] == "single"
