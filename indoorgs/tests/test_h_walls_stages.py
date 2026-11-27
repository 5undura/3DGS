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


def setup(tmp_path, extra=""):
    root = make_project(tmp_path / "p", extra_yaml=extra)
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
