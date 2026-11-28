"""e2e_v4 제안 1·2 (축척 확인 + 먼 끝 벽 2차 탐색 "fit_scaled") 와 3·4 (far_from_prior 문턱을 직접 맞춤 잔차에 비례,
뒤집힌 모델 검사)."""
import json

import numpy as np

from indoorgs import config, schemas
from indoorgs.colmap import model as cm
from indoorgs.colmap.pair_verify import rot2qvec
from indoorgs.project import Project
from indoorgs.stages import walls as walls_stage
from indoorgs.stages.base import StageContext
from indoorgs.stages.diagnose import (FAR_CAP_M, FAR_FROM_PRIOR_M, Diagnose, far_threshold, orientation_check)
from indoorgs.walls import planes as wp
from indoorgs.walls.refine import refine_planes, scale_check
from a_fixtures import make_model, make_project

# 도면: 복도 x -0.1..24.2, y -1.25..1.25, 바닥 0, 천장 2.6 (e2e_v4 와 같음)
PLAN = {"version": 1, "units": "m", "rooms": [
    {"id": "corr", "box": {"center": [12.05, 0.0], "size": [24.3, 2.5]}, "floor_z": 0.0, "ceiling_z": 2.6}]}
# 실제(모델 좌표) 복도: 축척 ~0.94 + 위치 오차 -> 옆 벽 5~10 cm, 끝 벽 70~90 cm (search 0.3 밖)
REAL = {"version": 1, "units": "m", "rooms": [
    {"id": "corr", "box": {"center": [11.95, 0.025], "size": [22.7, 2.35]}, "floor_z": 0.05, "ceiling_z": 2.5}]}


def _plans():
    return (wp.planes_from_plan(schemas.parse_plan(PLAN)), {p.id: p for p in wp.planes_from_plan(schemas.parse_plan(REAL))})


def _points(real, rng, drop=(), n_side=3000, n_end=400, sigma=0.005):
    pts = []
    for pid, p in real.items():
        if pid in drop:
            continue
        n = n_end if pid in ("corr:w1", "corr:w3") else n_side
        u = rng.uniform(0.02, 0.98, n) * p.size[0]
        v = rng.uniform(0.02, 0.98, n) * p.size[1]
        X = p.point(u, v) + rng.normal(0, sigma, n)[:, None] * p.n
        if p.polygon is not None:
            X = X[wp._in_polygon(X[:, 0], X[:, 1], p.polygon)]
        pts.append(X)
    return np.vstack(pts)


def _cams(x0=1.5, x1=22.5):
    xs = np.arange(x0, x1 + 1e-9, 0.1)
    return np.column_stack([xs, np.zeros_like(xs), np.full_like(xs, 1.4)])


def _off(q, real_p):
    """맞춘 면 가운데의 실제 면까지 부호 거리 (m)."""
    return float(real_p.signed_dist(q.point(q.size[0] / 2, q.size[1] / 2)))


def test_far_end_walls_need_scaled_search():
    planes, real = _plans()
    X = _points(real, np.random.default_rng(0))
    C = _cams()
    out, info = refine_planes(planes, X, centers=C)                       # 예전: 끝 벽은 탐색 밖 -> plan
    src = {i["id"]: i["source"] for i in info}
    assert src["corr:w1"] == "plan" and src["corr:w3"] == "plan"
    assert src["corr:w0"] == src["corr:w2"] == src["corr:floor"] == src["corr:ceiling"] == "fit"

    rep = {}
    out, info = refine_planes(planes, X, centers=C, scale_search=True, report=rep)
    by = {i["id"]: i for i in info}
    sc = rep["scale_check"]
    assert sc["warn"] and abs(sc["axes"]["y"]["ratio"] - 0.94) < 0.005 and abs(sc["axes"]["z"]["ratio"] - 2.45 / 2.6) < 0.005
    assert 0.93 < sc["scale"] < 0.95
    for k in ("corr:w1", "corr:w3"):
        assert by[k]["source"] == "fit_scaled", by[k]
        q = out[[p.id for p in planes].index(k)]
        assert abs(_off(q, real[k])) < 0.01                               # 실제 끝 벽 1 cm 안
        assert by[k]["shift_m"] > 0.6 and by[k]["search_m"] > 0.3
    assert set(rep["scale_search"]["accepted"]) == {"corr:w1", "corr:w3"}
    # 끝 벽까지 맞춘 뒤의 x 축 비율 = 22.7 / 24.3
    sc2 = scale_check(planes, out, info, sources=("fit", "fit_scaled"))
    assert abs(sc2["axes"]["x"]["ratio"] - 22.7 / 24.3) < 0.005


def test_scaled_search_rejects_clutter_and_wall_across_cameras():
    planes, real = _plans()
    rng = np.random.default_rng(1)
    X = _points(real, rng, drop=("corr:w1",))
    # 끝 벽 점 없이 흩어진 잡음만 -> 면 안 만듦
    clutter = np.column_stack([rng.uniform(22.4, 24.2, 300), rng.uniform(-1.1, 1.1, 300), rng.uniform(0.1, 2.4, 300)])
    out, info = refine_planes(planes, np.vstack([X, clutter]), centers=_cams(), scale_search=True)
    assert {i["id"]: i["source"] for i in info}["corr:w1"] == "plan"
    # 탐색 창 안의 촘촘한 판(x=22.9) 이지만 카메라가 그 뒤(23.0~23.4)까지 감 -> 동선을 가로지르는 가짜 면이라 버림
    pv = real["corr:w1"]
    u = rng.uniform(0.02, 0.98, 600) * pv.size[0]
    v = rng.uniform(0.02, 0.98, 600) * pv.size[1]
    panel = pv.point(u, v)
    panel[:, 0] = 22.9 + rng.normal(0, 0.005, 600)
    out, info = refine_planes(planes, np.vstack([X, panel]), centers=_cams(1.5, 23.4), scale_search=True)
    by = {i["id"]: i for i in info}
    assert by["corr:w1"]["source"] == "plan"
    assert "카메라" in (by["corr:w1"].get("reason_scaled") or "")
    # 같은 판인데 카메라가 앞에서 멈추면 받아들임 (판이 진짜 끝 벽일 수 있음)
    out, info = refine_planes(planes, np.vstack([X, panel]), centers=_cams(1.5, 22.5), scale_search=True)
    assert {i["id"]: i["source"] for i in info}["corr:w1"] == "fit_scaled"


def test_scale_check_no_warning_when_plan_matches_and_ignores_infer():
    planes = wp.planes_from_plan(schemas.parse_plan(PLAN))
    real = {p.id: p for p in planes}
    X = _points(real, np.random.default_rng(2), drop=("corr:w1", "corr:w3"))
    rep = {}
    out, info = refine_planes(planes, X, centers=_cams(), report=rep, trust_plan_dims=True)
    sc = rep["scale_check"]
    assert not sc["warn"] and abs(sc["scale"] - 1) < 0.005 and "x" not in sc["axes"]
    # infer 면(끝 벽이 있다면)은 축척 계산에 안 씀
    info2 = [dict(i, source="infer") if i["id"] == "corr:w2" else i for i in info]
    assert "y" not in scale_check(planes, out, info2)["axes"]
    assert scale_check(planes, planes, [dict(i, source="plan") for i in info])["scale"] is None


def test_seed_sources_default_excludes_fit_scaled():
    base = {"sections": [{"name": "a", "video": "v.mp4"}]}
    w = config.parse(base).walls
    assert w.scale_search and not w.seed_scaled
    assert "fit_scaled" not in walls_stage.seed_sources(w)
    w = config.parse({**base, "walls": {"seed_scaled": True}}).walls
    assert "fit_scaled" in walls_stage.seed_sources(w)


# ---------------- diagnose ----------------
def test_far_threshold_adapts_to_alignment_residual():
    assert far_threshold(None) == {"m": FAR_FROM_PRIOR_M, "basis": "default"}
    assert far_threshold({"mapper": "pose_prior_mapper"})["m"] == FAR_FROM_PRIOR_M
    t = far_threshold({"aligned": True, "align": {"resid_median_m": 0.284}})
    assert t["basis"] == "align_resid" and abs(t["m"] - 1.42) < 1e-6
    assert far_threshold({"aligned": True, "align": {"resid_median_m": 0.05}})["m"] == FAR_FROM_PRIOR_M
    assert far_threshold({"aligned": True, "align": {"resid_median_m": 3.0}})["m"] == FAR_CAP_M
    assert far_threshold({"aligned": False, "align": {"resid_median_m": 0.3}})["basis"] == "default"


def _gimbal_model(n=30, upside_down=False):
    """x 로 걷고 +x 를 보는 짐벌 카메라 (위 = +z). upside_down 이면 진행 축 둘레로 180도."""
    m = make_model(n, start=(0.5, 0.0, 1.4))
    fwd, up = np.array([1.0, 0, 0]), np.array([0, 0, -1.0 if upside_down else 1.0])
    R = np.stack([np.cross(-up, fwd), -up, fwd])     # 행: 카메라 x(오른쪽), y(아래), z(앞) 의 월드 방향
    assert np.linalg.det(R) > 0
    for im in m.images.values():
        c = im.center
        im.q = np.asarray(rot2qvec(R), float)
        im.t = -R @ c
    return m


def test_orientation_check():
    o = orientation_check(_gimbal_model())
    assert o["up_z_median"] > 0.99 and not o["flipped"] and not o["tilted"]
    o = orientation_check(_gimbal_model(upside_down=True))
    assert o["flipped"] and o["up_down_frac"] == 1.0 and "뒤집" in o["message"]
    mir = {"align": {"transform": {"rotation": np.diag([1.0, -1.0, 1.0]).tolist()}}}
    assert orientation_check(_gimbal_model(), mir)["mirrored"]


def _setup_diag(tmp_path, upside_down, aligned_resid=None):
    root = make_project(tmp_path / "p")
    p = Project(root)
    sec, paths = p.sections()[0]
    m = _gimbal_model(upside_down=upside_down)
    cm.write_model(m, paths.data_model)
    (paths.data / "images").mkdir()
    paths.colmap.mkdir(parents=True, exist_ok=True)
    reg = {"version": 1, "mapper": "pose_prior_mapper"}
    if aligned_resid is not None:
        reg.update(aligned=True, align={"resid_median_m": aligned_resid,
                                        "transform": {"scale": 1.0, "rotation": np.eye(3).tolist(), "translation": [0, 0, 0]}})
    (paths.colmap / "register.json").write_text(json.dumps(reg))
    frames = []
    for im in m.images.values():
        xyz = im.center.tolist()
        if im.id == 6:
            xyz[1] += 1.2                                          # 1.2 m 떨어짐: 1 m 문턱이면 빠지고 1.42 m 면 남음
        frames.append({"image": im.name, "xyz": xyz, "sigma": [0.5, 0.5, 0.3]})
    paths.priors.write_text(json.dumps({"version": 1, "coordinate_system": "CARTESIAN", "frames": frames}))
    return StageContext(p, sec, paths, lambda s: None)


def test_diagnose_records_adaptive_threshold(tmp_path):
    ctx = _setup_diag(tmp_path / "a", upside_down=False)
    assert Diagnose().run(ctx).ok
    d = schemas.read_json(ctx.paths.diagnose)
    assert d["far_from_prior"] == ["f_00006_000050.jpg"] and d["far_from_prior_threshold"]["basis"] == "default"
    assert d["orientation"]["flipped"] is False
    ctx = _setup_diag(tmp_path / "b", upside_down=False, aligned_resid=0.284)
    assert Diagnose().run(ctx).ok
    d = schemas.read_json(ctx.paths.diagnose)
    assert d["far_from_prior"] == [] and abs(d["far_from_prior_threshold"]["m"] - 1.42) < 1e-6


def test_diagnose_fails_on_flipped_model(tmp_path):
    ctx = _setup_diag(tmp_path, upside_down=True)
    old = ctx.paths.data_clean / "sparse" / "0"
    old.mkdir(parents=True)                                          # 예전 실행이 남긴 data_clean
    res = Diagnose().run(ctx)
    assert not res.ok and "뒤집" in res.message
    d = schemas.read_json(ctx.paths.diagnose)
    assert d["orientation"]["flipped"] and d["failed"]
    assert not ctx.paths.data_clean.exists()                         # train 이 못 쓰게
    assert any(q.name.startswith("data_clean_failed_") for q in ctx.paths.data_clean.parent.iterdir())  # 지우지 않고 옮김
    assert not Diagnose().is_done(ctx)
