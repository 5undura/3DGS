"""동선 고정 정합 (colmap/anchored.py, register.mode anchored). 합성 자세로 각 단계, 가짜 COLMAP 으로 명령 순서."""
from __future__ import annotations

import json
import math

import cv2
import numpy as np
import pytest

from a_fixtures import FakeRunner, arg, colmap_handlers, make_model, make_project
from indoorgs.colmap import anchored as A
from indoorgs.colmap import model as cm
from indoorgs.colmap import runner
from indoorgs.config import ConfigError, parse


def _rotz(a):
    return A._exp([0, 0, a])


# ---------------------------------------------------------------- 1) 동선 -> 요, 표시
def test_route_headings_straight_stop_and_turn():
    t = np.arange(0, 20, 0.2)
    x = np.where(t < 8, 0.5 * t, np.where(t < 10, 4.0, 4.0 + 0.5 * (t - 10)))        # 8~10 초 멈춤
    y = np.where(t < 14, 0.0, 0.5 * (t - 14))                                        # 14 초에 45° 꺾임
    P = np.stack([x, y, np.full_like(t, 1.4)], 1)
    h = A.route_headings(P, t, window_s=0.4, turn_deg_s=40.0)
    assert abs(math.degrees(h["yaw"][5])) < 1e-6
    assert h["flag"][(t > 8.5) & (t < 9.5)].all()                                     # 멈춤
    assert all(h["reason"][i] == "stop" for i in np.flatnonzero((t > 8.6) & (t < 9.4)))
    assert not h["flag"][(t > 2) & (t < 6)].any()
    assert h["flag"][np.argmin(np.abs(t - 14))]                                       # 급회전
    assert abs(math.degrees(h["yaw"][-1]) - 45) < 1                                  # 꺾인 뒤 45°
    assert abs(h["yaw"][np.argmin(np.abs(t - 9))]) < 0.1                              # 멈춘 곳은 이웃 요


def test_yaw_pitch_to_R_roundtrip_level_and_colmap_axes():
    for yaw in (-2.5, -1.0, 0.0, 0.3, 3.0):
        R = A.yaw_pitch_to_R(yaw)
        assert np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1)
        assert abs(R[2, 0]) < 1e-12                                       # 오른쪽 축(x) 수평 = 롤 0
        assert R[2, 1] < 0                                                # 카메라 y(아래) 가 월드 아래
        assert abs(A._wrap(A.R_yaw(R) - yaw)) < 1e-9


# ---------------------------------------------------------------- 2) 짝 고르기
def test_select_pairs_distance_yaw_and_sequential():
    n = 50
    P = np.stack([0.5 * np.arange(n), np.zeros(n), np.full(n, 1.4)], 1)
    P[30:40] = P[29:19:-1]                     # 30~39: 왔던 길을 뒷걸음으로 (같은 방향을 봄)
    P[40:50] = P[29:19:-1] + [0, 0.3, 0]       # 40~49: 같은 길을 돌아서서 (반대 방향)
    yaw = np.where(np.arange(n) < 40, 0.0, math.pi)
    pairs = A.select_pairs(P, yaw, seq_window=3, max_dist_m=1.2, max_yaw_deg=60, max_per_image=10)
    s = set(pairs)
    assert all((i, i + k) in s for i in range(n - 3) for k in (1, 2, 3))            # 앞뒤 3장 모두
    far = [(i, j) for i, j in pairs if j - i > 3]
    assert (20, 39) in s and (29, 30) in s                                          # 같은 자리, 같은 방향
    assert all(np.linalg.norm(P[i, :2] - P[j, :2]) <= 1.2 + 1e-9 for i, j in far)
    assert all(math.cos(yaw[i] - yaw[j]) >= math.cos(math.radians(60)) - 1e-9 for i, j in far)
    assert not any(i < 40 <= j for i, j in far)                                     # 반대 방향은 안 고름
    assert all(i < j for i, j in pairs)


def test_write_pairs_file(tmp_path):
    n = A.write_pairs_file(tmp_path / "p.txt", ["a.jpg", "b.jpg", "c.jpg"], [(0, 2), (1, 2)])
    assert n == 2 and (tmp_path / "p.txt").read_text(encoding="utf-8") == "a.jpg c.jpg\nb.jpg c.jpg\n"


# ---------------------------------------------------------------- 초점거리, 상대 자세 (합성 점)
def _scene(n_pts=300, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(-3, 3, n_pts), rng.uniform(-2, 2, n_pts), rng.uniform(4, 9, n_pts)], 1)


def _project(X, R_cw, t, K):
    x = (X @ R_cw.T + t) @ K.T
    return x[:, :2] / x[:, 2:]


def test_estimate_focal_from_fundamentals():
    f, w, h = 600.0, 960, 540
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    Fs = []
    rng = np.random.default_rng(1)
    for k in range(12):
        R = cv2.Rodrigues(rng.normal(0, 0.08, 3))[0]
        tvec = rng.normal(0, 1, 3)
        tvec /= np.linalg.norm(tvec)
        tx = np.array([[0, -tvec[2], tvec[1]], [tvec[2], 0, -tvec[0]], [-tvec[1], tvec[0], 0]])
        E = tx @ R
        Fs.append(np.linalg.inv(K).T @ E @ np.linalg.inv(K))
    est, info = A.estimate_focal(Fs, w, h)
    assert est is not None and abs(est - f) / f < 0.02, info
    assert A.estimate_focal(Fs[:3], w, h)[0] is None                                  # 너무 적으면 None


def test_relative_pose_translation_and_pure_rotation():
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    X = _scene()
    R2 = _rotz(0.05) @ cv2.Rodrigues(np.array([0.02, 0.0, 0.0]))[0]
    C2 = np.array([0.5, 0.0, 0.3])
    p1 = _project(X, np.eye(3), np.zeros(3), K)
    p2 = _project(X, R2, -R2 @ C2, K)
    R, d, par = A.relative_pose(p1, p2, K, np.zeros(4))
    assert np.degrees(np.linalg.norm(A._log(R.T @ R2))) < 0.2
    assert d is not None and np.degrees(math.acos(np.clip(d @ C2 / np.linalg.norm(C2), -1, 1))) < 1.0
    p3 = _project(X, R2, np.zeros(3), K)                                              # 제자리 회전
    R, d, par = A.relative_pose(p1, p3, K, np.zeros(4))
    assert d is None and par < 1.0 and np.degrees(np.linalg.norm(A._log(R.T @ R2))) < 0.2


# ---------------------------------------------------------------- 3) 회전·위치 평균 (합성 그래프)
def _graph(n=60, noise_deg=0.3, seed=0, wobble=0.4):
    rng = np.random.default_rng(seed)
    yaw_true = wobble * np.sin(np.arange(n) / 6.0)                     # 진행 방향(0) 둘레로 흔들림
    Rg = np.array([A.yaw_pitch_to_R(y, 0.05 * math.sin(k / 9)) for k, y in enumerate(yaw_true)])
    Cg = np.stack([0.15 * np.arange(n), 0.1 * np.sin(np.arange(n) / 10), np.full(n, 1.45)], 1)
    rels = []
    for i in range(n):
        for j in range(i + 1, min(n, i + 6)):
            Rij = Rg[j].T @ Rg[i]
            Rij = A._exp(rng.normal(0, math.radians(noise_deg), 3)) @ Rij
            dv = Rg[i].T @ (Cg[j] - Cg[i])
            rels.append(A.RelPose(i, j, Rij, dv / np.linalg.norm(dv), 100, 3.0))
    return Rg, Cg, rels


def test_solve_rotations_recovers_wobbling_heading_with_route_yaw():
    n = 60
    Rg, Cg, rels = _graph(n)
    rels.append(A.RelPose(3, 40, _rotz(1.0) @ (Rg[40].T @ Rg[3]), None, 20, 0.5))      # 가짜 짝 (회전 57° 틀림)
    yaw_route = np.zeros(n)                                                         # 동선은 직선 (흔들림 모름)
    R, conn, used, st = A.solve_rotations(n, rels, yaw_route, np.zeros(n, bool))
    assert conn.all()
    # 전체 요는 동선 요 평균으로 정해짐 (흔들림 평균만큼 어긋날 수 있음 -> 나중에 align_model_to_priors 가 맞춤).
    # 전체 요 하나를 뺀 뒤 사진마다 오차
    off = float(np.angle(np.mean(np.exp(1j * A._wrap(np.array([A.R_yaw(b) - A.R_yaw(a) for a, b in zip(R, Rg)]))))))
    err = [np.degrees(np.linalg.norm(A._log((_rotz(off) @ a).T @ b))) for a, b in zip(R, Rg)]
    assert np.median(err) < 1.0 and max(err) < 2.0, err
    assert not any(r.i == 3 and r.j == 40 for r in used)                           # 가짜 짝은 빠짐
    assert st["rel_res_median_deg"] < 1.0


def test_translation_averaging_pulls_noisy_route_to_directions():
    n = 60
    Rg, Cg, rels = _graph(n, noise_deg=0.0)
    rng = np.random.default_rng(3)
    P = Cg + np.stack([np.cumsum(rng.normal(0, 0.03, n)), rng.normal(0, 0.2, n), np.zeros(n)], 1)
    S = np.tile([0.5, 0.5, 0.3], (n, 1))
    C, st = A.translation_averaging(Rg, rels, P, S)
    e_prior = np.linalg.norm(P - Cg, axis=1)
    e = np.linalg.norm(C - Cg, axis=1)
    assert np.median(e) < 0.5 * np.median(e_prior), (np.median(e), np.median(e_prior))
    assert st["edges"] == len(rels)


def test_initial_model_poses_and_focal():
    m0 = make_model(5)
    names = sorted(im.name for im in m0.images.values())
    dbd = A.DbData({i + 1: nm for i, nm in enumerate(names)}, (1, 4, 960, 540, np.array([1152.0, 1152, 480, 270, 0.1, 0, 0, 0])),
                   {i + 1: 1 for i in range(5)}, {})
    R = np.array([A.yaw_pitch_to_R(0.1 * k) for k in range(5)])
    C = np.stack([np.arange(5.0), np.zeros(5), np.full(5, 1.4)], 1)
    use = np.array([True, True, False, True, True])
    m = A.initial_model(dbd, names, R, C, use, 570.0)
    assert len(m.images) == 4 and not m.points
    cam = m.cameras[1]
    assert np.allclose(cam.params, [570, 570, 480, 270, 0, 0, 0, 0])
    for im in m.images.values():
        k = names.index(im.name)
        assert np.allclose(im.center, C[k]) and np.allclose(im.R.T, R[k])


# ---------------------------------------------------------------- 점, 창
def test_weak_images_and_thin_points():
    m = make_model(10)
    obs = A.observation_counts(m)
    names = sorted(obs)
    assert obs[names[0]] == 1 and obs[names[1]] == 2 and obs[names[5]] == 3 and obs[names[-1]] == 1
    assert A.weak_images(m, 2) == {names[0], names[-1]}
    t, st = A.thin_points(m, max_per_image=1)
    assert st["points_before"] == len(m.points) and st["points_after"] <= len(m.points)
    t, st = A.thin_points(m, min_track=4)                                  # 트랙이 모두 3 -> 다 빠짐
    assert st["points_after"] == 0 and all((im.pid < 0).all() for im in t.images.values())


def test_windows_cover_every_frame_once_inside_its_window():
    order = [f"f{i:03d}" for i in range(103)]
    ws = A.windows(order, 20)
    cores = [nm for _, core in ws for nm in core]
    assert cores == order
    assert all(set(core) <= set(win) and len(win) == 20 for win, core in ws)
    assert A.windows(order[:7], 20) == [(order[:7], order[:7])]
    assert A.windows([], 20) == []


def test_merge_windows_undoes_window_gauge():
    base = make_model(12)
    names = sorted(im.name for im in base.images.values())
    from indoorgs.colmap.pair_verify import transform_model
    res = []
    for win, core in A.windows(names, 6, 3):
        sub = A.submodel(base, set(win))
        assert {im.name for im in sub.images.values()} == set(win)
        # BA 결과 흉내: 창마다 다른 닮음 변환 (자세 자체는 그대로)
        res.append((core, transform_model(sub, 2.0, _rotz(0.7), np.array([5.0, -1.0, 0.3]))))
    merged, st = A.merge_windows(base, res)
    assert st["windows"] == len(res) and st["moved_max_m"] < 1e-6
    for iid, im in base.images.items():
        assert np.allclose(merged.images[iid].center, im.center, atol=1e-6)
    assert not merged.points


# ---------------------------------------------------------------- 설정, register 흐름 (가짜 COLMAP)
def test_config_mode_and_finish_enums():
    base = {"sections": [{"name": "s1", "video": "v.mp4"}]}
    cfg = parse(base)
    assert cfg.register.mode == "two_pass" and cfg.register.anchored_finish == "prior_mapper"
    assert not cfg.register.ba_use_gpu
    assert parse({**base, "register": {"mode": "anchored", "anchored_finish": "windowed"}}).register.mode == "anchored"
    with pytest.raises(ConfigError):
        parse({**base, "register": {"mode": "fast"}})
    with pytest.raises(ConfigError):
        parse({**base, "register": {"anchored_finish": "global"}})


def _ctx(root):
    from indoorgs.project import Project
    from indoorgs.stages.base import StageContext
    pj = Project(root)
    sec, paths = pj.sections()[0]
    logs = []
    return StageContext(pj, sec, paths, logs.append), logs


def _priors(ctx, n=30):
    names = sorted(f.name for f in ctx.paths.frames.glob("f_*.jpg"))
    ctx.paths.colmap.mkdir(parents=True, exist_ok=True)
    fr = [{"image": nm, "t": 0.1 * k, "xyz": [0.1 * k, 0.0, 1.4], "sigma": [0.5, 0.5, 0.3]} for k, nm in enumerate(names)]
    ctx.paths.priors.write_text(json.dumps({"version": 1, "coordinate_system": "CARTESIAN", "frames": fr}),
                                encoding="utf-8")
    return names


def _fake_graph(names):
    n = len(names)
    R = np.array([A.yaw_pitch_to_R(0.0) for _ in range(n)])
    C = np.stack([0.1 * np.arange(n), np.zeros(n), np.full(n, 1.4)], 1)
    dbd = A.DbData({i + 1: nm for i, nm in enumerate(names)}, (1, 4, 64, 48, np.array([50.0, 50, 32, 24, 0, 0, 0, 0])),
                   {i + 1: 1 for i in range(n)}, {})
    head = A.route_headings(C, 0.1 * np.arange(n))
    return {"order": names, "R": R, "C": C, "use": np.ones(n, bool), "focal": 55.0, "K": np.eye(3), "dist": np.zeros(4),
            "heading": head, "rels": [], "dbd": dbd, "stats": {"frames": n, "used": n, "connected": n, "dropped_far": 0,
                                                               "focal": 55.0, "rotation": {}, "route_flagged": 0}}


@pytest.mark.parametrize("finish,expect", [
    ("prior_mapper", ["point_triangulator", "pose_prior_mapper"]),
    ("ba", ["point_triangulator", "bundle_adjuster", "point_triangulator", "bundle_adjuster", "point_triangulator",
            "image_registrator", "point_triangulator"]),
    ("none", ["point_triangulator", "image_registrator", "point_triangulator"]),
])
def test_register_anchored_flow(tmp_path, monkeypatch, finish, expect):
    root = make_project(tmp_path / "p", extra_yaml=f"register: {{mode: anchored, anchored_finish: {finish}}}\n")
    ctx, logs = _ctx(root)
    names = _priors(ctx)
    h = colmap_handlers()

    def copy_model(cmd):
        out = arg(cmd, "--output_path")
        cm.write_model(make_model(30), out)
    for k in ("point_triangulator", "bundle_adjuster", "image_registrator"):
        h[k] = copy_model
    h["pose_prior_mapper"] = copy_model                         # --input_path 이면 output_path 에 바로 씀
    fake = FakeRunner(h)
    monkeypatch.setattr(runner, "run", fake)
    monkeypatch.setattr(A, "solve_graph", lambda db, pri, nm, prm=None: _fake_graph(names))
    wrote = []
    monkeypatch.setattr("indoorgs.colmap.database.write_pose_priors",
                        lambda db, pri: wrote.append((str(db), pri["frames"][0]["sigma"])) or (len(names), []))
    monkeypatch.setattr("indoorgs.colmap.pair_verify.verify_pairs",
                        lambda db, ref, **kw: {"pairs": 0, "dropped": 0, "drop_rot": 0, "drop_no_pose": 0, "kw": kw})
    from indoorgs.stages.register import Register
    r = Register().run(ctx)
    assert r.ok, (r.message, logs)
    assert fake.names() == ["feature_extractor", "sequential_matcher", "matches_importer", *expect, "image_undistorter"]
    seq = fake.calls[1]
    assert arg(seq, "--SequentialMatching.quadratic_overlap") == "0" and arg(seq, "--SequentialMatching.loop_detection") == "0"
    imp = fake.calls[2]
    assert arg(imp, "--match_type") == "pairs"
    col = ctx.paths.colmap
    assert arg(imp, "--match_list_path") == str(col / "anchored_pairs.txt")
    tri = fake.calls[3]
    assert arg(tri, "--database_path") == str(col / "db_anchored.db")
    assert arg(tri, "--input_path") == str(col / "anchored" / "init")
    for c in fake.calls:
        if c[1] == "image_registrator":
            assert "--image_path" not in c                              # 4.2.1 image_registrator 에는 없음
        if c[1] == "bundle_adjuster":
            assert arg(c, "--BundleAdjustmentCeres.max_num_iterations") == "100"
        assert "--Mapper.ba_use_gpu" not in c and "--BundleAdjustmentCeres.use_gpu" not in c
    info = json.loads((col / "register.json").read_text(encoding="utf-8"))
    assert info["mode"] == "anchored" and info["anchored_finish"] == finish and info["aligned"]
    assert info["model"] == "sparse_aligned/0"
    assert info["mapper"] == ("pose_prior_mapper" if finish == "prior_mapper" else "anchored")
    assert (col / "anchored" / "anchored.json").exists()
    assert wrote[0] == (str(col / "db.db"), [0.5, 0.5, 0.3])
    if finish == "prior_mapper":                                   # 사본에는 느슨한 제약 (sigma x 4)
        assert wrote[1] == (str(col / "db_anchored.db"), [2.0, 2.0, 1.2])
    else:
        assert len(wrote) == 1
    assert (ctx.paths.data_model / "images.bin").exists()
    # 짝 후보 파일: 앞뒤 seq_overlap(20) 장 넘는 짝만
    lines = (col / "anchored_pairs.txt").read_text(encoding="utf-8").split("\n")
    for ln in filter(None, lines):
        a, b = ln.split()
        assert names.index(b) - names.index(a) > 20


def test_register_anchored_gpu_flags_and_settings_change(tmp_path, monkeypatch):
    root = make_project(tmp_path / "p", extra_yaml="register: {mode: anchored, ba_use_gpu: true, anchored_finish: ba,"
                                                     " anchored_ba_rounds: 1}\n")
    ctx, logs = _ctx(root)
    names = _priors(ctx)
    h = colmap_handlers()
    for k in ("point_triangulator", "bundle_adjuster", "image_registrator"):
        h[k] = lambda cmd: cm.write_model(make_model(30), arg(cmd, "--output_path"))
    fake = FakeRunner(h)
    monkeypatch.setattr(runner, "run", fake)
    monkeypatch.setattr(A, "solve_graph", lambda db, pri, nm, prm=None: _fake_graph(names))
    monkeypatch.setattr("indoorgs.colmap.database.write_pose_priors", lambda db, pri: (len(names), []))
    monkeypatch.setattr("indoorgs.colmap.pair_verify.verify_pairs",
                        lambda db, ref, **kw: {"pairs": 0, "dropped": 0, "drop_rot": 0, "drop_no_pose": 0})
    from indoorgs.stages.register import Register
    assert Register().run(ctx).ok
    tris = [c for c in fake.calls if c[1] == "point_triangulator"]
    bas = [c for c in fake.calls if c[1] == "bundle_adjuster"]
    assert all(arg(c, "--Mapper.ba_use_gpu") == "1" for c in tris)
    assert all(arg(c, "--BundleAdjustmentCeres.use_gpu") == "1" for c in bas)
    # 마무리 설정만 바꾸면 매칭은 그대로, 자세 그래프부터 다시 (끝난 data 는 run_all 이 옮기는 것 흉내)
    import shutil
    shutil.move(str(ctx.paths.data), str(ctx.paths.data.parent / "data_prev"))
    (root / "project.yaml").write_text((root / "project.yaml").read_text(encoding="utf-8").replace(
        "anchored_finish: ba", "anchored_finish: none"), encoding="utf-8")
    ctx, logs = _ctx(root)
    fake.calls.clear()
    assert Register().run(ctx).ok
    assert fake.names() == ["point_triangulator", "image_registrator", "point_triangulator", "image_undistorter"]
    assert any("설정이 바뀜" in m for m in logs)


def test_register_anchored_without_priors_falls_back(tmp_path, monkeypatch):
    root = make_project(tmp_path / "p", extra_yaml="register: {mode: anchored}\n")
    ctx, logs = _ctx(root)
    fake = FakeRunner(colmap_handlers())
    monkeypatch.setattr(runner, "run", fake)
    from indoorgs.stages.register import Register
    assert Register().run(ctx).ok
    assert "matches_importer" not in fake.names() and "mapper" in fake.names()
    assert any("anchored" in m for m in logs)


def test_solve_ls_without_scipy_matches(monkeypatch):
    import builtins
    rng = np.random.default_rng(0)
    rows, cols, vals, rhs = [], [], [], []
    for m in range(120):
        for c in rng.choice(30, int(rng.integers(1, 4)), replace=False):
            rows.append(m)
            cols.append(int(c))
            vals.append(float(rng.normal()))
        rhs.append(float(rng.normal()))
    x1 = A._solve_ls(rows, cols, vals, rhs, 30)
    real = builtins.__import__

    def no_scipy(name, *a, **k):
        if name.startswith("scipy"):
            raise ImportError(name)
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_scipy)
    x2 = A._solve_ls(rows, cols, vals, rhs, 30)
    assert np.allclose(x1, x2, atol=1e-8)
