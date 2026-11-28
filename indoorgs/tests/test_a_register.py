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
    # 4.2.1 기본 quadratic_overlap=1 은 2^k 장 떨어진 짝만 만듦 -> 항상 0
    assert arg(fake.calls[1], "--SequentialMatching.quadratic_overlap") == "0"
    fe = fake.calls[0]
    assert arg(fe, "--ImageReader.single_camera") == "1" and arg(fe, "--ImageReader.camera_model") == "OPENCV"
    # 가장 큰 모델(0) 을 썼는지
    assert arg(fake.calls[3], "--input_path").endswith("0")
    assert cm.count_images(ctx.paths.data_model) == 30
    assert json.loads((ctx.paths.colmap / "register.json").read_text())["mapper"] == "mapper"
    assert Register().is_done(ctx)


def test_register_prior_fallback(tmp_path, monkeypatch):
    root = make_project(tmp_path / "p", extra_yaml="register: {matcher: exhaustive, pair_verify: false}\n")
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
    root = make_project(tmp_path / "p", extra_yaml="register: {pair_verify: false}\n")
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
    assert arg(fake.calls[2], "--SequentialMatching.quadratic_overlap") == "0"
    assert "--SequentialMatching.vocab_tree_path" not in fake.calls[1]
    assert any("loop_detection 0" in m for m in logs)


def test_register_vocab_tree_path(tmp_path, monkeypatch):
    root = make_project(tmp_path / "p", extra_yaml="register: {vocab_tree: C:/tools/vocab.bin}\n")
    ctx, _ = ctx_of(root)
    fake = FakeRunner(colmap_handlers())
    monkeypatch.setattr(runner, "run", fake)
    assert Register().run(ctx).ok
    assert arg(fake.calls[1], "--SequentialMatching.vocab_tree_path") == "C:/tools/vocab.bin"


# ---------------------------------------------------------------- 두 번 정합 + 짝 검증 (pair_verify)
import numpy as np  # noqa: E402

from indoorgs.colmap import pair_verify  # noqa: E402
from indoorgs.stages.diagnose import paths_in_plan_coords  # noqa: E402

MATCHERS = {"sequential_matcher", "exhaustive_matcher", "spatial_matcher"}


def _write_priors(ctx, n=30):
    names = sorted(f.name for f in ctx.paths.frames.glob("*.jpg"))[:n]
    ctx.paths.colmap.mkdir(parents=True, exist_ok=True)
    ctx.paths.priors.write_text(json.dumps({"version": 1, "coordinate_system": "CARTESIAN", "frames": [
        {"image": nm, "xyz": [0.1 * i, 0, 1.4], "sigma": [0.5, 0.5, 0.3]} for i, nm in enumerate(names)]}))
    return names


def _patch_verify(monkeypatch, names, fake, n_ref=None):
    """load_reference / verify_pairs / align 를 가짜로: 부른 시점(명령 몇 개 뒤)과 DB 경로를 기록."""
    seen = {}

    def load_ref(sparse_dir, priors, max_far_m=1.5, min_model=10, level=True, **kw):
        seen["ref_kw"] = kw
        seen["ref_dir"] = str(sparse_dir)
        poses = {nm: (np.eye(3), np.zeros(3)) for nm in names[:len(names) if n_ref is None else n_ref]}
        return pair_verify.Reference(poses, np.eye(3), np.zeros(4), {"models": [30], "poses": len(poses),
                                     "far_dropped": 0, "unfit_models": 0, "leveled_deg": [1.0]})

    def verify(db_path, ref, **kw):
        seen["verify_db"] = str(db_path)
        seen["verify_at"] = len(fake.calls)
        seen["verify_kw"] = kw
        return {"pairs": 100, "kept_sequential": 50, "kept": 40, "drop_rot": 6, "drop_dir": 3, "drop_unknown": 1,
                "drop_no_pose": 0, "dropped": 10, "ref_poses": len(ref.poses)}

    def align(model, priors, level=True):
        seen["align_level"] = level
        return model, {"scale": 1.0, "yaw_deg": 0.0, "resid_median_m": 0.1, "n_priors": 30, "leveled_deg": 2.0,
                       "level": level, "far_1m": 0, "transform": {}}
    monkeypatch.setattr(pair_verify, "load_reference", load_ref)
    monkeypatch.setattr(pair_verify, "verify_pairs", verify)
    monkeypatch.setattr(pair_verify, "align_model_to_priors", align)
    return seen


def test_register_pair_verify_flow(tmp_path, monkeypatch):
    root = make_project(tmp_path / "p")
    ctx, logs = ctx_of(root)
    names = _write_priors(ctx)
    fake = FakeRunner(colmap_handlers())
    monkeypatch.setattr(runner, "run", fake)
    seen = _patch_verify(monkeypatch, names, fake)
    res = Register().run(ctx)
    assert res.ok, res.message
    assert fake.names() == ["feature_extractor", "sequential_matcher", "pose_prior_mapper", "spatial_matcher",
                            "pose_prior_mapper", "image_undistorter"]
    col = ctx.paths.colmap
    seq, p1, spa, p2, und = fake.calls[1:]
    assert arg(seq, "--SequentialMatching.quadratic_overlap") == "0"
    assert arg(seq, "--SequentialMatching.loop_detection") == "0"
    assert arg(seq, "--SequentialMatching.overlap") == "20"
    # 1차: db.db -> sparse_pass1
    assert arg(p1, "--database_path").endswith("db.db") and arg(p1, "--output_path").endswith("sparse_pass1")
    assert arg(p1, "--overwrite_priors_covariance") == "0"
    # 후보 짝: 같은 db.db 에 spatial 12 m, 이웃 300, ignore_z
    assert arg(spa, "--database_path").endswith("db.db")
    assert float(arg(spa, "--SpatialMatching.max_distance")) == 12.0
    assert arg(spa, "--SpatialMatching.max_num_neighbors") == "300"
    assert arg(spa, "--SpatialMatching.ignore_z") == "1"
    # 검증: 후보 추가 뒤, 사본에서. 사본 뒤로는 매처를 안 돌림
    assert seen["ref_dir"].endswith("sparse_pass1")
    assert seen["verify_db"].endswith("db_verified.db") and (col / "db_verified.db").exists()
    assert seen["verify_at"] == 4
    assert not any(fake.key(c) in MATCHERS for c in fake.calls[seen["verify_at"]:])
    assert seen["verify_kw"] == {"seq_keep": 10, "rot_tol_deg": 30.0, "dir_tol_deg": 45.0}
    # 2차: db_verified.db -> sparse
    assert arg(p2, "--database_path").endswith("db_verified.db")
    assert arg(p2, "--output_path") == str(ctx.paths.sparse)
    # 직접 맞춘 모델을 undistort
    assert arg(und, "--input_path").replace("\\", "/").endswith("sparse_aligned/0")
    info = json.loads((col / "register.json").read_text(encoding="utf-8"))
    assert info["mapper"] == "pose_prior_mapper" and info["aligned"] is True and info["mode"] == "pair_verify"
    assert info["pair_verify"]["dropped"] == 10 and info["pair_verify"]["reference"]["poses"] == 30
    assert info["model"] == "sparse_aligned/0"
    assert seen["align_level"] is True
    for f in ("matching.done", "pass1.done", "candidates.done", "verify.done", "mapper.done"):
        assert (col / f).exists(), f
    assert paths_in_plan_coords(ctx.paths)
    assert "짝 검증 10개 삭제" in res.message


def test_register_pair_verify_exhaustive_candidates(tmp_path, monkeypatch):
    root = make_project(tmp_path / "p", extra_yaml="register: {matcher: exhaustive, gimbal_level: false}\n")
    ctx, _ = ctx_of(root)
    names = _write_priors(ctx)
    fake = FakeRunner(colmap_handlers())
    monkeypatch.setattr(runner, "run", fake)
    seen = _patch_verify(monkeypatch, names, fake)
    assert Register().run(ctx).ok
    # 1차 매칭은 matcher 설정과 상관없이 sequential, 후보 추가가 exhaustive
    assert fake.names()[:4] == ["feature_extractor", "sequential_matcher", "pose_prior_mapper", "exhaustive_matcher"]
    assert seen["align_level"] is False


def test_register_pair_verify_pass2_falls_back_to_mapper(tmp_path, monkeypatch):
    root = make_project(tmp_path / "p")
    ctx, logs = ctx_of(root)
    names = _write_priors(ctx)
    h = colmap_handlers()
    ok_mapper = h["pose_prior_mapper"]
    h["pose_prior_mapper"] = lambda cmd: 1 if arg(cmd, "--database_path").endswith("db_verified.db") else ok_mapper(cmd)
    fake = FakeRunner(h)
    monkeypatch.setattr(runner, "run", fake)
    _patch_verify(monkeypatch, names, fake)
    assert Register().run(ctx).ok
    assert fake.names()[4:] == ["pose_prior_mapper", "mapper", "image_undistorter"]
    assert arg(fake.calls[5], "--database_path").endswith("db_verified.db")
    info = json.loads((ctx.paths.colmap / "register.json").read_text(encoding="utf-8"))
    assert info["mapper"] == "mapper" and info["aligned"] is True
    assert paths_in_plan_coords(ctx.paths)          # mapper 라도 직접 맞췄으면 도면 좌표
    assert any(p.name.startswith("sparse_prior_failed_") for p in ctx.paths.colmap.iterdir())


def test_register_pair_verify_small_reference_uses_pass1(tmp_path, monkeypatch):
    root = make_project(tmp_path / "p")
    ctx, logs = ctx_of(root)
    names = _write_priors(ctx)
    fake = FakeRunner(colmap_handlers())
    monkeypatch.setattr(runner, "run", fake)
    seen = _patch_verify(monkeypatch, names, fake, n_ref=10)          # 30장 중 10장 < 절반
    assert Register().run(ctx).ok
    assert fake.names() == ["feature_extractor", "sequential_matcher", "pose_prior_mapper", "spatial_matcher",
                            "image_undistorter"]
    assert "verify_db" not in seen and not (ctx.paths.colmap / "db_verified.db").exists()
    info = json.loads((ctx.paths.colmap / "register.json").read_text(encoding="utf-8"))
    assert info["pair_verify"]["skipped"] and info["aligned"] is True
    assert any("절반 미만" in m for m in logs)


def test_register_pair_verify_resume_after_candidate_failure(tmp_path, monkeypatch):
    root = make_project(tmp_path / "p")
    ctx, _ = ctx_of(root)
    names = _write_priors(ctx)
    h = colmap_handlers()
    h["spatial_matcher"] = lambda cmd: 1
    fake = FakeRunner(h)
    monkeypatch.setattr(runner, "run", fake)
    _patch_verify(monkeypatch, names, fake)
    res = Register().run(ctx)
    assert not res.ok and "후보 짝" in res.message
    col = ctx.paths.colmap
    assert (col / "pass1.done").exists() and (col / "sparse_pass1").is_dir() and ctx.paths.db.exists()
    # 다시: 특징·1차는 건너뛰고 후보 추가부터
    fake2 = FakeRunner(colmap_handlers())
    monkeypatch.setattr(runner, "run", fake2)
    _patch_verify(monkeypatch, names, fake2)
    assert Register().run(ctx).ok
    assert fake2.names() == ["spatial_matcher", "pose_prior_mapper", "image_undistorter"]


def test_register_pair_verify_pass1_failure_plain_mapper(tmp_path, monkeypatch):
    root = make_project(tmp_path / "p")
    ctx, logs = ctx_of(root)
    names = _write_priors(ctx)
    h = colmap_handlers()
    h["pose_prior_mapper"] = lambda cmd: 1
    fake = FakeRunner(h)
    monkeypatch.setattr(runner, "run", fake)
    _patch_verify(monkeypatch, names, fake)
    assert Register().run(ctx).ok
    assert fake.names() == ["feature_extractor", "sequential_matcher", "pose_prior_mapper", "mapper",
                            "image_undistorter"]
    info = json.loads((ctx.paths.colmap / "register.json").read_text(encoding="utf-8"))
    assert info["mapper"] == "mapper" and info["aligned"] is False
    assert not paths_in_plan_coords(ctx.paths)


def test_register_mode_change_rematches(tmp_path, monkeypatch):
    """예전 방식(plain)으로 매칭만 끝난 DB 는 pair_verify 로 바뀌면 옆으로 치우고 다시 매칭."""
    root = make_project(tmp_path / "p")
    ctx, logs = ctx_of(root)
    h = colmap_handlers()
    h["mapper"] = lambda cmd: 1
    fake = FakeRunner(h)
    monkeypatch.setattr(runner, "run", fake)
    assert not Register().run(ctx).ok                       # plain 매칭 끝, 정합 실패
    assert (ctx.paths.colmap / "matching.done").read_text() == "plain"
    names = _write_priors(ctx)
    fake2 = FakeRunner(colmap_handlers())
    monkeypatch.setattr(runner, "run", fake2)
    _patch_verify(monkeypatch, names, fake2)
    assert Register().run(ctx).ok
    assert fake2.names()[:2] == ["feature_extractor", "sequential_matcher"]
    assert any(p.name.startswith("db_failed_") for p in ctx.paths.colmap.iterdir())
    assert any("매칭 방식이 바뀜" in m for m in logs)


def test_register_pair_verify_priors_changed_restarts(tmp_path, monkeypatch):
    """1차 정합 뒤 (후보 짝에서 실패) 동선을 고쳐 priors.json 이 바뀌면 낡은 1차 자세·DB 를 이어 쓰지 않음."""
    root = make_project(tmp_path / "p")
    ctx, logs = ctx_of(root)
    names = _write_priors(ctx)
    h = colmap_handlers()
    h["spatial_matcher"] = lambda cmd: 1
    fake = FakeRunner(h)
    monkeypatch.setattr(runner, "run", fake)
    _patch_verify(monkeypatch, names, fake)
    assert not Register().run(ctx).ok
    col = ctx.paths.colmap
    assert (col / "pass1.done").read_text(encoding="utf-8").strip()
    d = json.loads(ctx.paths.priors.read_text(encoding="utf-8"))
    for f in d["frames"]:
        f["xyz"][1] += 2.0
    ctx.paths.priors.write_text(json.dumps(d), encoding="utf-8")
    fake2 = FakeRunner(colmap_handlers())
    monkeypatch.setattr(runner, "run", fake2)
    _patch_verify(monkeypatch, names, fake2)
    assert Register().run(ctx).ok
    assert fake2.names() == ["feature_extractor", "sequential_matcher", "pose_prior_mapper", "spatial_matcher",
                            "pose_prior_mapper", "image_undistorter"]
    assert any("priors.json" in m and "바뀜" in m for m in logs)
    # 낡은 중간 결과는 지우지 않고 옆으로
    assert any(x.name.startswith("db_failed_") for x in col.iterdir())
    assert any(x.name.startswith("sparse_pass1_failed_") for x in col.iterdir())


def test_register_rerun_does_not_resume_unfinished(tmp_path, monkeypatch):
    """--rerun (예: extract 부터 다시)인데 register 가 끝나지 않은 상태였으면 낡은 DB·1차 결과를 이어 쓰지 않음."""
    root = make_project(tmp_path / "p")
    ctx, logs = ctx_of(root)
    names = _write_priors(ctx)
    h = colmap_handlers()
    h["spatial_matcher"] = lambda cmd: 1
    fake = FakeRunner(h)
    monkeypatch.setattr(runner, "run", fake)
    _patch_verify(monkeypatch, names, fake)
    assert not Register().run(ctx).ok
    ctx.rerun = True
    fake2 = FakeRunner(colmap_handlers())
    monkeypatch.setattr(runner, "run", fake2)
    _patch_verify(monkeypatch, names, fake2)
    assert Register().run(ctx).ok
    assert fake2.names()[:3] == ["feature_extractor", "sequential_matcher", "pose_prior_mapper"]
    assert any(x.name.startswith("db_failed_") for x in ctx.paths.colmap.iterdir())


def test_register_pair_verify_mapper_fallback_without_level_not_aligned(tmp_path, monkeypatch):
    """2차가 일반 mapper 로 바뀌었고 gimbal_level 이 꺼져 있으면 위 방향을 모르므로 도면 좌표라고 하지 않음."""
    root = make_project(tmp_path / "p", extra_yaml="register: {gimbal_level: false}\n")
    ctx, logs = ctx_of(root)
    names = _write_priors(ctx)
    h = colmap_handlers()
    ok_mapper = h["pose_prior_mapper"]
    h["pose_prior_mapper"] = lambda cmd: 1 if arg(cmd, "--database_path").endswith("db_verified.db") else ok_mapper(cmd)
    fake = FakeRunner(h)
    monkeypatch.setattr(runner, "run", fake)
    seen = _patch_verify(monkeypatch, names, fake)
    assert Register().run(ctx).ok
    info = json.loads((ctx.paths.colmap / "register.json").read_text(encoding="utf-8"))
    assert info["mapper"] == "mapper" and info["aligned"] is False and "align_level" not in seen
    assert not paths_in_plan_coords(ctx.paths)
    assert arg(fake.calls[-1], "--input_path").replace("\\", "/").endswith("colmap/sparse/0")


def test_register_pair_verify_settings_changed_reverifies(tmp_path, monkeypatch):
    """2차 정합 전에 멈춘 뒤 짝 검증 설정을 바꾸면 옛 verify.done·db_verified.db 를 이어 쓰지 않음."""
    root = make_project(tmp_path / "p")
    ctx, logs = ctx_of(root)
    names = _write_priors(ctx)
    h = colmap_handlers()
    ok_prior = h["pose_prior_mapper"]
    calls = {"n": 0}

    def prior(cmd):
        calls["n"] += 1
        return ok_prior(cmd) if calls["n"] == 1 else 1
    h["pose_prior_mapper"] = prior
    h["mapper"] = lambda cmd: 1
    fake = FakeRunner(h)
    monkeypatch.setattr(runner, "run", fake)
    _patch_verify(monkeypatch, names, fake)
    assert not Register().run(ctx).ok
    col = ctx.paths.colmap
    assert (col / "verify.done").exists() and (col / "db_verified.db").exists()
    ctx.project.cfg.register.verify_rot_tol_deg = 20.0
    fake2 = FakeRunner(colmap_handlers())
    monkeypatch.setattr(runner, "run", fake2)
    seen = _patch_verify(monkeypatch, names, fake2)
    assert Register().run(ctx).ok
    assert seen["verify_kw"]["rot_tol_deg"] == 20.0
    assert fake2.names() == ["pose_prior_mapper", "image_undistorter"]
    assert any("짝 검증 설정" in m for m in logs)
    assert any(x.name.startswith("db_verified_failed_") for x in col.iterdir())
    st = json.loads((col / "verify.done").read_text(encoding="utf-8"))
    assert st["settings"]["rot_tol_deg"] == 20.0


# ---------------------------------------------------------------- 속도: pass1_stride, pass2_mapper (e2e_v4/SPEED.md)
from indoorgs.stages import register as reg_mod  # noqa: E402


def _global_handlers():
    h = colmap_handlers()
    h["view_graph_calibrator"] = lambda cmd: None
    h["global_mapper"] = h["mapper"]
    return h


def test_pass1_image_list():
    names = [f"f_{i:05d}.jpg" for i in range(1, 11)]
    assert reg_mod.pass1_image_list(names, 1) == names
    assert reg_mod.pass1_image_list(list(reversed(names)), 3) == [names[i] for i in (0, 3, 6, 9)]
    assert reg_mod.pass1_image_list(names, 4) == [names[i] for i in (0, 4, 8, 9)]     # 마지막 사진은 항상
    assert reg_mod.pass1_image_list([], 4) == []


def test_register_pass1_stride_image_list_and_interpolation(tmp_path, monkeypatch):
    root = make_project(tmp_path / "p", extra_yaml="register: {pass1_stride: 3}\n")
    ctx, logs = ctx_of(root)
    names = _write_priors(ctx)
    fake = FakeRunner(colmap_handlers())
    monkeypatch.setattr(runner, "run", fake)
    seen = _patch_verify(monkeypatch, names, fake)
    assert Register().run(ctx).ok
    assert fake.names() == ["feature_extractor", "sequential_matcher", "pose_prior_mapper", "spatial_matcher",
                            "pose_prior_mapper", "image_undistorter"]
    col = ctx.paths.colmap
    p1, p2 = fake.calls[2], fake.calls[4]
    lst = arg(p1, "--Mapper.image_list_path")
    assert lst == str(col / "pass1_images.txt")
    assert (col / "pass1_images.txt").read_text(encoding="utf-8").split() == names[::3] + [names[-1]]
    assert "--Mapper.image_list_path" not in p2                      # 2차는 전부
    assert arg(p1, "--database_path").endswith("db.db")
    assert seen["ref_kw"] == {"order": names, "interp_max_gap": 6}
    assert (col / "pass1.done").read_text(encoding="utf-8").endswith("|stride=3")
    info = json.loads((col / "register.json").read_text(encoding="utf-8"))
    assert info["pass1_stride"] == 3 and info["pass2_mapper"] == "incremental"
    assert json.loads((col / "verify.done").read_text(encoding="utf-8"))["settings"]["pass1_stride"] == 3


def test_register_pass1_stride_1_keeps_old_markers(tmp_path, monkeypatch):
    """기본(1)은 예전과 같음: 사진 목록 옵션 없음, 보간 없음, pass1.done·verify 설정 지문도 예전 그대로."""
    root = make_project(tmp_path / "p")
    ctx, _ = ctx_of(root)
    names = _write_priors(ctx)
    fake = FakeRunner(colmap_handlers())
    monkeypatch.setattr(runner, "run", fake)
    seen = _patch_verify(monkeypatch, names, fake)
    assert Register().run(ctx).ok
    col = ctx.paths.colmap
    assert "--Mapper.image_list_path" not in fake.calls[2] and not (col / "pass1_images.txt").exists()
    assert seen["ref_kw"] == {}
    assert (col / "pass1.done").read_text(encoding="utf-8") == reg_mod._priors_tag(ctx.paths.priors)
    assert "pass1_stride" not in json.loads((col / "verify.done").read_text(encoding="utf-8"))["settings"]


def test_register_pass1_stride_changed_before_candidates_redoes_pass1_only(tmp_path, monkeypatch):
    root = make_project(tmp_path / "p")
    ctx, logs = ctx_of(root)
    names = _write_priors(ctx)
    h = colmap_handlers()
    h["spatial_matcher"] = lambda cmd: 1
    fake = FakeRunner(h)
    monkeypatch.setattr(runner, "run", fake)
    _patch_verify(monkeypatch, names, fake)
    assert not Register().run(ctx).ok
    ctx.project.cfg.register.pass1_stride = 4
    fake2 = FakeRunner(colmap_handlers())
    monkeypatch.setattr(runner, "run", fake2)
    _patch_verify(monkeypatch, names, fake2)
    assert Register().run(ctx).ok
    assert fake2.names() == ["pose_prior_mapper", "spatial_matcher", "pose_prior_mapper", "image_undistorter"]
    assert "--Mapper.image_list_path" in fake2.calls[0]
    assert any("1차부터 다시" in m for m in logs)
    assert any(x.name.startswith("sparse_pass1_failed_") for x in ctx.paths.colmap.iterdir())   # 옛 1차는 옆으로


def test_register_pass1_stride_changed_after_candidates_rematches(tmp_path, monkeypatch):
    """후보 짝이 이미 db.db 에 들어갔으면 1차를 다시 돌릴 수 없음 (가짜 짝이 1차에 섞임) -> 매칭부터."""
    root = make_project(tmp_path / "p")
    ctx, logs = ctx_of(root)
    names = _write_priors(ctx)
    h = colmap_handlers()
    ok_prior = h["pose_prior_mapper"]
    h["pose_prior_mapper"] = lambda cmd: 1 if arg(cmd, "--database_path").endswith("db_verified.db") else ok_prior(cmd)
    h["mapper"] = lambda cmd: 1
    fake = FakeRunner(h)
    monkeypatch.setattr(runner, "run", fake)
    _patch_verify(monkeypatch, names, fake)
    assert not Register().run(ctx).ok
    assert (ctx.paths.colmap / "candidates.done").exists()
    ctx.project.cfg.register.pass1_stride = 4
    fake2 = FakeRunner(colmap_handlers())
    monkeypatch.setattr(runner, "run", fake2)
    _patch_verify(monkeypatch, names, fake2)
    assert Register().run(ctx).ok
    assert fake2.names()[0] == "feature_extractor"
    assert any("매칭부터 다시" in m for m in logs)
    assert any(x.name.startswith("db_failed_") for x in ctx.paths.colmap.iterdir())             # 옛 db.db 는 옆으로


def test_register_pass2_global(tmp_path, monkeypatch):
    root = make_project(tmp_path / "p", extra_yaml="register: {pass2_mapper: global}\n")
    ctx, _ = ctx_of(root)
    names = _write_priors(ctx)
    fake = FakeRunner(_global_handlers())
    monkeypatch.setattr(runner, "run", fake)
    seen = _patch_verify(monkeypatch, names, fake)
    assert Register().run(ctx).ok
    assert fake.names() == ["feature_extractor", "sequential_matcher", "pose_prior_mapper", "spatial_matcher",
                            "view_graph_calibrator", "global_mapper", "image_undistorter"]
    col = ctx.paths.colmap
    vgc, glo = fake.calls[4], fake.calls[5]
    # 검증한 사본의 사본(db_global.db)에서. db_verified.db 는 그대로 남음
    assert arg(vgc, "--database_path").endswith("db_global.db") and arg(glo, "--database_path").endswith("db_global.db")
    assert arg(glo, "--output_path") == str(ctx.paths.sparse) and arg(glo, "--image_path") == str(ctx.paths.frames)
    assert (col / "db_verified.db").exists() and (col / "db_global.db").exists()
    info = json.loads((col / "register.json").read_text(encoding="utf-8"))
    assert info["mapper"] == "global_mapper" and info["aligned"] is True and info["pass2_mapper"] == "global"
    assert info["model"] == "sparse_aligned/0" and seen["align_level"] is True
    assert paths_in_plan_coords(ctx.paths)
    assert arg(fake.calls[-1], "--input_path").replace("\\", "/").endswith("sparse_aligned/0")


def test_register_pass2_global_without_level_not_plan(tmp_path, monkeypatch):
    root = make_project(tmp_path / "p", extra_yaml="register: {pass2_mapper: global, gimbal_level: false}\n")
    ctx, logs = ctx_of(root)
    names = _write_priors(ctx)
    fake = FakeRunner(_global_handlers())
    monkeypatch.setattr(runner, "run", fake)
    seen = _patch_verify(monkeypatch, names, fake)
    assert Register().run(ctx).ok
    info = json.loads((ctx.paths.colmap / "register.json").read_text(encoding="utf-8"))
    assert info["mapper"] == "global_mapper" and info["aligned"] is False and "align_level" not in seen
    assert not paths_in_plan_coords(ctx.paths)
    assert any("global_mapper 결과인데 gimbal_level" in m for m in logs)


def test_register_pass2_global_failure_falls_back_to_prior_mapper(tmp_path, monkeypatch):
    root = make_project(tmp_path / "p", extra_yaml="register: {pass2_mapper: global}\n")
    ctx, logs = ctx_of(root)
    names = _write_priors(ctx)
    h = _global_handlers()
    h["global_mapper"] = lambda cmd: 1
    fake = FakeRunner(h)
    monkeypatch.setattr(runner, "run", fake)
    _patch_verify(monkeypatch, names, fake)
    assert Register().run(ctx).ok
    assert fake.names()[4:] == ["view_graph_calibrator", "global_mapper", "pose_prior_mapper", "image_undistorter"]
    assert arg(fake.calls[6], "--database_path").endswith("db_verified.db")
    info = json.loads((ctx.paths.colmap / "register.json").read_text(encoding="utf-8"))
    assert info["mapper"] == "pose_prior_mapper" and info["aligned"] is True
    assert any(p.name.startswith("sparse_global_failed_") for p in ctx.paths.colmap.iterdir())


def test_register_pass2_mapper_change_reruns_pass2_only(tmp_path, monkeypatch):
    root = make_project(tmp_path / "p")
    ctx, logs = ctx_of(root)
    names = _write_priors(ctx)
    fake = FakeRunner(colmap_handlers())
    monkeypatch.setattr(runner, "run", fake)
    _patch_verify(monkeypatch, names, fake)
    assert Register().run(ctx).ok
    ctx.project.cfg.register.pass2_mapper = "global"
    fake2 = FakeRunner(_global_handlers())
    monkeypatch.setattr(runner, "run", fake2)
    _patch_verify(monkeypatch, names, fake2)
    assert Register().run(ctx).ok
    assert fake2.names() == ["view_graph_calibrator", "global_mapper", "image_undistorter"]
    assert any("2차 정합 방식이 바뀜" in m for m in logs)
    col = ctx.paths.colmap
    assert any(x.name.startswith("sparse_failed_") for x in col.iterdir())          # 옛 2차 결과는 옆으로 (삭제 안 함)
    assert json.loads((col / "register.json").read_text(encoding="utf-8"))["mapper"] == "global_mapper"
    # 같은 설정으로 또 돌리면 정합은 건너뜀
    fake3 = FakeRunner(_global_handlers())
    monkeypatch.setattr(runner, "run", fake3)
    _patch_verify(monkeypatch, names, fake3)
    assert Register().run(ctx).ok
    assert fake3.names() == ["image_undistorter"]


def test_config_pass1_stride_and_pass2_mapper_checked(tmp_path):
    from indoorgs.config import ConfigError, parse
    base = {"sections": [{"name": "a", "video": "v.mp4"}]}
    assert parse({**base, "register": {"pass1_stride": 4, "pass2_mapper": "global"}}).register.pass1_stride == 4
    for bad in ({"pass1_stride": 0}, {"pass1_stride": 2.5}, {"pass1_stride": True}, {"pass2_mapper": "glomap"}):
        with pytest.raises(ConfigError):
            parse({**base, "register": bad})
