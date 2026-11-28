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

    def load_ref(sparse_dir, priors, max_far_m=1.5, min_model=10, level=True):
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
