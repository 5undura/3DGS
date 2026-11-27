"""cleanup 단계 (snap 뒤·repair 앞): 순서, 도면 좌표 아닌 구간, PLY 고르기(설정·시각·입력 이름), repair 입력, 설정·보고서."""
import json

import numpy as np
import pytest

from indoorgs import config, plyio, schemas
from indoorgs.stages import base, fill, merge, snap as snap_stage, walls as walls_stage
from indoorgs.stages.base import StageContext
from test_h_walls_stages import TRUE_NORTH_Y, _touch, _wall_ply, setup

S = snap_stage


def _with_junk(ply, cam):
    """벽 스플랫 + 카메라 자리 부유물 3개 + 불투명 바늘 10개 (벽 위, 긴 축 x)."""
    n0 = ply.n
    out = plyio.make(n0 + 13, ply.names)
    out.data[:n0] = ply.data
    P = np.vstack([cam + [[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0]],
                   np.column_stack([np.linspace(1, 5, 10), np.full(10, TRUE_NORTH_Y), np.full(10, 1.2)])])
    for k, c in enumerate("xyz"):
        out.data[c][n0:] = P[:, k]
    out.data["rot_0"][n0:] = 1.0
    out.data["opacity"][n0:] = 3.0
    for k in range(3):
        out.data[f"scale_{k}"][n0:] = np.log(0.02)
    out.data["scale_0"][n0 + 3:] = np.log(0.3)          # 바늘: 0.3 / 0.02 = 15
    return out


def _cam(ctx):
    from indoorgs.colmap import model as cm
    m = cm.read_model(ctx.paths.data_clean / "sparse" / "0")
    return next(iter(m.images.values())).center


def test_stage_order():
    names = [s.name for s in base.all_stages()]
    assert names.index("train") < names.index("snap") < names.index("cleanup") < names.index("repair") < names.index("fill")


def test_cleanup_runs_without_plan_coords(tmp_path):
    p, ctx, logs = setup(tmp_path)
    (ctx.paths.colmap / "register.json").write_text(json.dumps({"version": 1, "mapper": "mapper"}))
    assert not S.Snap().applies(ctx) and not S.Repair().applies(ctx)
    st = S.Cleanup()
    assert st.applies(ctx)
    run = ctx.paths.train_run("brush", "default")
    src = _with_junk(_wall_ply(), _cam(ctx))
    plyio.write(run / "splat.ply", src)
    assert not st.is_done(ctx)
    res = st.run(ctx)
    assert res.ok, res.message
    info = schemas.read_json(run / S.CLEAN_JSON)
    assert info["input"] == "splat.ply" and info["plan_coords"] is False and info["n_planes"] == 0
    assert info["floaters"]["removed_by"]["camera"] == 3
    assert info["needles"]["n_needles"] == 10 and info["needles"]["n_pieces"] > 10
    assert info["quality_before"]["count"] == src.n and info["quality_after"]["count"] == plyio.read(run / S.CLEAN_PLY).n
    assert plyio.read(run / S.REMOVED_PLY).n == info["n_removed_total"] >= 3
    assert plyio.read(run / "splat.ply").n == src.n                   # 원본 그대로
    assert st.is_done(ctx)
    assert fill.train_ply(ctx).name == S.CLEAN_PLY
    assert merge.section_ply(p, ctx.section, ctx.paths).name == S.CLEAN_PLY
    assert any("도면 좌표가 아님" in m for m in logs)
    # 입력이 새로워지면 다시
    _touch(run / "splat.ply", (run / S.CLEAN_PLY).stat().st_mtime + 10)
    assert not st.is_done(ctx) and fill.train_ply(ctx).name == "splat.ply"
    # 보고서
    from indoorgs.stages import report
    html = report.build_html(p)
    assert "학습 후 정리" in html and "camera 3" in html and "도면 좌표 아님" in html


def test_cleanup_plan_coords_then_repair_uses_clean(tmp_path):
    p, ctx, logs = setup(tmp_path, extra="walls: {seed: false}\n")
    assert walls_stage.Walls().run(ctx).ok
    run = ctx.paths.train_run("brush", "default")
    g = np.mgrid[0.3:5.7:0.04, 0.2:2.4:0.04].reshape(2, -1).T
    g = g[np.hypot(g[:, 0] - 3.0, g[:, 1] - 1.3) > 0.3]
    ply = plyio.make(len(g), plyio.brush_names(2))
    ply.data["x"], ply.data["y"], ply.data["z"] = g[:, 0], TRUE_NORTH_Y, g[:, 1]
    ply.data["rot_0"] = 1.0
    ply.data["scale_0"] = ply.data["scale_2"] = np.log(0.03)
    ply.data["scale_1"] = np.log(0.003)
    ply.data["opacity"] = 4.0
    n_wall = ply.n
    plyio.write(run / "splat.ply", _with_junk(ply, _cam(ctx)))
    assert S.Snap().run(ctx).ok
    assert S.cleanup_input(ctx).name == S.SNAP_PLY
    st = S.Cleanup()
    res = st.run(ctx)
    assert res.ok, res.message
    info = schemas.read_json(run / S.CLEAN_JSON)
    assert info["input"] == S.SNAP_PLY and info["plan_coords"] is True and info["n_planes"] > 0
    assert info["floaters"]["removed_by"]["camera"] == 3
    out = plyio.read(run / S.CLEAN_PLY)
    on_wall = np.abs(out.data["y"] - TRUE_NORTH_Y) < 0.02
    assert on_wall.sum() >= n_wall                                    # 면 근처 벽 스플랫은 하나도 안 지움
    assert st.is_done(ctx)
    # repair 는 정리한 것을 입력으로
    assert S.repair_input(ctx).name == S.CLEAN_PLY
    rp = S.Repair()
    assert not rp.is_done(ctx)
    r = rp.run(ctx)
    assert r.ok, r.message
    assert schemas.read_json(run / S.REPAIR_JSON)["input"] == S.CLEAN_PLY
    assert rp.is_done(ctx)
    assert fill.train_ply(ctx).name == S.REPAIR_PLY
    # 정리를 다시 돌리면 (clean 이 repaired 보다 새것) repair 는 낡음 -> clean 을 고르고 repair 다시 필요
    _touch(run / S.CLEAN_PLY, (run / S.REPAIR_PLY).stat().st_mtime + 10)
    assert not rp.is_done(ctx) and fill.train_ply(ctx).name == S.CLEAN_PLY
    # snap 다시 -> clean·repaired 둘 다 낡음
    _touch(run / S.SNAP_PLY, (run / S.CLEAN_PLY).stat().st_mtime + 10)
    assert not st.is_done(ctx) and fill.train_ply(ctx).name == S.SNAP_PLY
    assert S.repair_input(ctx).name == S.SNAP_PLY


def test_chain_selection_mtime_config_and_input(tmp_path):
    p, ctx, _ = setup(tmp_path)
    run = ctx.paths.train_run("brush", "default")
    t = 1_700_000_000
    for i, nm in enumerate(("splat.ply", S.SNAP_PLY, S.CLEAN_PLY, S.REPAIR_PLY)):
        _touch(run / nm, t + 10 * i)
    pick = lambda: fill.train_ply(ctx).name                           # noqa: E731
    assert pick() == S.REPAIR_PLY
    assert merge.section_ply(p, ctx.section, ctx.paths).name == S.REPAIR_PLY
    assert S.best_ply(run, True, False, True).name == S.CLEAN_PLY     # repair 꺼짐
    assert S.best_ply(run, False, True, True).name == S.CLEAN_PLY     # snap 꺼짐, 입력 기록 없음 -> 시각만 봄
    # clean.json 에 입력이 splat_walls.ply 로 적혀 있으면 snap 을 끈 지금은 안 씀
    schemas.write_json(run / S.CLEAN_JSON, {"version": 1, "input": S.SNAP_PLY})
    assert S.best_ply(run, False, True, True).name == "splat.ply"
    assert S.best_ply(run, True, True, False).name == S.REPAIR_PLY    # 정리 꺼짐, repaired 는 walls 보다 새것 (기록 없음)
    schemas.write_json(run / S.REPAIR_JSON, {"version": 1, "input": S.CLEAN_PLY})
    assert S.best_ply(run, True, True, False).name == S.SNAP_PLY      # repaired 는 clean 에서 만든 것 -> 정리 끄면 안 씀
    assert pick() == S.REPAIR_PLY
    _touch(run / S.SNAP_PLY, t + 100)                                  # snap 다시 -> clean·repaired 낡음
    assert pick() == S.SNAP_PLY
    _touch(run / "splat.ply", t + 200)
    assert pick() == "splat.ply"
    # 설정에서 정리를 끄면 예전 splat_clean.ply 안 씀
    p2, ctx2, _ = setup(tmp_path / "b", extra="cleanup: {enabled: false}\n")
    run2 = ctx2.paths.train_run("brush", "default")
    for i, nm in enumerate(("splat.ply", S.SNAP_PLY, S.CLEAN_PLY)):
        _touch(run2 / nm, t + i)
    assert fill.train_ply(ctx2).name == S.SNAP_PLY
    assert not S.Cleanup().applies(ctx2)
    assert S.repair_input(ctx2).name == S.SNAP_PLY                     # repair 는 cleanup 없이도 그대로


def test_prev_targets_and_snap_invalidates_clean(tmp_path):
    p, ctx, _ = setup(tmp_path)
    names = {q.name for q in S.Snap().prev_targets(ctx)}
    assert {S.CLEAN_PLY, S.CLEAN_JSON, S.REMOVED_PLY, S.REPAIR_PLY} <= names
    names = {q.name for q in S.Cleanup().prev_targets(ctx)}
    assert {S.CLEAN_PLY, S.CLEAN_JSON, S.REMOVED_PLY, S.REPAIR_PLY, S.REPAIR_JSON} <= names
    assert S.Cleanup.invalidate_when_skipped


def test_cleanup_moves_stale_outputs_aside(tmp_path):
    p, ctx, _ = setup(tmp_path)
    (ctx.paths.colmap / "register.json").write_text(json.dumps({"version": 1, "mapper": "mapper"}))
    run = ctx.paths.train_run("brush", "default")
    plyio.write(run / "splat.ply", _wall_ply())
    _touch(run / S.CLEAN_PLY, 1_600_000_000)                           # 낡은 결과
    assert not S.Cleanup().is_done(ctx)
    assert S.Cleanup().run(ctx).ok
    assert any(q.name.startswith("splat_clean_prev_") for q in run.iterdir())


def test_cleanup_config():
    base_d = {"sections": [{"name": "a", "video": "v.mp4"}]}
    c = config.parse(base_d)
    assert c.cleanup.enabled and c.cleanup.steps == ["floaters"]          # 11-28 RESULTS2 기본값
    w = c.walls
    assert (w.seed, w.snap, w.repair, w.remove_behind, w.flatten, w.trust_plan_dims) == (True, False, False, False, False,
                                                                                         False)
    c = config.parse({**base_d, "cleanup": {"steps": ["floaters", "sh"]}})
    assert c.cleanup.steps == ["floaters", "sh"]
    for bad in ({"steps": ["nope"]}, {"steps": "sh"}, {"steps": ["sh", "sh"]}, {"max_pieces": 4}):
        with pytest.raises(config.ConfigError):
            config.parse({**base_d, "cleanup": bad})


def test_cleanup_steps_from_config(tmp_path):
    p, ctx, _ = setup(tmp_path, extra="cleanup: {steps: [sh]}\n")
    run = ctx.paths.train_run("brush", "default")
    plyio.write(run / "splat.ply", _with_junk(_wall_ply(), _cam(ctx)))
    assert S.Cleanup().run(StageContext(p, ctx.section, ctx.paths, lambda m: None)).ok
    info = schemas.read_json(run / S.CLEAN_JSON)
    assert info["steps"] == ["sh"] and "floaters" not in info and "needles" not in info
