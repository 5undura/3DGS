"""후처리 뒤 짧은 재학습 (stages/finetune.py). 가짜 Brush 로 명령·고르기·낡음 판정만 봄."""
import json
import os
from pathlib import Path

import pytest

from indoorgs import config
from indoorgs.colmap import model as cm, runner
from indoorgs.colmap.runner import ToolResult
from indoorgs.project import Project
from indoorgs.stages import finetune as F
from indoorgs.stages import merge, snap as S
from indoorgs.stages.base import StageContext, all_stages
from indoorgs.trainers import REGISTRY
from indoorgs.trainers.base import Preset
from a_fixtures import FakeRunner, arg, make_model, make_project, write_brush_ply

BR = REGISTRY["brush"]
HELP_FULL = " ".join(BR.required_options + BR.finetune_options) + " brush 0.3.0"


def _touch(p: Path, t: float):
    os.utime(p, (t, t))


def setup(tmp_path, extra=""):
    root = make_project(tmp_path / "p", extra_yaml="train: {trainer: brush, preset: fast}\n" + extra)
    p = Project(root)
    sec, paths = p.sections()[0]
    cm.write_model(make_model(10), paths.data_clean / "sparse" / "0")
    (paths.data_clean / "images").mkdir()
    for i in range(10):
        (paths.data_clean / "images" / f"f_{i + 1:05d}_{i * 10:06d}.jpg").write_bytes(b"jpg")
    run = paths.train_run("brush", "fast")
    write_brush_ply(run / "export_5000.ply", 40, opacity_logit=1.0)
    write_brush_ply(run / "splat.ply", 40, opacity_logit=1.0)
    (run / "eval.json").write_text(json.dumps({"version": 1, "psnr": 25.0}))
    write_brush_ply(run / S.CLEAN_PLY, 30, opacity_logit=1.0)            # cleanup 결과 (기본 켬)
    (run / S.CLEAN_JSON).write_text(json.dumps({"version": 1, "input": "splat.ply"}))
    _touch(run / "splat.ply", 1_700_000_000)
    _touch(run / S.CLEAN_PLY, 1_700_000_100)
    return p, StageContext(p, sec, paths, lambda m: None), run


def fake_brush(ctx, seen):
    def brush(cmd):
        if "--help" in cmd:
            return ToolResult(0, [HELP_FULL])
        view = Path(cmd[1])
        seen["cmd"] = cmd
        seen["init"] = (view / "init.ply").read_bytes()
        seen["images"] = sorted(q.name for q in (view / "images").iterdir())
        out = Path(arg(cmd, "--export-path"))
        total = arg(cmd, "--total-steps")
        write_brush_ply(out / f"export_{total}.ply", 30, opacity_logit=3.0)
        return 0
    fake = FakeRunner()
    fake.key = lambda cmd: "brush"
    fake.handlers = {"brush": brush}
    return fake


def test_command():
    cmd = BR.finetune_command("brush.exe", Path("D"), Path("O"), Preset(1920, 3_000_000, 2, 30_000), 30_000,
                              1000, 1e-6, 6e-3, refine=False)
    assert cmd[:2] == ["brush.exe", "D"]
    assert arg(cmd, "--start-iter") == "30000" and arg(cmd, "--total-steps") == "31000"
    assert arg(cmd, "--lr-mean") == arg(cmd, "--lr-mean-end") == "1e-06"
    assert arg(cmd, "--lr-scale") == arg(cmd, "--lr-scale-end") == "0.006"
    assert arg(cmd, "--growth-stop-iter") == "30000"
    assert int(arg(cmd, "--refine-every")) > 31000                      # 지우고 다시 뽑기도 없음
    assert arg(cmd, "--sh-degree") == "2" and arg(cmd, "--max-resolution") == "1920"
    assert arg(cmd, "--export-every") == "31000" and arg(cmd, "--eval-split-every") == "8"
    assert "--max-splats" not in cmd
    cmd = BR.finetune_command("b", Path("D"), Path("O"), Preset(), 100, 10, 1e-6, 6e-3, refine=True)
    assert "--refine-every" not in cmd
    for o in BR.finetune_options:                                       # 옵션 상수와 명령이 맞음
        assert o in BR.finetune_command("b", Path("D"), Path("O"), Preset(), 1, 1, 1e-6, 1e-3, refine=False)
    with pytest.raises(NotImplementedError):
        REGISTRY["lichtfeld"].finetune_command("l", Path("D"), Path("O"), Preset(), 1, 1, 1e-6, 1e-3, False)


def test_order():
    names = [s.name for s in all_stages()]
    assert names.index("repair") < names.index("fill") < names.index("finetune") < names.index("merge")


def test_run_brush(tmp_path, monkeypatch):
    p, ctx, run = setup(tmp_path)
    seen = {}
    monkeypatch.setattr(runner, "run", fake_brush(ctx, seen))
    src_bytes = (run / S.CLEAN_PLY).read_bytes()
    st = F.Finetune()
    assert st.applies(ctx) and not st.is_done(ctx)
    assert merge.section_ply(p, None, ctx.paths).name == S.CLEAN_PLY
    res = st.run(ctx)
    assert res.ok, res.message
    cmd = seen["cmd"]
    assert arg(cmd, "--start-iter") == "5000" and arg(cmd, "--total-steps") == "6000"   # export_5000 에서 이어감
    assert seen["init"] == src_bytes and len(seen["images"]) == 10
    d = run / F.FT_DIR
    assert (d / F.FT_PLY).exists() and (d / "export_6000.ply").exists()
    rec = json.loads((d / F.FT_JSON).read_text())
    assert rec["input"].endswith(S.CLEAN_PLY) and rec["start_iter"] == 5000 and rec["end_iter"] == 6000
    assert rec["count_before"] == 30 and rec["psnr_train"] == 25.0
    # 원본은 그대로: 입력 PLY 내용, data_clean 에 init.ply 없음, 보기 폴더의 폴더 링크는 지워짐(원본 사진은 남음)
    assert (run / S.CLEAN_PLY).read_bytes() == src_bytes
    assert not list(ctx.paths.data_clean.rglob("init.ply"))
    assert len(list((ctx.paths.data_clean / "images").iterdir())) == 10
    assert not (d / "data" / "images").exists() or rec["data_view"]["images"] == "copy"
    assert st.is_done(ctx)
    assert merge.section_ply(p, None, ctx.paths) == d / F.FT_PLY
    # 다시 돌리면 이전 결과는 finetune_prev_<시각>
    assert st.run(ctx).ok
    assert len([q for q in run.iterdir() if q.name.startswith("finetune_prev_")]) == 1
    from indoorgs.stages import report
    html = report.build_html(p)
    assert "후처리 뒤 재학습" in html and "splat_ft.ply" in html and "5,000 → 6,000" in html


def test_invalid_when_input_or_config_changes(tmp_path, monkeypatch):
    p, ctx, run = setup(tmp_path)
    monkeypatch.setattr(runner, "run", fake_brush(ctx, {}))
    assert F.Finetune().run(ctx).ok
    ft = run / F.FT_DIR / F.FT_PLY
    assert F.finetune_valid(p, ctx.paths) == ft
    # 입력이 다른 파일로 바뀜 (시각은 결과보다 옛것) -> 안 씀
    write_brush_ply(run / S.CLEAN_PLY, 31, opacity_logit=1.0)
    _touch(run / S.CLEAN_PLY, 1_700_000_100)
    assert F.finetune_valid(p, ctx.paths) is None and not F.Finetune().is_done(ctx)
    assert merge.section_ply(p, None, ctx.paths).name == S.CLEAN_PLY
    assert F.Finetune().run(ctx).ok and F.finetune_valid(p, ctx.paths) == ft
    # 앞 단계를 다시 돌려 입력이 결과보다 새것 -> 안 씀
    _touch(ft, 1_700_000_050)
    assert F.finetune_valid(p, ctx.paths) is None
    assert F.Finetune().run(ctx).ok and F.finetune_valid(p, ctx.paths) == ft
    # fill 결과가 생기면 입력이 바뀜 -> 안 씀
    write_brush_ply(ctx.paths.fill_ply, 35)
    assert F.finetune_input(p, ctx.paths) == ctx.paths.fill_ply
    assert F.finetune_valid(p, ctx.paths) is None
    assert merge.section_ply(p, None, ctx.paths) == ctx.paths.fill_ply
    assert F.Finetune().run(ctx).ok
    assert json.loads((run / F.FT_DIR / F.FT_JSON).read_text())["input"] == "fill/holes_backdrop.ply"
    assert merge.section_ply(p, None, ctx.paths) == ft
    # 설정을 바꾸면 (반복 수) 안 씀, 끄면 안 씀
    p2 = Project(p.root)
    p2.cfg.finetune.iters = 500
    assert F.finetune_valid(p2, ctx.paths) is None
    p2.cfg.finetune.iters = 1000
    assert F.finetune_valid(p2, ctx.paths) == ft
    p2.cfg.finetune.enabled = False
    assert F.finetune_valid(p2, ctx.paths) is None and merge.section_ply(p2, None, ctx.paths) == ctx.paths.fill_ply


def test_lichtfeld_skipped(tmp_path, monkeypatch):
    p, ctx, run = setup(tmp_path)
    monkeypatch.setattr(runner, "run", fake_brush(ctx, {}))
    assert F.Finetune().run(ctx).ok                                       # Brush 로 만든 결과
    assert F.finetune_valid(p, ctx.paths) is not None
    p.cfg.train.trainer = "lichtfeld"
    logs = []
    ctx2 = StageContext(p, ctx.section, ctx.paths, logs.append)
    assert not F.Finetune().applies(ctx2)
    assert logs and "LichtFeld" in logs[0] and "건너뜀" in logs[0]
    # 예전 결과가 있어도 지원 안 하는 학습기면 안 씀
    assert F.finetune_valid(p, ctx.paths) is None
    p.cfg.finetune.enabled = False
    logs.clear()
    assert not F.Finetune().applies(ctx2) and not logs                   # 끔: 조용히 건너뜀


def test_missing_options_fail(tmp_path, monkeypatch):
    p, ctx, run = setup(tmp_path)
    fake = FakeRunner()
    fake.key = lambda cmd: "brush"
    fake.handlers = {"brush": lambda cmd: ToolResult(0, [" ".join(BR.required_options) + " v0.3.0"])}
    monkeypatch.setattr(runner, "run", fake)
    res = F.Finetune().run(ctx)
    assert not res.ok and "--start-iter" in res.message
    assert not (run / F.FT_DIR).exists() and len(fake.calls) == 2       # --help 만


def test_failed_run_moves_aside(tmp_path, monkeypatch):
    p, ctx, run = setup(tmp_path)
    fake = FakeRunner()
    fake.key = lambda cmd: "brush"
    fake.handlers = {"brush": lambda cmd: ToolResult(0, [HELP_FULL]) if "--help" in cmd else ToolResult(1, ["OOM"])}
    monkeypatch.setattr(runner, "run", fake)
    res = F.Finetune().run(ctx)
    assert not res.ok and "OOM" in res.message
    assert any(q.name.startswith("finetune_failed_") for q in run.iterdir())
    assert merge.section_ply(p, None, ctx.paths).name == S.CLEAN_PLY
    assert len(list((ctx.paths.data_clean / "images").iterdir())) == 10   # 원본 사진 그대로


def test_config_effective_walls_defaults():
    base = {"sections": [{"name": "a", "video": "v.mp4"}]}
    c = config.parse(base)
    assert c.finetune.enabled and c.finetune.iters == 1000 and c.finetune.lr_mean == 1e-6
    assert (c.walls.snap, c.walls.flatten, c.walls.remove_behind, c.walls.repair) == (True, True, False, False)
    for extra in ({"train": {"trainer": "lichtfeld"}}, {"finetune": {"enabled": False}}, {"walls": {"seed": False}}):
        w = config.parse({**base, **extra}).walls
        assert (w.snap, w.flatten) == (False, False), extra
    w = config.parse({**base, "walls": {"snap": False}}).walls                 # 적은 값은 그대로
    assert (w.snap, w.flatten) == (False, True)
    w = config.parse({**base, "finetune": {"enabled": False}, "walls": {"snap": True}}).walls
    assert (w.snap, w.flatten) == (True, False)
    assert config.parse({**base, "finetune": {"lr_mean": "2e-6"}}).finetune.lr_mean == 2e-6   # PyYAML 글자
    for bad in ({"iters": 0}, {"lr_mean": -1}, {"lr_scale": "x"}, {"nope": 1}):
        with pytest.raises(config.ConfigError):
            config.parse({**base, "finetune": bad})
