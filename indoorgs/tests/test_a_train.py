import json
from pathlib import Path

import cv2
import numpy as np

from indoorgs.colmap import model as cm, runner
from indoorgs.colmap.runner import ToolResult
from indoorgs.project import Project
from indoorgs.stages.base import StageContext
from indoorgs.stages.train import Train, ply_stats, read_ply_fields
from indoorgs.trainers import REGISTRY
from a_fixtures import FakeRunner, arg, make_model, make_project, write_brush_ply

BRUSH_HELP = " ".join(REGISTRY["brush"].required_options) + " brush 0.3.0"


def setup(tmp_path, preset="fast"):
    root = make_project(tmp_path / "p", extra_yaml=f"train: {{trainer: brush, preset: {preset}}}\n")
    p = Project(root)
    sec, paths = p.sections()[0]
    cm.write_model(make_model(10), paths.data_clean / "sparse" / "0")
    img = np.dstack([np.tile(np.linspace(0, 255, 64), (48, 1))] * 3).astype(np.uint8)   # 매끈한 그림 (JPEG 손실 작음)
    (paths.data_clean / "images").mkdir()
    for i in range(10):
        cv2.imencode(".jpg", img)[1].tofile(str(paths.data_clean / "images" / f"f_{i + 1:05d}_{i * 10:06d}.jpg"))
    return p, StageContext(p, sec, paths, lambda m: None), img


def test_ply_by_name(tmp_path):
    f = tmp_path / "a.ply"
    write_brush_ply(f, 10, opacity_logit=-2.0, scales=(-1.0, -4.0, -4.0 + np.log(20)))
    d = read_ply_fields(f, ["x", "opacity"])
    assert d["opacity"][0] == -2.0
    s = ply_stats(f)
    assert s["count"] == 10 and s["translucent_ratio"] == 1.0 and s["needle_ratio"] == 0.0


def test_train_brush(tmp_path, monkeypatch):
    p, ctx, img = setup(tmp_path)

    def brush(cmd):
        if "--help" in cmd:
            return ToolResult(0, [BRUSH_HELP])
        out = Path(arg(cmd, "--export-path"))
        assert Path(cmd[1]) == ctx.paths.data_clean
        assert arg(cmd, "--total-steps") == "5000"
        write_brush_ply(out / "export_5000.ply", 50, opacity_logit=2.0)
        (out / "eval_5000").mkdir()
        cv2.imencode(".png", img)[1].tofile(str(out / "eval_5000" / "f_00001_000000.png"))
        return 0

    fake = FakeRunner()
    fake.key = lambda cmd: "brush"
    fake.handlers = {"brush": brush}
    monkeypatch.setattr(runner, "run", fake)
    res = Train().run(ctx)
    assert res.ok, res.message
    run = ctx.paths.train_run("brush", "fast")
    ev = json.loads((run / "eval.json").read_text())
    st = json.loads((run / "stats.json").read_text())
    assert ev["n_eval"] == 1 and ev["psnr"] > 35 and ev["eval_every"] == 8
    assert st["count"] == 50 and st["translucent_ratio"] == 0.0
    assert (run / "splat.ply").exists() and (run / "export_5000.ply").exists()
    assert Train().is_done(ctx)


def test_train_fail_moves_aside(tmp_path, monkeypatch):
    p, ctx, _ = setup(tmp_path)
    fake = FakeRunner()
    fake.key = lambda cmd: "brush"
    fake.handlers = {"brush": lambda cmd: ToolResult(0, [BRUSH_HELP]) if "--help" in cmd else ToolResult(1, ["OOM"])}
    monkeypatch.setattr(runner, "run", fake)
    res = Train().run(ctx)
    assert not res.ok and "OOM" in res.message
    assert any(d.name.startswith("brush_fast_failed_") for d in ctx.paths.train.iterdir())


def test_train_detect_missing_option(tmp_path, monkeypatch):
    p, ctx, _ = setup(tmp_path)
    monkeypatch.setattr(runner, "run", lambda cmd, **k: ToolResult(0, ["--max-splats only, v0.4.0"]))
    res = Train().run(ctx)
    assert not res.ok and "옵션 없음" in res.message
    info = REGISTRY["brush"].detect(Project(p.root).cfg.tools.brush)
    assert info.found and info.version == "0.4.0" and "--total-steps" in info.missing_options


def test_lichtfeld_metrics(tmp_path):
    lf = REGISTRY["lichtfeld"]
    (tmp_path / "metrics.csv").write_text(
        "iteration,psnr,ssim,lpips,time_per_image,num_gaussians\n"
        "30000,26.1,0.87,,0.1,100\n80000,26.79,0.885,,0.1,100\n")
    ev = lf.evaluate(tmp_path, tmp_path)
    assert ev["psnr"] == 26.79 and ev["ssim"] == 0.885 and ev["iteration"] == 80000
    assert lf.evaluate(tmp_path / "none", tmp_path) is None
