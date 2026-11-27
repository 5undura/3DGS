from indoorgs import cli
from indoorgs.colmap import runner
from indoorgs.colmap.runner import ToolResult
from a_fixtures import make_project


def test_check_shows_colmap_version(tmp_path, monkeypatch, capsys):
    root = make_project(tmp_path / "p")

    def fake(cmd, **k):
        if cmd[1:] == ["help"]:
            return ToolResult(0, ["COLMAP 4.2.1 (Commit bd1fcf6 with CUDA)", "  pose_prior_mapper"])
        return ToolResult(0, ["nothing"])
    monkeypatch.setattr(runner, "run", fake)
    cli.main(["check", str(root)])
    out = capsys.readouterr().out
    assert "버전 4.2.1" in out and "pose_prior_mapper 못 찾음" not in out


def test_check_old_colmap(tmp_path, monkeypatch, capsys):
    root = make_project(tmp_path / "p")
    monkeypatch.setattr(runner, "run", lambda cmd, **k: ToolResult(0, ["COLMAP 3.9.1"]))
    cli.main(["check", str(root)])
    out = capsys.readouterr().out
    assert "버전 3.9.1" in out and "pose_prior_mapper 못 찾음" in out


def test_progress_echo():
    lines = []
    e = cli.progress_echo(3, lines.append)
    e("[10:00:00] extract: 시작")
    e("[10:00:01] 뭔가")
    e("[10:00:02] register: 시작")
    assert lines[0].startswith("[1/3 +00:00:0") and lines[2].startswith("[2/3 ")


def test_run_shows_plan(tmp_path, capsys):
    root = make_project(tmp_path / "p", colmap=False)
    assert cli.main(["run", str(root), "--until", "register"]) == 1
    out = capsys.readouterr().out
    assert "할 단계 1개: s1/register" in out and "COLMAP 없음" in out and "결과: 실패" in out


def test_run_until_train_end_to_end(tmp_path, monkeypatch):
    """extract(진짜 합성 영상) -> register(가짜 COLMAP) -> diagnose -> train(가짜 Brush) 를 run 한 번으로."""
    import json
    import shutil
    from pathlib import Path

    from test_a_extract import make_video
    from a_fixtures import FakeRunner, arg, colmap_handlers, write_brush_ply
    from indoorgs.colmap import model as cm
    from indoorgs.trainers import REGISTRY

    root = make_project(tmp_path / "p", n_frames=0, extra_yaml="train: {trainer: brush, preset: fast}\n")
    shutil.rmtree(root / "sections" / "s1" / "frames")
    make_video(root / "v.mp4")
    n_frames = {}
    h = colmap_handlers()

    def mapper(cmd):   # 추출된 프레임 이름 그대로 모델을 만듦
        names = sorted(f.name for f in Path(arg(cmd, "--image_path")).glob("*.jpg"))
        n_frames["n"] = len(names)
        m = cm.read_model(_tmp_model(tmp_path, len(names)))
        for im, nm in zip(m.images.values(), names):
            im.name = nm
        cm.write_model(m, Path(arg(cmd, "--output_path")) / "0")
    h["mapper"] = mapper

    def brush(cmd):
        if "--help" in cmd:
            return ToolResult(0, [" ".join(REGISTRY["brush"].required_options)])
        write_brush_ply(Path(arg(cmd, "--export-path")) / "export_5000.ply", 20)
        return 0
    fake = FakeRunner(h)
    fake.key = lambda cmd: cmd[1] if cmd[0].endswith("colmap.bat") else "brush"
    fake.handlers["brush"] = brush
    monkeypatch.setattr(runner, "run", fake)
    assert cli.main(["run", str(root), "--until", "train"]) == 0
    run = root / "sections" / "s1" / "train" / "brush_fast"
    assert json.loads((run / "stats.json").read_text())["count"] == 20
    assert json.loads((run / "eval.json").read_text())["psnr"] is None      # 평가 렌더 없음
    d = json.loads((root / "sections" / "s1" / "diagnose.json").read_text(encoding="utf-8"))
    assert d["registered"] == n_frames["n"]


def _tmp_model(tmp_path, n):
    from a_fixtures import make_model
    from indoorgs.colmap import model as cm
    out = tmp_path / f"m{n}"
    if not out.exists():
        cm.write_model(make_model(n), out)
    return out
