import shutil
from pathlib import Path

from indoorgs import schemas
from indoorgs.project import Project
from indoorgs.stages.base import StageContext
from indoorgs.stages.report import Report, write_report

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo"


def _copy(tmp_path):
    root = tmp_path / "demo"
    shutil.copytree(DEMO, root)
    return root


def test_report_without_results(tmp_path):
    root = _copy(tmp_path)
    p = Project(root)
    res = Report().run(StageContext(p, None, None, lambda m: None))
    assert res.ok
    html = (root / "report.html").read_text(encoding="utf-8")
    assert "아직 없음" in html and "corridor1" in html and "<svg" in html     # 공간은 있음
    assert "http://" not in html.replace('xmlns="http://www.w3.org/2000/svg"', "")   # 외부 리소스 없음


def test_report_without_space(tmp_path):
    root = _copy(tmp_path)
    (root / "floorplan" / "plan.json").unlink()
    html = write_report(Project(root)).read_text(encoding="utf-8")
    assert "<svg" not in html and "plan.json 없음" in html


def test_report_with_results(tmp_path):
    root = _copy(tmp_path)
    sec = root / "sections" / "room_b117"
    schemas.write_json(sec / "frames" / "done.json", {"version": 1, "fps": 24.0, "count": 1540})
    schemas.write_json(sec / "diagnose.json", {"version": 1, "registered": 1072, "total": 1084, "reproj_error_px": 0.96,
                                               "teleports": [], "outside_space": ["f_00002_000024.jpg"],
                                               "far_from_prior": [], "excluded": ["f_00002_000024.jpg"]})
    schemas.write_json(sec / "colmap" / "priors.json", {"version": 1, "coordinate_system": "CARTESIAN", "frames": [
        {"image": "f_00001_000000.jpg", "t": 0.0, "xyz": [-2.9, 3.97, 1.4], "sigma": [0.5, 0.5, 0.3]},
        {"image": "f_00002_000024.jpg", "t": 1.0, "xyz": [0.5, 3.97, 1.4], "sigma": [0.5, 0.5, 0.3]}]})
    run = sec / "train" / "brush_default"
    schemas.write_json(run / "eval.json", {"version": 1, "psnr": 26.79, "ssim": 0.885, "n_eval": 133, "eval_every": 8})
    schemas.write_json(run / "stats.json", {"version": 1, "count": 3000000, "translucent_ratio": 0.869, "needle_ratio": 0.01})
    run2 = sec / "train" / "lichtfeld_quality"
    schemas.write_json(run2 / "stats.json", {"version": 1, "count": 2900000, "translucent_ratio": 0.494, "needle_ratio": 0.01})
    (sec / "train" / "brush_x_failed_20260101").mkdir()
    schemas.write_json(root / "merged" / "layout.json", {"version": 1, "sections": {"room_b117": [[1, 0, 0, 0]] * 4}})

    html = write_report(Project(root)).read_text(encoding="utf-8")
    assert "1,540" in html and "1,072" in html and "26.79" in html
    assert "86.9%" in html and "VR에서 비칠 수 있음" in html
    assert "49.4%" in html and html.count("VR에서 비칠 수 있음") == 1
    assert "lichtfeld_quality" in html and "_failed_" not in html
    assert 'fill="#dc2626"' in html                     # 제외 사진 빨간 점
    assert "구간 1개" in html


def test_report_escapes(tmp_path):
    root = _copy(tmp_path)
    plan = schemas.read_json(root / "floorplan" / "plan.json")
    plan["rooms"][0]["name"] = "<script>alert(1)</script>"
    schemas.write_json(root / "floorplan" / "plan.json", plan)
    html = write_report(Project(root)).read_text(encoding="utf-8")
    assert "<script>" not in html and "&lt;script&gt;" in html
