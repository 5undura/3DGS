"""GUI: 벽·정리 결과 API, 면 텍스처 파일, 후처리 설정 저장 (gui/post.py)."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
import cv2  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from indoorgs import config  # noqa: E402
from indoorgs.gui import post  # noqa: E402
from indoorgs.gui.server import create_app  # noqa: E402

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo"


def _wall(pid, x, src="fit"):
    return {"id": pid, "kind": "wall", "room": "corridor1", "n": [-1, 0, 0], "d": -x,
            "origin": [x, -1, 0], "U": [0, 1, 0], "V": [0, 0, 1], "size": [20, 2.7], "holes": []}


def make_project(tmp_path):
    """데모 프로젝트 + corridor1 의 planes/walls/clean/repair JSON, 면 텍스처."""
    root = tmp_path / "demo"
    shutil.copytree(DEMO, root)
    sec = root / "sections" / "corridor1"
    (sec / "walls" / "tex").mkdir(parents=True)
    planes = {"version": 1, "rooms": ["corridor1"], "source_model": "data_clean",
              "planes": [_wall("corridor1:w1", -1.13), _wall("corridor1:w2", 1.2)],
              "plan_planes": [_wall("corridor1:w1", -1.2), _wall("corridor1:w2", 1.2)],
              "fit": [{"id": "corridor1:w1", "kind": "wall", "source": "fit", "shift_m": 0.07, "tilt_deg": 0.4,
                       "n_points": 812, "coverage": 0.6},
                      {"id": "corridor1:w2", "kind": "wall", "source": "plan", "shift_m": 0.0, "tilt_deg": 0.0,
                       "n_points": 3, "coverage": 0.01, "reason": "점 부족"}],
              "seed": {"n_added": 182000}}
    (sec / "walls" / "planes.json").write_text(json.dumps(planes), encoding="utf-8")
    run = sec / "train" / "brush_default"
    run.mkdir(parents=True)
    (run / "splat.ply").write_bytes(b"ply\n")
    (run / "splat_walls.ply").write_bytes(b"ply\n")
    # NaN 이 든 walls.json (snap 이 실제로 쓸 수 있는 값)
    (run / "walls.json").write_text(
        '{"version": 1, "flatness_before": {"thick_m": 0.05}, "flatness_after": {"thick_m": NaN},'
        ' "n_moved": 120, "n_removed": 7, "planes_used": ["corridor1:w1"], "warnings": ["경고: x"],'
        ' "remove": {"skipped_faces": {"corridor1:w1": "too many"}}}', encoding="utf-8")
    (run / "clean.json").write_text(json.dumps({
        "version": 1, "input": "splat_walls.ply", "plan_coords": True,
        "floaters": {"removed_by": {"camera": 5, "air": 3}}, "needles": {"n_needles": 10, "n_pieces": 70},
        "huge": {"n_removed": 2, "n_split": 1}, "sh": {"n_damped": 4},
        "quality_before": {"count": 1000}, "quality_after": {"count": 1050}}), encoding="utf-8")
    (run / "repair.json").write_text(json.dumps({
        "version": 1, "input": "splat_clean.ply", "n_added": 300, "hole_m2_before": 1.5, "hole_m2_after": 0.2,
        "faces": {"corridor1:w1": {"hole_m2_before": 1.5, "hole_m2_after": 0.2, "n_added": 300, "n_textured": 280}}}),
        encoding="utf-8")
    tex = sec / "walls" / "tex"
    img = np.zeros((8, 16, 3), np.uint8)
    img[:] = (200, 210, 220)
    cv2.imencode(".png", img)[1].tofile(str(tex / "corridor1_w1.png"))
    src = np.full((8, 16), 85, np.uint8)
    src[:, 8:] = 170
    cv2.imencode(".png", src)[1].tofile(str(tex / "corridor1_w1.src.png"))
    (tex / "corridor1_w1.json").write_text(json.dumps({"plane_id": "corridor1:w1", "res": 0.015,
                                                       "stats": {"photo_m2": 1.0, "inpaint_m2": 0.5}}),
                                           encoding="utf-8")
    (root / "secret.png").write_bytes(b"SECRET")
    return root


@pytest.fixture
def proj(tmp_path):
    return make_project(tmp_path)




@pytest.fixture
def client(proj):
    return TestClient(create_app(proj))


def test_walls_endpoint(client):
    r = client.get("/api/walls/corridor1")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["run"] == "brush_default" and d["runs"] == ["brush_default"]
    assert [f["id"] for f in d["planes"]["fit"]] == ["corridor1:w1", "corridor1:w2"]
    assert d["snap"]["flatness_after"]["thick_m"] is None          # NaN -> null
    assert d["snap"]["n_moved"] == 120
    assert d["clean"]["needles"]["n_pieces"] == 70 and "1,000 -> 1,050" in d["clean_summary"]
    assert d["repair"]["faces"]["corridor1:w1"]["n_textured"] == 280
    assert d["files"]["walls"] and not d["files"]["repaired"]
    assert d["textures"] == [{"name": "corridor1_w1.png", "src": "corridor1_w1.src.png", "plane_id": "corridor1:w1",
                              "res": 0.015, "size": None, "shape": None,
                              "stats": {"photo_m2": 1.0, "inpaint_m2": 0.5}}]
    assert d["errors"] == {}


def test_walls_missing_files(client):
    d = client.get("/api/walls/room_b117").json()
    assert d["planes"] is None and d["run"] is None and d["snap"] is None and d["textures"] == []
    assert client.get("/api/walls/nope").status_code == 404
    assert client.get("/api/walls/corridor1?run=zzz").status_code == 404
    assert client.get("/api/walls/corridor1?run=..").status_code == 404


def test_walls_broken_json(client, proj):
    (proj / "sections" / "corridor1" / "train" / "brush_default" / "repair.json").write_text("{", encoding="utf-8")
    d = client.get("/api/walls/corridor1").json()
    assert d["repair"] is None and "repair" in d["errors"]


def test_texture_files(client):
    r = client.get("/api/walls/corridor1/tex/corridor1_w1.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert client.get("/api/walls/corridor1/tex/corridor1_w1.src.png").status_code == 200
    assert client.get("/api/walls/corridor1/tex/none.png").status_code == 404
    assert client.get("/api/walls/room_b117/tex/corridor1_w1.png").status_code == 404


@pytest.mark.parametrize("name", ["..%2F..%2F..%2Fsecret.png", "..%2Fplanes.json", "corridor1_w1.json",
                                  "..png", "%2Fetc%2Fpasswd", "..%5C..%5Csecret.png", "a..b.png"])
def test_texture_traversal_rejected(client, name):
    r = client.get(f"/api/walls/corridor1/tex/{name}")
    assert r.status_code in (400, 404)
    assert b"SECRET" not in r.content and b"plane_id" not in r.content


def test_texture_file_rejects_symlink_out(proj):
    tex = proj / "sections" / "corridor1" / "walls" / "tex"
    try:
        (tex / "link.png").symlink_to(proj / "secret.png")
    except OSError:
        pytest.skip("symlink 불가")
    from indoorgs.project import Project
    sp = Project(proj).section("corridor1")
    with pytest.raises(ValueError):
        post.texture_file(sp, "link.png")


def test_settings_roundtrip(client, proj):
    yml = proj / "project.yaml"
    before = yml.read_text(encoding="utf-8")
    s = client.get("/api/settings/post").json()
    assert s["walls"]["band"] == 0.06 and s["cleanup"]["steps"] == ["floaters", "huge", "needles", "sh"]
    body = {"walls": {"enabled": True, "seed": False, "snap": True, "repair": False, "texture": False,
                      "band": 0.04, "color_tol": 0.2},
            "cleanup": {"enabled": True, "steps": ["needles", "floaters"]}}
    r = client.put("/api/settings/post", json=body)
    assert r.status_code == 200, r.text
    text = yml.read_text(encoding="utf-8")
    cfg = config.load(yml)
    assert (cfg.walls.seed, cfg.walls.repair, cfg.walls.texture, cfg.walls.band, cfg.walls.color_tol) == (False, False, False, 0.04, 0.2)
    assert cfg.walls.remove_behind is True                         # 안 보낸 항목 유지
    assert cfg.cleanup.steps == ["needles", "floaters"]
    # 다른 줄·주석은 그대로
    assert "# 형식 예시" in text and "# 학습 후 정리" in text
    keep = [ln for ln in before.splitlines() if not ln.startswith(("walls:", "cleanup:"))]
    assert [ln for ln in text.splitlines() if not ln.startswith(("walls:", "cleanup:"))] == keep
    assert client.get("/api/settings/post").json()["walls"]["band"] == 0.04
    assert client.get("/api/project").status_code == 200


@pytest.mark.parametrize("bad, msg", [
    ({"walls": {"band": -1}}, "band"),
    ({"walls": {"band": "x"}}, "band"),
    ({"walls": {"seed": "yes"}}, "seed"),
    ({"walls": {"search": 0.5}}, "못 고치는"),
    ({"cleanup": {"steps": ["zzz"]}}, "cleanup.steps"),
    ({"cleanup": {"steps": ["sh", "sh"]}}, "cleanup.steps"),
    ({"tools": {}}, "모르는"),
])
def test_settings_bad_is_400(client, proj, bad, msg):
    yml = proj / "project.yaml"
    before = yml.read_bytes()
    r = client.put("/api/settings/post", json=bad)
    assert r.status_code == 400 and msg in r.json()["detail"]
    assert yml.read_bytes() == before


def test_settings_block_style_keeps_comments(tmp_path):
    yml = tmp_path / "project.yaml"
    yml.write_text(
        "# 머리 주석\n"
        "sections:\n  - name: a\n    video: a.mp4\n"
        "walls:\n"
        "  enabled: true   # 켬\n"
        "  # 안쪽 주석\n"
        "  band: 0.03  # 액자 보호\n"
        "  seed_spacing: 0.1\n"
        "cleanup:\n"
        "  steps:\n"
        "    - floaters\n"
        "    - sh\n"
        "  enabled: true\n"
        "\n# 끝 주석\n", encoding="utf-8")
    post.save_post_settings(yml, {"walls": {"band": 0.05, "repair": False},
                                  "cleanup": {"steps": ["huge"], "enabled": False}})
    text = yml.read_text(encoding="utf-8")
    for c in ("# 머리 주석", "# 켬", "# 안쪽 주석", "# 액자 보호", "# 끝 주석"):
        assert c in text
    raw = yaml.safe_load(text)
    assert raw["walls"] == {"enabled": True, "band": 0.05, "seed_spacing": 0.1, "repair": False}
    assert raw["cleanup"] == {"steps": ["huge"], "enabled": False}
    config.parse(raw)


def test_settings_missing_block_appended(tmp_path):
    yml = tmp_path / "project.yaml"
    yml.write_text("sections:\n  - {name: a, video: a.mp4}\n", encoding="utf-8")
    post.save_post_settings(yml, {"walls": {"snap": False}})
    assert config.load(yml).walls.snap is False
