import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from indoorgs import schemas  # noqa: E402
from indoorgs.gui.server import create_app  # noqa: E402

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo"
STATIC = Path(__file__).resolve().parents[1] / "src" / "indoorgs" / "gui" / "static"


@pytest.fixture
def proj(tmp_path):
    root = tmp_path / "demo"
    shutil.copytree(DEMO, root)
    # 가짜 영상: room_b117 의 영상 경로를 프로젝트 안 파일로
    video = root / "v.mp4"
    video.write_bytes(bytes(range(256)) * 4)          # 1024 바이트
    y = (root / "project.yaml").read_text(encoding="utf-8").replace("D:/capture/5.MP4", video.as_posix())
    (root / "project.yaml").write_text(y, encoding="utf-8")
    return root


@pytest.fixture
def client(proj):
    return TestClient(create_app(proj))


def test_index_and_project(client):
    assert "IndoorGS" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    p = client.get("/api/project").json()
    assert [s["name"] for s in p["sections"]] == ["corridor1", "room_b117"]
    assert p["has_space"] and "report" in p["stages"]


def test_plan_roundtrip(client, proj):
    plan = client.get("/api/plan").json()
    plan["rooms"][0]["box"]["center"] = [1.0, 2.0]          # 박스 옮기기
    plan["rooms"].append({"id": "r3", "box": {"center": [10, 0], "size": [3, 4], "yaw_deg": 15}, "floor_z": 0, "ceiling_z": 2.5})
    plan["doors"].append({"id": "d3", "rooms": ["r3"], "center": [8.5, 0], "width": 0.9, "height": 2.1})
    r = client.put("/api/plan", json=plan)
    assert r.status_code == 200, r.text
    saved = schemas.load_plan(proj / "floorplan" / "plan.json")
    assert {x.id for x in saved.rooms} == {"corridor1", "b117", "r3"}
    assert saved.room_at(1.0, 2.0).id == "corridor1"
    assert client.get("/api/plan").json()["rooms"][0]["box"]["center"] == [1.0, 2.0]


@pytest.mark.parametrize("bad, msg", [
    ({"version": 1, "rooms": []}, "비어"),
    ({"version": 1, "rooms": [{"id": "a", "box": {"center": [0, 0], "size": [-1, 1]}}]}, "양수"),
    ({"version": 1, "rooms": [{"id": "a", "box": {"center": [0, 0], "size": [1, 1]}}],
      "doors": [{"id": "d", "rooms": ["x"], "center": [0, 0], "width": 1}]}, "없는 방"),
    ({"version": 1, "rooms": [{"id": "a", "box": {"center": [0], "size": [1, 1]}}]}, "두 숫자"),
])
def test_plan_bad_is_400(client, proj, bad, msg):
    before = (proj / "floorplan" / "plan.json").read_bytes()
    r = client.put("/api/plan", json=bad)
    assert r.status_code == 400 and msg in r.json()["detail"]
    assert (proj / "floorplan" / "plan.json").read_bytes() == before      # 안 바뀜


def test_path_roundtrip(client, proj):
    assert client.get("/api/sections/corridor1/path").status_code == 404
    body = {"version": 1, "camera_height": 1.5, "sigma_xy": 0.5, "sigma_z": 0.3,
            "stroke": [[-2.4, 0], [-2.4, 20]],
            "anchors": [{"t": 0, "point": [-2.4, 0]}, {"t": 30.5, "point": [-2.4, 20]}]}
    assert client.put("/api/sections/corridor1/path", json=body).status_code == 200
    cam = schemas.load_path(proj / "sections" / "corridor1" / "path.json")
    assert cam.camera_height == 1.5 and len(cam.anchors) == 2
    assert client.get("/api/sections/corridor1/path").json()["anchors"][1]["t"] == 30.5
    r = client.put("/api/sections/corridor1/path", json={"version": 1, "anchors": [{"t": 0, "point": [0, 0]}]})
    assert r.status_code == 400 and "최소 2개" in r.json()["detail"]
    assert client.get("/api/sections/nope/path").status_code == 404


def test_status(client):
    st = client.get("/api/status").json()
    cells = {(r["section"], r["stage"]): r["state"] for r in st["rows"]}
    assert cells[("room_b117", "extract")] == "todo"
    assert cells[("corridor1", "prior")] == "skip"
    assert cells[("*", "report")] == "todo"


def test_status_fail_from_log(client, proj):
    (proj / "logs").mkdir(exist_ok=True)
    (proj / "logs" / "room_b117_extract.log").write_text("[10:00:00] extract: 시작\n[10:00:01] extract: 실패 영상 없음\n", encoding="utf-8")
    cells = {(r["section"], r["stage"]): r for r in client.get("/api/status").json()["rows"]}
    assert cells[("room_b117", "extract")]["state"] == "fail"
    assert cells[("room_b117", "extract")]["log"] == "room_b117_extract.log"
    logs = client.get("/api/logs").json()
    assert logs[0]["name"] == "room_b117_extract.log"
    tail = client.get("/api/logs/room_b117_extract.log?n=1").json()["lines"]
    assert tail == ["[10:00:01] extract: 실패 영상 없음"]


@pytest.mark.parametrize("name", ["..%2Fproject.yaml", "..%2F..%2Fetc%2Fpasswd", "a.txt", "%2Fetc%2Fpasswd.log", "..%5Cx.log"])
def test_log_traversal_rejected(client, name):
    r = client.get(f"/api/logs/{name}")
    assert r.status_code in (400, 404)
    assert "root:" not in r.text and "sections" not in r.text


def test_video_range(client):
    full = client.get("/api/sections/room_b117/video")
    assert full.status_code == 200 and len(full.content) == 1024
    assert full.headers["accept-ranges"] == "bytes"
    r = client.get("/api/sections/room_b117/video", headers={"Range": "bytes=10-19"})
    assert r.status_code == 206
    assert r.headers["content-range"] == "bytes 10-19/1024"
    assert r.content == bytes(range(10, 20))
    r = client.get("/api/sections/room_b117/video", headers={"Range": "bytes=1000-"})
    assert r.status_code == 206 and len(r.content) == 24
    r = client.get("/api/sections/room_b117/video", headers={"Range": "bytes=-4"})
    assert r.content == bytes([252, 253, 254, 255])
    assert client.get("/api/sections/room_b117/video", headers={"Range": "bytes=5000-"}).status_code == 416
    assert client.get("/api/sections/corridor1/video").status_code == 404        # 영상 없음
    assert client.get("/api/sections/..%2F..%2Fproject.yaml/video").status_code == 404


def test_run_in_background(client, proj):
    r = client.post("/api/run", json={"section": "room_b117", "until": "extract"})
    assert r.status_code == 200
    for _ in range(100):
        s = client.get("/api/run/state").json()
        if not s["running"]:
            break
        time.sleep(0.05)
    assert s["ok"] is False                             # extract 는 아직 빈 틀이거나 영상이 가짜
    assert any("extract" in line for line in s["lines"])
    assert client.post("/api/run", json={"section": "nope"}).status_code == 404
    assert client.post("/api/run", json={"until": "zzz"}).status_code == 400


def test_run_only_one(client):
    import threading
    app_runner = client.app.state.runner
    gate = threading.Event()
    app_runner.thread = threading.Thread(target=gate.wait, daemon=True)
    app_runner.thread.start()
    try:
        assert client.post("/api/run", json={}).status_code == 409
    finally:
        gate.set()


def test_report_endpoint(client, proj):
    r = client.get("/api/report?rebuild=1")
    assert r.status_code == 200 and "IndoorGS 보고서" in r.text
    assert (proj / "report.html").exists()


@pytest.mark.skipif(shutil.which("node") is None, reason="node 없음")
@pytest.mark.parametrize("js", sorted(p.name for p in STATIC.glob("*.js")))
def test_js_syntax(js):
    r = subprocess.run(["node", "--check", str(STATIC / js)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
