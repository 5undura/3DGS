"""E: 도면 불러오기·검사 API."""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient  # noqa: E402

from indoorgs.gui.server import create_app  # noqa: E402
from test_b_image import PPM, draw_plan  # noqa: E402
from test_b_svg import SVG  # noqa: E402

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo"


@pytest.fixture
def proj(tmp_path):
    root = tmp_path / "demo"
    shutil.copytree(DEMO, root)
    return root


@pytest.fixture
def client(proj):
    return TestClient(create_app(proj))


def up(client, name, data, **opt):
    return client.post("/api/plan/import", files={"file": (name, data)}, data={"options": json.dumps(opt)})


def test_import_svg(client, proj):
    before = (proj / "floorplan" / "plan.json").read_bytes()
    r = up(client, "도면.svg", SVG.encode(), px_per_m=100)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["kind"] == "svg" and j["n_rooms"] == 4
    assert len(j["plan"]["rooms"]) == 4 and j["plan"]["source"]["px_per_m"] == 100
    assert j["segments"] and len(j["segments"][0]) == 2 and len(j["segments"][0][0]) == 2
    assert j["bounds"][0] == pytest.approx(0) and j["bounds"][1] == pytest.approx(0)   # recenter 기본
    assert (proj / "floorplan" / "source" / "도면.svg").read_text(encoding="utf-8") == SVG
    assert (proj / "floorplan" / "plan.json").read_bytes() == before                # 저장 안 함
    # 같은 내용 다시 -> 같은 파일, 다른 내용 -> 번호
    up(client, "도면.svg", SVG.encode(), px_per_m=50)
    assert len(list((proj / "floorplan" / "source").iterdir())) == 1
    r2 = up(client, "도면.svg", SVG.replace("B117", "B118").encode(), px_per_m=100)
    assert r2.json()["source_file"] == "floorplan/source/도면_2.svg"


def test_import_svg_needs_scale(client, proj):
    r = up(client, "a.svg", SVG.encode())
    assert r.status_code == 400 and "축척" in r.json()["detail"]
    assert not (proj / "floorplan" / "source" / "a.svg").exists()     # 실패하면 방금 만든 원본 치움
    assert up(client, "a.txt", b"x").status_code == 400
    assert up(client, "a.svg", b"<svg", px_per_m=10).status_code == 400


def test_import_dxf(client, proj, tmp_path):
    pytest.importorskip("ezdxf")
    from test_b_dxf import make_dxf
    f = tmp_path / "p.dxf"
    make_dxf(f)
    j = up(client, "p.dxf", f.read_bytes(), snap_tol=0.03).json()
    assert j["kind"] == "dxf" and j["n_rooms"] == 3, j
    assert j["plan"]["source"]["scale_m"] == 0.001
    # 축척 덮어쓰기 (두 점 실측: 1000 단위 = 2m -> 2배)
    j2 = up(client, "p.dxf", f.read_bytes(), snap_tol=0.06, scale_points=[[0, 0], [1000, 0], 2.0]).json()
    a1 = sorted(r["box"]["size"][0] * r["box"]["size"][1] for r in j["plan"]["rooms"])
    a2 = sorted(r["box"]["size"][0] * r["box"]["size"][1] for r in j2["plan"]["rooms"])
    assert a2 == pytest.approx([4 * a for a in a1], rel=0.02)


def test_import_image(client, proj):
    ok, buf = cv2.imencode(".png", draw_plan())
    # 위에서 본 그림: 네 점 없이
    j = up(client, "plan.png", buf.tobytes(), px_per_m=PPM).json()
    assert j["kind"] == "image" and j["n_rooms"] == 3, j.get("warnings")
    assert j["rectified_png"].startswith("data:image/png;base64,")
    al = j["plan"]["source"]["image_align"]
    assert al["px_per_m"] == PPM and al["flip_y"] is True and al["origin_px"][1] == j["image_size"][1]
    assert not (proj / "floorplan" / "plan.png").exists()
    # 네 점 + plan.png 저장 -> 기존 plan.png 옮김
    (proj / "floorplan" / "plan.png").write_bytes(b"old")
    h, w = draw_plan().shape[:2]
    o = 40
    corners = [[o, o], [o + 12 * PPM, o], [o + 12 * PPM, h - o], [o, h - o]]
    j = up(client, "photo.jpg", buf.tobytes(), corners=corners, size_m=[12, 8], px_per_m=50, save_png=True).json()
    assert j["plan_png"] == "floorplan/plan.png" and j["moved_png"].startswith("floorplan/plan_")
    assert (proj / j["moved_png"]).read_bytes() == b"old"
    img = cv2.imdecode(np.fromfile(str(proj / "floorplan" / "plan.png"), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (400, 600)
    assert j["plan"]["source"]["size_m"] == [12, 8]
    assert (proj / "floorplan" / "source" / "photo.jpg").exists()


def test_validate(client, proj):
    assert client.get("/api/plan/validate").json()["issues"] == []
    plan = client.get("/api/plan").json()
    plan["rooms"].append({"id": "x", "box": {"center": [2, 5], "size": [2, 2], "yaw_deg": 0}, "floor_z": 0, "ceiling_z": 2.6})
    plan["doors"].append({"id": "dx", "rooms": ["x"], "center": [5, 5], "width": 0.9, "height": 2.1})
    iss = client.post("/api/plan/validate", json=plan).json()["issues"]
    kinds = {i["kind"] for i in iss}
    assert {"overlap", "door_off_wall"} <= kinds
    ov = next(i for i in iss if i["kind"] == "overlap")
    assert ov["level"] == "error" and "x" in ov["ids"]
    # 저장은 막지 않고 issues 를 돌려줌
    r = client.put("/api/plan", json=plan)
    assert r.status_code == 200 and any(i["kind"] == "overlap" for i in r.json()["issues"])
    assert any(i["kind"] == "overlap" for i in client.get("/api/plan/validate").json()["issues"])
    assert client.post("/api/plan/validate", json={"rooms": []}).json()["issues"] == []


def test_image_align_roundtrip(client, proj):
    plan = client.get("/api/plan").json()
    plan["source"] = {"kind": "boxes", "image_align": {"px_per_m": 40, "origin_px": [12, 300], "flip_y": True}}
    assert client.put("/api/plan", json=plan).status_code == 200
    assert client.get("/api/plan").json()["source"]["image_align"]["origin_px"] == [12, 300]
