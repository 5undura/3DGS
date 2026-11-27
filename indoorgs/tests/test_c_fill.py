"""fill: 작은 방(바닥에 구멍 하나) 합성 스플랫 + 직접 만든 COLMAP 모델 -> 구멍에만 바탕층."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from indoorgs import plyio, schemas
from indoorgs.fillcore import backdrop as bd
from indoorgs.fillcore import colmap_io
from indoorgs.project import Project
from indoorgs.stages.base import StageContext
from indoorgs.stages.fill import Fill

ROOM = (-3.0, 3.0, -1.5, 1.5)      # x0 x1 y0 y1 (m)
HEIGHT = 2.5
HOLE = (1.0, 0.0, 0.5)             # 바닥 구멍 중심 x, y, 반지름
CAM_H = 1.4
COLOR = (200, 40, 40)              # 사진 색 (RGB)


def surface_points(spacing, rng, hole=None, jitter=0.0):
    x0, x1, y0, y1 = ROOM
    P, N = [], []
    xs, ys = np.arange(x0, x1, spacing), np.arange(y0, y1, spacing)
    X, Y = np.meshgrid(xs, ys)
    X, Y = X.ravel(), Y.ravel()
    keep = np.ones(len(X), bool) if hole is None else np.hypot(X - hole[0], Y - hole[1]) > hole[2]
    P += [np.stack([X[keep], Y[keep], np.zeros(keep.sum())], 1), np.stack([X, Y, np.full(len(X), HEIGHT)], 1)]
    zs = np.arange(0, HEIGHT, spacing)
    for x in (x0, x1):
        Yw, Zw = np.meshgrid(ys, zs); P.append(np.stack([np.full(Yw.size, x), Yw.ravel(), Zw.ravel()], 1))
    for y in (y0, y1):
        Xw, Zw = np.meshgrid(xs, zs); P.append(np.stack([Xw.ravel(), np.full(Xw.size, y), Zw.ravel()], 1))
    P = np.concatenate(P)
    return P + rng.normal(0, jitter, P.shape)


def look_rot(d):
    """세계(z 위) -> 카메라 (x 오른쪽, y 아래, z 앞)."""
    z = np.asarray(d, float); z /= np.linalg.norm(z)
    y = np.array([0, 0, -1.0]); y -= (y @ z) * z; y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return np.stack([x, y, z])


def make_model(folder, image_dir, offset=np.zeros(3)):
    """카메라 24대 (방 가운데 선, 1.4m, 사방을 봄) + 면 위 점. offset 만큼 옮긴 좌표계로 씀."""
    rng = np.random.default_rng(1)
    cams = {1: dict(model=1, w=64, h=48, params=np.array([40.0, 40.0, 32.0, 24.0]))}
    ims, C = [], []
    for i in range(24):
        c = np.array([-2.5 + 5 * (i / 23), 0.0, CAM_H]) + offset
        a = i * 2 * np.pi / 6
        R = look_rot([np.cos(a), np.sin(a), -0.3])
        ims.append(dict(q=plyio.mat_to_quat(R), t=-R @ c, cam=1, name=f"f_{i + 1:05d}_{i * 10:06d}.jpg"))
        C.append(c)
    X = surface_points(0.08, rng) + offset
    colmap_io.write_model(folder, cams, ims, X)
    image_dir.mkdir(parents=True, exist_ok=True)
    img = np.zeros((48, 64, 3), np.uint8); img[:] = COLOR[::-1]
    ok, buf = cv2.imencode(".jpg", img)
    for im in ims:
        (image_dir / im["name"]).write_bytes(buf.tobytes())
    return np.array(C)


def train_splats(names, offset=np.zeros(3), n_rng=2):
    rng = np.random.default_rng(n_rng)
    P = surface_points(0.03, rng, hole=HOLE) + offset
    p = plyio.make(len(P), names, ["comment Exported from Brush"])
    for i, k in enumerate("xyz"):
        p.data[k] = P[:, i]
    p.data["opacity"] = 3.0
    p.data["rot_0"] = 1
    for i in range(3):
        p.data[f"scale_{i}"] = np.log(0.02)
    return p


def make_project(root: Path, plan: bool, names=None):
    root.mkdir(parents=True, exist_ok=True)
    (root / "project.yaml").write_text("sections:\n  - {name: room, video: x.mp4}\n", encoding="utf-8")
    sec = root / "sections" / "room"
    make_model(sec / "data_clean" / "sparse" / "0", sec / "data_clean" / "images")
    plyio.write(sec / "train" / "brush_default" / "splat.ply", train_splats(names or plyio.brush_names(2)))
    if plan:
        x0, x1, y0, y1 = ROOM
        schemas.write_json(root / "floorplan" / "plan.json", {
            "version": 1, "units": "m",
            "rooms": [{"id": "r", "floor": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "floor_z": 0, "ceiling_z": HEIGHT},
                      {"id": "far", "box": {"center": [20, 20], "size": [2, 2]}, "floor_z": 0, "ceiling_z": 2}]})
        schemas.write_json(sec / "colmap" / "priors.json", {"version": 1, "coordinate_system": "CARTESIAN", "frames": []})
    return Project(root)


def run_fill(project):
    sec, paths = project.sections()[0]
    logs = []
    res = Fill().run(StageContext(project, sec, paths, logs.append))
    return res, paths, logs


def check_added(paths, n_train, names):
    out = plyio.read(paths.fill_ply)
    assert out.names == names                          # 학습 PLY 속성 순서 그대로
    add = out.data[n_train:]
    assert len(add) > 20
    P = np.stack([add["x"], add["y"], add["z"]], 1)
    on_floor = P[:, 2] < 0.2
    # 구멍만: 바닥 바탕층은 구멍 근처에만, 벽 없음, 천장은 덮여 있으니 없음
    assert on_floor.all()
    d = np.hypot(P[:, 0] - HOLE[0], P[:, 1] - HOLE[1])
    assert d.max() < HOLE[2] + 0.1
    assert d.min() < 0.1                               # 구멍 가운데도 채워짐
    # 색: 사진 색(빨강). f_dc -> RGB
    rgb = 0.5 + plyio.SH_C0 * np.stack([add["f_dc_0"], add["f_dc_1"], add["f_dc_2"]], 1)
    np.testing.assert_allclose(np.median(rgb, 0), np.array(COLOR) / 255, atol=0.08)
    # 불투명, 납작, 법선 = 위
    assert (out.opacity()[n_train:] > 0.98).all()
    R = plyio.quat_to_mat(np.stack([add[f"rot_{i}"] for i in range(4)], 1).astype(float))
    assert np.abs(R[:, 2, 2]).min() > 0.99
    return out


def test_fill_plan_mode(tmp_path):
    pr = make_project(tmp_path / "p", plan=True)
    res, paths, logs = run_fill(pr)
    assert res.ok, res.message
    info = json.loads((paths.fill / "fill.json").read_text(encoding="utf-8"))
    assert info["mode"] == "plan" and list(info["rooms"]) == ["r"]
    out = check_added(paths, train_splats(plyio.brush_names(2)).n, plyio.brush_names(2))
    # 바탕층은 바닥 면(z=0) 바로 아래
    assert np.all(out.data["z"][-10:] < 0) and np.all(out.data["z"][-10:] > -0.05)


def test_fill_points_mode_lichtfeld(tmp_path):
    names = plyio.lichtfeld_names(2)
    pr = make_project(tmp_path / "p", plan=False, names=names)
    res, paths, logs = run_fill(pr)
    assert res.ok, res.message
    assert json.loads((paths.fill / "fill.json").read_text(encoding="utf-8"))["mode"] == "points"
    check_added(paths, train_splats(names).n, names)


def test_fill_keep_walls_when_not_holes_only(tmp_path):
    """holes_only=False: 덮이지 않은 벽 바탕층도 들어감 (학습 스플랫에서 벽 하나를 뺌)."""
    pr = make_project(tmp_path / "p", plan=True)
    sec, paths = pr.sections()[0]
    t = plyio.read(paths.train_run("brush", "default") / "splat.ply")
    keep = t.data["x"] > ROOM[0] + 0.05                   # x0 벽 제거
    plyio.write(paths.train_run("brush", "default") / "splat.ply", plyio.GaussianPly(t.data[keep], t.comments))
    pr.cfg.fill.holes_only = False
    res, paths, _ = run_fill(pr)
    assert res.ok, res.message
    add = plyio.read(paths.fill_ply).data[keep.sum():]
    wall = add[(add["z"] > 0.3) & (add["z"] < HEIGHT - 0.3)]
    assert len(wall) > 100 and np.all(wall["x"] < ROOM[0] + 0.1)


def test_fill_missing_inputs(tmp_path):
    root = tmp_path / "p"; root.mkdir()
    (root / "project.yaml").write_text("sections:\n  - {name: room, video: x.mp4}\n", encoding="utf-8")
    res, _, _ = run_fill(Project(root))
    assert not res.ok and "학습 결과 없음" in res.message


def test_points_in_polygon():
    poly = [(0, 0), (2, 0), (2, 1), (0, 1)]
    x = np.array([1, 3, 0.5]); y = np.array([0.5, 0.5, 0.9])
    assert list(bd.points_in_polygon(x, y, poly)) == [True, False, True]
