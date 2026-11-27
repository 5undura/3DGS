"""walls.refine (정합 점으로 면 다시 맞추기) · walls.seed (면 위 초기 점 심기) 합성 테스트."""
import math

import cv2
import numpy as np
import pytest

from indoorgs import schemas
from indoorgs.colmap import model as cm
from indoorgs.walls import planes as wp
from indoorgs.walls.refine import assign_points, edge_distance, flatness, refine_planes, room_widths
from indoorgs.walls.seed import seed_points, write_seeded_model

H = 2.6


def make_plan(door_ab=True, door_a0=False, room_b=True):
    rooms = [{"id": "A", "floor": [[0, 0], [6, 0], [6, 3], [0, 3]], "floor_z": 0.0, "ceiling_z": H}]
    if room_b:
        rooms.append({"id": "B", "floor": [[6, 0], [9, 0], [9, 3], [6, 3]], "floor_z": 0.0, "ceiling_z": H})
    doors = []
    if door_ab:
        doors.append({"id": "d1", "rooms": ["A", "B"], "center": [6, 1.5], "width": 1.0, "height": 2.1})
    if door_a0:
        doors.append({"id": "d0", "rooms": ["A"], "center": [3, 0], "width": 1.0, "height": 2.1})
    return schemas.parse_plan({"version": schemas.VERSION, "units": "m", "rooms": rooms, "doors": doors})


def by_id(planes):
    return {p.id: p for p in planes}


def rot_z(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])


# ---------------- refine ----------------
def sample_on(p, rng, n, u_rng=(0, 1), v_rng=(0, 1), sigma=0.01, outlier=0.10, search=0.30):
    u = rng.uniform(*u_rng, n) * p.size[0]
    v = rng.uniform(*v_rng, n) * p.size[1]
    X = p.point(u, v) + rng.normal(0, sigma, n)[:, None] * p.n
    k = int(outlier * n)
    X[:k] += rng.uniform(-search, search, k)[:, None] * p.n + rng.normal(0, 0.05, (k, 3))
    return X


def build_refine_case():
    plan = make_plan(door_ab=False, door_a0=True, room_b=False)
    P = by_id(wp.planes_from_plan(plan))
    rng = np.random.default_rng(3)
    true = {
        "A:w0": P["A:w0"].shifted(0.12),                                   # 도면보다 12 cm 안쪽
        "A:w1": P["A:w1"].shifted(0.04, rot_z(1.0) @ P["A:w1"].n),         # 4 cm + 1° 기울어짐
        "A:w2": P["A:w2"].shifted(-0.05),
        "A:w3": P["A:w3"],
        "A:floor": P["A:floor"].shifted(-0.03),
        "A:ceiling": P["A:ceiling"].shifted(0.02),
    }
    pts = [sample_on(true["A:w0"], rng, 1500, (0.0, 0.4)),       # 문(u 2.5~3.5) 앞까지만
           sample_on(true["A:w1"], rng, 1500, (0.1, 0.9)),
           sample_on(true["A:w2"], rng, 1500, (0.3, 0.9)),
           sample_on(true["A:w3"], rng, 800),
           sample_on(true["A:floor"], rng, 2000, (0.1, 0.7), (0.1, 0.9)),
           sample_on(true["A:ceiling"], rng, 15)]                 # 너무 적음 -> plan
    # 문 구멍 안 (실제 벽 위치 근처) 점
    u = rng.uniform(2.6, 3.4, 200)
    v = rng.uniform(0.5, 2.0, 200)
    hole = P["A:w0"].point(u, v) + 0.12 * P["A:w0"].n
    xyz = np.concatenate(pts + [hole])
    return P, true, xyz, hole


@pytest.fixture(scope="module")
def refine_case():
    return build_refine_case()


def face_center(p):
    return p.point(p.size[0] / 2, p.size[1] / 2)


def test_refine_recovers_shift_and_tilt(refine_case):
    P, true, xyz, _ = refine_case
    planes = list(P.values())
    out, info = refine_planes(planes, xyz)
    R, I = by_id(out), {d["id"]: d for d in info}
    for pid in ("A:w0", "A:w1", "A:w2", "A:w3", "A:floor"):
        assert I[pid]["source"] == "fit", I[pid]
        assert abs(R[pid].signed_dist(face_center(true[pid]))) < 0.015, pid
        assert wp.angle_deg(R[pid].n, true[pid].n) < 0.3, pid
        assert I[pid]["rms_after_m"] < 0.02
    assert abs(I["A:w0"]["shift_m"] - 0.12) < 0.015
    assert abs(I["A:w2"]["shift_m"] + 0.05) < 0.015
    assert abs(I["A:w1"]["tilt_deg"] - 1.0) < 0.3
    assert I["A:w1"]["rms_before_m"] > I["A:w1"]["rms_after_m"]
    assert I["A:ceiling"]["source"] == "plan" and "reason" in I["A:ceiling"]
    assert R["A:ceiling"] is P["A:ceiling"]
    # 폭: 도면 3 m
    W = {(w["a"], w["b"]): w for w in room_widths(planes, out)}
    assert abs(W[("A:w0", "A:w2")]["plan_m"] - 3.0) < 1e-9
    assert abs(W[("A:w0", "A:w2")]["fit_m"] - (3.0 - 0.12 + 0.05)) < 0.02   # w2 는 바깥으로 5 cm
    assert ("A:w1", "A:w3") in W


def test_door_hole_points_not_assigned(refine_case):
    P, _, xyz, hole = refine_case
    assert (assign_points(list(P.values()), hole) == -1).all()


def test_refine_low_coverage_and_bound():
    plan = make_plan(door_ab=False, room_b=False)
    P = by_id(wp.planes_from_plan(plan))
    rng = np.random.default_rng(5)
    # 많지만 작은 한 곳에 몰린 점 -> 덮임 부족
    X1 = sample_on(P["A:w2"], rng, 500, (0.45, 0.5), (0.4, 0.5), outlier=0)
    # 탐색 한계 근처 (0.29 m) -> plan + at_bound
    X2 = sample_on(P["A:w3"].shifted(0.29), rng, 800, outlier=0, sigma=0.003)
    out, info = refine_planes(list(P.values()), np.concatenate([X1, X2]))
    I = {d["id"]: d for d in info}
    assert I["A:w2"]["source"] == "plan" and I["A:w2"]["coverage"] < 0.25
    assert I["A:w3"]["source"] == "plan" and I["A:w3"].get("at_bound")


def test_flatness():
    plan = make_plan(door_ab=False, room_b=False)
    planes = wp.planes_from_plan(plan)
    P = by_id(planes)
    rng = np.random.default_rng(7)
    X = sample_on(P["A:w2"], rng, 1000, sigma=0.01, outlier=0)
    f = flatness(X, planes, band=0.05)
    assert f["planes"]["A:w2"]["n"] > 990
    assert 0.008 < f["planes"]["A:w2"]["rms_m"] < 0.012
    assert f["overall"]["n"] >= f["planes"]["A:w2"]["n"]


# ---------------- seed ----------------
W_IMG, H_IMG = 64, 48
FX = FY = 40.0


def look_rot(d):
    z = np.asarray(d, float); z /= np.linalg.norm(z)
    y = np.array([0, 0, -1.0]); y -= (y @ z) * z; y /= np.linalg.norm(y)
    return np.stack([np.cross(y, z), y, z])


def rot2q(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


def colors_for(planes):
    rng = np.random.default_rng(11)
    return {p.id: tuple(int(c) for c in rng.integers(30, 230, 3)) for p in planes}


def render(planes, colors, R, C):
    """광선 쏘기: 가장 가까운 앞면(카메라가 면 안쪽) 색. RGB 이미지."""
    j, i = np.meshgrid(np.arange(W_IMG), np.arange(H_IMG))
    d = np.stack([(j + 0.5 - W_IMG / 2) / FX, (i + 0.5 - H_IMG / 2) / FY, np.ones_like(j, float)], -1).reshape(-1, 3)
    d = d @ R           # 카메라 -> 세계 (R^T d)
    best = np.full(len(d), np.inf)
    img = np.zeros((len(d), 3), np.uint8)
    for p in planes:
        den = d @ p.n
        if p.signed_dist(C) <= 0:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (p.d - p.n @ C) / den
        ok = (den < 0) & (t > 1e-6) & (t < best)
        k = np.flatnonzero(ok)
        hit = p.contains_uv(p.uv(C + t[k, None] * d[k]))
        k = k[hit]
        best[k] = t[k]
        img[k] = colors[p.id]
    return img.reshape(H_IMG, W_IMG, 3)


def make_scene(tmp_path, planes, views):
    image_dir = tmp_path / "사진"
    image_dir.mkdir(exist_ok=True)
    colors = colors_for(planes)
    cams = {1: cm.Camera(1, 1, W_IMG, H_IMG, np.array([FX, FY, W_IMG / 2, H_IMG / 2]))}
    images = {}
    for k, (C, look) in enumerate(views, 1):
        C = np.asarray(C, float)
        R = look_rot(look)
        name = f"f_{k:03d}.png"
        ok, buf = cv2.imencode(".png", render(planes, colors, R, C)[:, :, ::-1])
        buf.tofile(str(image_dir / name))
        images[k] = cm.Image(k, rot2q(R), -R @ C, 1, name)
    pts = {1: cm.Point3D(1, np.array([3.0, 1.5, 1.0]), np.array([1, 2, 3], np.uint8), 0.7,
                         np.array([[1, 0], [2, 0]], np.uint32))}
    return cm.Model(cams, images, pts), image_dir, colors


def views_forward():
    return [((x, 0.6, 1.4), (dx, 1.0, 0.0)) for x in (1.5, 2.5, 3.5, 4.5) for dx in (-0.2, 0.2)]


def check_on_planes_and_colors(planes, xyz, rgb, colors, info):
    P = by_id(planes)
    start = 0
    for pid, c in info["planes"].items():
        n = c["added"]
        if n == 0:
            continue
        # seed 는 면 순서대로 돌려줌
        X, Cc = xyz[start:start + n], rgb[start:start + n].astype(int)
        start += n
        assert np.abs(P[pid].signed_dist(X)).max() < 1e-6, pid
        close = np.abs(Cc - np.array(colors[pid])).max(1) <= 3
        uv = P[pid].uv(X)
        inner = edge_distance(P[pid], uv) > 0.15      # 64x48 사진: 모서리·문틀 근처는 색이 섞임
        for (u0, u1, v0, v1) in P[pid].holes:
            inner &= ~((uv[:, 0] > u0 - 0.15) & (uv[:, 0] < u1 + 0.15) & (uv[:, 1] < v1 + 0.15))
        need = 0.8 if pid.startswith("B:") else 0.97   # 문으로 본 옆 방: 문틀 실루엣 근처도 섞임
        assert close[inner].mean() > need, (pid, close[inner].mean())
    assert start == len(xyz)


def test_seed_occlusion_and_out_of_view(tmp_path):
    planes = wp.planes_from_plan(make_plan(door_ab=False))
    model, image_dir, colors = make_scene(tmp_path, planes, views_forward())
    xyz, rgb, info = seed_points(planes, model, image_dir, spacing=0.15)
    pl = info["planes"]
    assert info["n_added"] == len(xyz) == len(rgb) > 100
    assert pl["A:w2"]["added"] > 0.5 * pl["A:w2"]["sampled"]     # 정면 벽
    assert pl["A:w0"]["added"] == 0                               # 뒤쪽 벽: 안 보임
    assert all(pl[p.id]["added"] == 0 for p in planes if p.room == "B")   # 문 없는 옆 방: 가림
    assert info["n_rejected_visibility"] == info["n_sampled"] - info["n_added"]
    check_on_planes_and_colors(planes, xyz, rgb, colors, info)


def test_seed_through_door(tmp_path):
    planes = wp.planes_from_plan(make_plan(door_ab=True))
    views = views_forward() + [((x, y, 1.4), (1.0, dy, 0.0)) for x in (2.0, 3.0, 4.0)
                               for y, dy in ((1.3, 0.1), (1.7, -0.1))]
    model, image_dir, colors = make_scene(tmp_path, planes, views)
    xyz, rgb, info = seed_points(planes, model, image_dir, spacing=0.12)
    pl = info["planes"]
    assert pl["B:w3"]["added"] == 0                                # 옆 방 쪽 같은 벽 뒷면
    assert 0 < pl["B:w1"]["added"] < pl["B:w1"]["sampled"]         # 문으로만 보이는 먼 벽 일부
    P = by_id(planes)
    a1 = P["A:w1"]
    on_a1 = np.abs(a1.signed_dist(xyz)) < 1e-6
    uv = a1.uv(xyz[on_a1])
    u0, u1, v0, v1 = a1.holes[0]
    assert len(uv) > 0
    assert not ((uv[:, 0] > u0) & (uv[:, 0] < u1) & (uv[:, 1] > v0) & (uv[:, 1] < v1)).any()
    check_on_planes_and_colors(planes, xyz, rgb, colors, info)

    # 새 모델 쓰기 + 다시 읽기
    out = write_seeded_model(model, xyz, rgb, tmp_path / "seeded" / "sparse" / "0")
    back = cm.read_model(out)
    assert len(back.points) == 1 + len(xyz)
    assert back.points[1].error == pytest.approx(0.7) and len(back.points[1].track) == 2
    ids = sorted(back.points)[1:]
    assert ids[0] == 2
    bx = np.array([back.points[i].xyz for i in ids])
    bc = np.array([back.points[i].rgb for i in ids])
    assert np.allclose(bx, xyz) and (bc == rgb).all()
    assert all(len(back.points[i].track) == 0 and back.points[i].error == 0 for i in ids)
    assert set(back.images) == set(model.images) and back.images[3].name == model.images[3].name
    assert len(model.points) == 1                                  # 원래 모델은 그대로
    with pytest.raises(FileExistsError):
        write_seeded_model(model, xyz, rgb, out)


def test_seed_skip_near_existing(tmp_path):
    planes = wp.planes_from_plan(make_plan(door_ab=False, room_b=False))
    model, image_dir, _ = make_scene(tmp_path, planes, views_forward())
    P = by_id(planes)
    model.points[1].xyz = P["A:w2"].point(3.0, 1.3)
    xyz, _, info = seed_points(planes, model, image_dir, spacing=0.15, skip_near_existing=0.3)
    assert info["n_skipped_near_existing"] > 0
    assert np.linalg.norm(xyz - model.points[1].xyz, axis=1).min() > 0.3
