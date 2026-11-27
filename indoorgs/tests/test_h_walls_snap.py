"""walls/snap: 스플랫을 도면 면에 붙이기 (합성 데이터)."""
import numpy as np
import pytest

from indoorgs import plyio
from indoorgs.schemas import parse_plan
from indoorgs.walls import snap
from indoorgs.walls.planes import planes_from_plan

THICK = 0.004


def make_plan():
    return parse_plan({"version": 1, "units": "m", "rooms": [
        {"id": "A", "floor": [[0, 0], [6, 0], [6, 3], [0, 3]], "floor_z": 0.0, "ceiling_z": 2.6},
        {"id": "B", "floor": [[6.2, 0], [10, 0], [10, 3], [6.2, 3]], "floor_z": 0.0, "ceiling_z": 2.6}],
        "doors": [{"id": "d_ab", "rooms": ["A", "B"], "center": [6.1, 1.5], "width": 1.0, "height": 2.1},
                  {"id": "d_out", "rooms": ["A"], "center": [2.0, 0.0], "width": 1.0, "height": 2.1}]})


def rand_quat(rng, n):
    q = rng.normal(size=(n, 4))
    return q * rng.uniform(0.5, 2.0, (n, 1))          # 일부러 정규화 안 함


def surface_points(rng, n):
    """방 A 의 여섯 면 위 무작위 점 + 법선 방향 ±3 cm."""
    face = rng.integers(0, 6, n)
    P = np.column_stack([rng.uniform(0.1, 5.9, n), rng.uniform(0.1, 2.9, n), rng.uniform(0.1, 2.5, n)])
    P[face == 0, 1] = 0.0
    P[face == 1, 1] = 3.0
    P[face == 2, 0] = 0.0
    P[face == 3, 0] = 6.0
    P[face == 4, 2] = 0.0
    P[face == 5, 2] = 2.6
    return P + rng.uniform(-0.03, 0.03, (n, 3))


FLOATERS = {  # 이름: (점, 지워져야 하나)
    "behind_north": ([3.0, 3.5, 1.0], True),
    "behind_north_20cm": ([3.0, 3.2, 1.0], True),
    "in_wall_body_near_B": ([6.18, 0.5, 1.0], False),   # 방 B 의 margin 안
    "behind_door_out": ([2.0, -0.6, 1.0], False),       # 문 구멍 너머 도면 밖
    "neighbour_room": ([8.0, 0.5, 1.0], False),          # 문 아닌 벽 뒤지만 방 B 안
    "through_door_ab": ([8.0, 1.5, 1.0], False),
    "below_floor": ([3.0, 1.5, -0.5], True),
    "above_ceiling": ([3.0, 1.5, 3.0], True),
    "near_wall_inside_margin": ([6.1, 0.5, 1.0], False),  # 벽 뒤 10 cm < margin
    "just_outside_west": ([-0.05, 1.0, 1.0], False),      # 방 B 서쪽 벽(먼 벽) 뒤지만 A 바로 바깥
}


def build(names, n=4000, seed=1):
    rng = np.random.default_rng(seed)
    P = surface_points(rng, n)
    big = np.array([[3.0, 2.99, 1.2], [0.01, 1.5, 1.0]])    # 면 안쪽 축이 큰 배경 덩어리
    inner = rng.uniform([1, 1, 1], [5, 2, 1.8], (50, 3))      # 방 가운데 (배정 안 됨)
    fl = np.array([v[0] for v in FLOATERS.values()], float)
    P = np.vstack([P, big, inner, fl])
    N = len(P)
    ply = plyio.make(N, names)
    for k in names:
        ply.data[k] = rng.normal(size=N)
    for i, c in enumerate("xyz"):
        ply.data[c] = P[:, i]
    q = rand_quat(rng, N)
    for i in range(4):
        ply.data[f"rot_{i}"] = q[:, i]
    for i in range(3):
        ply.data[f"scale_{i}"] = np.log(rng.uniform(0.003, 0.08, N))
    nb = n
    ply.data["scale_0"][nb:nb + 2] = np.log(1.5)
    ply.data["scale_1"][nb:nb + 2] = np.log(1.2)
    ply.data["scale_2"][nb:nb + 2] = np.log(0.9)
    ply.data["opacity"] = rng.uniform(-1, 4, N)
    return ply, {"big": np.arange(nb, nb + 2), "floaters": np.arange(N - len(fl), N)}


LAYOUTS = [plyio.brush_names(2), plyio.lichtfeld_names(2)]


@pytest.mark.parametrize("names", LAYOUTS, ids=["brush", "lichtfeld"])
def test_snap(names):
    plan = make_plan()
    planes = planes_from_plan(plan, rooms=["A"])
    ply, ids = build(names)
    src = ply.data.copy()
    out, info = snap.snap_to_planes(ply, planes, band=0.06, thickness=THICK, color_tol=None)
    np.testing.assert_array_equal(ply.data, src)                      # 입력은 안 바뀜
    assert out.names == names

    idx, _ = snap.assign(ply.xyz, planes, 0.06)
    touched = idx >= 0
    touched[ids["big"]] = False
    assert idx[ids["big"]].min() >= 0                                  # 배정은 되지만 안 건드림
    assert info["n_skipped_big"] == 2
    assert info["n_moved"] == info["n_flattened"] == touched.sum() > 3000
    assert sum(info["n_assigned"].values()) == (idx >= 0).sum()

    # 안 건드린 것은 비트 그대로
    np.testing.assert_array_equal(out.data[~touched], src[~touched])
    # 색·불투명도·법선은 모두 비트 그대로
    for k in names:
        if k.startswith(("f_dc", "f_rest", "opacity", "nx", "ny", "nz")):
            np.testing.assert_array_equal(out.data[k], src[k])

    t = np.nonzero(touched)[0]
    n = np.array([planes[j].n for j in idx[t]])
    dd = np.array([planes[j].d for j in idx[t]])
    assert np.abs(np.einsum("mi,mi->m", out.xyz[t], n) - dd).max() < 1e-5
    q_in = np.stack([src[f"rot_{i}"][t] for i in range(4)], 1).astype(float)
    q_out = np.stack([out.data[f"rot_{i}"][t] for i in range(4)], 1).astype(float)
    assert np.abs(np.linalg.norm(q_out, axis=1) - 1).max() < 1e-6
    R0, R1 = plyio.quat_to_mat(q_in), plyio.quat_to_mat(q_out)
    k = np.argmax(np.abs(np.einsum("mik,mi->mk", R0, n)), 1)
    ar = np.arange(len(t))
    assert np.abs(np.einsum("mi,mi->m", R1[ar, :, k], n)).min() > 0.9999
    assert np.linalg.det(R1).min() > 0.999
    s_in = np.stack([src[f"scale_{i}"][t] for i in range(3)], 1)
    s_out = np.stack([out.data[f"scale_{i}"][t] for i in range(3)], 1)
    assert np.exp(s_out[ar, k].astype(float)).max() <= THICK * (1 + 1e-6)
    np.testing.assert_array_equal(s_out[ar, k], np.minimum(s_in[ar, k], np.float32(np.log(THICK))))
    other = np.ones((len(t), 3), bool)
    other[ar, k] = False
    np.testing.assert_array_equal(s_out[other], s_in[other])


def test_move_only_and_flatten_only():
    planes = planes_from_plan(make_plan(), rooms=["A"])
    ply, _ = build(plyio.brush_names(2), n=500)
    a, ia = snap.snap_to_planes(ply, planes, flatten=False, band=0.06, color_tol=None)
    np.testing.assert_array_equal(a.data["rot_0"], ply.data["rot_0"])
    assert ia["n_flattened"] == 0 and ia["n_moved"] > 0
    b, ib = snap.snap_to_planes(ply, planes, move=False, band=0.06, color_tol=None)
    np.testing.assert_array_equal(b.data["x"], ply.data["x"])
    assert ib["n_moved"] == 0 and ib["n_flattened"] > 0


@pytest.mark.parametrize("names", LAYOUTS, ids=["brush", "lichtfeld"])
def test_remove_behind_walls(names):
    plan = make_plan()
    planes = planes_from_plan(plan, rooms=["A"])
    ply, ids = build(names)
    out, info = snap.remove_behind_walls(ply, planes, plan, margin=0.15)
    fl = ids["floaters"]
    rm_expected = np.zeros(ply.n, bool)
    rm_expected[fl] = [v[1] for v in FLOATERS.values()]
    kept = ply.data[~rm_expected]
    np.testing.assert_array_equal(out.data, kept)                  # 정확히 진짜 떠다니는 것만
    assert info["removed_by"] == {"wall": 2, "floor": 1, "ceiling": 1}
    assert info["n_removed"] == 4 and out.names == names
    # 다각형 목록만 줘도 (2D) 같은 결과
    out2, _ = snap.remove_behind_walls(ply, planes, [r.floor for r in plan.rooms], margin=0.15)
    assert out2.n == out.n
    # 두 방의 면 전부: A 벽 근처 점이 B 의 먼 벽 뒤로 잡혀 지워지면 안 됨
    out3, info3 = snap.remove_behind_walls(ply, planes_from_plan(plan), plan, margin=0.15)
    np.testing.assert_array_equal(out3.data, kept)


def test_flatness_and_file(tmp_path):
    plan = make_plan()
    planes = planes_from_plan(plan, rooms=["A"])
    ply, _ = build(plyio.brush_names(2))
    f_in = plyio.write(tmp_path / "in.ply", ply)
    before = snap.splat_flatness(ply, planes)
    assert before["all"]["count"] > 1000 and before["wall"]["count"] > 0
    info = snap.snap_file(f_in, tmp_path / "out.ply", planes, plan, thickness=THICK, band=0.06, color_tol=None)
    after = info["flatness_after"]
    assert after["all"]["rms_dist"] < before["all"]["rms_dist"]
    assert after["all"]["rms_thick"] < before["all"]["rms_thick"]
    assert after["wall"]["within_1cm"] > before["wall"]["within_1cm"]
    assert info["flatness_before"]["all"]["rms_dist"] == pytest.approx(before["all"]["rms_dist"])
    assert info["remove"]["n_removed"] == info["n_removed"] == 4
    assert info["n_moved"] > 3000 and after["rms_m"] < before["rms_m"]
    import json
    json.dumps(info)                                                  # snap.json 으로 쓸 수 있어야 함
    out = plyio.read(tmp_path / "out.ply")
    assert out.names == ply.names and out.n == ply.n - 4
    np.testing.assert_array_equal(plyio.read(f_in).data, ply.data)    # 원본 그대로
    with pytest.raises(ValueError):
        snap.snap_file(f_in, f_in, planes, plan)
    with pytest.raises(TypeError):
        snap.snap_file(f_in, tmp_path / "o2.ply", planes, plan, bogus=1)


def test_color_gate_keeps_wall_objects():
    """지운 의견: 벽 ±3 cm 안이라도 벽 색과 다른 것(액자 등)은 붙이지 않아 입체감 유지."""
    plan = make_plan()
    planes = planes_from_plan(plan, ["A"])
    rng = np.random.default_rng(5)
    n_wall, n_frame = 3000, 200
    ply = plyio.make(n_wall + n_frame, plyio.brush_names(1))
    d = ply.data
    x = rng.uniform(0.5, 5.5, n_wall + n_frame)
    z = rng.uniform(0.3, 2.3, n_wall + n_frame)
    y = 3.0 - rng.uniform(-0.02, 0.02, n_wall + n_frame)        # 북쪽 벽 ±2 cm
    # 액자: x 2.0~2.5, z 1.2~1.6, 벽에서 2.5 cm 앞, 진한 갈색
    x[n_wall:] = rng.uniform(2.0, 2.5, n_frame)
    z[n_wall:] = rng.uniform(1.2, 1.6, n_frame)
    y[n_wall:] = 3.0 - 0.025
    d["x"], d["y"], d["z"] = x, y, z
    d["rot_0"] = 1.0
    for k in range(3):
        d[f"scale_{k}"] = np.log(0.02)
    d["opacity"] = 3.0
    wall = (np.array([0.85, 0.83, 0.78]) - 0.5) / plyio.SH_C0           # 흰 벽 (조금 흔들림)
    frame = (np.array([0.35, 0.22, 0.12]) - 0.5) / plyio.SH_C0
    for c in range(3):
        d[f"f_dc_{c}"][:n_wall] = wall[c] + rng.normal(0, 0.05, n_wall)
        d[f"f_dc_{c}"][n_wall:] = frame[c]
    out, info = snap.snap_to_planes(ply, planes)                          # 기본값 band 3 cm, color_tol 0.12
    assert info["n_skipped_color"] >= n_frame
    np.testing.assert_array_equal(out.data[n_wall:], d[n_wall:])          # 액자는 그대로
    moved = np.abs(out.data["y"][:n_wall] - 3.0) < 1e-5
    assert moved.mean() > 0.95                                             # 벽은 거의 다 붙음
