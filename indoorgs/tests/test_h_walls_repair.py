"""walls/repair: 면의 비치는 곳·빈 곳을 둘레 스플랫 통째 복사로 메우기 (합성 데이터)."""
import numpy as np
import pytest

from indoorgs import plyio
from indoorgs.schemas import parse_plan
from indoorgs.walls import repair
from indoorgs.walls.planes import planes_from_plan

CELL = 0.05
MIN_COVER = 0.5
HOLE = (3.0, 1.2, 0.25)          # 빈 원 (u, v, 반지름)
THIN = (0.5, 2.5, 0.6, 1.2)      # 반투명 사각형 (u0, u1, v0, v1)


def make_planes():
    plan = parse_plan({"version": 1, "units": "m", "rooms": [
        {"id": "A", "floor": [[0, 0], [4, 0], [4, 3], [0, 3]], "floor_z": 0.0, "ceiling_z": 2.6}],
        "doors": [{"id": "d", "rooms": ["A"], "center": [1.5, 0.0], "width": 1.0, "height": 2.1}]})
    planes = planes_from_plan(plan)
    wall = next(p for p in planes if p.id == "A:w0")     # y = 0 벽, n = +y, 문 구멍 u 1..2, v 0..2.1
    return planes, wall


def build(names, seed=0, spacing=0.03):
    """A:w0 위 납작한 가우시안 격자. 문 구멍과 HOLE 은 비우고 THIN 은 거의 투명하게."""
    planes, wall = make_planes()
    rng = np.random.default_rng(seed)
    uu, vv = np.meshgrid(np.arange(0.015, wall.size[0], spacing), np.arange(0.015, wall.size[1], spacing))
    U, V = uu.ravel() + rng.normal(0, 0.003, uu.size), vv.ravel() + rng.normal(0, 0.003, uu.size)
    keep = wall.contains_uv(np.stack([U, V], 1)) & ((U - HOLE[0]) ** 2 + (V - HOLE[1]) ** 2 > HOLE[2] ** 2)
    U, V = U[keep], V[keep]
    N = len(U)
    ply = plyio.make(N, names)
    for k in names:
        ply.data[k] = rng.normal(0, 0.3, N)
    P = wall.point(U, V) + wall.n * rng.normal(0, 0.002, N)[:, None]
    for i, c in enumerate("xyz"):
        ply.data[c] = P[:, i]
    # 회전: 법선(+y) 축이 가우시안 z 축이 되게 x 축으로 -90도, 그 다음 법선 둘레 무작위
    th = rng.uniform(0, np.pi, N)
    qn = np.stack([np.cos(th / 2), np.zeros(N), np.sin(th / 2), np.zeros(N)], 1)     # 법선(+y) 둘레
    q0 = np.array([np.cos(-np.pi / 4), np.sin(-np.pi / 4), 0, 0])                    # z -> +y
    q = plyio.quat_mul(qn, np.broadcast_to(q0, (N, 4)).copy())
    for i in range(4):
        ply.data[f"rot_{i}"] = q[:, i]
    ply.data["scale_0"] = np.log(rng.uniform(0.018, 0.03, N))
    ply.data["scale_1"] = np.log(rng.uniform(0.018, 0.03, N))
    ply.data["scale_2"] = np.log(0.002)
    thin = (U > THIN[0]) & (U < THIN[1]) & (V > THIN[2]) & (V < THIN[3])
    ply.data["opacity"] = np.where(thin, -4.0, rng.uniform(2.0, 4.0, N))
    return ply, planes, wall


LAYOUTS = [plyio.brush_names(2), plyio.lichtfeld_names(2)]


def _rows_key(data, names):
    keys = [k for k in names if k.startswith(("scale_", "rot_", "f_rest_", "opacity"))]
    return [tuple(r) for r in np.stack([data[k] for k in keys], 1).view(np.uint32)]


@pytest.mark.parametrize("names", LAYOUTS, ids=["brush", "lichtfeld"])
def test_repair_wall(names):
    ply, planes, wall = build(names)
    src = ply.data.copy()
    N = len(src)
    cov0, inside = repair.coverage_map(ply, wall, CELL)
    assert (inside & (cov0 < MIN_COVER)).sum() * CELL ** 2 > 0.6     # 빈 원 + 반투명 사각형

    out, info = repair.repair_faces(ply, [wall], cell=CELL, min_cover=MIN_COVER, rng=1)
    np.testing.assert_array_equal(ply.data, src)                      # 입력 안 바뀜
    assert out.names == names
    np.testing.assert_array_equal(out.data[:N], src)                  # 원본 비트 그대로
    new = out.data[N:]
    assert len(new) == info["n_added"] > 100

    cov1, _ = repair.coverage_map(out, wall, CELL)
    assert not (inside & (cov1 < MIN_COVER)).any()                    # 면 안에 덮임 모자란 셀 없음
    fi = info["faces"]["A:w0"]
    assert fi["hole_m2_before"] > 0.6 and fi["hole_m2_after"] == 0 and fi["cover_after"] == 1.0
    assert fi["cover_before"] < 1.0

    P = np.stack([new["x"], new["y"], new["z"]], 1).astype(float)
    assert np.abs(wall.signed_dist(P)).max() < 1e-5                   # 면 위
    uv = wall.uv(P)
    assert wall.contains_uv(uv).all()                                 # 문 구멍·면 밖에는 안 놓음
    door = (uv[:, 0] > 1.0) & (uv[:, 0] < 2.0) & (uv[:, 1] < 2.1)
    assert not door.any()
    # 새 것은 구멍 쪽에만
    in_hole = (uv[:, 0] - HOLE[0]) ** 2 + (uv[:, 1] - HOLE[1]) ** 2 < (HOLE[2] + 0.15) ** 2
    in_thin = (uv[:, 0] > THIN[0] - 0.15) & (uv[:, 0] < THIN[1] + 0.15) & (uv[:, 1] > THIN[2] - 0.15) & \
              (uv[:, 1] < THIN[3] + 0.15)
    assert (in_hole | in_thin).mean() > 0.95
    assert in_hole.sum() > 20 and in_thin.sum() > 20

    # 통째 복사: scale·rot·SH·opacity 가 어떤 원본 하나와 비트까지 같음 (불투명한 기증자만)
    orig = set(_rows_key(src, names))
    assert all(k in orig for k in _rows_key(new, names))
    assert repair._sigmoid(new["opacity"]).min() >= repair.DONOR_MIN_ALPHA
    if "nx" in names:
        orig_n = {tuple(r) for r in np.stack([src[k] for k in ("nx", "ny", "nz")], 1).view(np.uint32)}
        assert all(tuple(r) in orig_n for r in np.stack([new[k] for k in ("nx", "ny", "nz")], 1).view(np.uint32))
    # DC 는 저주파 색장만큼만 옮김 (원본 DC 범위 안팎)
    assert np.abs(new["f_dc_0"]).max() < np.abs(src["f_dc_0"]).max() + 0.5


def test_no_tone_shift_keeps_dc():
    ply, planes, wall = build(plyio.brush_names(2), seed=3)
    N = ply.n
    out, info = repair.repair_faces(ply, [wall], color_smooth=0.0, rng=2)
    keys = ["f_dc_0", "f_dc_1", "f_dc_2", "scale_0", "rot_0", "f_rest_0"]
    orig = {tuple(r) for r in np.stack([ply.data[k] for k in keys], 1).view(np.uint32)}
    assert all(tuple(r) in orig for r in np.stack([out.data[k][N:] for k in keys], 1).view(np.uint32))


def test_big_empty_faces_skipped_and_others_untouched():
    """나머지 면(벽 3개·바닥·천장)은 스플랫이 없음 -> 하나의 큰 구멍 > max_hole_m2 -> 안 메움. 기증자는 면을 넘지 않음."""
    ply, planes, wall = build(plyio.lichtfeld_names(2), seed=4)
    out, info = repair.repair_faces(ply, planes, rng=0)
    for p in planes:
        fi = info["faces"][p.id]
        if p.id == "A:w0":
            assert fi["n_added"] > 0 and fi["hole_m2_after"] == 0
        else:
            assert fi["n_added"] == 0 and fi["skipped_large_m2"] > 4.0 and fi["hole_m2_after"] > 4.0
    P = out.xyz[ply.n:]
    assert np.abs(wall.signed_dist(P)).max() < 1e-5
    # 제한을 풀어도 기증자가 없는 면은 못 메움
    _, info2 = repair.repair_faces(ply, planes, max_hole_m2=None, rng=0)
    assert info2["faces"]["A:floor"]["n_added"] == 0 and info2["faces"]["A:floor"]["no_donor_m2"] > 4.0


def test_cameras_visibility():
    ply, planes, wall = build(plyio.brush_names(2), seed=5)
    inside_cam = np.array([[2.5, 2.0, 1.4]])          # 방 안, 벽 앞
    behind_cam = np.array([[2.5, -2.0, 1.4]])         # 벽 뒤
    _, a = repair.repair_faces(ply, planes, cameras=inside_cam, rng=0)
    _, b = repair.repair_faces(ply, [wall], cameras=behind_cam, rng=0)
    assert a["faces"]["A:w0"]["n_added"] > 0 and a["faces"]["A:w0"]["hole_m2_after"] == 0
    assert b["n_added"] == 0 and b["faces"]["A:w0"]["unseen_m2"] == pytest.approx(
        b["faces"]["A:w0"]["hole_m2_before"])
    # 다른 벽(A:w2, x = 4)이 가리는 카메라: 방 밖 x > 4 에서 비스듬히 -> 가려서 못 봄
    occl_cam = np.array([[6.0, 1.0, 1.2]])
    _, c = repair.repair_faces(ply, planes, cameras=occl_cam, max_hole_m2=None, rng=0)
    assert c["faces"]["A:w0"]["n_added"] == 0


def test_repair_file(tmp_path):
    ply, planes, wall = build(plyio.brush_names(2), seed=6)
    src = tmp_path / "splat_walls.ply"
    plyio.write(src, ply)
    with pytest.raises(ValueError):
        repair.repair_file(src, src, [wall])
    with pytest.raises(TypeError):
        repair.repair_file(src, tmp_path / "o.ply", [wall], bogus=1)
    info = repair.repair_file(src, tmp_path / "o.ply", [wall], rng=0)
    back = plyio.read(tmp_path / "o.ply")
    assert back.n == ply.n + info["n_added"] and back.names == ply.names
    np.testing.assert_array_equal(plyio.read(src).data, ply.data)
    assert info["stats_after"]["count"] == back.n


def test_no_holes_no_change():
    ply, planes, wall = build(plyio.brush_names(1), seed=7)
    full = ply.data[repair._sigmoid(ply.data["opacity"]) > 0.5]
    ply2 = plyio.GaussianPly(full, ["comment x"])
    cov, inside = repair.coverage_map(ply2, wall)
    # 빈 원이 남아 있으니 그것만 메워짐; 원 안쪽이 아닌 셀은 처음부터 덮여 있어야 함
    out, info = repair.repair_faces(ply2, [wall], rng=0)
    assert out.comments == ["comment x"]
    assert info["faces"]["A:w0"]["hole_m2_after"] == 0


# ---------------- 11-28 검토 회귀 ----------------
def test_tilted_real_wall_is_not_a_hole():
    """실제 벽이 면과 1.5~2도 기울어 band(6 cm) 밖으로 나가도 ±0.3 m 안에서 덮였으면 구멍 아님
    (예전: 실제 벽 앞 몇 cm 에 두 번째 벽을 지음)."""
    from indoorgs.walls.planes import Plane, WALL
    names = plyio.brush_names(2)
    L, H = 6.0, 2.6
    plan_pl = Plane("A:w0", WALL, "A", np.array([0, 1., 0]), 0.0, np.zeros(3), np.array([1., 0, 0]),
                    np.array([0, 0, 1.]), (L, H))
    t = np.radians(2.0)
    Ut, nt = np.array([np.cos(t), np.sin(t), 0]), np.array([-np.sin(t), np.cos(t), 0])
    true_pl = Plane("t", WALL, "A", nt, 0.0, np.zeros(3), Ut, np.array([0, 0, 1.]), (L, H))
    uu, vv = np.meshgrid(np.arange(0.015, L, 0.03), np.arange(0.015, H, 0.03))
    P = true_pl.point(uu.ravel(), vv.ravel())
    ply = plyio.make(len(P), names)
    for i, c in enumerate("xyz"):
        ply.data[c] = P[:, i]
    q = plyio.mat_to_quat(np.stack([Ut, np.array([0, 0, 1.]), np.cross(Ut, [0, 0, 1.])], 1))
    for i in range(4):
        ply.data[f"rot_{i}"] = q[i]
    ply.data["scale_0"] = ply.data["scale_1"] = np.log(0.025)
    ply.data["scale_2"] = np.log(0.002)
    ply.data["opacity"] = 3.0
    cams = np.array([[x, 1.2, 1.4] for x in np.linspace(0.3, 5.7, 20)])
    out, info = repair.repair_faces(ply, [plan_pl], cameras=cams, max_hole_m2=None)
    fi = info["faces"]["A:w0"]
    assert info["n_added"] == 0 and fi["hole_m2_before"] == 0
    assert fi["covered_off_plane_m2"] > 1.0
    _, old = repair.repair_faces(ply, [plan_pl], cameras=cams, max_hole_m2=None, hole_search=None)
    assert old["n_added"] > 0                                  # 예전 동작: 두 번째 벽


def test_ray_coverage_skips_wall_drawn_behind_plane():
    """wall_ab 버그 5: 학습된 3DGS 가 빈 원(HOLE) 자리를 면 40 cm 뒤 가우시안으로 그리고 있으면 (±30 cm 밖 = hole_search 로도
    못 봄) 학습 카메라에서는 불투명 -> 카메라 광선 덮임으로 구멍 아님. 반투명 사각형(THIN)은 그대로 메움.
    ray_check=False 면 예전처럼 빈 원도 메움."""
    names = plyio.brush_names(2)
    ply, planes, wall = build(names, seed=6)
    rng = np.random.default_rng(6)
    uu, vv = np.meshgrid(np.arange(2.6, 3.4, 0.04), np.arange(0.8, 1.6, 0.04))
    M = uu.size
    back = plyio.make(M, names)
    for k in names:
        back.data[k] = 0.0
    P = wall.point(uu.ravel(), vv.ravel()) - 0.40 * wall.n
    for i, c in enumerate("xyz"):
        back.data[c] = P[:, i]
    back.data["rot_0"] = 1.0
    for i in range(3):
        back.data[f"scale_{i}"] = np.log(0.04)
    back.data["opacity"] = 4.0
    both = plyio.GaussianPly(np.concatenate([ply.data, back.data]), [])
    cam = np.array([[2.8, 2.0, 1.3]])
    out, info = repair.repair_faces(both, [wall], cameras=cam, rng=0)
    fi = info["faces"]["A:w0"]
    assert fi["covered_by_rays_m2"] > 0.1
    uv = wall.uv(out.xyz[both.n:])
    in_hole = (uv[:, 0] - HOLE[0]) ** 2 + (uv[:, 1] - HOLE[1]) ** 2 < HOLE[2] ** 2
    in_thin = (uv[:, 0] > THIN[0]) & (uv[:, 0] < THIN[1]) & (uv[:, 1] > THIN[2]) & (uv[:, 1] < THIN[3])
    assert in_hole.sum() <= 2 and in_thin.sum() > 20
    old, oinfo = repair.repair_faces(both, [wall], cameras=cam, rng=0, ray_check=False)
    uv = wall.uv(old.xyz[both.n:])
    assert ((uv[:, 0] - HOLE[0]) ** 2 + (uv[:, 1] - HOLE[1]) ** 2 < HOLE[2] ** 2).sum() > 20
