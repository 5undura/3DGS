"""colmap/pair_verify.py: 짝 검증(가짜 짝 삭제, 진짜·앞뒤 짝 유지), 1차 기준 자세, 제약에 직접 맞추기."""
import math
import shutil
import sqlite3
from contextlib import closing

import numpy as np
import pytest

from indoorgs.colmap import model as cm
from indoorgs.colmap import pair_verify as pv

K = np.array([[500.0, 0, 480.0], [0, 500.0, 270.0], [0, 0, 1.0]])
W, H = 960, 540
# 카메라가 +y 를 봄 (카메라 x = 월드 x, 카메라 y = 월드 -z, 카메라 z = 월드 y)
R_CW0 = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])


def Rz(deg):
    t = math.radians(deg)
    return np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1.0]])


def Rx(deg):
    t = math.radians(deg)
    return np.array([[1.0, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])


def name(i):
    return f"f_{i:05d}_{i * 10:06d}.jpg"


def project(R_cw, C, X):
    Xc = (X - C) @ R_cw.T
    uv = Xc[:, :2] / Xc[:, 2:3] @ K[:2, :2].T + K[:2, 2]
    ok = (Xc[:, 2] > 0.5) & (uv[:, 0] > 0) & (uv[:, 0] < W) & (uv[:, 1] > 0) & (uv[:, 1] < H)
    return uv, ok


# ---------------------------------------------------------------- verify_pairs
def _center(i):
    return np.array([0.3 * (i - 1), 0.0, 1.4])


def _make_db(path, rng):
    n = 30
    X = np.column_stack([rng.uniform(-5, 15, 4000), rng.uniform(5, 9, 4000), rng.uniform(0, 2.7, 4000)])
    kps = {i: [] for i in range(1, n + 1)}

    def add(i, uv):
        k0 = len(kps[i])
        kps[i].extend((uv + rng.normal(0, 0.3, uv.shape)).tolist())
        return np.arange(k0, k0 + len(uv))

    def real_pair(i1, cam1, i2, cam2, m=200):
        uv1, ok1 = project(*cam1, X)
        uv2, ok2 = project(*cam2, X)
        sel = np.where(ok1 & ok2)[0][:m]
        assert len(sel) >= 50
        return np.column_stack([add(i1, uv1[sel]), add(i2, uv2[sel])])

    true = {i: (R_CW0, _center(i)) for i in range(1, n + 1)}
    pairs = {
        (1, 20): real_pair(1, true[1], 20, true[20]),                       # 진짜, 5.7 m
        (3, 25): real_pair(3, true[3], 25, true[25]),                       # 진짜, 6.6 m
        # 가짜: 실제로는 1번 왼쪽 3 m 에서 찍힌 것처럼 맞음 (기준은 오른쪽 6.3 m) -> 방향 틀림
        (1, 22): real_pair(1, true[1], 22, (R_CW0, _center(1) + [-3.0, 0, 0])),
        # 가짜: 50° 돌아간 카메라처럼 맞음 -> 회전 틀림
        (2, 24): real_pair(2, true[2], 24, (R_CW0 @ Rz(-50).T, _center(2) + [1.5, 0, 0])),
        (5, 30): real_pair(5, true[5], 30, true[30]),                       # 30번은 기준 자세 없음
    }
    g = rng.uniform([0, 0], [W, H], (40, 2))                                # 앞뒤 짝: 엉터리여도 그대로 둠
    pairs[(4, 8)] = np.column_stack([add(4, g), add(8, g[::-1])])
    with closing(sqlite3.connect(str(path))) as c, c:
        c.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)")
        c.execute("CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB)")
        c.execute("CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER,"
                  " data BLOB, config INTEGER)")
        for i in range(1, n + 1):
            c.execute("INSERT INTO images VALUES (?,?,1)", (i, name(i)))
            a = np.zeros((len(kps[i]), 6), np.float32)
            if kps[i]:
                a[:, :2] = kps[i]
                a[:, 2] = a[:, 5] = 1.0
            c.execute("INSERT INTO keypoints VALUES (?,?,?,?)", (i, len(a), 6, a.tobytes()))
        for (i1, i2), m in pairs.items():
            c.execute("INSERT INTO two_view_geometries VALUES (?,?,?,?,2)",
                      (pv.pair_id(i1, i2), len(m), 2, m.astype(np.uint32).tobytes()))
    ref = pv.Reference({name(i): (R_CW0.T, _center(i)) for i in range(1, n)}, K, np.zeros(4))
    return ref


def _pairs_left(path):
    with closing(sqlite3.connect(str(path))) as c:
        ids = [r[0] for r in c.execute("SELECT pair_id FROM two_view_geometries")]
    return {(p // pv.MAX_ID, p % pv.MAX_ID) for p in ids}


def test_verify_pairs_removes_fake_keeps_true_and_sequential(tmp_path):
    db = tmp_path / "db_verified.db"
    ref = _make_db(db, np.random.default_rng(0))
    st = pv.verify_pairs(db, ref)
    assert _pairs_left(db) == {(1, 20), (3, 25), (4, 8)}
    assert st["pairs"] == 6 and st["dropped"] == 3
    assert (st["kept"], st["kept_sequential"]) == (2, 1)
    assert (st["drop_dir"], st["drop_rot"], st["drop_unknown"], st["drop_no_pose"]) == (1, 1, 1, 0)


def test_verify_pairs_keep_unknown_and_tolerance(tmp_path):
    db = tmp_path / "a.db"
    ref = _make_db(db, np.random.default_rng(1))
    db2 = tmp_path / "b.db"
    shutil.copy2(db, db2)
    st = pv.verify_pairs(db, ref, drop_unknown=False)
    assert (5, 30) in _pairs_left(db) and st["drop_unknown"] == 0
    # 회전 허용을 60° 로 넓히면 50° 가짜는 회전 검사를 통과 (방향 검사에는 걸릴 수 있음)
    st2 = pv.verify_pairs(db2, ref, rot_tol_deg=60.0)
    assert st2["drop_rot"] == 0


# ---------------------------------------------------------------- 모델 만들기
def _corridor(n=40, yaw_wiggle=20.0, rng=None, x0=0.0, back=True):
    """x 축 복도를 걷는 카메라 (갔다 돌아옴), 요를 흔듦. 돌려줌: (names, R_cw 목록, 중심)."""
    rng = rng or np.random.default_rng(0)
    Rs, Cs = [], []
    for k in range(n):
        half = back and k >= n // 2
        x = x0 + (k if not half else n - 1 - k) * 0.25
        yaw = (180.0 if half else 0.0) - 90.0 + rng.uniform(-yaw_wiggle, yaw_wiggle)   # 진행 방향(±x)을 봄
        R_wc = Rz(yaw) @ R_CW0.T
        Rs.append(R_wc.T)
        Cs.append(np.array([x, rng.uniform(-0.3, 0.3), 1.4 + rng.uniform(-0.05, 0.05)]))
    return Rs, np.array(Cs)


def _model(Rs, Cs, names, rng, n_pts=50):
    cams = {1: cm.Camera(1, 4, W, H, np.array([500.0, 500.0, 480.0, 270.0, 0.01, -0.002, 0, 0]))}
    images, frames = {}, []
    for k, (R, C, nm) in enumerate(zip(Rs, Cs, names)):
        q, t = pv.rot2qvec(R), -R @ C
        images[k + 1] = cm.Image(k + 1, q, t, 1, nm, np.zeros((0, 2)), np.zeros(0, np.int64))
        frames.append(cm.Frame(k + 1, 1, tuple(map(float, (*q, *t))), [(0, 1, k + 1)]))
    pts = {i + 1: cm.Point3D(i + 1, rng.uniform([-1, -1.2, 0], [12, 1.2, 2.7]), np.array([1, 2, 3], np.uint8), 0.5,
                             np.array([[1, 0], [2, 0]], np.uint32)) for i in range(n_pts)}
    return cm.Model(cams, images, pts, frames, b"rig")


def _priors(names, Cs):
    return {"version": 1, "coordinate_system": "CARTESIAN",
            "frames": [{"image": n, "xyz": c.tolist(), "sigma": [0.5, 0.5, 0.3]} for n, c in zip(names, Cs)]}


# ---------------------------------------------------------------- align_model_to_priors
def test_align_model_to_priors_recovers_similarity_with_roll(tmp_path):
    rng = np.random.default_rng(3)
    Rs, Cs = _corridor(rng=rng)
    names = [name(i + 1) for i in range(len(Cs))]
    truth = _model(Rs, Cs, names, rng)
    # COLMAP 이 놓은 좌표처럼: 복도 축(x) 둘레 35° 기울기 + 요 70° + 축척 0.37 + 이동
    s0, A0, b0 = 0.37, Rz(70) @ Rx(35), np.array([4.0, -2.0, 7.0])
    dist = pv.transform_model(truth, s0, A0, b0)
    im = dist.images[5]
    assert np.allclose(im.center, s0 * A0 @ Cs[4] + b0)                       # 변환 식 자체 확인
    assert np.allclose(cm.qvec2rot(dist.frames[4].pose[:4]), im.R)            # frames.bin 자세도 같이

    new, st = pv.align_model_to_priors(dist, _priors(names, Cs))
    for k, im in new.images.items():
        assert np.allclose(im.center, Cs[k - 1], atol=1e-6)
        assert np.allclose(im.R, Rs[k - 1], atol=1e-6)
        fr = new.frames[k - 1]
        assert np.allclose(fr.pose[:4], im.q) and np.allclose(fr.pose[4:], im.t)
    for pid, p in new.points.items():
        assert np.allclose(p.xyz, truth.points[pid].xyz, atol=1e-6)
    assert st["scale"] == pytest.approx(1 / s0, rel=1e-6)
    assert st["leveled_deg"] == pytest.approx(35.0, abs=1e-3)
    assert st["far_1m"] == 0 and st["resid_median_m"] < 1e-6
    # 쓰고 다시 읽어도 같음 (undistort 입력)
    cm.write_model(new, tmp_path / "a" / "0")
    back = cm.read_model(tmp_path / "a" / "0")
    assert np.allclose(back.images[3].center, Cs[2], atol=1e-6)


def test_align_without_level_keeps_roll():
    """수평 맞추기 없이는 위치 제약만으로 복도 축 둘레 기울기가 안 풀림 (RESULTS 7)."""
    rng = np.random.default_rng(4)
    Rs, Cs = _corridor(rng=rng)
    names = [name(i + 1) for i in range(len(Cs))]
    dist = pv.transform_model(_model(Rs, Cs, names, rng), 1.0, Rx(30), np.zeros(3))
    new, st = pv.align_model_to_priors(dist, _priors(names, Cs), level=False)
    up = np.array([new.images[k].R.T @ [0, -1.0, 0] for k in new.images])   # 카메라 위축 (월드)
    assert np.degrees(np.arccos(np.clip(up[:, 2], -1, 1))).mean() > 20


def test_align_needs_three_priors():
    rng = np.random.default_rng(5)
    Rs, Cs = _corridor(n=10, rng=rng)
    names = [name(i + 1) for i in range(10)]
    with pytest.raises(ValueError):
        pv.align_model_to_priors(_model(Rs, Cs, names, rng), _priors(names[:2], Cs[:2]))


# ---------------------------------------------------------------- load_reference
def test_load_reference_levels_fragments_and_drops_far(tmp_path):
    rng = np.random.default_rng(6)
    Rs, Cs = _corridor(n=60, back=False, rng=rng)
    names = [name(i + 1) for i in range(60)]
    pri = _priors(names, Cs)
    sp = tmp_path / "sparse_pass1"
    # 조각 두 개 (서로 다른 기울기·축척·좌표, 하나는 뒤집힘) + 작은 조각
    Cbad = Cs.copy()
    Cbad[10] += [0, 3.0, 0]                                           # 1차에서 잘못 놓인 사진
    m0 = pv.transform_model(_model(Rs[:35], Cbad[:35], names[:35], rng), 2.0, Rz(30) @ Rx(20), np.array([1, 2, 3.0]))
    m1 = pv.transform_model(_model(Rs[30:], Cs[30:], names[30:], rng), 0.5, Rx(170), np.array([-5, 0, 1.0]))
    m2 = _model(Rs[:5], Cs[:5], names[:5], rng)
    cm.write_model(m0, sp / "0")
    cm.write_model(m1, sp / "1")
    cm.write_model(m2, sp / "2")
    ref = pv.load_reference(sp, pri)
    assert ref.info["models"] == [35, 30]                            # 10장 미만 조각은 안 씀
    assert ref.info["far_dropped"] == 1 and names[10] not in ref.poses
    assert len(ref.poses) == 59
    for k, nm in enumerate(names):
        if nm in ref.poses:
            R, C = ref.poses[nm]
            assert np.linalg.norm(C - Cs[k]) < 0.15, nm
            assert np.allclose(R, Rs[k].T, atol=0.02), nm
    assert ref.K[0, 0] == 500.0 and ref.dist[0] == pytest.approx(0.01)
