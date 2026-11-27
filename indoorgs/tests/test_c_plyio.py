"""plyio: Brush·LichtFeld 속성 순서 왕복, SH 차수 맞추기, 변환, 통계."""
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest

from indoorgs import plyio

REPO = Path(__file__).resolve().parents[2]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def random_ply(names, n=500, seed=0):
    rng = np.random.default_rng(seed)
    p = plyio.make(n, names)
    for k in names:
        p.data[k] = rng.normal(size=n)
    for i in range(3):
        p.data[f"scale_{i}"] = rng.uniform(-6, -1, n)
    return p


@pytest.mark.parametrize("names", [plyio.brush_names(2), plyio.lichtfeld_names(2), plyio.lichtfeld_names(3)])
def test_roundtrip_by_name(tmp_path, names):
    p = random_ply(names)
    f = plyio.write(tmp_path / "a.ply", p)
    q = plyio.read(f)
    assert q.names == names
    for k in names:
        np.testing.assert_array_equal(q.data[k], p.data[k])
    h = plyio.read_header(f)
    assert h.count == 500 and q.sh_degree == (3 if "f_rest_44" in names else 2)
    assert not (tmp_path / "a.ply.tmp").exists()


def test_brush_and_lichtfeld_same_content(tmp_path):
    """같은 스플랫을 두 순서로 쓰면 이름으로 읽은 값이 같음 (순서로 읽으면 다름)."""
    b = random_ply(plyio.brush_names(2))
    lf = plyio.make(b.n, plyio.lichtfeld_names(2))
    for k in b.names:
        lf.data[k] = b.data[k]
    rb, rl = plyio.read(plyio.write(tmp_path / "b.ply", b)), plyio.read(plyio.write(tmp_path / "l.ply", lf))
    np.testing.assert_array_equal(rb.xyz, rl.xyz)
    raw_b = np.fromfile(tmp_path / "b.ply", np.uint8)
    assert raw_b.size > 0
    assert rb.names[0] == "f_dc_0" and rl.names[0] == "x"


def test_merge_ply_tool_compat(tmp_path):
    """tools/fill/merge_ply.read 로도 같은 값이 읽힘."""
    p = random_ply(plyio.lichtfeld_names(1), 50)
    plyio.write(tmp_path / "a.ply", p)
    import sys
    sys.path.insert(0, str(REPO / "tools" / "fill"))
    try:
        import merge_ply
        names, A, _ = merge_ply.read(str(tmp_path / "a.ply"))
    finally:
        sys.path.pop(0)
    assert names == p.names
    np.testing.assert_array_equal(A[:, names.index("x")], p.data["x"])


def test_vertex_not_first_element(tmp_path):
    p = random_ply(plyio.brush_names(0), 10)
    body = p.data.tobytes()
    head = ("ply\nformat binary_little_endian 1.0\nelement extra 2\nproperty uchar a\n"
            f"element vertex 10\n" + "".join(f"property float {k}\n" for k in p.names) + "end_header\n").encode()
    (tmp_path / "x.ply").write_bytes(head + b"\x01\x02" + body)
    q = plyio.read(tmp_path / "x.ply")
    np.testing.assert_array_equal(q.data["x"], p.data["x"])


def test_ascii_rejected(tmp_path):
    (tmp_path / "a.ply").write_text("ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nend_header\n")
    with pytest.raises(plyio.PlyError):
        plyio.read(tmp_path / "a.ply")


def test_concat_sh_degree_and_order():
    a = random_ply(plyio.brush_names(1), 4, 1)          # SH1: K=3 -> f_rest 9 개
    b = random_ply(plyio.lichtfeld_names(2), 5, 2)      # SH2: K=8 -> 24 개, 법선 있음
    m = plyio.concat([a, b])
    assert m.n == 9 and m.sh_degree == 2
    assert "nx" not in m.names                          # 한쪽에만 있는 속성은 버림
    assert m.names[:3] == ["f_dc_0", "f_dc_1", "f_dc_2"] and m.names[-3:] == ["x", "y", "z"]  # 첫 PLY(Brush) 순서
    # 채널 c 의 계수 k: SH1 은 f_rest_{c*3+k}, SH2 는 f_rest_{c*8+k}. 없는 고차 계수는 0
    for c in range(3):
        for k in range(8):
            want = a.data[f"f_rest_{c * 3 + k}"] if k < 3 else 0
            np.testing.assert_array_equal(m.data[f"f_rest_{c * 8 + k}"][:4], want)
            np.testing.assert_array_equal(m.data[f"f_rest_{c * 8 + k}"][4:], b.data[f"f_rest_{c * 8 + k}"])
    with pytest.raises(plyio.PlyError):
        plyio.common_dtype([["x", "y"], plyio.brush_names(0)])


def rot(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    t = np.radians(deg)
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def test_transform_similarity():
    p = random_ply(plyio.lichtfeld_names(2), 200)
    R = rot([0.3, 1, -0.2], 37)
    s = 1.7
    M = np.eye(4); M[:3, :3] = s * R; M[:3, 3] = [1, -2, 3]
    with pytest.warns(plyio.ShNotRotatedWarning):
        q = plyio.transform(p, M)
    np.testing.assert_allclose(q.xyz, p.xyz @ (s * R).T + M[:3, 3], atol=1e-4)
    Rq0 = plyio.quat_to_mat(np.stack([p.data[f"rot_{i}"] for i in range(4)], 1).astype(float))
    Rq1 = plyio.quat_to_mat(np.stack([q.data[f"rot_{i}"] for i in range(4)], 1).astype(float))
    np.testing.assert_allclose(Rq1, R @ Rq0, atol=1e-5)
    np.testing.assert_allclose(q.data["scale_0"], p.data["scale_0"] + np.log(s), atol=1e-5)
    N0 = np.stack([p.data["nx"], p.data["ny"], p.data["nz"]], 1)
    np.testing.assert_allclose(np.stack([q.data["nx"], q.data["ny"], q.data["nz"]], 1), N0 @ R.T, atol=1e-5)
    np.testing.assert_array_equal(q.data["f_dc_0"], p.data["f_dc_0"])
    # 이동만이면 경고 없음
    T = np.eye(4); T[:3, 3] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plyio.transform(p, T)


def test_similarity_rejects_bad():
    M = np.diag([1, 2, 1, 1.0])
    with pytest.raises(ValueError):
        plyio.similarity_parts(M)
    with pytest.raises(ValueError):
        plyio.similarity_parts(np.diag([-1, 1, 1, 1.0]))


def test_mat_quat_roundtrip():
    rng = np.random.default_rng(3)
    q = rng.normal(size=(1000, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    q *= np.sign(q[:, :1])
    np.testing.assert_allclose(plyio.mat_to_quat(plyio.quat_to_mat(q)), q, atol=1e-9)


def test_stats_matches_ply_stats_tool(tmp_path):
    p = random_ply(plyio.brush_names(2), 2000)
    p.data["scale_0"][:300] += 4                         # 바늘형 몇 개
    f = plyio.write(tmp_path / "s.ply", p)
    tool = load_tool("ply_stats").stats(load_tool("ply_stats").load(str(f)))
    mine = plyio.stats(plyio.read(f))
    assert mine["count"] == tool["n"] == 2000
    assert mine["translucent_ratio"] == pytest.approx(tool["transl05"] / 100, abs=1e-4)
    assert mine["needle_ratio"] == pytest.approx(tool["needle10"] / 100, abs=1e-4)
    assert mine["needle_ratio"] > 0


def test_merge_files_streaming(tmp_path):
    a = random_ply(plyio.brush_names(2), 30, 1)
    b = random_ply(plyio.lichtfeld_names(2), 20, 2)
    fa, fb = plyio.write(tmp_path / "a.ply", a), plyio.write(tmp_path / "b.ply", b)
    T = np.eye(4); T[:3, 3] = [5, 0, 0]
    r = plyio.merge_files(tmp_path / "m.ply", [(fa, None), (fb, T)])
    m = plyio.read(tmp_path / "m.ply")
    assert r["count"] == m.n == 50 and r["parts"] == [30, 20]
    np.testing.assert_allclose(m.xyz[30:], b.xyz + [5, 0, 0], atol=1e-5)
    np.testing.assert_array_equal(m.data["f_rest_7"][:30], a.data["f_rest_7"])
