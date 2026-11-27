"""merge: layout_v4 형식 변환이 기존 수식(tools/realign_layout.py apply)과 같은지, 도면 구간 문 근처 ICP 복원."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from indoorgs import plyio, schemas
from indoorgs.fillcore import align
from indoorgs.project import Project
from indoorgs.stages.base import StageContext
from indoorgs.stages.merge import Merge, read_layout
from test_c_plyio import load_tool, random_ply, rot, REPO


def project(root: Path, names):
    root.mkdir(parents=True, exist_ok=True)
    (root / "project.yaml").write_text(
        "sections:\n" + "".join(f"  - {{name: {n}, video: {n}.mp4}}\n" for n in names), encoding="utf-8")
    return Project(root)


def run_merge(pr):
    logs = []
    res = Merge().run(StageContext(pr, None, None, logs.append))
    return res, logs


def put_train(pr, name, ply, fill=False):
    p = pr.section(name)
    plyio.write(p.fill_ply if fill else p.train_run("brush", "default") / "splat.ply", ply)


def test_layout_v4_matches_existing_formula(tmp_path):
    """완료 기준: tools/layout_v4.json 의 구간 1·2 변환으로 합친 결과 = realign_layout.apply(M, P) (기존 배치 수식)."""
    pr = project(tmp_path / "p", ["west", "hall"])
    a = random_ply(plyio.brush_names(2), 300, 1)           # Brush 순서
    b = random_ply(plyio.lichtfeld_names(2), 200, 2)       # LichtFeld 순서
    put_train(pr, "west", a)
    put_train(pr, "hall", b, fill=True)                    # fill 결과가 있으면 그것을 씀
    v4 = json.loads((REPO / "tools" / "layout_v4.json").read_text(encoding="utf-8"))
    v4["section_keys"] = {"west": "1", "hall": "2"}
    schemas.write_json(pr.layout_json, v4)
    res, logs = run_merge(pr)
    assert res.ok, res.message
    out = plyio.read(pr.building_ply)
    assert out.n == 500 and out.names == plyio.brush_names(2)   # 첫 구간(Brush) 순서, 법선은 버림
    ref = load_tool("realign_layout")
    for sl, src, key in ((slice(0, 300), a, "1"), (slice(300, 500), b, "2")):
        M = np.array(v4[key]["matrix_4x4"])
        np.testing.assert_allclose(out.xyz[sl], ref.apply(M, src.xyz), atol=2e-4)
        s, R, _ = plyio.similarity_parts(M)
        R0 = plyio.quat_to_mat(np.stack([src.data[f"rot_{i}"] for i in range(4)], 1).astype(float))
        R1 = plyio.quat_to_mat(np.stack([out.data[f"rot_{i}"][sl] for i in range(4)], 1).astype(float))
        np.testing.assert_allclose(R1, R @ R0, atol=1e-5)
        np.testing.assert_allclose(out.data["scale_1"][sl], src.data["scale_1"] + np.log(s), atol=1e-5)
        assert s == pytest.approx(v4[key]["scale"], abs=1e-3)
        np.testing.assert_array_equal(out.data["f_dc_2"][sl], src.data["f_dc_2"])
    lay = schemas.read_json(pr.layout_json)
    assert lay["sections"]["west"]["source"] == "layout.json"
    np.testing.assert_allclose(lay["sections"]["hall"]["matrix_4x4"], v4["2"]["matrix_4x4"])
    assert lay["sections"]["hall"]["ply"].endswith("holes_backdrop.ply")
    assert not any("SH" in n for n in lay["notes"])         # SH 도 돌리므로 '회전 생략' 메모 없음
    kept = list(pr.merged.glob("layout_input_*.json"))       # 사용자 입력 보존
    assert len(kept) == 1 and json.loads(kept[0].read_text(encoding="utf-8"))["1"] == v4["1"]


def test_layout_v4_unity_fix_noted(tmp_path):
    v4 = json.loads((REPO / "tools" / "layout_v4.json").read_text(encoding="utf-8"))
    v4["section_keys"] = {"r5": "5"}
    f = tmp_path / "l.json"; schemas.write_json(f, v4)
    m, notes = read_layout(f, ["r5", "other"])
    assert list(m) == ["r5"] and any("unity_fix_4x4" in n for n in notes)


def test_missing_layout_fails(tmp_path):
    pr = project(tmp_path / "p", ["a", "b"])
    put_train(pr, "a", random_ply(plyio.brush_names(0), 10))
    put_train(pr, "b", random_ply(plyio.brush_names(0), 10))
    res, _ = run_merge(pr)
    assert not res.ok and "배치가 없는 구간" in res.message


def test_single_section_identity(tmp_path):
    pr = project(tmp_path / "p", ["a"])
    a = random_ply(plyio.brush_names(1), 10)
    put_train(pr, "a", a)
    res, _ = run_merge(pr)
    assert res.ok
    np.testing.assert_array_equal(plyio.read(pr.building_ply).data, a.data)


# ---------------- 도면 구간 + 문 근처 ICP ----------------
def door_scene(rng, n=40000):
    """문(x=0 벽, y -0.5~0.5, 높이 2.1) 양쪽 3m x 3m 공간의 바닥·벽·문틀·상자 몇 개 위 점 (스플랫 중심)."""
    P = []
    k = n // 8
    P.append(np.c_[rng.uniform(-1.5, 1.5, 2 * k), rng.uniform(-1.5, 1.5, 2 * k), np.zeros(2 * k)])        # 바닥
    y, z = rng.uniform(-1.5, 1.5, 3 * k), rng.uniform(0, 2.6, 3 * k)
    m = ~((np.abs(y) < 0.5) & (z < 2.1))
    P.append(np.c_[np.zeros(m.sum()), y[m], z[m]])                                                          # 문 있는 벽
    for yy in (-1.5, 1.5):
        P.append(np.c_[rng.uniform(-1.5, 1.5, k), np.full(k, yy), rng.uniform(0, 2.6, k)])                # 옆벽
    for c in ((0.8, 0.9, 0.3), (-0.7, -1.0, 0.5)):                                                           # 상자
        q = rng.uniform(-0.3, 0.3, (k // 2, 3)); i = rng.integers(0, 3, len(q)); q[np.arange(len(q)), i] = np.sign(q[np.arange(len(q)), i]) * 0.3
        P.append(q + c)
    return np.concatenate(P)


def splats_at(P):
    p = plyio.make(len(P), plyio.brush_names(0))
    for i, k in enumerate("xyz"):
        p.data[k] = P[:, i]
    p.data["opacity"] = 4.0; p.data["rot_0"] = 1
    return p


def space_project(tmp_path, names):
    pr = project(tmp_path / "p", names)
    schemas.write_json(pr.plan_json, {"version": 1, "units": "m",
        "rooms": [{"id": "A", "box": {"center": [-2, 0], "size": [4, 4]}, "floor_z": 0, "ceiling_z": 2.6},
                  {"id": "B", "box": {"center": [2, 0], "size": [4, 4]}, "floor_z": 0, "ceiling_z": 2.6}],
        "doors": [{"id": "AB", "rooms": ["A", "B"], "center": [0, 0], "width": 1.0, "height": 2.1}]})
    for n in names:
        schemas.write_json(pr.section(n).priors, {"version": 1, "coordinate_system": "CARTESIAN", "frames": []})
        schemas.write_json(pr.section(n).colmap / "register.json", {"version": 1, "mapper": "pose_prior_mapper"})
    return Project(pr.root)


def test_icp_recovers_small_offset_at_door(tmp_path):
    pr = space_project(tmp_path, ["a", "b"])
    rng = np.random.default_rng(0)
    Pa = door_scene(rng)
    put_train(pr, "a", splats_at(Pa))
    D = np.eye(4); D[:3, :3] = rot([0.2, 0.1, 1], 1.5); D[:3, 3] = [0.08, -0.05, 0.03]   # b 구간 정합 오차
    Pb = align.apply(D, door_scene(rng))
    put_train(pr, "b", splats_at(Pb))
    res, logs = run_merge(pr)
    assert res.ok, res.message
    lay = schemas.read_json(pr.layout_json)["sections"]
    assert lay["a"]["source"] == "floorplan" and np.allclose(lay["a"]["matrix_4x4"], np.eye(4))
    assert lay["b"]["source"] == "floorplan+icp" and lay["b"]["icp_door"] == "AB"
    T = np.array(lay["b"]["matrix_4x4"])
    E = T @ D                                                  # 단위 행렬에 가까워야 함
    assert align.rot_deg(E) < 0.3
    assert np.linalg.norm(E[:3, 3]) < 0.02
    out = plyio.read(pr.building_ply)
    assert out.n == len(Pa) + len(Pb)


def test_icp_rejects_large_or_bad(tmp_path):
    rng = np.random.default_rng(1)
    P = door_scene(rng, 8000)
    D = np.eye(4); D[:3, :3] = rot([0, 0, 1], 10)             # 10도: 제한(3도) 넘음 -> 변환 안 함
    T, info = align.icp(align.apply(D, door_scene(rng, 8000)), P)
    assert T is None and "reason" in info
    far = door_scene(rng, 8000) + [10, 0, 0]                   # 겹치지 않음
    T, info = align.icp(far, P)
    assert T is None


def test_no_icp_when_already_aligned(tmp_path):
    pr = space_project(tmp_path, ["a", "b"])
    rng = np.random.default_rng(2)
    put_train(pr, "a", splats_at(door_scene(rng)))
    put_train(pr, "b", splats_at(door_scene(rng)))
    res, logs = run_merge(pr)
    assert res.ok
    lay = schemas.read_json(pr.layout_json)["sections"]["b"]
    T = np.array(lay["matrix_4x4"])
    assert align.rot_deg(T) < 0.3 and np.linalg.norm(T[:3, 3]) < 0.02


# ---- experiments/accuracy/merge_eval.py: 벽 두께 함정 ----
def two_sided_door(rng, side, wall_t=0.2, sp=0.04):
    """문 난 벽(두께 wall_t, y=0~wall_t) 근처 1.5m. side='a'(복도, y<0) 는 복도 쪽 면 y=0 만,
    'b'(방, y>wall_t) 는 방 쪽 면 y=wall_t 만 봄. 둘 다 문틀(양옆 x=±0.5, 위 z=2.1)과 바닥을 봄."""
    g = np.arange(-1.5, 1.5, sp); z = np.arange(0, 2.6, sp); d = np.arange(0, wall_t, sp / 2)
    gw, zw = np.arange(-1.5, 1.5, sp * 0.7), np.arange(0, 2.6, sp * 0.7)      # 벽 점이 바닥보다 촘촘 (벽 점 > 절반)
    X, Z = np.meshgrid(gw, zw); X, Z = X.ravel(), Z.ravel()
    face = ~((np.abs(X) < 0.5) & (Z < 2.1))
    yf = 0.0 if side == "a" else wall_t
    P = [np.c_[X[face], np.full(face.sum(), yf), Z[face]]]
    Xf, Yf = np.meshgrid(g, np.arange(-1.5, 1.5 + wall_t, sp)); Xf, Yf = Xf.ravel(), Yf.ravel()
    near = Yf < wall_t if side == "a" else Yf > 0                              # 제 쪽 바닥 + 문턱
    cone = np.abs(Xf) < 0.5 + (Yf - wall_t if side == "a" else -Yf) * 0.8      # 문 너머 보이는 부채꼴
    keep = near | cone
    P.append(np.c_[Xf[keep], Yf[keep], np.zeros(keep.sum())])                   # 바닥
    for x0 in (-0.5, 0.5):
        Dd, Zz = np.meshgrid(d, np.arange(0, 2.1, sp))
        P.append(np.c_[np.full(Dd.size, x0), Dd.ravel(), Zz.ravel()])          # 문틀 옆
    Xx, Dd = np.meshgrid(np.arange(-0.5, 0.5, sp), d)
    P.append(np.c_[Xx.ravel(), Dd.ravel(), np.full(Xx.size, 2.1)])             # 문틀 위
    P = np.concatenate(P)
    P = P[np.hypot(P[:, 0], P[:, 1] - wall_t / 2) < 1.5]                       # merge.door_points 처럼 문 1.5m 안
    return P + rng.normal(0, 0.015, P.shape)


def test_icp_does_not_glue_wall_faces():
    # 예전 기본 max_corr 0.3: 이미 맞은 구간을 벽 두께만큼(0.18m) 옮겨 붙였음 -> 0.08
    rng = np.random.default_rng(3)
    A, B = two_sided_door(rng, "a"), two_sided_door(rng, "b")
    T, info = align.icp(B, A)
    assert T is None or np.linalg.norm(T[:3, 3]) < 0.03, info


def test_icp_still_fixes_height_offset():
    rng = np.random.default_rng(4)
    A, B = two_sided_door(rng, "a"), two_sided_door(rng, "b") + [0, 0, 0.06]
    T, info = align.icp(B, A)
    assert T is not None, info
    assert T[2, 3] == pytest.approx(-0.06, abs=0.02)
