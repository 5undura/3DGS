"""walls/cleanup: 바늘 쪼개기·부유물·아주 큰 것·SH 튐 (합성 데이터, Brush·LichtFeld 속성 순서)."""
import numpy as np
import pytest

from indoorgs import plyio
from indoorgs.colmap import model as cm
from indoorgs.schemas import parse_plan
from indoorgs.walls import cleanup
from indoorgs.walls.planes import planes_from_plan

LAYOUTS = [plyio.brush_names(2), plyio.lichtfeld_names(2)]
IDS = ["brush", "lichtfeld"]


def logit(o):
    return np.log(o / (1 - o))


def rand_quat(rng, n):
    q = rng.normal(size=(n, 4))
    return q * rng.uniform(0.5, 2.0, (n, 1))          # 일부러 정규화 안 함


def random_ply(names, n, rng, needle_frac=0.3):
    """무작위 가우시안. needle_frac 만큼은 긴 축/중간 축 = 12~40 (축 번호도 무작위)."""
    ply = plyio.make(n, names)
    for k in names:
        ply.data[k] = rng.normal(size=n)
    q = rand_quat(rng, n)
    for i in range(4):
        ply.data[f"rot_{i}"] = q[:, i]
    S = rng.uniform(0.005, 0.03, (n, 3))
    nd = int(n * needle_frac)
    ax = rng.integers(0, 3, nd)
    S[np.arange(nd), ax] = S[np.arange(nd), (ax + 1) % 3] * 0 + S[:nd].max(1) * rng.uniform(12, 40, nd)
    for i in range(3):
        ply.data[f"scale_{i}"] = np.log(S[:, i])
    ply.data["opacity"] = rng.uniform(-3, 4, n)
    return ply


def rows_bytes(a):
    return np.ascontiguousarray(a).view(np.uint8).reshape(len(a), -1)


# ---------------- 1D 쪼개기 수학 ----------------
@pytest.mark.parametrize("ratio_lp", [10 / 3, 6.0])
def test_split_profile_matches_original(ratio_lp):
    sL, spm = ratio_lp, 1.0
    k, sp = cleanup.split_layout(np.array([sL]), np.array([spm]), max_pieces=8)
    t, f = cleanup.split_weights(int(k[0]), np.array([sL]), sp)
    x = np.linspace(-4 * sL, 4 * sL, 4001)
    orig = np.exp(-x ** 2 / (2 * sL ** 2))
    mix = (f[0][:, None] * np.exp(-(x[None] - t[0][:, None]) ** 2 / (2 * sp[0] ** 2))).sum(0)
    assert abs(mix.sum() / orig.sum() - 1) < 0.01          # 축 방향 적분 알파 보존
    assert np.abs(mix - orig).max() < 0.08                  # 정면 프로파일 (최대 오차, 봉우리 = 1)
    assert sp[0] / 1.0 <= 3.0 + 1e-9 or k[0] == 8


def test_split_layout_cap_and_noop():
    k, sp = cleanup.split_layout(np.array([1.0, 100.0, 0.5]), np.array([1.0, 1.0, 1.0]), max_pieces=8)
    assert k[0] == 1 and k[2] == 1 and k[1] == 8
    assert sp[1] > 1.0                                       # 상한에 걸리면 조각이 길어짐


# ---------------- split_needles ----------------
@pytest.mark.parametrize("names", LAYOUTS, ids=IDS)
def test_split_needles(names):
    rng = np.random.default_rng(3)
    ply = random_ply(names, 3000, rng)
    S = np.exp(np.stack([ply.data[f"scale_{i}"] for i in range(3)], 1).astype(float))
    Ss = np.sort(S, 1)
    needle = Ss[:, 2] / Ss[:, 1] > 10
    out, info = cleanup.split_needles(ply, target=3.0, max_pieces=8)
    assert out.names == ply.names and out.comments == ply.comments
    nk = int((~needle).sum())
    assert info["n_needles"] == needle.sum() > 500
    assert np.array_equal(rows_bytes(out.data[:nk]), rows_bytes(ply.data[~needle]))   # 비트 그대로, 순서 그대로
    new = out.data[nk:]
    assert len(new) == info["n_pieces"] == out.n - nk
    # 조각 -> 원본 (회전 값이 고유하므로 rot_0 로 찾음)
    src_of = {float(v): i for i, v in enumerate(ply.data["rot_0"])}
    src = np.array([src_of[float(v)] for v in new["rot_0"]])
    assert set(src) == set(np.nonzero(needle)[0])
    counts = np.bincount(src, minlength=ply.n)[needle]
    assert counts.min() >= 2 and counts.max() <= 8
    for k in names:
        if k.startswith(("f_dc", "f_rest", "rot_")) or k in ("nx", "ny", "nz"):
            assert np.array_equal(new[k], ply.data[k][src]), k              # 색·SH·회전 그대로
    Sn = np.exp(np.stack([new[f"scale_{i}"] for i in range(3)], 1).astype(float))
    So = S[src]
    ax = np.argmax(So, 1)
    r = np.arange(len(src))
    other = np.ones((len(src), 3), bool)
    other[r, ax] = False
    assert np.allclose(Sn[other], So[other], rtol=1e-6)                     # 다른 두 축 그대로
    mid = np.sort(So, 1)[:, 1]
    piece_ratio = Sn[r, ax] / mid
    k_src = np.bincount(src, minlength=ply.n)[src]
    assert np.all(piece_ratio[k_src < 8] <= 3.0 + 1e-4)                     # 상한 안 걸린 것은 종횡비 <= target
    assert np.all(Sn[r, ax] < So[r, ax])
    # 위치: 원래 긴 축 위, 원래 중심 대칭, ±1.8 s 안
    q = np.stack([ply.data[f"rot_{i}"] for i in range(4)], 1).astype(float)[src]
    a = plyio.quat_to_mat(q)[r, :, ax]
    D = np.stack([new[c] for c in "xyz"], 1).astype(float) - ply.xyz[src]
    assert np.abs(np.cross(D, a)).max() < 1e-4 * max(1.0, np.abs(D).max())
    tt = np.einsum("ij,ij->i", D, a)
    s = np.sqrt(So[r, ax] ** 2 - Sn[r, ax] ** 2)
    assert np.all(np.abs(tt) <= 1.8 * s * (1 + 1e-4) + 1e-6)
    mean_t = np.bincount(src, weights=tt, minlength=ply.n)[needle] / counts
    assert np.abs(mean_t).max() < 1e-5
    q_after = cleanup.ply_quality(out)
    assert q_after["needle_ratio"] < cleanup.ply_quality(ply)["needle_ratio"]


def test_split_needles_opacity_integral_low_alpha():
    """투명한 쪽: Σ o_i σ_p ≈ o σ_L (축 방향 적분 알파 보존). keep_volume=False 면 원래 불투명도."""
    names = plyio.brush_names(1)
    ply = plyio.make(1, names)
    ply.data["rot_0"] = 1
    ply.data["scale_0"], ply.data["scale_1"], ply.data["scale_2"] = np.log(0.2), np.log(0.01), np.log(0.002)
    ply.data["opacity"] = logit(0.02)
    out, info = cleanup.split_needles(ply)
    o = 1 / (1 + np.exp(-out.data["opacity"].astype(float)))
    sp = np.exp(out.data["scale_0"].astype(float))
    assert abs((o * sp).sum() / (0.02 * 0.2) - 1) < 0.02
    out2, _ = cleanup.split_needles(ply, keep_volume=False)
    assert np.all(out2.data["opacity"] == ply.data["opacity"][0])


def test_split_needles_min_opacity_and_errors():
    names = plyio.brush_names(0)
    ply = plyio.make(2, names)
    ply.data["rot_0"] = 1
    ply.data["scale_0"] = np.log(0.3)
    ply.data["scale_1"] = ply.data["scale_2"] = np.log(0.01)
    ply.data["opacity"] = [logit(0.01), logit(0.9)]
    out, info = cleanup.split_needles(ply, min_opacity=0.05)
    assert info["n_needles"] == 1 and np.array_equal(rows_bytes(out.data[:1]), rows_bytes(ply.data[:1]))
    with pytest.raises(ValueError):
        cleanup.split_needles(ply, ratio=3, target=3)


# ---------------- 줄무늬 측정 (numpy 정사영 아님: 원근 + 3DGS 아핀 근사 vs 광선 정확값) ----------------
def look_at(c, target, up=(0, 0, 1)):
    c = np.asarray(c, float)
    f = np.asarray(target, float) - c
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    return np.stack([r, np.cross(f, r), f]), c


def render_alpha(ply, cam, W=320, H=240, f=240.0, mode="affine", near=0.05):
    """알파만 (1 - Π(1-α), 순서 무관). affine = 3DGS 래스터라이저 (중심 야코비안, +0.3 px² 확대),
    exact = 광선 위 마할라노비스 최소점에서의 3D 가우시안 값 (같은 가우시안의 원근 정확값)."""
    Rw, c = cam
    d = ply.data
    P = ply.xyz
    R = plyio.quat_to_mat(np.stack([d[f"rot_{i}"] for i in range(4)], 1).astype(float))
    S = np.exp(np.stack([d[f"scale_{i}"] for i in range(3)], 1).astype(float))
    o = 1 / (1 + np.exp(-d["opacity"].astype(float)))
    mu = (P - c) @ Rw.T
    A = Rw[None] @ R
    Sig = (A * S[:, None, :] ** 2) @ A.transpose(0, 2, 1)
    cx, cy = W / 2, H / 2
    tau = np.zeros(H * W)
    corners = np.array([[a, b, e] for a in (-3, 3) for b in (-3, 3) for e in (-3, 3)], float)
    for i in np.nonzero(mu[:, 2] > near)[0]:
        x, y, z = mu[i]
        if mode == "affine":
            J = np.array([[f / z, 0, -f * x / z ** 2], [0, f / z, -f * y / z ** 2]])
            S2 = J @ Sig[i] @ J.T + 0.3 * np.eye(2)
            m = np.array([f * x / z + cx, f * y / z + cy])
            r = 3 * np.sqrt(np.linalg.eigvalsh(S2).max())
            box = (m[0] - r, m[0] + r, m[1] - r, m[1] + r)
        else:
            C = mu[i] + (corners * S[i]) @ A[i].T
            C = C[C[:, 2] > near]
            if len(C) == 0:
                continue
            us, vs = f * C[:, 0] / C[:, 2] + cx, f * C[:, 1] / C[:, 2] + cy
            box = (us.min(), us.max(), vs.min(), vs.max())
        u0, u1 = int(max(0, np.floor(box[0]))), int(min(W, np.ceil(box[1]) + 1))
        v0, v1 = int(max(0, np.floor(box[2]))), int(min(H, np.ceil(box[3]) + 1))
        if u0 >= u1 or v0 >= v1:
            continue
        uu, vv = np.meshgrid(np.arange(u0, u1) + 0.5, np.arange(v0, v1) + 0.5)
        if mode == "affine":
            Ai = np.linalg.inv(S2)
            du, dv = uu - m[0], vv - m[1]
            qd = Ai[0, 0] * du * du + 2 * Ai[0, 1] * du * dv + Ai[1, 1] * dv * dv
        else:
            V = np.stack([(uu - cx) / f, (vv - cy) / f, np.ones_like(uu)], -1)
            Ai = np.linalg.inv(Sig[i])
            Am = Ai @ mu[i]
            qd = mu[i] @ Am - (V @ Am) ** 2 / np.einsum("...i,ij,...j->...", V, Ai, V)
        al = np.minimum(o[i] * np.exp(-0.5 * qd), 0.99)
        al[al < 1 / 255] = 0
        idx = (np.arange(v0, v1)[:, None] * W + np.arange(u0, u1)[None]).ravel()
        tau[idx] += -np.log1p(-al.ravel())
    return (1 - np.exp(-tau)).reshape(H, W)


def needle_wall(names, sL=0.10, sM=0.008, sS=0.002, x1=3.0, z0=0.6, z1=1.6, dx=0.25, op=0.7, tilt_deg=1.0, seed=0):
    """y = 0 벽 위 가로(x) 바늘 줄 (복도 벽을 걸으며 보는 방향). 줄 간격 2σ_M, 줄마다 x 위치 무작위로 엇갈림."""
    rng = np.random.default_rng(seed)
    pts = []
    for z in np.arange(z0, z1, 2 * sM):
        pts += [(x, 0.0, z) for x in np.arange(0, x1, dx) + rng.uniform(0, dx)]
    P = np.array(pts)
    n = len(P)
    ply = plyio.make(n, names)
    for k in names:
        if k.startswith("f_"):
            ply.data[k] = rng.normal(0, 0.3, n)
    for i, c in enumerate("xyz"):
        ply.data[c] = P[:, i]
    ang = np.radians(rng.normal(0, tilt_deg, n))
    ply.data["rot_0"], ply.data["rot_3"] = np.cos(ang / 2), np.sin(ang / 2)
    ply.data["scale_0"], ply.data["scale_1"], ply.data["scale_2"] = np.log(sL), np.log(sS), np.log(sM)
    ply.data["opacity"] = logit(op)
    return ply


def stripe(alpha, mask):
    """바늘을 가로지르는 방향(화면 세로) 1차 차분 RMS, 벽 영역 안."""
    dd = np.diff(alpha, axis=0)
    m = mask[1:] & mask[:-1]
    return float(np.sqrt(np.mean(dd[m] ** 2)))


def test_split_needles_reduces_grazing_artefact():
    w = needle_wall(plyio.brush_names(1), x1=2.0)
    s, _ = cleanup.split_needles(w)
    front = look_at((1.0, 1.5, 1.1), (1.0, 0, 1.1))
    graze = look_at((-0.3, 0.25, 1.1), (2.0, 0, 1.1))           # 벽과 약 7°
    res = {}
    for name, cam in (("front", front), ("graze", graze)):
        e0, a0 = render_alpha(w, cam, mode="exact"), render_alpha(w, cam)
        e1, a1 = render_alpha(s, cam, mode="exact"), render_alpha(s, cam)
        m = e0 > 0.1
        res[name] = dict(cov=(a0[m].mean(), a1[m].mean()), err=(np.abs(a0 - e0)[m].mean(), np.abs(a1 - e1)[m].mean()),
                         stripe=(stripe(a0, m), stripe(a1, m)))
    c0, c1 = res["front"]["cov"]
    assert abs(c1 / c0 - 1) < 0.04, res                          # 정면 덮임 몇 % 안
    g = res["graze"]
    assert g["err"][1] < 0.4 * g["err"][0], res                  # 래스터라이저 원근 오차 (줄무늬 원인) 크게 줄어듦
    assert g["stripe"][1] < g["stripe"][0], res


# ---------------- remove_floaters ----------------
def make_plan():
    return parse_plan({"version": 1, "units": "m", "rooms": [
        {"id": "A", "floor": [[0, 0], [6, 0], [6, 3], [0, 3]], "floor_z": 0.0, "ceiling_z": 2.6}],
        "doors": []})


def room_scene(names, seed=0, n_surf=20000):
    """방 A 의 여섯 면 위 촘촘한 스플랫 (남쪽 벽 x 2.5~3.1, z 1.0~1.6 은 비움 = 무늬 없어 덜 덮인 곳) + 시험용 무리."""
    rng = np.random.default_rng(seed)
    face = rng.integers(0, 6, n_surf * 2)
    P = np.column_stack([rng.uniform(0, 6, len(face)), rng.uniform(0, 3, len(face)), rng.uniform(0, 2.6, len(face))])
    P[face == 0, 1], P[face == 1, 1], P[face == 2, 0] = 0.0, 3.0, 0.0
    P[face == 3, 0], P[face == 4, 2], P[face == 5, 2] = 6.0, 0.0, 2.6
    hole = (face == 0) & (P[:, 0] > 2.5) & (P[:, 0] < 3.1) & (P[:, 2] > 1.0) & (P[:, 2] < 1.6)
    P = P[~hole][:n_surf] + rng.normal(0, 0.005, (n_surf, 3))
    groups = {"surface": np.arange(n_surf)}
    parts = [P]
    ops = [rng.uniform(0.3, 1.0, n_surf)]
    sizes = [rng.uniform(0.01, 0.04, n_surf)]
    wall_faint = np.column_stack([rng.uniform(0, 6, 400), np.full(400, 0.004), rng.uniform(0.2, 2.4, 400)])
    wall_faint = wall_faint[~((wall_faint[:, 0] > 2.3) & (wall_faint[:, 0] < 3.3) & (wall_faint[:, 2] > 0.8)
                              & (wall_faint[:, 2] < 1.8))][:300]
    def add(name, pts, op, size):
        start = sum(len(p) for p in parts)
        parts.append(np.asarray(pts, float))
        ops.append(np.broadcast_to(op, len(pts)).astype(float))
        sizes.append(np.broadcast_to(size, len(pts)).astype(float))
        groups[name] = np.arange(start, start + len(pts))
    add("wall_faint", wall_faint, 0.05, 0.02)                                  # 벽 속 옅은 것 (밀집 -> 고립 아님)
    add("cluster", rng.normal([2.0, 2.3, 1.0], 0.05, (300, 3)), 0.9, 0.02)    # 방 가운데 물체 (300개)
    add("faint_air", rng.uniform([1, 0.8, 0.4], [5, 2.2, 1.0], (30, 3)), 0.04, 0.03)
    add("big_air", np.array([[4.5, 1.0, 1.0], [1.2, 2.0, 1.8]]), 0.3, 0.25)
    add("blob", rng.normal([4.0, 2.2, 0.9], 0.03, (5, 3)), 0.9, 0.02)          # 작은 불투명 덩어리 (떨어짐)
    add("cam_fog", np.array([[3.0, 1.5, 1.45], [3.05, 1.45, 1.4], [1.0, 1.5, 1.35]]), 0.8, 0.02)
    add("hole_faint", np.column_stack([rng.uniform(2.6, 3.0, 6), np.full(6, 0.08), rng.uniform(1.1, 1.5, 6)]),
        0.05, 0.03)                                                              # 덜 덮인 벽 앞 8 cm (면 보호 대상)
    add("outside", np.array([[3.0, -1.0, 1.0], [8.0, 1.5, 1.2]]), 0.04, 0.03)  # 방 밖 (plan 이면 보호)
    P = np.vstack(parts)
    op, sz = np.concatenate(ops), np.concatenate(sizes)
    n = len(P)
    ply = plyio.make(n, names)
    for k in names:
        ply.data[k] = rng.normal(size=n)
    for i, c in enumerate("xyz"):
        ply.data[c] = P[:, i]
    q = rand_quat(rng, n)
    for i in range(4):
        ply.data[f"rot_{i}"] = q[:, i]
    for i in range(3):
        ply.data[f"scale_{i}"] = np.log(sz * rng.uniform(0.7, 1.0, n))
    ply.data["opacity"] = logit(np.clip(op, 1e-4, 1 - 1e-4))
    cams = np.column_stack([np.linspace(0.5, 5.5, 40), np.full(40, 1.5), np.full(40, 1.4)])
    return ply, groups, cams


def kept_mask(ply, out):
    keys = {bytes(r) for r in rows_bytes(out.data)}
    return np.array([bytes(r) in keys for r in rows_bytes(ply.data)])


@pytest.mark.parametrize("names", LAYOUTS, ids=IDS)
def test_remove_floaters_planes(names):
    ply, G, cams = room_scene(names)
    plan = make_plan()
    planes = planes_from_plan(plan)
    out, info = cleanup.remove_floaters(ply, planes=planes, plan=plan, cameras=cams)
    assert out.names == ply.names
    k = kept_mask(ply, out)
    assert np.array_equal(rows_bytes(out.data), rows_bytes(ply.data[k]))       # 남은 것 비트 그대로·순서 그대로
    for g in ("surface", "wall_faint", "cluster", "hole_faint", "outside"):
        assert k[G[g]].all(), (g, info)
    for g in ("faint_air", "big_air", "blob", "cam_fog"):
        assert not k[G[g]].any(), (g, info)
    rb = info["removed_by"]
    assert rb["camera"] == 3 and rb["detached"] >= 5 and rb["faint"] >= 25 and rb["big"] + rb["faint"] >= 30
    assert info["n_removed"] == ply.n - out.n == sum(rb.values())
    assert info["protected_face"] >= 6 and info["protected_outside"] >= 2


def test_remove_floaters_no_planes_and_removed_rows():
    names = plyio.brush_names(1)
    ply, G, _ = room_scene(names, seed=2)
    out, info = cleanup.remove_floaters(ply, return_removed=True)
    k = kept_mask(ply, out)
    assert k[G["surface"]].all() and k[G["cluster"]].all() and k[G["wall_faint"]].all()
    assert not k[G["faint_air"]].any() and not k[G["blob"]].any()
    assert not k[G["hole_faint"]].any()                    # 면 정보가 없으면 덜 덮인 벽 앞 옅은 것도 지워짐
    assert info["removed_by"]["camera"] == 0
    rem = info["removed"]
    assert rem.n == info["n_removed"] and rem.names == ply.names


def _synthetic_model(cams, pts, W=320, H=240, f=200.0):
    """방 가운데에서 남·북 벽을 보는 카메라 + 무늬 있는 곳의 점 (가려짐은 무시: 볼록한 방)."""
    camera = cm.Camera(1, 1, W, H, np.array([f, f, W / 2, H / 2]))
    points = {i + 1: cm.Point3D(i + 1, p, np.zeros(3, np.uint8), 0.5, np.zeros((0, 2), np.uint32))
              for i, p in enumerate(pts)}
    images = {}
    for j, c in enumerate(cams):
        tgt = c + np.array([0.3 * np.sin(j), 1.0 if j % 2 else -1.0, 0.0])
        Rw, _ = look_at(c, tgt)
        t = -Rw @ c
        Pc = pts @ Rw.T + t
        z = Pc[:, 2]
        u = f * Pc[:, 0] / np.maximum(z, 1e-9) + W / 2
        v = f * Pc[:, 1] / np.maximum(z, 1e-9) + H / 2
        vis = (z > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        images[j + 1] = cm.Image(j + 1, plyio.mat_to_quat(Rw), t, 1, f"f_{j:05d}.jpg",
                                 np.stack([u[vis], v[vis]], 1), np.nonzero(vis)[0] + 1)
    return cm.Model({1: camera}, images, points)


def test_remove_floaters_model_through_and_air():
    """clean_section.py 기준 C (투과)·D (허공): 점이 있는 남·북 벽 앞 허공의 것은 지우고, 점 없는 흰 벽(동·서)·
    바닥 근처는 air_wall 로 보호."""
    names = plyio.brush_names(1)
    rng = np.random.default_rng(5)
    plan = make_plan()
    planes = planes_from_plan(plan)
    cams = np.column_stack([np.linspace(1.0, 5.0, 30), np.full(30, 1.5), np.full(30, 1.4)])
    npt = 15000                                                # 사진마다 칸(48x27)당 점 3개 넘게
    pts = np.vstack([np.column_stack([rng.uniform(0, 6, npt), np.full(npt, y), rng.uniform(0.2, 2.4, npt)])
                     for y in (0.0, 3.0)])
    model = _synthetic_model(cams, pts)
    n_surf = 4000
    S = np.vstack([pts[rng.integers(0, len(pts), n_surf)] + rng.normal(0, 0.003, (n_surf, 3)),
                   np.column_stack([np.full(500, 0.0), rng.uniform(0.2, 2.8, 500), rng.uniform(0.2, 2.4, 500)])])
    air_opaque = np.array([[2.5, 0.6, 1.4], [3.5, 2.4, 1.2], [4.2, 0.7, 1.6]])  # 벽에서 60 cm, 불투명 (고립 아님 정도로 큼)
    white_near = np.column_stack([np.full(4, 0.15), rng.uniform(0.5, 2.5, 4), rng.uniform(0.5, 2.0, 4)])
    P = np.vstack([S, air_opaque, white_near])
    n = len(P)
    ply = plyio.make(n, names)
    for i, c in enumerate("xyz"):
        ply.data[c] = P[:, i]
    ply.data["rot_0"] = 1
    for i in range(3):
        ply.data[f"scale_{i}"] = np.log(0.02)
    ply.data["opacity"] = logit(0.9)
    out, info = cleanup.remove_floaters(ply, planes=planes, plan=plan, model=model, cam_radius=0.0,
                                        cluster_max=1)
    k = kept_mask(ply, out)
    ia = np.arange(len(S), len(S) + 3)
    iw = np.arange(len(S) + 3, n)
    assert not k[ia].any(), info
    assert k[:len(S)].all() and k[iw].all(), info
    rb = info["removed_by"]
    assert rb["through"] + rb["air"] == 3 and rb["through"] >= 1, info
    # model 이 없으면 (불투명·덩어리 아님 판정이 어려운) 이 셋은 남음: 새 기준이 실제로 일함
    out2, info2 = cleanup.remove_floaters(ply, planes=planes, plan=plan, cluster_max=1)
    assert kept_mask(ply, out2)[ia].all()


# ---------------- fix_huge ----------------
@pytest.mark.parametrize("names", LAYOUTS, ids=IDS)
def test_fix_huge(names):
    rng = np.random.default_rng(7)
    ply = random_ply(names, 500, rng, needle_frac=0.0)
    # 0: 투명 큰 것 (지움), 1: 불투명 한 축 큰 것, 2: 불투명 두 축 큰 것
    for i, (s, o) in enumerate([((2.0, 1.5, 0.01), 0.1), ((1.6, 0.05, 0.02), 0.8), ((1.2, 0.9, 0.01), 0.6)]):
        for a in range(3):
            ply.data[f"scale_{a}"][i] = np.log(s[a])
        ply.data["opacity"][i] = logit(o)
    out, info = cleanup.fix_huge(ply, max_axis=0.5)
    assert info["n_huge"] == 3 and info["n_removed"] == 1 and info["n_split"] == 2
    assert np.array_equal(rows_bytes(out.data[:497]), rows_bytes(ply.data[3:]))
    new = out.data[497:]
    Sn = np.exp(np.stack([new[f"scale_{a}"] for a in range(3)], 1).astype(float))
    assert np.all(Sn.max(1) <= 0.5 + 1e-5)
    src1 = new["rot_0"] == ply.data["rot_0"][1]
    src2 = new["rot_0"] == ply.data["rot_0"][2]
    assert src1.sum() >= 4 and src2.sum() >= 4 * 3 and src1.sum() + src2.sum() == len(new)
    assert np.allclose(Sn[src1, 1], 0.05, rtol=1e-6)                     # 작은 축은 그대로
    for k in names:
        if k.startswith("f_"):
            assert np.all(new[k][src2] == ply.data[k][2])
    # 조각 중심의 평균 = 원래 중심
    c2 = np.stack([new[c][src2] for c in "xyz"], 1).astype(float).mean(0)
    assert np.allclose(c2, ply.xyz[2], atol=1e-4)
    assert cleanup.ply_quality(out)["n_huge"] == 0


# ---------------- damp_sh ----------------
@pytest.mark.parametrize("names", LAYOUTS, ids=IDS)
def test_damp_sh(names):
    rng = np.random.default_rng(9)
    n = 2000
    ply = random_ply(names, n, rng, needle_frac=0.0)
    rest = [k for k in names if k.startswith("f_rest_")]
    for k in rest:
        ply.data[k] = rng.normal(0, 0.02, n)
    for i in range(3):
        ply.data[f"f_dc_{i}"] = rng.normal(1.0, 0.2, n)                    # 밝은 벽 (rgb ≈ 0.78)
    spikes = np.arange(10)
    for k in rest[:6]:
        ply.data[k][spikes] = 1.5
    out, info = cleanup.damp_sh(ply, factor=0.3)
    assert info["n_damped"] == 10
    others = np.setdiff1d(np.arange(n), spikes)
    assert np.array_equal(rows_bytes(out.data[others]), rows_bytes(ply.data[others]))
    for k in rest[:6]:
        assert np.allclose(out.data[k][spikes], 1.5 * 0.3, rtol=1e-6)
    for k in names:
        if not k.startswith("f_rest_"):
            assert np.array_equal(out.data[k][spikes], ply.data[k][spikes])
    assert cleanup.ply_quality(ply)["n_sh_outliers"] == 10
    sh0 = plyio.make(5, plyio.brush_names(0))
    o0, i0 = cleanup.damp_sh(sh0)
    assert i0["n_damped"] == 0


# ---------------- cleanup_file ----------------
def test_cleanup_file(tmp_path):
    names = plyio.lichtfeld_names(2)
    ply, G, cams = room_scene(names, n_surf=6000)
    sel = G["surface"][:200]                                             # 바늘 200 (긴/중간 = 15)
    ply.data["scale_0"][sel], ply.data["scale_1"][sel] = np.log(0.45), np.log(0.03)
    ply.data["scale_2"][sel] = np.log(0.01)
    src = tmp_path / "splat.ply"
    plyio.write(src, plyio.GaussianPly(ply.data, ["comment test"]))
    with pytest.raises(ValueError):
        cleanup.cleanup_file(src, src)
    with pytest.raises(TypeError):
        cleanup.cleanup_file(src, tmp_path / "o.ply", bogus=1)
    with pytest.raises(TypeError):
        cleanup.cleanup_file(src, tmp_path / "o.ply", needles={"nope": 1})
    with pytest.raises(ValueError):
        cleanup.cleanup_file(src, tmp_path / "o.ply", steps=("floaters", "magic"))
    plan = make_plan()
    info = cleanup.cleanup_file(src, tmp_path / "o.ply", planes=planes_from_plan(plan), plan=plan, cameras=cams,
                                removed_ply=tmp_path / "removed.ply", needles={"target": 4.0})
    out = plyio.read(tmp_path / "o.ply")
    assert out.names == names and out.comments == ["comment test"]
    qb, qa = info["quality_before"], info["quality_after"]
    assert qb["n_needles"] >= 200 and qa["n_needles"] < qb["n_needles"]
    assert qa["count"] == out.n
    assert info["floaters"]["n_removed"] > 30 and info["needles"]["n_needles"] >= 200
    rem = plyio.read(tmp_path / "removed.ply")
    assert rem.n == info["n_removed_total"] == info["floaters"]["n_removed"] + info["huge"]["n_removed"]
    assert set(info["seconds"]) >= {"read", "floaters", "huge", "needles", "sh", "write"}
    import json
    json.dumps({k: v for k, v in info.items()})                          # 그대로 JSON 으로 쓸 수 있음
