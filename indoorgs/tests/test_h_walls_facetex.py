"""walls/facetex: 사진을 면에 투영한 텍스처 + 못 본 곳 패치 메우기 + repair 색 출처 (합성 실내)."""
import math

import cv2
import numpy as np
import pytest

from indoorgs import plyio, schemas
from indoorgs.colmap import model as cm
from indoorgs.walls import facetex as ft
from indoorgs.walls import planes as wp
from indoorgs.walls import repair

H_ROOM = 2.6
RES = 0.015
IMG_W, IMG_H, F = 320, 240, 250.0
SS = 2                                   # 광선 초과 표본 (사진 한 픽셀 = SS x SS 광선 평균)
DOOR = (4.4, 5.2, 0.0, 2.0)              # 벽 A:w0 의 문 구멍 (u0, u1, v0, v1)
TILE = 0.30                              # 타일 한 변 (체크 무늬 주기 0.6 m)
POSTER = (1.0, 2.0, 1.2, 2.0)
GAINS = [0.85, 1 / 0.85, 1.1, 1 / 1.1, 1.0, 1.0]   # 사진별 노출 (기하 평균 1)


def make_planes():
    plan = schemas.parse_plan({"version": schemas.VERSION, "units": "m", "rooms": [
        {"id": "A", "floor": [[0, 0], [6, 0], [6, 3], [0, 3]], "floor_z": 0.0, "ceiling_z": H_ROOM}],
        "doors": [{"id": "d", "rooms": ["A"], "center": [4.8, 0.0], "width": 0.8, "height": 2.0}]})
    planes = wp.planes_from_plan(plan)
    wall = next(p for p in planes if p.id == "A:w0")      # y = 0, n = +y, U = +x, V = +z
    return planes, wall


def wall_rgb(u, v):
    """벽 A:w0 의 참 색 (float RGB): 부드러운 줄눈의 체크 타일 + 부드러운 포스터."""
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    iu, iv = np.floor(u / TILE), np.floor(v / TILE)
    chk = ((iu + iv) % 2)[..., None]
    base = np.where(chk > 0, np.array([190.0, 175.0, 150.0]), np.array([150.0, 160.0, 175.0]))
    du = np.abs(u / TILE - np.round(u / TILE)) * TILE
    dv = np.abs(v / TILE - np.round(v / TILE)) * TILE
    g = np.exp(-0.5 * (np.minimum(du, dv) / 0.008) ** 2)[..., None]
    col = base * (1 - g) + np.array([90.0, 85.0, 80.0]) * g
    u0, u1, v0, v1 = POSTER

    def ramp(x, a, b, s=0.01):
        return 1 / (1 + np.exp(-(x - a) / s)) * (1 - 1 / (1 + np.exp(-(x - b) / s)))
    m = (ramp(u, u0, u1) * ramp(v, v0, v1))[..., None]
    pu, pv = (u - u0) / (u1 - u0), (v - v0) / (v1 - v0)
    poster = np.stack([120 + 80 * np.sin(2 * np.pi * pu), 110 + 70 * np.cos(2 * np.pi * pv),
                       100 + 60 * np.sin(np.pi * (pu + pv))], -1)
    return col * (1 - m) + poster * m


OTHER = {"A:w1": (120, 140, 90), "A:w2": (140, 120, 100), "A:w3": (100, 110, 140), "A:floor": (90, 90, 90),
         "A:ceiling": (200, 200, 200)}


def look_rot(d):
    z = np.asarray(d, float)
    z /= np.linalg.norm(z)
    y = np.array([0, 0, -1.0])
    y -= (y @ z) * z
    y /= np.linalg.norm(y)
    return np.stack([np.cross(y, z), y, z])


def rot2q(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


def render(planes, R, C, w=IMG_W, h=IMG_H, f=F):
    """광선 쏘기 (SS x SS 초과 표본 평균). 벽 A:w0 은 wall_rgb, 다른 면은 단색, 문 너머는 검정."""
    W2, H2 = w * SS, h * SS
    j, i = np.meshgrid(np.arange(W2), np.arange(H2))
    d = np.stack([((j + 0.5) / SS - w / 2) / f, ((i + 0.5) / SS - h / 2) / f, np.ones_like(j, float)],
                 -1).reshape(-1, 3) @ R
    best = np.full(len(d), np.inf)
    img = np.zeros((len(d), 3))
    for p in planes:
        if p.signed_dist(C) <= 0:
            continue
        den = d @ p.n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (p.d - p.n @ C) / den
        k = np.flatnonzero((den < 0) & (t > 1e-6) & (t < best))
        X = C + t[k, None] * d[k]
        uv = p.uv(X)
        hit = p.contains_uv(uv)
        k, uv = k[hit], uv[hit]
        best[k] = t[k]
        img[k] = wall_rgb(uv[:, 0], uv[:, 1]) if p.id == "A:w0" else OTHER[p.id]
    return img.reshape(H2, W2, 3).reshape(h, SS, w, SS, 3).mean((1, 3))


def views():
    return [((x, 2.4, z), (0.0, -1.0, 0.0)) for x in (1.0, 2.0, 3.0) for z in (0.9, 1.7)]


def make_scene(tmp_path, planes, gains=GAINS):
    image_dir = tmp_path / "사진"
    image_dir.mkdir(exist_ok=True)
    cams = {1: cm.Camera(1, 1, IMG_W, IMG_H, np.array([F, F, IMG_W / 2, IMG_H / 2]))}
    images = {}
    for k, ((C, look), g) in enumerate(zip(views(), gains), 1):
        C = np.asarray(C, float)
        R = look_rot(look)
        img = np.clip(np.round(render(planes, R, C) * g), 0, 255).astype(np.uint8)
        name = f"f_{k:03d}.png"
        ok, buf = cv2.imencode(".png", img[:, :, ::-1])
        buf.tofile(str(image_dir / name))
        images[k] = cm.Image(k, rot2q(R), -R @ C, 1, name)
    return cm.Model(cams, images, {}), image_dir


def truth_texture(wall, res=RES, ss=4):
    H, W, uv, inside = ft.texture_grid(wall, res)
    o = (np.arange(ss) + 0.5) / ss - 0.5
    acc = np.zeros((H, W, 3))
    for a in o:
        for b in o:
            acc += wall_rgb(uv[..., 0] + a * res, uv[..., 1] + b * res)
    return acc / ss / ss, inside


def psnr(a, b):
    mse = float(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2))
    return 10 * math.log10(255 ** 2 / max(mse, 1e-12))


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("facetex")
    planes, wall = make_planes()
    model, image_dir = make_scene(tmp, planes)
    tex = ft.face_texture_from_photos(wall, model, image_dir, res=RES, occluders=planes)
    truth, inside = truth_texture(wall)
    return planes, wall, model, image_dir, tex, truth, inside


def edge_margin(tex, px=3):
    """사진 텍스처 경계(못 본 곳·문·면 가장자리)에서 px 픽셀 안쪽만 (경계 픽셀은 시점 수가 적어 섞임)."""
    from scipy import ndimage
    return ndimage.binary_erosion(tex.source == ft.SRC_PHOTO, np.ones((2 * px + 1, 2 * px + 1)))


def test_photo_texture_matches_truth(scene):
    planes, wall, model, image_dir, tex, truth, inside = scene
    H, W = tex.shape
    assert (H, W) == (math.ceil(H_ROOM / RES - 1e-9), math.ceil(6.0 / RES - 1e-9))
    assert (tex.inside == inside).all()
    seen = tex.source == ft.SRC_PHOTO
    assert seen.sum() * RES ** 2 > 8.0                       # 보인 곳 ~ 4.5 m x 2.6 m 중 대부분
    # 문 구멍·못 본 쪽은 사진이 없음
    _, _, uv, _ = ft.texture_grid(wall, RES)
    door = (uv[..., 0] > DOOR[0]) & (uv[..., 0] < DOOR[1]) & (uv[..., 1] < DOOR[3])
    assert not tex.inside[door].any() and (tex.source[door] == 0).all()
    assert (tex.source[:, uv[0, :, 0] > 4.7] == 0).all()     # x 3.0 카메라 끝 (3.0 + 1.54 m) 너머
    assert (tex.support[seen] >= 2).all()
    m = edge_margin(tex)
    p = psnr(tex.image[m], truth[m])
    assert p > 30, p
    # 노출 보정이 효과가 있음 (끄면 낮음)
    g = tex.meta["gains"]
    assert abs(g["f_001.png"][0] * 0.85 - 1) < 0.05 and abs(g["f_002.png"][1] * (1 / 0.85) - 1) < 0.05
    raw = ft.face_texture_from_photos(wall, model, image_dir, res=RES, occluders=planes, gain_comp=False)
    p_raw = psnr(raw.image[m], truth[m])
    assert p > p_raw + 2, (p, p_raw)
    print(f"\n[facetex] photo PSNR {p:.1f} dB (gain off {p_raw:.1f} dB), seen {seen.sum() * RES ** 2:.2f} m2")


def test_inpaint_patchmatch_beats_telea(scene):
    planes, wall, model, image_dir, tex, truth, inside = scene
    pm, filled = ft.inpaint_texture(tex, method="patchmatch", rng=0)
    te, filled_t = ft.inpaint_texture(tex, method="telea")
    assert (filled == filled_t).all()
    assert (tex.source == pm.source)[~filled].all()                   # 나머지 안 바뀜
    assert (pm.image[tex.source == 1] == tex.image[tex.source == 1]).all()
    assert (pm.source[filled] == ft.SRC_INPAINT).all()
    assert filled.sum() * RES ** 2 > 1.5
    assert not (filled & ~inside).any()                              # 문 구멍 비어 있음
    assert (pm.source[~inside] == 0).all() and (pm.image[~inside] == 0).all()
    assert pm.stats()["empty_m2"] == pytest.approx(0, abs=1e-6)
    # (1) 사진 텍스처 가까이 메운 곳 (모서리, 문 왼쪽·위): 무늬 위상까지 맞음
    # (2) 문 뒤 넓은 띠 (u > 5.25, 문 위 0.6 m 띠로만 이어짐): 위상은 못 맞혀도 타일 주기는 이어 감
    _, _, uv, _ = ft.texture_grid(wall, RES)
    band = filled & (uv[..., 0] > 5.25)
    near = filled & ~band

    def mae(im, m):
        return np.abs(im[m].astype(float) - truth[m]).mean()
    e_pm, e_te = mae(pm.image, near), mae(te.image, near)
    b_pm, b_te = mae(pm.image, band), mae(te.image, band)

    def vcorr(im, lag):
        im = np.asarray(im, float)
        sub = im[2:132, 352:398, 0] - im[2:132, 352:398, 2]          # 띠 안 (v 0.03..1.98), R-B = 체크 색
        sub = sub - sub.mean()
        return (sub[:-lag] * sub[lag:]).mean() / sub.var()
    per = int(round(2 * TILE / RES))
    c_pm, c_te, c_tr = vcorr(pm.image, per), vcorr(te.image, per), vcorr(np.round(truth), per)
    h_pm, h_tr = vcorr(pm.image, per // 2), vcorr(np.round(truth), per // 2)
    print(f"\n[facetex] inpaint MAE vs truth near: patchmatch {e_pm:.1f}, telea {e_te:.1f}; far band: "
          f"patchmatch {b_pm:.1f}, telea {b_te:.1f}; band autocorr @0.6 m pm {c_pm:.2f} telea {c_te:.2f} "
          f"truth {c_tr:.2f}, @0.3 m pm {h_pm:.2f} truth {h_tr:.2f} (filled {filled.sum() * RES ** 2:.2f} m2, "
          f"{pm.meta['inpaint']['seconds']} s)")
    assert near.sum() * RES ** 2 > 1.0 and band.sum() * RES ** 2 > 1.0
    assert e_pm < 0.3 * e_te, (e_pm, e_te)
    assert c_pm > 0.4 and h_pm < -0.2 and c_pm > c_te + 0.2, (c_pm, h_pm, c_te)


def test_inpaint_cap_and_mask(scene):
    planes, wall, model, image_dir, tex, truth, inside = scene
    out, filled = ft.inpaint_texture(tex, method="telea", max_fill_m2=0.05)
    assert filled.sum() * RES ** 2 <= 0.05 + 1e-9
    assert out.meta["inpaint"]["skipped_m2"] > 1.5
    mask = np.zeros(tex.shape, bool)
    mask[10:20, 380:390] = True
    mask[100:110, 150:160] = True                                     # 사진 픽셀 -> 안 바뀜
    out, filled = ft.inpaint_texture(tex, mask, method="patchmatch", rng=1)
    assert (tex.source[100:110, 150:160] == 1).all()
    assert filled[10:20, 380:390].all() and not filled[100:110, 150:160].any()
    with pytest.raises(ValueError):
        ft.inpaint_texture(tex, method="nope")


def test_save_load_and_edit(scene, tmp_path):
    planes, wall, model, image_dir, tex, truth, inside = scene
    pm, _ = ft.inpaint_texture(tex, method="telea")
    out = tmp_path / "tex 텍스처"
    paths = ft.save_textures({wall.id: pm}, out, planes)
    assert paths[0].name == "A_w0.png" and (out / "A_w0.json").is_file()
    back = ft.load_textures(out, planes)[wall.id]
    assert (back.image == pm.image).all() and (back.source == pm.source).all()
    assert (back.support == pm.support).all() and back.meta["edited"] is False
    assert ft.load_textures(out)[wall.id].inside.sum() == inside.sum()     # JSON 의 면으로
    # PNG 위 = 큰 v (천장 쪽)
    png = cv2.imdecode(np.fromfile(str(paths[0]), np.uint8), cv2.IMREAD_COLOR)
    assert (png[0, 5, ::-1] == pm.image[-1, 5]).all()
    # 포토샵처럼 고침: 반 크기로 줄이고 알파로 한 곳 지우기 + 빈 곳 칠하기
    H, W = pm.shape
    small = cv2.resize(png, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    a = np.full(small.shape[:2], 255, np.uint8)
    a[:10, :10] = 0                                                   # 위 왼쪽 (큰 v, 작은 u) 지움
    small[-5:, 80:120] = (0, 0, 255)                                  # 문 아래쪽 (문 구멍 = inside 아님) 은 무시
    edited = np.dstack([small, a])
    ok, buf = cv2.imencode(".png", edited)
    buf.tofile(str(paths[0]))
    e = ft.load_texture(out / "A_w0.json", wall)
    assert e.meta["edited"] and e.meta["resized"] and e.shape == (H, W)
    assert (e.source[-18:, :18] == 0).all()
    assert (e.source[~inside] == 0).all()
    # 빈 곳에 알파 > 0 으로 칠한 곳 = 손 (source 3): 원래 비었던 곳을 비워 두고 저장한 뒤 칠함
    empty = tex.copy()
    ft.save_texture(empty, out / "b", wall)
    png = cv2.imdecode(np.fromfile(str(out / "b" / "A_w0.png"), np.uint8), cv2.IMREAD_COLOR)
    png[60:80, 380:390] = (10, 200, 10)                               # u 5.7..5.85, v 1.4..1.7: 못 본 곳
    rgba = np.dstack([png, np.full(png.shape[:2], 255, np.uint8)])
    ok, buf = cv2.imencode(".png", rgba)
    buf.tofile(str(out / "b" / "A_w0.png"))
    e2 = ft.load_texture(out / "b" / "A_w0.png", wall)
    r = H - 70
    assert e2.source[r, 385] == ft.SRC_MANUAL and (e2.image[r, 385] == (10, 200, 10)).all()


def _repair_ply(wall, names, seed=0, spacing=0.03):
    """A:w0 위 납작한 가우시안 격자, 구멍 하나 (u 3.0, v 1.2, r 0.3)."""
    rng = np.random.default_rng(seed)
    uu, vv = np.meshgrid(np.arange(0.015, wall.size[0], spacing), np.arange(0.015, wall.size[1], spacing))
    U, V = uu.ravel() + rng.normal(0, 0.003, uu.size), vv.ravel() + rng.normal(0, 0.003, uu.size)
    keep = wall.contains_uv(np.stack([U, V], 1)) & ((U - 3.0) ** 2 + (V - 1.2) ** 2 > 0.3 ** 2)
    U, V = U[keep], V[keep]
    N = len(U)
    ply = plyio.make(N, names)
    for k in names:
        ply.data[k] = rng.normal(0, 0.3, N)
    P = wall.point(U, V)
    for i, c in enumerate("xyz"):
        ply.data[c] = P[:, i]
    q0 = np.array([np.cos(-np.pi / 4), np.sin(-np.pi / 4), 0, 0])      # 가우시안 z -> +y (벽 법선)
    for i in range(4):
        ply.data[f"rot_{i}"] = q0[i]
    ply.data["scale_0"] = np.log(0.022)
    ply.data["scale_1"] = np.log(0.022)
    ply.data["scale_2"] = np.log(0.002)
    ply.data["opacity"] = 3.0
    return ply


@pytest.mark.parametrize("rest", ["keep", "zero"])
def test_repair_with_texture(rest):
    planes, wall = make_planes()
    names = plyio.brush_names(2)
    ply = _repair_ply(wall, names)
    N = ply.n
    truth, inside = truth_texture(wall)
    tex = ft._empty_texture(wall, RES)
    tex.image[inside] = np.round(truth[inside]).astype(np.uint8)
    tex.source[inside] = ft.SRC_PHOTO
    out, info = repair.repair_faces(ply, [wall], cell=0.05, min_cover=0.5, rng=1, texture={wall.id: tex},
                                    texture_rest=rest)
    fi = info["faces"][wall.id]
    new = out.data[N:]
    assert len(new) > 50 and fi["n_textured"] == len(new)
    np.testing.assert_array_equal(out.data[:N], ply.data)
    uv = wall.uv(np.stack([new["x"], new["y"], new["z"]], 1).astype(float))
    rgb = (np.stack([new[f"f_dc_{i}"] for i in range(3)], 1) * plyio.SH_C0 + 0.5) * 255
    want, src = tex.sample(uv)
    assert (src == 1).all()
    assert np.abs(rgb - want).max() < 0.05
    assert np.abs(rgb - wall_rgb(uv[:, 0], uv[:, 1])).mean() < 4             # 참 무늬와도 맞음
    rest_names = [k for k in names if k.startswith("f_rest_")]
    rest_vals = np.stack([new[k] for k in rest_names], 1)
    assert (rest_vals == 0).all() if rest == "zero" else (np.abs(rest_vals).max() > 0)
    # 텍스처 없는 면/곳은 예전 방식 그대로 (같은 rng -> 같은 결과)
    a, _ = repair.repair_faces(ply, [wall], cell=0.05, min_cover=0.5, rng=1)
    b, _ = repair.repair_faces(ply, [wall], cell=0.05, min_cover=0.5, rng=1, texture={"X:w9": tex})
    np.testing.assert_array_equal(a.data, b.data)


def test_build_face_textures(scene):
    planes, wall, model, image_dir, tex, truth, inside = scene
    texs = ft.build_face_textures(planes, model, image_dir, res=0.03, faces=["A:w0", "A:w2"], method="telea")
    assert set(texs) == {"A:w0", "A:w2"}
    assert (texs["A:w2"].source == 0).all()                         # 카메라 뒤 벽
    t = texs["A:w0"]
    assert t.res == 0.03 and (t.source == ft.SRC_INPAINT).any() and (t.source == ft.SRC_PHOTO).any()
