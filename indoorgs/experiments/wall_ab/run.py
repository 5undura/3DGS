"""wall_ab: IndoorGS 학습 후 벽 파이프라인(snap -> cleanup -> repair[±facetex])이 3DGS 화질을 올리는지 (CPU mini3dgs).

  python run.py prep  --out <S>                       장면 + 면 맞추기(refine) + 초기 점 심기(seed)
  python run.py train --out <S> --variant A|B [--iters 2000 --res 256 --threads 2]
  python run.py post  --out <S> --variant A|B         snap -> cleanup -> repair / repair_tex (누적, 파이프라인 기본값)
  python run.py eval  --out <S> --variant A|B         모든 단계 PLY 평가 -> <S>/<v>/eval.json
  python run.py table --out <S>                       표 출력 (markdown)

변형: A = COLMAP 비슷한 성긴 점으로 시작, B = A 점 + seed.seed_points(맞춘 면) (stages/walls.py 와 같은 호출).
후처리는 stages/snap.py 와 같은 인자: snap 은 맞춘 면(fit)만, cleanup 은 모든 면(맞춘 면 + 못 맞춘 면은 도면 위치) +
plan + 카메라 + 모델, repair 는 snap 이 쓴 면 + 카메라 (+ 면 텍스처).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scene  # noqa: E402  (mini3dgs, src 경로도 넣음)
import gs  # noqa: E402
from gs import Camera  # noqa: E402

PINK = (1.0, 0.0, 1.0)
STEPS = ["raw", "snap", "clean", "repair", "repair_tex"]
FILES = {"raw": "point_cloud.ply", "snap": "splat_walls.ply", "clean": "splat_clean.ply",
         "repair": "splat_repaired.ply", "repair_tex": "splat_repaired_tex.ply"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def load_ctx(S: Path):
    from indoorgs import schemas
    from indoorgs.colmap import model as cm
    from indoorgs.walls import planes as wp
    d = json.loads((S / "planes.json").read_text(encoding="utf-8"))
    planes = [wp.plane_from_json(q) for q in d["planes"]]
    fit = {f["id"]: f for f in d["fit"]}
    trusted = [p for p in planes if fit[p.id]["source"] in ("fit", "infer")]
    plan = schemas.load_plan(S / "scene" / "plan.json")
    model = cm.read_model(S / "scene" / "sparse" / "0")
    cams = np.array([im.center for im in model.images.values()])
    return planes, trusted, plan, model, cams


# ---------------------------------------------------------------- prep
def prep(S: Path, n_views=128, width=384, height=240):
    from indoorgs import schemas
    from indoorgs.colmap import model as cm
    from indoorgs.walls import planes as wp, refine, seed
    t0 = time.time()
    sc = scene.make_scene(S / "scene", n_views, width, height)
    log("장면", sc, f"{time.time() - t0:.0f}s")
    m = cm.read_model(Path(sc["model_dir"]))
    plan = schemas.load_plan(sc["plan_json"])
    rooms = wp.rooms_for_cameras(plan, np.array([im.center for im in m.images.values()]))
    planes = wp.planes_from_plan(plan, rooms)
    xyz = np.array([p.xyz for p in m.points.values()])
    w = np.array([len(p.track) for p in m.points.values()], float)
    fitted, info = refine.refine_planes(planes, xyz, weights=w, search=0.30)
    trusted = [q for q, i in zip(fitted, info) if i["source"] == "fit"]
    # 맞춘 면이 실제 면과 얼마나 다른지 (모서리 4곳의 부호 거리, 실제 면 기준)
    true = {p.id: p for p in scene.true_planes()}
    for q, i in zip(fitted, info):
        tp = true[q.id]
        corners = [q.point(u, v) for u in (0, q.size[0]) for v in (0, q.size[1])]
        i["true_err_corners_m"] = [round(float(tp.signed_dist(c)), 4) for c in corners]
    out = {"version": 1, "rooms": rooms, "planes": [wp.plane_to_json(q) for q in fitted],
           "plan_planes": [wp.plane_to_json(q) for q in planes], "fit": info}
    sxyz, srgb, sinfo = seed.seed_points(trusted, m, sc["image_dir"], spacing=0.08)
    sinfo["planes"] = [q.id for q in trusted]
    sd = S / "seed" / "sparse" / "0"
    if not (sd / "points3D.bin").exists():
        seed.write_seeded_model(m, sxyz, srgb, sd)
    out["seed"] = sinfo
    (S / "planes.json").write_text(json.dumps(jsonable(out), indent=1, ensure_ascii=False), encoding="utf-8")
    for i in info:
        log(f"  {i['id']:13s} {i['source']:4s} pts {i['n_points']:4d} cov {i['coverage']:.2f} "
            f"shift {i['shift_m'] * 100:+.1f}cm tilt {i['tilt_deg']:.2f}° 실제와 모서리 차 {i['true_err_corners_m']} "
            f"{i.get('reason', '')}")
    log(f"초기 점 심기: {sinfo['n_added']} 개 (기존 {len(m.points)}), 면 {sinfo['planes']}")


# ---------------------------------------------------------------- train
def train_variant(S: Path, v: str, iters: int, res: int, threads: int, cap: int):
    from train import train
    init = None if v == "A" else str(S / "seed" / "sparse" / "0")
    m = train(S / "scene" / "sparse" / "0", S / "scene" / "images", S / v, iters, res, init_points=init,
              novel_views=None, cap=cap, threads=threads, seed=0, save_renders=4)
    log(json.dumps({"test": m["test"]["mean"], "count": m["count"], "stats": m["ply_stats"], "time": m["time"]}))


# ---------------------------------------------------------------- post
def post(S: Path, v: str, oracle: bool = False):
    """oracle=True: 맞춘 면 대신 실제 면 12개 전부 (refine 이 완벽했다면) -> <S>/<v>_oracle/."""
    from indoorgs.walls import snap, cleanup, repair, facetex
    planes, trusted, plan, model, cams = load_ctx(S)
    D = S / v
    if oracle:
        planes = trusted = scene.true_planes()
        plan = scene.true_plan()
        D = S / f"{v}_oracle"
        D.mkdir(exist_ok=True)
        if not (D / FILES["raw"]).exists():
            (D / FILES["raw"]).symlink_to(S / v / FILES["raw"])
    res = {}
    t = time.time()
    res["snap"] = snap.snap_file(D / FILES["raw"], D / FILES["snap"], trusted, plan, band=0.03, color_tol=0.12,
                                 remove_behind=True)
    res["snap"]["seconds"] = round(time.time() - t, 1)
    log(f"snap: 지움 {res['snap']['n_removed']} 붙임 {res['snap']['n_moved']} 납작 {res['snap']['n_flattened']} "
        f"색으로 건너뜀 {res['snap']['snap']['n_skipped_color']} 면 {res['snap']['planes_used']} 경고 {res['snap']['warnings']}")
    t = time.time()
    res["clean"] = cleanup.cleanup_file(D / FILES["snap"], D / FILES["clean"], planes=planes, plan=plan, cameras=cams,
                                        model=model, removed_ply=D / "removed_clean.ply")
    res["clean"]["seconds_total"] = round(time.time() - t, 1)
    log(f"cleanup: {res['clean']['quality_before']} -> {res['clean']['quality_after']} "
        f"floaters {res['clean']['floaters']['removed_by']}")
    used = [p for p in trusted if p.id in set(res["snap"]["planes_used"])]
    t = time.time()
    res["repair"] = repair.repair_file(D / FILES["clean"], D / FILES["repair"], used, cameras=cams,
                                       min_cover=0.5, max_hole_m2=4.0)
    res["repair"]["seconds_total"] = round(time.time() - t, 1)
    log(f"repair: +{res['repair']['n_added']}")
    tex_dir = S / ("tex_oracle" if oracle else "tex")
    t = time.time()
    if (tex_dir / "done").exists():
        tex = facetex.load_textures(tex_dir, used)
    else:
        tex = facetex.build_face_textures(used, model, S / "scene" / "images")
        facetex.save_textures(tex, tex_dir, used)
        (tex_dir / "done").write_text("ok")
    res["tex_seconds"] = round(time.time() - t, 1)
    res["repair_tex"] = repair.repair_file(D / FILES["clean"], D / FILES["repair_tex"], used, cameras=cams,
                                           texture=tex, min_cover=0.5, max_hole_m2=4.0)
    log(f"repair+tex: +{res['repair_tex']['n_added']}")
    (D / "post.json").write_text(json.dumps(jsonable(res), indent=1, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------- eval
def eval_views(S: Path, res: int):
    """(test 시점 [(name, cam)], novel 시점 [(name, cam, kind)]) + 정답 캐시 (rgb, 입사각, 깊이)."""
    from train import load_scene
    sc = load_scene(S / "scene" / "sparse" / "0", S / "scene" / "images", res)
    test = [(v.name, v.camera) for v in sc.test]
    nov = scene.novel_cameras([c for _, c in test])
    novel = [(c.name, c, ("graze" if c.name.startswith("graze") else c.name.rsplit("_", 1)[1])) for c in nov]
    cache = S / f"gt_{res}.npz"
    if cache.exists():
        z = np.load(cache)
        gt = {k: (z[k + "/rgb"], z[k + "/inc"], z[k + "/depth"]) for k in [n for n, _ in test] + [n for n, _, _ in novel]}
    else:
        gt, arrs = {}, {}
        for n, c in test + [(n, c) for n, c, _ in novel]:
            rgb, inc, dep, _ = scene.render_gt_full(c, 2)
            if not n.startswith("graze") and n in dict(test):      # 테스트 시점은 학습 사진 (노이즈 포함) 그대로
                rgb = sc_img(sc, n)
            gt[n] = (rgb.astype(np.float32), inc.astype(np.float32), dep.astype(np.float32))
            arrs[n + "/rgb"], arrs[n + "/inc"], arrs[n + "/depth"] = gt[n]
        np.savez_compressed(cache, **arrs)
    return test, novel, gt


def sc_img(sc, name):
    for v in sc.test:
        if v.name == name:
            return v.image.numpy()
    raise KeyError(name)


def poster_cam(res_w: int):
    """포스터 정면 (벽에서 1.5 m) 시점 - 렌더 깊이로 포스터가 벽에서 튀어나온 정도 재기."""
    w, h = res_w, int(round(res_w * 240 / 384))
    x = sum(scene.POSTER["x"]) / 2
    z = sum(scene.POSTER["z"]) / 2
    return scene._cam(scene.look(-math.pi / 2, 0.0), [x, -scene.CW + 1.5, z], w, h, "poster_front")


def poster_metrics(ply, cam_front):
    """포스터 가우시안 (불투명 0.5 이상, 앞면 범위 안쪽 3 cm, 벽에서 ±8 cm) 의 벽까지 거리 (불투명도 가중 중앙값) +
    정면 렌더 깊이로 잰 '포스터 면 - 벽 면' 차 (정답 2.5 cm)."""
    d = ply.data
    P = np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)
    op = ply.opacity()
    x0, x1 = scene.POSTER["x"]
    z0, z1 = scene.POSTER["z"]
    sd = P[:, 1] + scene.CW
    m = (op >= 0.5) & (P[:, 0] > x0 + 0.03) & (P[:, 0] < x1 - 0.03) & (P[:, 2] > z0 + 0.03) & (P[:, 2] < z1 - 0.03) \
        & (np.abs(sd) < 0.08)
    out = {"n": int(m.sum())}
    if m.any():
        o = np.argsort(sd[m])
        cw = np.cumsum(op[m][o])
        out["median_sd_cm"] = round(float(sd[m][o][np.searchsorted(cw, cw[-1] / 2)]) * 100, 2)
        out["frac_on_wall"] = round(float(np.mean(np.abs(sd[m]) < 0.005)), 3)
    r = gs.render(ply, cam_front, with_depth=True)
    _, inc, gdep, qi = scene.render_gt_full(cam_front, 1)
    names = [q.name for q in scene.QUADS]
    pm = qi == names.index("poster")
    wm = qi == names.index("c_south")
    if pm.any() and wm.any():
        # 정면 시점: 렌더 깊이(알파 가중) - 정답 깊이, 포스터 픽셀과 그 옆 흰 벽 픽셀 각각 (중앙값, cm).
        # relief = 벽 렌더 깊이 - 포스터 렌더 깊이 (정답 2.5 cm). 알파가 낮은 픽셀(<0.5)은 뺌.
        dep, al = r["depth"], r["alpha"]
        pm2, wm2 = pm & (al > 0.5), wm & (al > 0.5)
        if pm2.any() and wm2.any():
            out["poster_depth_err_cm"] = round(float(np.median(dep[pm2] - gdep[pm2])) * 100, 2)
            out["wall_depth_err_cm"] = round(float(np.median(dep[wm2] - gdep[wm2])) * 100, 2)
            out["render_relief_cm"] = round(float(np.median(dep[wm2]) - np.median(dep[pm2])) * 100, 2)
            out["gt_relief_cm"] = round(float(np.median(gdep[wm2]) - np.median(gdep[pm2])) * 100, 2)
    return out


def masked_psnr(se_sum, n):
    return float(10 * math.log10(1.0 / max(se_sum / max(n, 1), 1e-12))) if n else None


def evaluate_ply(path: Path, test, novel, gt, true_planes, res_w: int, save_dir: Path | None = None):
    from indoorgs import plyio
    from indoorgs.walls import snap
    import torch
    ply = plyio.read(path)
    p, deg = gs.ply_to_params(ply)
    rows = {}

    def one(n, cam):
        with torch.no_grad():
            out, _ = gs.render_params(p, cam, deg, (0.0, 0.0, 0.0))
        img = out["image"].clamp(0, 1).numpy()
        a = out["alpha"].numpy()
        g, inc, _ = gt[n]
        se = ((img - g) ** 2).mean(-1)
        r = {"psnr": float(10 * math.log10(1 / max(se.mean(), 1e-12))),
             "ssim": float(gs.ssim(torch.from_numpy(img), torch.from_numpy(g))),
             "leak": float(np.mean(1 - a > 0.05)), "leak_mean": float(np.mean(1 - a)),
             "g_se": float(se[inc > 70].sum()), "g_n": int((inc > 70).sum())}
        return r, img, a

    saved = 0
    for n, cam in test:
        rows[n], _, _ = one(n, cam)
        rows[n]["kind"] = "test"
    for n, cam, kind in novel:
        rows[n], img, a = one(n, cam)
        rows[n]["kind"] = kind
        if save_dir is not None and kind in ("graze", "U+", "Y+") and saved < 8 and int(n[-2:] if kind == "graze" else 0) % 3 == 0:
            save_dir.mkdir(parents=True, exist_ok=True)
            pink = img + (1 - a[..., None]) * np.array(PINK)
            strip = np.concatenate([gt[n][0], img, np.clip(pink, 0, 1)], 1)
            cv2.imwrite(str(save_dir / f"{path.stem}_{n.replace('+', 'p').replace('.png', '')}.png"),
                        (strip[..., ::-1] * 255).round().astype(np.uint8))
            saved += 1

    def agg(sel):
        rr = [r for r in rows.values() if sel(r)]
        if not rr:
            return {}
        return {"n": len(rr), "psnr": float(np.mean([r["psnr"] for r in rr])), "ssim": float(np.mean([r["ssim"] for r in rr])),
                "leak": float(np.mean([r["leak"] for r in rr])), "leak_mean": float(np.mean([r["leak_mean"] for r in rr])),
                "graze_px_psnr": masked_psnr(sum(r["g_se"] for r in rr), sum(r["g_n"] for r in rr))}
    out = {"test": agg(lambda r: r["kind"] == "test"),
           "novel": agg(lambda r: r["kind"] != "test"),
           "novel_offset": agg(lambda r: r["kind"] in ("R+", "R-", "U+", "U-")),
           "novel_yaw60": agg(lambda r: r["kind"] in ("Y+", "Y-")),
           "graze_views": agg(lambda r: r["kind"] == "graze"),
           "stats": plyio.stats(ply)}
    fl = snap.splat_flatness(ply, true_planes, band=0.05)
    out["flat"] = {"thick_all_cm": fl["all"]["rms_thick"] * 100, "dist_all_cm": fl["all"]["rms_dist"] * 100,
                   "thick_wall_cm": fl["wall"]["rms_thick"] * 100, "count_wall": fl["wall"]["count"],
                   "within_1cm": fl["all"]["within_1cm"]}
    out["poster"] = poster_metrics(ply, poster_cam(res_w))
    out["views"] = rows
    return out


def eval_variant(S: Path, v: str, res: int):
    test, novel, gt = eval_views(S, res)
    tp = scene.true_planes()
    D = S / v
    res_all = {}
    for st in STEPS:
        f = D / FILES[st]
        if not f.exists():
            continue
        t = time.time()
        res_all[st] = evaluate_ply(f, test, novel, gt, tp, res, D / "eval_renders")
        e = res_all[st]
        log(f"{v} {st:10s} test {e['test']['psnr']:.2f} novel {e['novel']['psnr']:.2f} graze {e['graze_views']['psnr']:.2f} "
            f"gpx {e['novel']['graze_px_psnr']:.2f} leak {e['novel']['leak']:.4f} thick {e['flat']['thick_wall_cm']:.2f}cm "
            f"n {e['stats']['count']} poster {e['poster']}  {time.time() - t:.0f}s")
    (D / "eval.json").write_text(json.dumps(jsonable(res_all), indent=1, ensure_ascii=False), encoding="utf-8")


def table(S: Path, variants=("A", "B", "A_oracle", "B_oracle")):
    lines = ["| 변형 | 단계 | 테스트 PSNR / SSIM | 새 시점 PSNR / SSIM | 비스듬 20장 PSNR | 입사각>70° 픽셀 PSNR | 비침(새 시점) | 벽 두께 RMS cm | 개수 | 반투명 | 바늘 | 포스터 깊이 cm (가우시안 / 렌더) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in variants:
        f = S / v / "eval.json"
        if not f.exists():
            continue
        E = json.loads(f.read_text(encoding="utf-8"))
        for st in STEPS:
            if st not in E:
                continue
            e = E[st]
            po = e["poster"]
            lines.append(
                f"| {v} | {st} | {e['test']['psnr']:.2f} / {e['test']['ssim']:.3f} | {e['novel']['psnr']:.2f} / {e['novel']['ssim']:.3f} | "
                f"{e['graze_views']['psnr']:.2f} | {e['novel']['graze_px_psnr']:.2f} | {e['novel']['leak'] * 100:.2f}% | "
                f"{e['flat']['thick_wall_cm']:.2f} | {e['stats']['count']:,} | {e['stats']['translucent_ratio']:.3f} | "
                f"{e['stats']['needle_ratio']:.3f} | {po.get('median_sd_cm')} / {po.get('render_relief_cm')} (정답 {po.get('gt_relief_cm')}) |")
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "train", "post", "eval", "table"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="A")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--cap", type=int, default=100_000)
    ap.add_argument("--oracle", action="store_true", help="post: 실제 면으로 (refine 완벽 가정), 결과 <v>_oracle")
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(a.threads)
    S = Path(a.out)
    S.mkdir(parents=True, exist_ok=True)
    if a.cmd == "prep":
        prep(S)
    elif a.cmd == "train":
        train_variant(S, a.variant, a.iters, a.res, a.threads, a.cap)
    elif a.cmd == "post":
        post(S, a.variant, a.oracle)
    elif a.cmd == "eval":
        eval_variant(S, a.variant, a.res)
    else:
        table(S)


if __name__ == "__main__":
    main()
