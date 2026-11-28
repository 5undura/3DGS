"""wall_ab 2차 (RESULTS2.md): 버그 1·2·3·5 수정 뒤 다시 재기 + "후처리 뒤 짧은 재학습(fine-tune)".

  python run2.py prep  --out <S2> --old <S>       장면·정답 캐시·A 학습 결과는 <S> 것을 링크. 새 refine(뒤쪽 층 + 느슨한 기준)
                                                  으로 planes.json, 맞춘 면에 seed -> <S2>/seed. refine 비교표 refine_cmp.json
  python run.py train --out <S2> --variant B --iters 2500     (새 seed 로 B 다시 학습)
  python run.py post / eval --out <S2> --variant A|B           (새 코드, 예전과 같은 인자) ; python ablate.py --out <S2> ...
  python run2.py rec   --out <S2> --variant B     추천 기본값 묶음 -> <S2>/<v>_rec/ (+ 평가 rec.json)
  python run2.py ft    --out <S2> --variant B --src raw|snap|clean|repair_tex|floaters|snap_move|rec [--iters 1000] [--flat 0.1]
                                                  그 PLY 에서 이어 학습 (같은 손실, 밀도 조절 없음), 300/600/끝 체크포인트 평가
                                                  -> <S2>/ft/<v>_<src>[_flatX]/ft.json
  python run2.py table --out <S2>                 표 (markdown)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import run as R
import scene

SRC = {"raw": ("{v}", "point_cloud.ply"), "snap": ("{v}", "splat_walls.ply"), "clean": ("{v}", "splat_clean.ply"),
       "repair_tex": ("{v}", "splat_repaired_tex.ply"), "floaters": ("{v}_abl", "clean_floaters.ply"),
       "snap_move": ("{v}_abl", "snap_move_only.ply"), "rec": ("{v}_rec", "rec.ply"),
       "rec_snap": ("{v}_rec", "rec_snap.ply")}
CKPT = (300, 600)


def link(dst: Path, src: Path):
    if not dst.exists() and not dst.is_symlink():
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src)


# ---------------------------------------------------------------- prep
def refine_variants(planes, xyz, w):
    from indoorgs.walls import refine
    true = {p.id: p for p in scene.true_planes()}
    V = {"예전 (backmost=False, relaxed=False)": dict(backmost=False, relaxed=False),
         "뒤쪽 층만": dict(relaxed=False),
         "뒤쪽 층 + 도면 치수 믿기 (느슨한 기준 없음)": dict(relaxed=False, trust_plan_dims=True),
         "뒤쪽 층 + 느슨한 기준 (새 기본)": dict(),
         "뒤쪽 층 + 느슨한 기준 + 도면 치수 믿기": dict(trust_plan_dims=True)}
    out = {}
    for name, kw in V.items():
        fitted, info = refine.refine_planes(planes, xyz, weights=w, search=0.30, **kw)
        rows = []
        for q, i in zip(fitted, info):
            tp = true[q.id]
            err = [float(tp.signed_dist(q.point(u, v))) for u in (0, q.size[0]) for v in (0, q.size[1])]
            rows.append({"id": q.id, "source": i["source"], "n_points": i.get("n_points"), "coverage": i.get("coverage"),
                         "tilt_deg": round(float(i.get("tilt_deg", 0.0)), 3), "max_err_cm": round(max(map(abs, err)) * 100, 2),
                         "err_cm": [round(e * 100, 2) for e in err], "n_front_excluded": i.get("n_front_excluded"),
                         "relaxed": bool(i.get("relaxed")), "offset_only": bool(i.get("offset_only")),
                         "inferred_from": i.get("inferred_from"),
                         "reason": i.get("reason") or i.get("reason_relaxed") or i.get("reason_infer") or ""})
        out[name] = rows
    return out


def prep(S2: Path, S: Path):
    from indoorgs import schemas
    from indoorgs.colmap import model as cm
    from indoorgs.walls import planes as wp, refine, seed
    S2.mkdir(parents=True, exist_ok=True)
    link(S2 / "scene", S / "scene")
    link(S2 / "gt_256.npz", S / "gt_256.npz")
    for f in ("point_cloud.ply", "metrics.json"):
        link(S2 / "A" / f, S / "A" / f)
    m = cm.read_model(S2 / "scene" / "sparse" / "0")
    plan = schemas.load_plan(S2 / "scene" / "plan.json")
    rooms = wp.rooms_for_cameras(plan, np.array([im.center for im in m.images.values()]))
    planes = wp.planes_from_plan(plan, rooms)
    xyz = np.array([p.xyz for p in m.points.values()])
    w = np.array([len(p.track) for p in m.points.values()], float)
    cmp_ = refine_variants(planes, xyz, w)
    (S2 / "refine_cmp.json").write_text(json.dumps(R.jsonable(cmp_), indent=1, ensure_ascii=False), encoding="utf-8")
    for name, rows in cmp_.items():
        nf = sum(r["source"] != "plan" for r in rows)
        errs = [r["max_err_cm"] for r in rows if r["source"] != "plan"]
        R.log(f"{name}: 맞춘 면 {nf}/12, 모서리 오차 최대 {max(errs):.2f} cm, 평균 {np.mean(errs):.2f} cm")
    fitted, info = refine.refine_planes(planes, xyz, weights=w, search=0.30)
    trusted = [q for q, i in zip(fitted, info) if i["source"] in ("fit", "infer")]
    true = {p.id: p for p in scene.true_planes()}
    for q, i in zip(fitted, info):
        tp = true[q.id]
        i["true_err_corners_m"] = [round(float(tp.signed_dist(q.point(u, v))), 4)
                                   for u in (0, q.size[0]) for v in (0, q.size[1])]
    out = {"version": 1, "rooms": rooms, "planes": [wp.plane_to_json(q) for q in fitted],
           "plan_planes": [wp.plane_to_json(q) for q in planes], "fit": info}
    sxyz, srgb, sinfo = seed.seed_points(trusted, m, S2 / "scene" / "images", spacing=0.08)
    sinfo["planes"] = [q.id for q in trusted]
    sd = S2 / "seed" / "sparse" / "0"
    if not (sd / "points3D.bin").exists():
        seed.write_seeded_model(m, sxyz, srgb, sd)
    out["seed"] = sinfo
    (S2 / "planes.json").write_text(json.dumps(R.jsonable(out), indent=1, ensure_ascii=False), encoding="utf-8")
    R.log(f"초기 점 심기: {sinfo['n_added']} 개 (기존 {len(m.points)}), 면 {len(trusted)}개 {sinfo['planes']}")


# ---------------------------------------------------------------- 추천 묶음
REC = {"snap": dict(band=0.03, color_tol=0.12, flatten=False, remove_behind=True),
       "cleanup_steps": ("floaters",), "repair": False}


def rec(S2: Path, v: str, res: int):
    """추천 기본값 묶음: snap(옮기기만, 넓이 가중 색 검사, 벽 뒤 지우기 안전장치 그대로) -> cleanup(floaters) (repair 없음)."""
    from indoorgs.walls import snap, cleanup
    planes, trusted, plan, model, cams = R.load_ctx(S2)
    D = S2 / f"{v}_rec"
    D.mkdir(exist_ok=True)
    raw = S2 / v / R.FILES["raw"]
    info = {"snap": snap.snap_file(raw, D / "rec_snap.ply", trusted, plan, **REC["snap"])}
    info["clean"] = cleanup.cleanup_file(D / "rec_snap.ply", D / "rec.ply", planes=planes, plan=plan, cameras=cams,
                                         model=model, steps=REC["cleanup_steps"])
    test, novel, gt = R.eval_views(S2, res)
    tp = scene.true_planes()
    ev = {}
    for k in ("rec_snap", "rec"):
        e = R.evaluate_ply(D / f"{k}.ply", test, novel, gt, tp, res)
        e.pop("views")
        ev[k] = e
        R.log(f"{v} {k}: test {e['test']['psnr']:.2f} novel {e['novel']['psnr']:.2f} graze {e['graze_views']['psnr']:.2f} "
              f"leak {e['novel']['leak'] * 100:.2f}% thick {e['flat']['thick_wall_cm']:.2f}")
    (D / "rec.json").write_text(json.dumps(R.jsonable({"eval": ev, "post": {
        "snap": {k: info["snap"].get(k) for k in ("n_removed", "n_moved", "n_flattened", "planes_used")},
        "skipped_color": info["snap"]["snap"]["n_skipped_color"],
        "floaters": info["clean"]["floaters"]["removed_by"]}}), indent=1, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------- fine-tune
def ft(S2: Path, v: str, src: str, iters: int, res: int, threads: int, flat: float):
    from train import train
    d, f = SRC[src]
    init = S2 / d.format(v=v) / f
    if not init.exists():
        raise FileNotFoundError(init)
    tag = f"{v}_{src}" + (f"_flat{flat:g}" if flat else "")
    D = S2 / "ft" / tag
    opts = {}
    if flat:
        planes, trusted, plan, model, cams = R.load_ctx(S2)
        opts = {"flat_planes": [(p.n, p.d) for p in trusted], "flat_lambda": flat}
    t = time.time()
    if (D / "point_cloud.ply").exists() and all((D / f"point_cloud_{c}.ply").exists() for c in CKPT):
        m = {"time": {"s_per_iter": None}}           # 학습은 끝났고 평가만 다시 (예: metrics.json 쓰다 멈춘 경우)
    else:
        m = train(S2 / "scene" / "sparse" / "0", S2 / "scene" / "images", D, iters, res, init_ply=init,
                  threads=threads, seed=0, checkpoints=CKPT, save_renders=0, **opts)
    tsec = time.time() - t
    test, novel, gt = R.eval_views(S2, res)
    tp = scene.true_planes()
    out = {"src": str(init), "iters": iters, "train_s": round(tsec, 1), "s_per_iter": m["time"]["s_per_iter"],
           "flat": flat, "ckpt": {}}
    e0 = R.evaluate_ply(init, test, novel, gt, tp, res)
    e0.pop("views")
    out["ckpt"]["0"] = e0
    for it in list(CKPT) + [iters]:
        p = D / ("point_cloud.ply" if it == iters else f"point_cloud_{it}.ply")
        e = R.evaluate_ply(p, test, novel, gt, tp, res)
        e.pop("views")
        out["ckpt"][str(it)] = e
        R.log(f"ft {tag} it {it}: test {e['test']['psnr']:.2f} novel {e['novel']['psnr']:.2f} "
              f"graze {e['graze_views']['psnr']:.2f} gpx {e['novel']['graze_px_psnr']:.2f} leak {e['novel']['leak'] * 100:.2f}% "
              f"thick {e['flat']['thick_wall_cm']:.2f} n {e['stats']['count']} tr {e['stats']['translucent_ratio']:.3f}")
    (D / "ft.json").write_text(json.dumps(R.jsonable(out), indent=1, ensure_ascii=False), encoding="utf-8")


def table(S2: Path):
    rows = ["| 시작 PLY | 반복 | 테스트 PSNR | 새 시점 | 스침 20장 | 입사각>70° | 비침 | 벽 두께 cm | 개수 | 반투명 |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for f in sorted((S2 / "ft").glob("*/ft.json")):
        E = json.loads(f.read_text(encoding="utf-8"))
        for it, e in E["ckpt"].items():
            rows.append(f"| {f.parent.name} | {it} | {e['test']['psnr']:.2f} | {e['novel']['psnr']:.2f} | "
                        f"{e['graze_views']['psnr']:.2f} | {e['novel']['graze_px_psnr']:.2f} | {e['novel']['leak'] * 100:.2f}% | "
                        f"{e['flat']['thick_wall_cm']:.2f} | {e['stats']['count']:,} | {e['stats']['translucent_ratio']:.3f} |")
    print("\n".join(rows))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "rec", "ft", "table"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--old", default=None)
    ap.add_argument("--variant", default="B")
    ap.add_argument("--src", default="raw", choices=sorted(SRC))
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--flat", type=float, default=0.0, help="ft: 납작하게 정규화 가중치 (0 = 끔)")
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(a.threads)
    S2 = Path(a.out)
    if a.cmd == "prep":
        prep(S2, Path(a.old))
    elif a.cmd == "rec":
        rec(S2, a.variant, a.res)
    elif a.cmd == "ft":
        ft(S2, a.variant, a.src, a.iters, a.res, a.threads, a.flat)
    else:
        table(S2)


if __name__ == "__main__":
    main()
