"""run_e2e.py 결과 평가 (정답 gt_poses.json, gt_planes.json 과 비교). 표(markdown)와 eval.json.

  python eval_e2e.py <v4 장면 폴더> <작업 폴더> [<작업 폴더> ...]
"""
from __future__ import annotations

import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parent / "prior_matching"))
import pairlib as pl  # noqa: E402
from indoorgs import schemas  # noqa: E402
from indoorgs.colmap import model as cm  # noqa: E402
from indoorgs.stages.diagnose import find_teleports  # noqa: E402
from indoorgs.walls import planes as wp  # noqa: E402

GT_NAME = {"corr:floor": "floor", "corr:ceiling": "ceiling"}


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / xs.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def robust_sim3(src, dst, thr=0.5, iters=1000, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        idx = rng.choice(len(src), 3, replace=False)
        s, R, t = umeyama(src[idx], dst[idx])
        if not np.isfinite(s):
            continue
        inl = np.linalg.norm(s * src @ R.T + t - dst, axis=1) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    for _ in range(3):
        s, R, t = umeyama(src[best], dst[best])
        best = np.linalg.norm(s * src @ R.T + t - dst, axis=1) < thr
    return s, R, t


def mean_rotation(Ms):
    U, _, Vt = np.linalg.svd(np.sum(Ms, 0))
    D = np.diag([1, 1, np.linalg.det(U @ Vt)])
    return U @ D @ Vt


def pose_metrics(m: cm.Model, gt: dict, pri: dict) -> dict:
    ims = sorted(m.images.values(), key=lambda im: im.name)
    names = [im.name for im in ims]
    C = np.array([im.center for im in ims])
    Rm = np.array([im.R.T for im in ims])                 # world_from_cam
    Cg = np.array([gt[n][1] for n in names])
    Rg = np.array([gt[n][0] for n in names])
    e = np.linalg.norm(C - Cg, axis=1)                     # 도면 좌표 그대로
    s, R, t = robust_sim3(C, Cg)
    es = np.linalg.norm(s * C @ R.T + t - Cg, axis=1)      # 모델 안에서 맞게 놓였나
    D = mean_rotation(np.einsum("nij,nkj->nik", Rm, Rg))   # 정답 세계 -> 모델 세계 (평균)
    up = D @ np.array([0, 0, 1.0])
    P = np.array([pri.get(n, [np.nan] * 3) for n in names], float)
    dp = np.linalg.norm(C - P, axis=1)
    return {"registered": len(ims), "plan_bad05": int((e > 0.5).sum()), "plan_bad1": int((e > 1.0).sum()),
            "plan_median_m": float(np.median(e)), "sim3_bad05": int((es > 0.5).sum()), "sim3_bad1": int((es > 1.0).sum()),
            "teleports": len(find_teleports(names, C)),
            "tilt_up_deg": math.degrees(math.acos(min(1.0, up[2]))),
            "roll_about_x_deg": math.degrees(math.atan2(D[2, 1], D[2, 2])),
            "pitch_about_y_deg": math.degrees(math.atan2(-D[2, 0], math.hypot(D[2, 1], D[2, 2]))),
            "far_prior_1m": int((dp > 1.0).sum()), "far_prior_1_5m": int((dp > 1.5).sum()),
            "far_prior_1m_truly_bad": int(((dp > 1.0) & (e > 1.0)).sum()),
            "far_prior_1_5m_truly_bad": int(((dp > 1.5) & (e > 1.0)).sum()),
            "scale_vs_gt": float(s)}


def step_times(tool_log: Path) -> list[tuple[str, float, datetime]]:
    """[(명령, 초, 시작 시각)] (colmap_shim 이 찍은 SHIM <명령> <초>s)."""
    out, cur = [], None
    for line in tool_log.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] \$ \S+ (\S+)", line)
        if m:
            cur = (m.group(2), datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
            continue
        m = re.match(r"SHIM (\S+) ([\d.]+)s$", line)
        if m and cur:
            out.append((cur[0], float(m.group(2)), cur[1]))
    return out


def false_pairs(db: Path, sc_gt, K, px) -> tuple[int, int]:
    d = pl.read_db(db)
    lab = pl.label_pairs(d, sc_gt, K, px)
    return len(lab), int(sum(lab.values()))


def walls_metrics(sec: Path, gtp: dict) -> list[dict]:
    pj = schemas.read_json(sec / "walls" / "planes.json")
    gtn = {p["name"]: (np.array(p["n"], float), float(p["d"])) for p in gtp["planes"]}
    rows = []
    for fj, pj0, info in zip(pj["planes"], pj["plan_planes"], pj["fit"]):
        q, q0 = wp.plane_from_json(fj), wp.plane_from_json(pj0)
        best = max(gtn, key=lambda k: float(gtn[k][0] @ q.n))
        n, d = gtn[best]
        c = q.point(q.size[0] / 2, q.size[1] / 2) if q.polygon is None else q.point(q.size[0] / 2, q.size[1] / 2)
        c0 = q0.point(q0.size[0] / 2, q0.size[1] / 2)
        rows.append({"id": q.id, "gt": best, "source": info.get("source"),
                     "offset_cm": round(100 * float(n @ c - d), 1),          # + = 정답 면보다 방 안쪽
                     "plan_offset_cm": round(100 * float(n @ c0 - d), 1),
                     "tilt_deg": round(wp.angle_deg(q.n, n), 2), "n_assigned": info.get("n_assigned"),
                     "relaxed": bool(info.get("relaxed")), "reason": info.get("reason") or info.get("reason_relaxed")})
    return rows, pj.get("seed", {})


def evaluate(scene: Path, work: Path) -> dict:
    g = schemas.read_json(scene / "gt_poses.json")
    gt = {f["image"]: (np.array(f["R_world_from_cam"]), np.array(f["center"])) for f in g["frames"]}
    K = np.array(g["K"])
    px = 4.0 * g["size"][0] / 640
    gtp = schemas.read_json(scene / "gt_planes.json")
    sec = work / "sections" / "v4"
    col = sec / "colmap"
    pri = {f["image"]: np.array(f["xyz"]) for f in schemas.read_json(col / "priors.json")["frames"]}
    reg = schemas.read_json(col / "register.json")
    out = {"work": str(work), "register": reg}
    out["sparse_models"] = sorted((cm.count_images(d) for d in (col / "sparse").iterdir() if d.is_dir()), reverse=True)
    if (col / "sparse_pass1").exists():
        out["pass1_models"] = sorted((cm.count_images(d) for d in (col / "sparse_pass1").iterdir() if d.is_dir()),
                                     reverse=True)
    out["final"] = pose_metrics(cm.read_model(sec / "data" / "sparse" / "0"), gt, pri)
    # COLMAP 이 놓은 그대로(직접 맞추기 전): 가장 큰 모델
    from indoorgs.stages.register import best_model
    b, _ = best_model(col / "sparse")
    out["raw_colmap"] = pose_metrics(cm.read_model(b), gt, pri)
    out["steps"] = [(c, s) for c, s, _ in step_times(next((work / "logs").glob("*_register_colmap.log")))]
    st = step_times(next((work / "logs").glob("*_register_colmap.log")))
    # 짝 검증 시간 = 후보 짝 끝 ~ 2차 mapper 시작 (파이썬 검증 + DB 복사)
    for i, (c, s, t0) in enumerate(st):
        if c in ("spatial_matcher", "exhaustive_matcher") and i + 1 < len(st):
            out["verify_seconds"] = (st[i + 1][2] - t0).total_seconds() - s
    out["pairs_db"] = false_pairs(col / "db.db", gt, K, px)
    if (col / "db_verified.db").exists():
        out["pairs_verified"] = false_pairs(col / "db_verified.db", gt, K, px)
    dj = schemas.read_json(sec / "diagnose.json")
    out["diagnose"] = {k: (len(v) if isinstance(v, list) else v) for k, v in dj.items()
                       if k in ("registered", "teleports", "outside_space", "far_from_prior", "excluded")}
    if (sec / "walls" / "planes.json").exists():
        out["walls"], out["seed"] = walls_metrics(sec, gtp)
        out["clean_model"] = pose_metrics(cm.read_model(sec / "data_clean" / "sparse" / "0"), gt, pri)
    schemas.write_json(work / "eval.json", out)
    return out


def fmt(out: dict) -> str:
    L = [f"## {out['work']}"]
    r = out["register"]
    L.append(f"mapper {r.get('mapper')}, aligned {r.get('aligned')}, model {r.get('model')}, "
             f"sparse 모델 {out['sparse_models']}, 1차 모델 {out.get('pass1_models')}")
    L.append(f"짝 (가짜): db.db {out['pairs_db']}, db_verified {out.get('pairs_verified')}")
    L.append("| 모델 | 등록 | 도면 >0.5/>1 m | 도면 중앙 m | Sim3 >0.5/>1 | 순간이동 | 위 기울기° | x축 둘레° | y축 둘레° | 축척 | 제약 >1 m (진짜 틀림) | >1.5 m (진짜) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in ("raw_colmap", "final", "clean_model"):
        if k in out:
            p = out[k]
            L.append(f"| {k} | {p['registered']} | {p['plan_bad05']}/{p['plan_bad1']} | {p['plan_median_m']:.2f} | "
                     f"{p['sim3_bad05']}/{p['sim3_bad1']} | {p['teleports']} | {p['tilt_up_deg']:.2f} | "
                     f"{p['roll_about_x_deg']:.2f} | {p['pitch_about_y_deg']:.2f} | {p['scale_vs_gt']:.3f} | "
                     f"{p['far_prior_1m']} ({p['far_prior_1m_truly_bad']}) | {p['far_prior_1_5m']} ({p['far_prior_1_5m_truly_bad']}) |")
    L.append(f"단계 시간: {[(c, round(s)) for c, s in out['steps']]}, 검증 {out.get('verify_seconds', 0):.0f}s")
    L.append(f"diagnose: {out['diagnose']}")
    if "walls" in out:
        L.append("| 면 | 정답 | source | 오프셋 cm (도면 cm) | 기울기° | 배정 점 | relaxed | 이유 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for w in out["walls"]:
            L.append(f"| {w['id']} | {w['gt']} | {w['source']} | {w['offset_cm']:+.1f} ({w['plan_offset_cm']:+.1f}) | "
                     f"{w['tilt_deg']:.2f} | {w['n_assigned']} | {w['relaxed']} | {w['reason'] or ''} |")
        L.append(f"seed: n_added {out['seed'].get('n_added')}, planes {out['seed'].get('planes')}")
    return "\n".join(L)


if __name__ == "__main__":
    scene = Path(sys.argv[1]).resolve()
    for w in sys.argv[2:]:
        print(fmt(evaluate(scene, Path(w).resolve())))
        print()
