"""합성 복도 + 실제 COLMAP(pycolmap 4.2.1, CPU) 실험: 동선 위치 제약이 '순간이동' 오정합을 막는지.

  python run.py --out <dir> [--stages render,features,map,eval] [--runs r1,r2,r3,...] [--ambiguity strong|weak]

PYTHONPATH 에 indoorgs/src 가 없으면 자동으로 넣음 (indoorgs.pathing / colmap.database / stages.diagnose 를 그대로 씀).
단계마다 결과가 있으면 건너뜀 (다시 하려면 해당 폴더 지움).
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import scene  # noqa: E402
from indoorgs import pathing, schemas  # noqa: E402
from indoorgs.colmap import database  # noqa: E402
from indoorgs.stages.diagnose import find_teleports  # noqa: E402

# 실행 목록: 이름 -> (매칭, mapper 에 줄 priors 종류, 강건 손실)
# 매칭 "spatial:<kind>" = priors(<kind>)를 DB 에 넣고 COLMAP spatial_matcher (제약 위치가 가까운 사진끼리만 매칭)
RUNS = {
    "r1_exh": ("exhaustive", None, False),          # 예전 설정
    "r2_seq": ("sequential", None, False),
    "r3_exh_prior": ("exhaustive", "good", False),  # = pose_prior_mapper
    "r3_seq_prior": ("sequential", "good", False),
    "r5_spa": ("spatial:good", None, False),        # 동선은 매칭 짝 고르기에만
    "r5_spa_prior": ("spatial:good", "good", False),  # 동선으로 짝 고르기 + pose_prior_mapper
    "r4_spa_prior_t2s": ("spatial:t2s", "t2s", False),  # 틀린 동선: 가운데 고정점 시간 +2초
    "r4_spa_prior_x1m": ("spatial:x1m", "x1m", False),        # 틀린 동선: 그린 선·고정점 전체가 x 로 1m 어긋남
    "r4_exh_prior_t2s": ("exhaustive", "t2s", False),
    # 진단용 (현실에선 불가능): 정답 기하와 안 맞는 '가짜 짝'을 DB 에서 지움 -> 매칭만 맞으면 재구성이 되는지
    "r6_exh_oracle": ("exhaustive+oracle", None, False),
    "r6_spa_prior_oracle": ("spatial:good+oracle", "good", False),
    # 현실적 거르기: 동선 진행 방향이 120도 넘게 다른 두 사진(갈 때 vs 올 때 같은 자리)의 짝을 지움
    "r7_spa_heading": ("spatial:good+heading", None, False),
    "r7_spa_heading_prior": ("spatial:good+heading", "good", False),
}
SPATIAL_MAX_DIST = 6.0       # m. 제약 오차(최대 ~1.6m) + 복도 끝을 멀리 보는 사진 고려
SPATIAL_MAX_NEIGHBORS = 80


def split_seed(run: str) -> tuple[str, int]:
    """"r1_exh_s2" -> ("r1_exh", 2): 같은 설정을 mapper random_seed 만 바꿔 다시 (결과가 얼마나 흔들리는지)."""
    head, _, tail = run.rpartition("_s")
    if head and tail.isdigit() and head in RUNS:
        return head, int(tail)
    return run, 0


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- 1. 렌더
_TEX = None


def _init(amb):
    global _TEX
    _TEX = scene.make_textures(0, amb)


def _render_one(args):
    i, name, R, C, out = args
    img = scene.render(_TEX, R, C, rng=np.random.default_rng(1000 + i))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    Path(out, name).write_bytes(buf.tobytes())
    return name


def stage_render(out: Path, amb: str):
    img_dir = out / "images"
    traj = scene.gt_trajectory()
    if img_dir.is_dir() and len(list(img_dir.glob("*.jpg"))) == len(traj):
        log("render: 있음, 건너뜀")
        return
    img_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with mp.Pool(4, initializer=_init, initargs=(amb,)) as pool:
        pool.map(_render_one, [(i, n, R, C, str(img_dir)) for i, (n, t, R, C) in enumerate(traj)], chunksize=4)
    gt = {"version": 1, "fps": scene.SRC_FPS, "K": scene.K.tolist(), "size": [scene.IMG_W, scene.IMG_H],
          "ambiguity": amb,
          "frames": [{"image": n, "t": t, "R_world_from_cam": R.tolist(), "center": C.tolist()}
                     for n, t, R, C in traj]}
    schemas.write_json(out / "gt_poses.json", gt)
    schemas.write_json(out / "gt_planes.json", scene.planes_json())
    log(f"render: {len(traj)}장 {time.time() - t0:.0f}s")


def make_priors(out: Path, kind: str) -> Path:
    shift = {"good": (0, 0), "t2s": (2.0, 0), "x1m": (0, 1.0)}[kind]
    pj = scene.drawn_path(time_shift=shift[0], offset_m=shift[1])
    path = schemas.parse_path(pj)
    names = sorted(p.name for p in (out / "images").glob("*.jpg"))
    pri, warns = pathing.build_priors(path, names, scene.SRC_FPS)
    d = out / "priors"
    d.mkdir(exist_ok=True)
    schemas.write_json(d / f"path_{kind}.json", pj)
    schemas.write_json(d / f"priors_{kind}.json", pri)
    # 제약 자체의 오차 (정답 대비)
    gt = {f["image"]: np.array(f["center"]) for f in schemas.read_json(out / "gt_poses.json")["frames"]}
    e = np.array([np.linalg.norm(np.array(f["xyz"]) - gt[f["image"]]) for f in pri["frames"]])
    exy = np.array([np.linalg.norm(np.array(f["xyz"][:2]) - gt[f["image"]][:2]) for f in pri["frames"]])
    schemas.write_json(d / f"priors_{kind}_err.json", {"mean": float(e.mean()), "median": float(np.median(e)),
                                                         "max": float(e.max()), "xy_rmse": float(np.sqrt((exy ** 2).mean())),
                                                         "warnings": warns})
    log(f"priors {kind}: 정답과 거리 평균 {e.mean():.2f} 중앙 {np.median(e):.2f} 최대 {e.max():.2f} m, 경고 {warns}")
    return d / f"priors_{kind}.json"


# ---------------------------------------------------------------- 2. 특징·매칭
def stage_features(out: Path, runs: list[str]):
    import pycolmap
    db = out / "db" / "features.db"
    if db.exists():
        log("features: 있음")
    else:
        db.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        eo = pycolmap.FeatureExtractionOptions()
        eo.num_threads = 4
        eo.use_gpu = False
        ro = pycolmap.ImageReaderOptions()
        ro.camera_model = "SIMPLE_RADIAL"   # 초점거리 모름 (EXIF 없음) -> COLMAP 이 추정
        pycolmap.extract_features(db, out / "images", camera_mode=pycolmap.CameraMode.SINGLE,
                                  reader_options=ro, extraction_options=eo, device=pycolmap.Device.cpu)
        log(f"features: {time.time() - t0:.0f}s")
    times = {}
    want = {RUNS[split_seed(r)[0]][0] for r in runs}
    kinds = ["exhaustive", "sequential"] + sorted({k.split("+")[0] for k in want if k.startswith("spatial:")})
    kinds += sorted(k for k in want if "+" in k)
    for kind in kinds:
        mdb = out / "db" / f"{kind.replace(':', '_')}.db"
        if mdb.exists():
            continue
        if "+" in kind:
            base, filt = kind.split("+")
            shutil.copy(out / "db" / f"{base.replace(':', '_')}.db", mdb.with_suffix(".tmp"))
            if filt == "oracle":
                n, m = oracle_clean(out, mdb.with_suffix(".tmp"))
            else:
                n, m = heading_filter(out, mdb.with_suffix(".tmp"), base.split(":")[1])
            mdb.with_suffix(".tmp").rename(mdb)
            log(f"filter {kind}: 짝 {n}/{m} 지움")
            continue
        tmp = mdb.with_suffix(".tmp")
        shutil.copy(db, tmp)
        mo = pycolmap.FeatureMatchingOptions()
        mo.num_threads = 4
        mo.use_gpu = False
        t0 = time.time()
        if kind == "exhaustive":
            pycolmap.match_exhaustive(tmp, matching_options=mo, device=pycolmap.Device.cpu)
        elif kind.startswith("spatial:"):
            n, missing = database.write_pose_priors(tmp, schemas.read_json(make_priors(out, kind.split(":")[1])))
            assert n and not missing, missing
            po = pycolmap.SpatialPairingOptions()
            po.max_distance = SPATIAL_MAX_DIST
            po.max_num_neighbors = SPATIAL_MAX_NEIGHBORS
            po.ignore_z = True
            pycolmap.match_spatial(tmp, matching_options=mo, pairing_options=po, device=pycolmap.Device.cpu)
        else:
            po = pycolmap.SequentialPairingOptions()
            po.overlap = 20
            po.loop_detection = False   # 어휘 트리 파일이 오프라인에 없음 -> 끔
            pycolmap.match_sequential(tmp, matching_options=mo, pairing_options=po, device=pycolmap.Device.cpu)
        tmp.rename(mdb)
        times[kind] = time.time() - t0
        log(f"match {kind}: {times[kind]:.0f}s")
    if times:
        f = out / "db" / "match_times.json"
        old = schemas.read_json(f) if f.exists() else {"version": 1}
        old.update(times)
        schemas.write_json(f, old)


def heading_filter(out: Path, db: Path, prior_kind: str, max_deg: float = 120.0) -> tuple[int, int]:
    """동선(priors)에서 프레임별 진행 방향 = 앞뒤 프레임 제약 위치 차이. 두 사진의 진행 방향이 max_deg 넘게 다르면 짝 삭제.
    (멈춤 등으로 방향을 모르면 지우지 않음). 정답 정보는 안 씀."""
    import sqlite3
    from contextlib import closing
    fr = schemas.read_json(out / "priors" / f"priors_{prior_kind}.json")["frames"]
    xy = np.array([f["xyz"][:2] for f in fr])
    d = np.zeros_like(xy)
    d[1:-1] = xy[2:] - xy[:-2]
    n = np.linalg.norm(d, axis=1)
    ok = n > 0.05
    d[ok] /= n[ok, None]
    head = {f["image"]: (d[i] if ok[i] else None) for i, f in enumerate(fr)}
    cmax = math.cos(math.radians(max_deg))
    with closing(sqlite3.connect(str(db))) as c, c:
        ims = dict(c.execute("SELECT image_id, name FROM images"))
        rows = [p for (p,) in c.execute("SELECT pair_id FROM two_view_geometries WHERE rows > 0")]
        drop = []
        for pid in rows:
            i2 = pid % 2147483647
            i1 = (pid - i2) // 2147483647
            a, b = head[ims[i1]], head[ims[i2]]
            if a is not None and b is not None and float(a @ b) < cmax:
                drop.append(pid)
        c.executemany("DELETE FROM two_view_geometries WHERE pair_id=?", [(p,) for p in drop])
    return len(drop), len(rows)


def oracle_clean(out: Path, db: Path, max_px: float = 4.0) -> tuple[int, int]:
    """정답 자세로 만든 F 에 대해 inlier 매칭의 Sampson 오차 중앙값이 max_px 넘는 짝을 two_view_geometries 에서 지움."""
    import sqlite3
    from contextlib import closing
    g = schemas.read_json(out / "gt_poses.json")
    Ki = np.linalg.inv(np.array(g["K"]))
    gt = {f["image"]: (np.array(f["R_world_from_cam"]), np.array(f["center"])) for f in g["frames"]}

    def skew(t):
        return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    with closing(sqlite3.connect(str(db))) as c, c:
        ims = dict(c.execute("SELECT image_id, name FROM images"))
        kp = {i: np.frombuffer(b, np.float32).reshape(r, cc)[:, :2]
              for i, r, cc, b in c.execute("SELECT image_id, rows, cols, data FROM keypoints")}
        rows = c.execute("SELECT pair_id, rows, data FROM two_view_geometries WHERE rows > 0").fetchall()
        drop = []
        for pid, r, b in rows:
            i2 = pid % 2147483647
            i1 = (pid - i2) // 2147483647
            m = np.frombuffer(b, np.uint32).reshape(r, 2)
            (R1, C1), (R2, C2) = gt[ims[i1]], gt[ims[i2]]
            F = Ki.T @ skew(R2.T @ (C1 - C2)) @ (R2.T @ R1) @ Ki
            x1 = np.c_[kp[i1][m[:, 0]], np.ones(r)]
            x2 = np.c_[kp[i2][m[:, 1]], np.ones(r)]
            Fx1, Ftx2 = x1 @ F.T, x2 @ F
            e = np.sqrt(np.sum(x2 * Fx1, 1) ** 2 / (Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2))
            if np.median(e) > max_px:
                drop.append(pid)
        c.executemany("DELETE FROM two_view_geometries WHERE pair_id=?", [(p,) for p in drop])
    return len(drop), len(rows)


# ---------------------------------------------------------------- 3. 정합
def stage_map(out: Path, run: str):
    import pycolmap
    base, seed = split_seed(run)
    match, prior, robust = RUNS[base]
    rd = out / "runs" / run
    if (rd / "done.json").exists():
        log(f"map {run}: 있음")
        return
    if rd.exists():
        shutil.rmtree(rd)   # 이 스크립트가 만든 실패한 중간 산출물
    (rd / "sparse").mkdir(parents=True)
    db = rd / "db.db"
    shutil.copy(out / "db" / f"{match.replace(':', '_')}.db", db)
    n_pri = 0
    if prior:   # spatial DB 에는 이미 같은 priors 가 있음 (write_pose_priors 는 같은 사진 것을 바꿔 씀)
        pj = make_priors(out, prior)
        n_pri, missing = database.write_pose_priors(db, schemas.read_json(pj))
        assert not missing, missing
        assert database.count_pose_priors(db) == n_pri
    o = pycolmap.IncrementalPipelineOptions()
    o.num_threads = 4
    o.random_seed = seed
    o.use_prior_position = bool(prior)
    o.use_robust_loss_on_prior_position = robust
    t0 = time.time()
    recs = pycolmap.incremental_mapping(db, out / "images", rd / "sparse", o)
    dt = time.time() - t0
    schemas.write_json(rd / "done.json", {"version": 1, "match": match, "prior": prior, "robust": robust,
                                          "n_priors": n_pri, "seconds": dt, "models": len(recs)})
    log(f"map {run}: 모델 {len(recs)}개 {[r.num_reg_images() for r in recs.values()]} {dt:.0f}s")


# ---------------------------------------------------------------- 4. 평가
def umeyama(src, dst, scale=True):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / xs.var(0).sum() if scale else 1.0
    return s, R, mu_d - s * R @ mu_s


def robust_sim3(src, dst, thr=0.5, iters=2000, seed=0):
    """RANSAC + Umeyama (잘못 붙은 카메라에 끌려가지 않게). 돌려줌 s,R,t, inlier."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        idx = rng.choice(len(src), 3, replace=False)
        try:
            s, R, t = umeyama(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        inl = np.linalg.norm((s * src @ R.T + t) - dst, axis=1) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    for _ in range(3):
        s, R, t = umeyama(src[best], dst[best])
        best = np.linalg.norm((s * src @ R.T + t) - dst, axis=1) < thr
    s, R, t = umeyama(src[best], dst[best])
    return s, R, t, best


def rot_err_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return math.degrees(math.acos(max(-1, min(1, c))))


def plane_stats(X, planes):
    """점 -> 가장 가까운 정답 면까지 거리."""
    D = np.stack([X @ np.array(p["n"]) - p["d"] for p in planes], 1)  # 안쪽 +
    j = np.argmin(np.abs(D), 1)
    d = D[np.arange(len(X)), j]
    near = np.abs(d) < 0.3
    out = {"n_points": int(len(X)), "frac_within_5cm": float(np.mean(np.abs(d) < 0.05)),
           "frac_within_30cm": float(np.mean(near)),
           "rms_near_m": float(np.sqrt(np.mean(d[near] ** 2))) if near.any() else None,
           "median_abs_near_m": float(np.median(np.abs(d[near]))) if near.any() else None}
    return out


def plot_xy(path: Path, C, Cg, title):
    """위에서 본 그림: 회색 = 정답, 색 = 추정 (시간 순 파랑->빨강), 노란 선 = 같은 사진의 정답-추정 차이."""
    path.parent.mkdir(exist_ok=True)
    sc, mx, my = 50, 60, 120
    Wp, Hp = int(scene.L * sc) + 2 * mx, int(10 * sc) + 2 * my

    def px(p):
        return int(mx + p[0] * sc), int(Hp / 2 - p[1] * sc)
    img = np.full((Hp, Wp, 3), 255, np.uint8)
    cv2.rectangle(img, px((0, scene.W / 2)), px((scene.L, -scene.W / 2)), (0, 0, 0), 2)
    for a, b in zip(Cg, Cg[1:]):
        cv2.line(img, px(a), px(b), (180, 180, 180), 3)
    n = len(C)
    for i in range(n):
        cv2.line(img, px(C[i]), px(Cg[i]), (0, 200, 255), 1)
    for i in range(n):
        f = i / max(1, n - 1)
        col = (int(255 * (1 - f)), 0, int(255 * f))
        if i:
            cv2.line(img, px(C[i - 1]), px(C[i]), col, 1)
        cv2.circle(img, px(C[i]), 3, col, -1)
    cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "gray=GT  blue->red=estimate (time)  yellow=error", (10, Hp - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.imwrite(str(path), img)


def eval_run(out: Path, run: str) -> dict:
    import pycolmap
    rd = out / "runs" / run
    done = schemas.read_json(rd / "done.json")
    gtj = schemas.read_json(out / "gt_poses.json")
    gt = {f["image"]: (np.array(f["R_world_from_cam"]), np.array(f["center"])) for f in gtj["frames"]}
    planes = schemas.read_json(out / "gt_planes.json")["planes"]
    models = sorted([p for p in (rd / "sparse").iterdir() if p.is_dir()], key=lambda p: p.name)
    res = {"run": run, **{k: done[k] for k in ("match", "prior", "robust", "seconds")},
           "total": len(gt), "models": len(models)}
    if not models:
        return res
    recs = [pycolmap.Reconstruction(m) for m in models]
    sizes = [r.num_reg_images() for r in recs]
    res["model_sizes"] = sizes
    k = int(np.argmax(sizes))
    rec, res["best_model"] = recs[k], str(models[k])
    names, C, Rwc = [], [], []
    for im in rec.images.values():
        if not im.has_pose:
            continue
        names.append(im.name)
        C.append(im.projection_center())
        Rwc.append(im.cam_from_world().rotation.matrix().T)
    o = np.argsort(names)
    names = [names[i] for i in o]
    C, Rwc = np.array(C)[o], np.array(Rwc)[o]
    res["registered"] = len(names)
    tel = find_teleports(names, C)
    res["teleports"] = len(tel)
    res["teleport_list"] = tel
    Cg = np.array([gt[n][1] for n in names])
    Rg = np.array([gt[n][0] for n in names])
    # 사용자 기준 순간이동: 중앙 속도 6배 넘게 + 30도 넘게 회전 (연속 등록 사진 사이)
    v = np.linalg.norm(np.diff(C, axis=0), axis=1)
    rot = np.array([rot_err_deg(Rwc[i], Rwc[i + 1]) for i in range(len(names) - 1)])
    res["jumps_6x_30deg"] = int(np.sum((v > 6 * np.median(v)) & (rot > 30)))

    def pose_metrics(s, R, t, tag):
        Ca = s * C @ R.T + t
        e = np.linalg.norm(Ca - Cg, axis=1)
        re = np.array([rot_err_deg(R @ Rwc[i], Rg[i]) for i in range(len(names))])
        Xa = s * np.array([p.xyz for p in rec.points3D.values()]) @ R.T + t
        return {f"{tag}_rmse_m": float(np.sqrt(np.mean(e ** 2))), f"{tag}_median_m": float(np.median(e)),
                f"{tag}_max_m": float(e.max()), f"{tag}_n_err_gt1m": int(np.sum(e > 1.0)),
                f"{tag}_rot_median_deg": float(np.median(re)), f"{tag}_rot_max_deg": float(re.max()),
                f"{tag}_planes": plane_stats(Xa, planes)}

    s, R, t, inl = robust_sim3(C, Cg)
    plot_xy(out / "plots" / f"{run}_sim3.png", s * C @ R.T + t, Cg, f"{run} (Sim3 aligned)")
    if done["prior"]:
        plot_xy(out / "plots" / f"{run}_plan.png", C, Cg, f"{run} (plan coords, no alignment)")
    res["sim3_scale"] = float(s)
    res["sim3_inliers"] = int(inl.sum())
    res.update(pose_metrics(s, R, t, "sim3"))
    if done["prior"]:
        res.update(pose_metrics(1.0, np.eye(3), np.zeros(3), "plan"))   # 정렬 없이 = 도면 좌표 그대로
        # diagnose 의 far_from_prior (제약에서 1m 넘게) 가 실제 오정합(정답에서 1m 넘게)을 얼마나 잡나
        pri = {f["image"]: np.array(f["xyz"]) for f in
               schemas.read_json(out / "priors" / f"priors_{done['prior']}.json")["frames"]}
        far = np.linalg.norm(C - np.array([pri[n] for n in names]), axis=1) > 1.0
        bad = np.linalg.norm(C - Cg, axis=1) > 1.0
        res["far_from_prior"] = {"flagged": int(far.sum()), "truly_bad": int(bad.sum()),
                                 "caught": int((far & bad).sum()), "false_alarm": int((far & ~bad).sum())}
        # 모델이 여러 개여도 priors 로 각자 도면 좌표에 놓이므로 합쳐서 봄 (10장 이상 모델, 겹치면 큰 모델 것)
        seen = {}
        for r in sorted(recs, key=lambda r: -r.num_reg_images()):
            if r.num_reg_images() < 10:
                continue
            for im in r.images.values():
                if im.has_pose and im.name not in seen:
                    seen[im.name] = (im.projection_center(), r)
        an = sorted(seen)
        AC = np.array([seen[n][0] for n in an])
        ACg = np.array([gt[n][1] for n in an])
        e = np.linalg.norm(AC - ACg, axis=1)
        Xa = np.concatenate([np.array([p.xyz for p in r.points3D.values()])
                             for r in recs if r.num_reg_images() >= 10])
        far = np.linalg.norm(AC - np.array([pri[n] for n in an]), axis=1) > 1.0
        res["all_models_plan"] = {"registered": len(an), "rmse_m": float(np.sqrt(np.mean(e ** 2))),
                                  "median_m": float(np.median(e)), "n_err_gt1m": int(np.sum(e > 1.0)),
                                  "n_err_lt0p5m": int(np.sum(e < 0.5)), "teleports": len(find_teleports(an, AC)),
                                  "far_from_prior_flagged": int(far.sum()), "far_caught": int((far & (e > 1)).sum()),
                                  "planes": plane_stats(Xa, planes)}
        if len(recs) > 1:
            plot_xy(out / "plots" / f"{run}_plan_all.png", AC, ACg, f"{run} (all models, plan coords)")
    schemas.write_json(rd / "eval.json", res)
    return res


def fmt_table(rows):
    hdr = ("| run | 매칭 | prior | 등록/전체 | 모델 크기 | 맞게 등록 (Sim3 후 <0.5m) | 오정합 (Sim3 후 >1m) | find_teleports"
           " | 회전오차 중앙 (°) | 도면좌표 오차 중앙/RMSE (m) | far_from_prior 잡음/실제오정합 | 점 중 벽 ±5cm Sim3/도면 | 정합 시간 (s) |")
    lines = [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    short = {"exhaustive": "exh", "sequential": "seq"}
    for r in rows:
        m = r["match"]
        for k, v in short.items():
            m = m.replace(k, v)
        m = m.replace("spatial:", "spa:")
        if "registered" not in r:
            lines.append(f"| {r['run']} | {m} | {r['prior'] or '-'} | 0/{r['total']} | - | | | | | | | | {r['seconds']:.0f} |")
            continue
        if r.get("prior"):
            plan = f"{r['plan_median_m']:.2f} / {r['plan_rmse_m']:.1f}"
            ff = r["far_from_prior"]
            far = f"{ff['caught']}/{ff['truly_bad']} (오경보 {ff['false_alarm']})"
            pl = f"{r['sim3_planes']['frac_within_5cm']:.2f} / {r['plan_planes']['frac_within_5cm']:.2f}"
        else:
            plan, far, pl = "-", "-", f"{r['sim3_planes']['frac_within_5cm']:.2f} / -"
        lines.append(f"| {r['run']} | {m} | {r['prior'] or '-'} | {r['registered']}/{r['total']} | {r['model_sizes']} | "
                     f"{r['sim3_inliers']} | {r['sim3_n_err_gt1m']} | {r['teleports']} | {r['sim3_rot_median_deg']:.1f} | "
                     f"{plan} | {far} | {pl} | {r['seconds']:.0f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--stages", default="render,features,map,eval")
    ap.add_argument("--runs", default=",".join(RUNS))
    ap.add_argument("--ambiguity", default="ends", choices=["extreme", "ends", "ends_unique"])
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    st = a.stages.split(",")
    runs = [r for r in a.runs.split(",") if r]
    if "render" in st:
        stage_render(out, a.ambiguity)
    if "features" in st:
        stage_features(out, runs)
    if "map" in st:
        for r in runs:
            stage_map(out, r)
    if "eval" in st:
        rows = [eval_run(out, r) for r in runs if (out / "runs" / r / "done.json").exists()]
        mt = out / "db" / "match_times.json"
        table = fmt_table(rows)
        (out / "results.md").write_text(table + "\n\nmatch times: " + (mt.read_text() if mt.exists() else "") + "\n",
                                        encoding="utf-8")
        print(table)


if __name__ == "__main__":
    main()
