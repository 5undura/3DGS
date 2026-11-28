"""2차만 여러 seed 로: <작업>/sections/v4/colmap/db_verified.db 사본 -> shim (pose_prior_mapper | vgc+global_mapper)
-> 가장 큰 모델 align_model_to_priors -> 정답과 비교. 사용: pass2_sweep.py <v4> <작업> <incremental|global> <seed> <out>"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path
E2E = Path("/home/user/3DGS/indoorgs/experiments/e2e_v4")
sys.path.insert(0, str(E2E)); sys.path.insert(0, str(E2E.parents[1] / "src"))
import eval_e2e as ev
from indoorgs import schemas
from indoorgs.colmap import model as cm, pair_verify as pv
from indoorgs.stages.register import best_model
import numpy as np
scene, work, kind, seed, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], Path(sys.argv[5])
SC = "/tmp/claude-0/-home-user-3DGS/5698f89c-b779-591a-9900-84c73f83daf3/scratchpad"
col = work / "sections/v4/colmap"; fr = work / "sections/v4/frames"
out.mkdir(parents=True, exist_ok=True)
db = out / "db.db"; shutil.copy2(col / "db_verified.db", db)
sp = out / "sparse"; sp.mkdir(exist_ok=True)
env = dict(os.environ, SHIM_SEED=seed)
def shim(*a):
    t = time.time()
    r = subprocess.run([f"{SC}/venv/bin/python", str(E2E / "colmap_shim.py"), *map(str, a)], env=env, capture_output=True, text=True)
    (out / "log.txt").open("a").write(r.stdout + r.stderr)
    return r.returncode, time.time() - t
t0 = time.time(); times = {}
if kind == "global":
    rc, times["vgc"] = shim("view_graph_calibrator", "--database_path", db)
    rc, times["global"] = shim("global_mapper", "--database_path", db, "--image_path", fr, "--output_path", sp)
else:
    rc, times["ppm"] = shim("pose_prior_mapper", "--database_path", db, "--image_path", fr, "--output_path", sp, "--overwrite_priors_covariance", "0")
g = schemas.read_json(scene / "gt_poses.json")
gt = {f["image"]: (np.array(f["R_world_from_cam"]), np.array(f["center"])) for f in g["frames"]}
priors = schemas.read_json(col / "priors.json")
pri = {f["image"]: np.array(f["xyz"]) for f in priors["frames"]}
b, n = best_model(sp)
res = {"work": str(work), "kind": kind, "seed": seed, "rc": rc, "times": times,
       "models": sorted((cm.count_images(d) for d in sp.iterdir() if d.is_dir()), reverse=True)}
if b is not None:
    new, ast = pv.align_model_to_priors(cm.read_model(b), priors, level=True)
    m = ev.pose_metrics(new, gt, pri)
    res.update({"metrics": m, "align": {k: ast[k] for k in ("leveled_deg", "scale", "resid_median_m", "far_1m")}})
schemas.write_json(out / "res.json", res)
mm = res.get("metrics", {})
print(json.dumps({k: res[k] for k in ("work", "kind", "seed", "models", "times")}),
      {k: mm.get(k) for k in ("registered", "sim3_bad1", "teleports", "plan_bad1", "tilt_up_deg")}, flush=True)
