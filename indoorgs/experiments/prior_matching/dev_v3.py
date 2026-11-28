import json, sys, time, pickle
from pathlib import Path
import numpy as np
import pairlib as pl
out = Path(sys.argv[1]); dbname = sys.argv[2]; fscale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.15
g = json.load(open(out / "gt_poses.json"))
K = np.array(g["K"]); gt = {f["image"]: (np.array(f["R_world_from_cam"]), np.array(f["center"])) for f in g["frames"]}
tt = {f["image"]: f["t"] for f in g["frames"]}
d = pl.read_db(out / "db" / dbname)
lab = pl.label_pairs(d, gt, K, 4.0 * g["size"][0] / 640)
print("pairs", len(lab), "false", sum(lab.values()))
pri = json.load(open(out / "priors" / "priors_good.json"))
pos = pl.prior_arrays(pri, d["images"])
order = d["order"]; inv = {k: i for i, k in order.items()}
P = np.array([pos[inv[k]] for k in range(len(inv))]); t = np.array([tt[d["images"][inv[k]]] for k in range(len(inv))])
cache = out / "db" / f"{dbname}.rel_{fscale}.pkl"
if cache.exists():
    rel = pickle.load(open(cache, "rb"))
else:
    t0 = time.time(); rel = pl.relative_poses(d, K[0, 0] * fscale); print("relposes", len(rel), f"{time.time()-t0:.0f}s")
    pickle.dump(rel, open(cache, "wb"))
head = pl.path_headings(P, t)
print("heading unknown frames", int(np.isnan(head[:, 0]).sum()))
def rep(name, drop):
    drop = set(drop)
    f = sum(1 for p in lab if lab[p]); tr = len(lab) - f
    df = sum(1 for p in drop if lab[p]); dt = len(drop) - df
    print(f"{name}: drop {len(drop)} | false removed {df}/{f} (left {f-df}) | true removed {dt}/{tr}")
for tol in (60, 80, 100):
    rep(f"T_rot tol{tol}", pl.check_heading(rel, order, head, tol))
for md in (30, 45, 60):
    rep(f"T_depth max{md}", pl.check_depth(rel, order, P, md))
a = pl.check_heading(rel, order, head, 80) | pl.check_depth(rel, order, P, 30)
rep("T_rot80+T_depth30", a)
keep = pl.select_pairs(pos, order, 6.0, 0, 10)
rep("spatial6", set(lab) - keep)
rep("spatial6+T_rot80+T_depth30", (set(lab) - keep) | a)
# 남은 가짜 짝 성격
left = [p for p in lab if lab[p] and p not in a]
import collections
c = collections.Counter()
for p in left:
    i, j = (order[x] for x in pl.id_pair(p))
    r = rel.get(p)
    c[("seq" if abs(i - j) <= 10 else "far", "nohead" if (np.isnan(head[i, 0]) or np.isnan(head[j, 0])) else "head", "norel" if r is None else "rel")] += 1
print(c)
