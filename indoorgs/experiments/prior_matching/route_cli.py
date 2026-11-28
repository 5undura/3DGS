"""권하는 순서를 실제 매처로 끝까지 돌려 확인 (exhaustive DB 에서 짝을 지우는 흉내가 아님).
pycolmap 함수 = COLMAP 명령: match_sequential = sequential_matcher, match_spatial = spatial_matcher,
incremental_mapping(use_prior_position) = pose_prior_mapper.

  python route_cli.py <scene> <이름> [prior_kind] [spatial max_distance] [max_num_neighbors]
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import pm  # noqa: E402
import prior_verify as pv  # noqa: E402
from indoorgs import schemas  # noqa: E402
from indoorgs.colmap import database  # noqa: E402


def main(root: Path, name: str, kind: str = "good", max_dist: float = 6.0, max_nn: int = 80):
    import pycolmap
    sc = pm.Scene(root)
    rd = root / "pm" / name
    if rd.exists():
        shutil.rmtree(rd)
    (rd / "pass1").mkdir(parents=True)
    (rd / "sparse").mkdir()
    T = {}
    db1 = rd / "pass1.db"
    shutil.copy(root / "db" / "features.db", db1)
    mo = pycolmap.FeatureMatchingOptions()
    mo.num_threads = 4
    mo.use_gpu = False
    # 1) sequential_matcher --SequentialMatching.overlap 20 --SequentialMatching.quadratic_overlap 0 --SequentialMatching.loop_detection 0
    so = pycolmap.SequentialPairingOptions()
    so.overlap, so.quadratic_overlap, so.loop_detection = 20, False, False
    t0 = time.time()
    pycolmap.match_sequential(db1, matching_options=mo, pairing_options=so, device=pycolmap.Device.cpu)
    T["sequential_matcher"] = time.time() - t0
    # 2) priors -> DB, pose_prior_mapper (1차)
    pri = sc.priors(kind)
    database.write_pose_priors(db1, pri)
    o = pycolmap.IncrementalPipelineOptions()
    o.num_threads, o.random_seed, o.use_prior_position = 4, 0, True
    t0 = time.time()
    r1 = pycolmap.incremental_mapping(db1, root / "images", rd / "pass1", o)
    T["pass1_mapper"] = time.time() - t0
    pm.log(f"pass1 models {[r.num_reg_images() for r in r1.values()]}")
    # 3) spatial_matcher --SpatialMatching.max_distance D --SpatialMatching.max_num_neighbors K --SpatialMatching.ignore_z 1 (같은 DB 에 짝 추가)
    po = pycolmap.SpatialPairingOptions()
    po.max_distance, po.max_num_neighbors, po.ignore_z = max_dist, max_nn, True
    t0 = time.time()
    pycolmap.match_spatial(db1, matching_options=mo, pairing_options=po, device=pycolmap.Device.cpu)
    T["spatial_matcher"] = time.time() - t0
    # 4) 사본에서 짝 검증 (prior_verify)
    db2 = rd / "db.db"
    shutil.copy(db1, db2)
    t0 = time.time()
    ref = pv.load_reference(rd / "pass1", pri)
    st = pv.verify_pairs(db2, ref)
    T["verify"] = time.time() - t0
    dropped = set(st.pop("dropped_ids"))
    # 정답 라벨 (평가용)
    d = pm.pl.read_db(db1)
    lab = pm.pl.label_pairs(d, sc.gt, sc.K, sc.px)
    st["false_before"] = int(sum(lab.values()))
    st["false_after"] = int(sum(v for p, v in lab.items() if p not in dropped))
    st["true_dropped"] = int(sum(1 for p in dropped if not lab[p]))
    pm.log(f"verify {st} ref {ref.info}")
    # 5) pose_prior_mapper (2차)
    t0 = time.time()
    r2 = pycolmap.incremental_mapping(db2, root / "images", rd / "sparse", o)
    T["pass2_mapper"] = time.time() - t0
    half = 0
    fst = {"pairs": st["pairs"] - st["dropped"], "false": st["false_after"], "true_far": half, "filter_seconds": T["verify"],
           "steps": [{"filter": "route", **st, "ref": ref.info, "times": T}]}
    schemas.write_json(rd / "done.json", {"version": 1, "spec": {"pairs": ["route"], "prior": kind}, "seed": 0,
                                          "seconds": T["pass2_mapper"], "models": len(r2), "filter": fst})
    print(pm.fmt_table([pm.eval_run(sc, name)]))
    print("times", {k: round(v) for k, v in T.items()})


if __name__ == "__main__":
    a = sys.argv
    main(Path(a[1]).resolve(), a[2], a[3] if len(a) > 3 else "good", float(a[4]) if len(a) > 4 else 6.0,
         int(a[5]) if len(a) > 5 else 80)
