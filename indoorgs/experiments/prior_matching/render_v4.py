"""v4 장면 렌더: python render_v4.py <out>  -> <out>/images, gt_poses.json, gt_planes.json, priors/"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import scene_v4 as scene  # noqa: E402

_TEX = _BOX = None


def _init():
    global _TEX, _BOX
    _TEX, _BOX = scene.make_textures(0), scene.make_boxes(0)


def _one(a):
    i, name, R, C, out = a
    img = scene.render(_TEX, _BOX, R, C, rng=np.random.default_rng(1000 + i))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    Path(out, name).write_bytes(buf.tobytes())


def main(out: Path, procs: int = 4):
    from indoorgs import pathing, schemas
    (out / "images").mkdir(parents=True, exist_ok=True)
    traj = scene.gt_trajectory()
    t0 = time.time()
    todo = [(i, n, R, C, str(out / "images")) for i, (n, t, R, C) in enumerate(traj) if not (out / "images" / n).exists()]
    with mp.Pool(procs, initializer=_init) as pool:
        pool.map(_one, todo, chunksize=2)
    gt = {"version": 1, "fps": scene.SRC_FPS, "K": scene.K.tolist(), "size": [scene.IMG_W, scene.IMG_H], "scene": "v4",
          "frames": [{"image": n, "t": t, "R_world_from_cam": R.tolist(), "center": C.tolist()} for n, t, R, C in traj]}
    schemas.write_json(out / "gt_poses.json", gt)
    schemas.write_json(out / "gt_planes.json", scene.planes_json())
    (out / "priors").mkdir(exist_ok=True)
    names = [n for n, *_ in traj]
    for kind, (shift, off) in {"good": (0, 0), "t2s": (2.0, 0), "x1m": (0, 1.0)}.items():
        pj = scene.drawn_path(time_shift=shift, offset_m=off)
        pri, warns = pathing.build_priors(schemas.parse_path(pj), names, scene.SRC_FPS)
        schemas.write_json(out / "priors" / f"path_{kind}.json", pj)
        schemas.write_json(out / "priors" / f"priors_{kind}.json", pri)
    print(f"render {len(todo)} frames {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
