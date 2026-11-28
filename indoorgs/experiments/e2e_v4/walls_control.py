"""대조: 같은 data_clean 모델을 정답에 Sim3 로 맞춘 뒤(동선 오차 제거) 같은 walls 코드로 면을 맞춤.
walls 코드 자체 오차와 동선(그린 선)에서 온 축척·위치 오차를 나눠 보려는 것.

  python walls_control.py <v4 장면 폴더> <작업 폴더>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))
from eval_e2e import robust_sim3  # noqa: E402
from indoorgs import schemas  # noqa: E402
from indoorgs.colmap import model as cm  # noqa: E402
from indoorgs.walls import planes as wp, refine  # noqa: E402


def main(scene: Path, work: Path):
    g = schemas.read_json(scene / "gt_poses.json")
    gt = {f["image"]: np.array(f["center"]) for f in g["frames"]}
    gtp = {p["name"]: (np.array(p["n"], float), float(p["d"])) for p in schemas.read_json(scene / "gt_planes.json")["planes"]}
    sec = work / "sections" / "v4"
    m = cm.read_model(sec / "data_clean" / "sparse" / "0")
    ims = list(m.images.values())
    C = np.array([im.center for im in ims])
    Cg = np.array([gt[im.name] for im in ims])
    s, R, t = robust_sim3(C, Cg)
    xyz = np.array([p.xyz for p in m.points.values()])
    w = np.array([len(p.track) for p in m.points.values()], float)
    plan = schemas.load_plan(work / "floorplan" / "plan.json")
    planes = wp.planes_from_plan(plan)
    print(f"Sim3 to gt: scale {s:.4f}")
    print("| 면 | 정답 | source | 오프셋 cm (정답 정렬) | 오프셋 cm (파이프라인) | 기울기° |")
    print("|---|---|---|---|---|---|")
    pipe = {f["id"]: f for f in schemas.read_json(work / "eval.json").get("walls", [])}
    for label, X in (("oracle", s * xyz @ R.T + t),):
        fitted, info = refine.refine_planes(planes, X, weights=w)
        for q, i in zip(fitted, info):
            best = max(gtp, key=lambda k: float(gtp[k][0] @ q.n))
            n, d = gtp[best]
            c = q.point(q.size[0] / 2, q.size[1] / 2)
            pp = pipe.get(q.id, {})
            print(f"| {q.id} | {best} | {i['source']} | {100 * float(n @ c - d):+.1f} | "
                  f"{pp.get('offset_cm', float('nan')):+.1f} ({pp.get('source')}) | {wp.angle_deg(q.n, n):.2f} |")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
