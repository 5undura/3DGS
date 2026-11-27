"""merge 문 근처 ICP 정확도. 복도 구간(A) + 방 구간(B), 둘 다 도면 좌표로 정합됐는데 B 에 작은 오차가 있을 때
refine_at_doors 가 오차를 되돌리는지, 나빠지게 하지는 않는지, 크게 어긋나면 거절하는지.

장면 (m): 복도 x 0~14, y -1.2~1.2, 방 x 7~13, y 1.4~7 (벽 두께 0.2), 문 (10, 1.3) 폭 1.0 높이 2.1, 천장 2.6.
구간마다 '보이는 면'만 스플랫 중심으로: A = 복도 바닥·벽(문 틈 뺌)·문틀(양옆·위) + 문 너머 방 바닥(부채꼴),
B = 방 바닥·벽·문틀 + 문 너머 복도 바닥. 간격 3cm, 면에 수직 잡음 σ, 떠다니는 점(불투명) 5%.
벽 두 면(복도 쪽 y=1.2, 방 쪽 y=1.4)은 서로 다른 구간에만 보임 -> 점-점 ICP 가 0.2m 벽 두께를 붙이는 함정.

python experiments/accuracy/merge_eval.py
"""
from __future__ import annotations

import json
import math

import numpy as np

from common import OUT, stats
from indoorgs import plyio, schemas
from indoorgs.fillcore import align
from indoorgs.stages.merge import door_points, refine_at_doors

SP = 0.03
import os
WT = float(os.environ.get("WALL_T", "0.2"))   # 벽 두께 (문이 난 벽)
Y2 = 1.2 + WT
YD = 1.2 + WT / 2
DOOR_C = (10.0, YD)
PLAN = schemas.parse_plan({"version": 1, "rooms": [
    {"id": "corr", "floor": [[0, -1.2], [14, -1.2], [14, 1.2], [0, 1.2]], "floor_z": 0, "ceiling_z": 2.6},
    {"id": "room", "floor": [[7, Y2], [13, Y2], [13, 7], [7, 7]], "floor_z": 0, "ceiling_z": 2.6}],
    "doors": [{"id": "d1", "rooms": ["corr", "room"], "center": list(DOOR_C), "width": 1.0, "height": 2.1}]})


def grid(a0, a1, b0, b1, sp=SP):
    A, B = np.meshgrid(np.arange(a0 + sp / 2, a1, sp), np.arange(b0 + sp / 2, b1, sp))
    return A.ravel(), B.ravel()


def plane_x(x, y0, y1, z0, z1, n):
    y, z = grid(y0, y1, z0, z1); return np.stack([np.full_like(y, x), y, z], 1), np.tile(n, (len(y), 1))


def plane_y(y, x0, x1, z0, z1, n):
    x, z = grid(x0, x1, z0, z1); return np.stack([x, np.full_like(x, y), z], 1), np.tile(n, (len(x), 1))


def plane_z(z, x0, x1, y0, y1, n, keep=None):
    x, y = grid(x0, x1, y0, y1)
    P = np.stack([x, y, np.full_like(x, z)], 1)
    if keep is not None:
        P = P[keep(P)]
    return P, np.tile(n, (len(P), 1))


def door_frame():
    """문틀: 양옆 면 (x=9.5 법선 +x, x=10.5 법선 -x), 위 면 (z=2.1 법선 -z), y 1.2~Y2"""
    return [plane_x(9.5, 1.2, Y2, 0, 2.1, (1, 0, 0)), plane_x(10.5, 1.2, Y2, 0, 2.1, (-1, 0, 0)),
            plane_z(2.1, 9.5, 10.5, 1.2, Y2, (0, 0, -1))]


def section_A():
    parts = [plane_z(0, 6, 14, -1.2, 1.2, (0, 0, 1)),
             plane_y(-1.2, 6, 14, 0, 2.6, (0, 1, 0)),
             plane_y(1.2, 6, 9.5, 0, 2.6, (0, -1, 0)), plane_y(1.2, 10.5, 14, 0, 2.6, (0, -1, 0)),
             plane_y(1.2, 9.5, 10.5, 2.1, 2.6, (0, -1, 0)),
             plane_z(0, 9.5, 10.5, 1.2, Y2, (0, 0, 1)),        # 문턱
             plane_z(0, 7, 13, Y2, 7, (0, 0, 1), keep=lambda P: np.abs(P[:, 0] - 10) < 0.5 + (P[:, 1] - Y2) * 0.8)]
    return parts + door_frame()


def section_B():
    parts = [plane_z(0, 7, 13, Y2, 7, (0, 0, 1)),
             plane_y(Y2, 7, 9.5, 0, 2.6, (0, 1, 0)), plane_y(Y2, 10.5, 13, 0, 2.6, (0, 1, 0)),
             plane_y(Y2, 9.5, 10.5, 2.1, 2.6, (0, 1, 0)),
             plane_x(7, Y2, 7, 0, 2.6, (1, 0, 0)), plane_x(13, Y2, 7, 0, 2.6, (-1, 0, 0)),
             plane_z(0, 9.5, 10.5, 1.2, Y2, (0, 0, 1)),
             plane_z(0, 6, 14, -1.2, 1.2, (0, 0, 1), keep=lambda P: np.abs(P[:, 0] - 10) < 0.5 + (1.2 - P[:, 1]) * 0.8)]
    return parts + door_frame()


def to_ply(parts, rng, noise=0.02, floaters=0.05):
    P = np.concatenate([p for p, _ in parts]); N = np.concatenate([n for _, n in parts]).astype(float)
    P = P + N * rng.normal(0, noise, (len(P), 1)) + rng.normal(0, noise / 3, P.shape)
    k = int(floaters * len(P))
    F = np.stack([rng.uniform(8, 12, k), rng.uniform(-1, 3, k), rng.uniform(0, 2.5, k)], 1)
    P = np.concatenate([P, F])
    ply = plyio.make(len(P), plyio.brush_names(0))
    for i, c in enumerate("xyz"):
        ply.data[c] = P[:, i]
    ply.data["opacity"] = rng.normal(2.0, 1.0, len(P))      # 대부분 불투명 (sigmoid(2)=0.88)
    return ply


def rigid(rot_deg=0.0, axis=(0, 0, 1), t=(0, 0, 0), about=(10.0, YD, 1.0)):
    a = np.radians(rot_deg); k = np.asarray(axis, float); k /= np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K
    c = np.asarray(about, float)
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = c - R @ c + np.asarray(t, float)
    return M


def door_err(M, P):
    """문 근처 정답 점 P 가 M 으로 움직인 평균 거리"""
    return float(np.linalg.norm(align.apply(M, P) - P, axis=1).mean())


def trial(rng, E, noise=0.02, floaters=0.05, **icp_kw):
    A = to_ply(section_A(), rng, noise, floaters)
    B = to_ply(section_B(), rng, noise, floaters)
    pa = {"d1": door_points(A, PLAN.doors[0], PLAN)}
    pb_true = door_points(B, PLAN.doors[0], PLAN)
    pb = {"d1": align.apply(E, pb_true)}
    res = refine_at_doors(["A", "B"], {"A": pa, "B": pb}, **icp_kw)
    T, rec = res["B"]
    before = door_err(E, pb_true)
    after = door_err(T @ E, pb_true)
    return before, after, not np.allclose(T, np.eye(4)), rec


CASES = [
    ("0.1m along wall (x)", dict(t=(0.1, 0, 0))),
    ("0.1m across wall (y)", dict(t=(0, 0.1, 0))),
    ("0.1m up (z)", dict(t=(0, 0, 0.1))),
    ("1deg yaw", dict(rot_deg=1.0)),
    ("1deg roll (axis x)", dict(rot_deg=1.0, axis=(1, 0, 0))),
    ("0.1m random dir + 1deg random axis", "rand_small"),
    ("0.2m random dir + 2deg", "rand_mid"),
    ("0.5m along wall (x) -> must refuse", dict(t=(0.5, 0, 0))),
    ("0.5m random dir -> must refuse", "rand_big"),
    ("5deg yaw -> must refuse", dict(rot_deg=5.0)),
    ("no error (identity)", dict()),
]


def make_E(rng, spec):
    if isinstance(spec, dict):
        return rigid(**spec)
    d = rng.normal(size=3); d /= np.linalg.norm(d)
    ax = rng.normal(size=3)
    if spec == "rand_small":
        return rigid(1.0, ax, 0.1 * d)
    if spec == "rand_mid":
        return rigid(2.0, ax, 0.2 * d)
    return rigid(0.0, ax, 0.5 * d)


def main(runs=20, tag="", **icp_kw):
    out = []
    for name, spec in CASES:
        rng = np.random.default_rng(abs(hash(name)) % 2**32)
        bs, afs, acc, worse = [], [], 0, 0
        reasons = {}
        for _ in range(runs):
            E = make_E(rng, spec)
            b, a, ok, rec = trial(rng, E, **icp_kw)
            bs.append(b); afs.append(a); acc += ok
            worse += a > b + 0.01
            if not ok:
                r = rec.get("reason", "?").split(" ")[0]
                reasons[r] = reasons.get(r, 0) + 1
        r = {"case": name, "before_m": stats(bs), "after_m": stats(afs), "accepted": acc, "runs": runs,
             "worse_by_1cm": worse, "refuse_reasons": reasons}
        out.append(r)
        print(f"{name:38s} before med {r['before_m']['median']:.3f}  after med {r['after_m']['median']:.3f} "
              f"p95 {r['after_m']['p95']:.3f}  accepted {acc}/{runs}  worse {worse}  {reasons}")
    OUT.mkdir(exist_ok=True)
    (OUT / f"merge_wt{WT}{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    import sys
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    kw = {k: float(v) for k, v in (a.split("=") for a in sys.argv[2:])}
    print("icp 옵션:", kw or "기본")
    main(runs, "_" + "_".join(f"{k}{v}" for k, v in kw.items()) if kw else "", **kw)
