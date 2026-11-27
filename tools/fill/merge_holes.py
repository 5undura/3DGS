"""바탕층(backdrop.ply)을 '뚫린 곳'에만 남기고 학습 결과와 합침
바탕층을 전부 깔면 반투명 스플랫 뒤로 흐린 바탕층이 비쳐 화면이 뭉개져 보임 -> 학습 스플랫이 이미 덮은 곳의 바탕층은 버림.

사용: python merge_holes.py <out.ply> <학습 결과.ply> <backdrop.ply> [덮임 기준=1.0] [반경 배수=1.5]
  - 바탕층 점마다, 반경(바탕층 점 간격 x 배수) 안 학습 스플랫의 불투명도 합을 구함
  - 합 < 기준 인 바탕층 점만 남김 (기준을 낮추면 바탕층이 덜 남음)
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge_ply import read  # noqa: E402  (merge_ply 는 import 시 실행되지 않도록 __main__ 가드 필요)

out, run_p, bd_p = sys.argv[1], sys.argv[2], sys.argv[3]
thr = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
rmul = float(sys.argv[5]) if len(sys.argv) > 5 else 1.5

names, A, comments = read(run_p)
bn, B, _ = read(bd_p)
xyz = lambda nm, M: M[:, [nm.index("x"), nm.index("y"), nm.index("z")]].astype(np.float64)
P, Q = xyz(names, A), xyz(bn, B)
op = 1 / (1 + np.exp(-A[:, names.index("opacity")].astype(np.float64)))

sp = np.median(cKDTree(Q).query(Q, k=2)[0][:, 1])   # 바탕층 점 간격
r = sp * rmul
keep_run = op > 0.05
tree = cKDTree(P[keep_run]); opk = op[keep_run]
cover = np.zeros(len(Q))
for s in range(0, len(Q), 20000):
    for j, idx in enumerate(tree.query_ball_point(Q[s:s + 20000], r)):
        cover[s + j] = opk[idx].sum()
for t in (0.3, 0.5, 1.0, 2.0):
    print(f"  기준 {t}: 바탕층 남는 비율 {np.mean(cover < t) * 100:.1f}%")
hole = cover < thr
# 벽 바탕층은 평면도 경계로 만든 것이라 실제 벽과 어긋나는 일이 많음 (복도 1: 80% 가 5cm 넘게 떨어짐)
# -> 반투명 벽 뒤로 비쳐 뭉개져 보임. 기본은 벽 바탕층을 빼고 바닥·천장만 (KEEP_WALLS=1 이면 벽도 남김)
q = B[:, [bn.index("rot_%d" % i) for i in range(4)]].astype(np.float64)
w_, x_, y_, z_ = q.T
nrm_y = 2 * (y_ * z_ - w_ * x_)               # 면 법선(회전 행렬 3번째 열)의 y 성분. y 가 수직축
is_wall = np.abs(nrm_y) < 0.5
if os.environ.get("KEEP_WALLS") != "1":
    print(f"벽 바탕층 {is_wall.sum():,} 개 제외 (KEEP_WALLS=1 이면 포함)")
    hole &= ~is_wall
print(f"바탕층 간격 {sp:.4f}, 반경 {r:.4f}, 기준 {thr} -> {hole.sum():,} / {len(Q):,} 남김 ({hole.mean() * 100:.1f}%)")

Bm = np.zeros((hole.sum(), len(names)), "<f4")
for i, k in enumerate(names):
    if k in bn: Bm[:, i] = B[hole, bn.index(k)]
M = np.concatenate([A, Bm])
with open(out, "wb") as f:
    h = "ply\nformat binary_little_endian 1.0\n" + "".join(c + "\n" for c in comments)
    h += "element vertex %d\n" % len(M) + "".join("property float %s\n" % k for k in names) + "end_header\n"
    f.write(h.encode("ascii")); f.write(M.astype("<f4").tobytes())
print(out, f"{len(M):,} splats")
