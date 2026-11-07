"""PLY 여러 개의 스플랫 통계 비교 (학습 설정 실험용)
사용: python ply_stats.py 이름1=a.ply 이름2=b.ply ...
출력: 스플랫 수, 바늘형 비율(긴 축/중간 축 > 5, > 10), 가장 긴 축 크기(중앙값, 상위 1%), 반투명 비율(<0.5, <0.2)
"""
import sys
import numpy as np


def load(path):
    f = open(path, "rb"); names = []; n = 0
    while True:
        line = f.readline().decode().strip()
        if line.startswith("element vertex"):
            n = int(line.split()[-1])
        elif line.startswith("property"):
            names.append(line.split()[-1])
        elif line == "end_header":
            break
    a = np.fromfile(f, dtype=np.float32, count=n * len(names)).reshape(n, len(names))
    return {k: a[:, i] for i, k in enumerate(names)}


for arg in sys.argv[1:]:
    name, path = arg.split("=", 1) if "=" in arg else (arg, arg)
    d = load(path)
    s = np.exp(np.stack([d["scale_0"], d["scale_1"], d["scale_2"]], 1)); s.sort(1)
    op = 1 / (1 + np.exp(-d["opacity"]))
    ratio = s[:, 2] / np.maximum(s[:, 1], 1e-12)
    print(f"{name}: N={len(op):,} | 바늘형 >5 {np.mean(ratio > 5) * 100:.1f}% >10 {np.mean(ratio > 10) * 100:.1f}% "
          f"| 긴 축 중앙 {np.median(s[:, 2]):.4f} 상위1% {np.percentile(s[:, 2], 99):.3f} "
          f"| 반투명 <0.5 {np.mean(op < 0.5) * 100:.1f}% <0.2 {np.mean(op < 0.2) * 100:.1f}%")
