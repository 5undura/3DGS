"""PLY 의 천장·바닥이 얼마나 채워졌는지 그림으로 비교 (PC 에서 실행, numpy + matplotlib)
- 각 PLY 를 복도 바닥 평면 기준으로 펼쳐서, 천장 높이 근처 / 바닥 근처 스플랫의 '덮인 정도'(불투명도 합)를 위에서 본 지도로 그림
- 하얗게 = 꽉 참, 검게 = 비어 있음
사용: python check_fill.py frame2.npz out.png 이름1=a.ply 이름2=b.ply ...
"""
import sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def read_ply(path):
    with open(path, "rb") as f:
        names, n = [], 0
        while True:
            l = f.readline().decode("ascii").strip()
            if l.startswith("element vertex"): n = int(l.split()[-1])
            elif l.startswith("property"): names.append(l.split()[-1])
            elif l == "end_header": break
        need = n * len(names) * 4; buf = bytearray(need); mv = memoryview(buf); got = 0
        while got < need:
            c = f.read(min(32 << 20, need - got))
            if not c: break
            mv[got:got + len(c)] = c; got += len(c)
    a = np.frombuffer(bytes(mv[:got]), "<f4").reshape(-1, len(names))
    return {k: a[:, i] for i, k in enumerate(names)}

fr = np.load(sys.argv[1]); ax1, ax2, up, c0 = fr["ax1"], fr["ax2"], fr["up"], fr["c"]
H = float(fr["H"]) if "H" in fr else 1.60
out = sys.argv[2]; runs = [a.split("=", 1) for a in sys.argv[3:]]
CELL = 0.05; X0, X1, Y0, Y1 = -8, 8, -3, 2
fig, axs = plt.subplots(len(runs), 2, figsize=(18, 3.2 * len(runs)), squeeze=False)
for r, (name, path) in enumerate(runs):
    d = read_ply(path)
    p = np.stack([d["x"], d["y"], d["z"]], 1) - c0
    q = np.stack([p @ ax1, p @ ax2, p @ up], 1)
    op = 1 / (1 + np.exp(-d["opacity"]))
    for c, (lo, hi, title) in enumerate([(H - 0.25, H + 0.1, "ceiling (z %.2f~%.2f)"), (-0.05, 0.08, "floor (z %.2f~%.2f)")]):
        m = (q[:, 2] > lo) & (q[:, 2] < hi)
        img, _, _ = np.histogram2d(q[m, 1], q[m, 0], bins=[int((Y1 - Y0) / CELL), int((X1 - X0) / CELL)],
                                   range=[[Y0, Y1], [X0, X1]], weights=op[m])
        cov = 1 - np.exp(-img)          # 불투명도 합 -> 덮인 정도 (0~1)
        axs[r, c].imshow(cov, origin="lower", extent=[X0, X1, Y0, Y1], cmap="gray", vmin=0, vmax=1)
        axs[r, c].set_title(f"{name}: " + title % (lo, hi) + f"  mean cover {cov[cov > 0].mean():.2f}")
    print(name, "splats", len(op))
plt.tight_layout(); plt.savefig(out, dpi=70); print("saved", out)
