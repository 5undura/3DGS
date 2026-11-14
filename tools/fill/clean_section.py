"""복도·방 안의 부유물 정리 (fill_section.py 와 같은 바닥·평면도·천장 분석을 사용)

사용: python clean_section.py <Nb\\data 폴더> <입력.ply> <출력.ply> [지운것.ply]
지우는 기준 (모두 '바닥 위 카메라 높이' 기준으로 축척 자동 조정)
  A. 바닥 아래로 일정 이상 내려간 스플랫
  B. 천장 위로 일정 이상 올라간 스플랫
  C. 투과 부유물: 여러 사진에서 '실제로 보인 표면(COLMAP 점)보다 확실히 앞 허공' 에 있는 스플랫
     (점이 없는 천장 같은 곳은 판정 안 함. A·B 는 평면도 안쪽에만 적용)
지운 스플랫만 모은 PLY 를 함께 저장 -> 열어서 진짜 물체가 지워지지 않았는지 확인
"""
import sys, os, struct
import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage as ndi

SRC, PIN, POUT = sys.argv[1:4]
PREM = sys.argv[4] if len(sys.argv) > 4 else None
SP0 = os.path.join(SRC, "sparse", "0")

# ---------- COLMAP (fill_section.py 와 같은 코드) ----------
def qvec2rot(q):
    w, x, y, z = q
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
                     [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
                     [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]])
def read_images(p):
    data = open(p, "rb").read(); off = 8; out = []
    dt = np.dtype([("x", "<f8"), ("y", "<f8"), ("p", "<i8")])
    for _ in range(struct.unpack_from("<Q", data, 0)[0]):
        off += 4; q = struct.unpack_from("<4d", data, off); off += 32
        t = struct.unpack_from("<3d", data, off); off += 24
        cid, = struct.unpack_from("<i", data, off); off += 4
        e = data.index(b"\0", off); off = e + 1
        n2, = struct.unpack_from("<Q", data, off); off += 8
        a = np.frombuffer(data, dt, count=n2, offset=off); off += n2 * 24
        k = a["p"] >= 0
        out.append((qvec2rot(q), np.array(t), np.stack([a["x"][k], a["y"][k]], 1), a["p"][k]))
    return out
def read_cameras(p):
    with open(p, "rb") as f:
        f.read(8); cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
        npar = {0: 3, 1: 4}[model]; pp = struct.unpack("<%dd" % npar, f.read(8 * npar))
    return (w, h) + ((pp[0], pp[0], pp[1], pp[2]) if model == 0 else tuple(pp))
def read_points(p):
    data = open(p, "rb").read(); off = 8; n = struct.unpack_from("<Q", data, 0)[0]
    xyz = np.empty((n, 3)); err = np.empty(n); ids = np.empty(n, np.int64)
    for i in range(n):
        v = struct.unpack_from("<Q3d3BdQ", data, off); off += 51 + 8 * v[-1]
        ids[i] = v[0]; xyz[i] = v[1:4]; err[i] = v[7]
    return xyz, err, ids

ims = read_images(os.path.join(SP0, "images.bin"))
X, ERR, PID = read_points(os.path.join(SP0, "points3D.bin"))
CAM = read_cameras(os.path.join(SP0, "cameras.bin"))
C = np.array([-im[0].T @ im[1] for im in ims])

# ---------- 바닥·축척·평면도·천장 (fill_section.py 와 같은 방법) ----------
rng = np.random.default_rng(0)
up0 = -np.array([im[0].T @ [0, 1, 0] for im in ims]).mean(0); up0 /= np.linalg.norm(up0)
spread = np.median(np.linalg.norm(C - np.median(C, 0), axis=1))
dist, _ = cKDTree(C).query(X); near = (ERR < 2) & (dist < 3 * spread)
Xn = X[near]; h0 = (Xn - np.median(C, 0)) @ up0
low = Xn[h0 < np.percentile(h0, 30)]
thr0 = np.percentile(np.abs(h0 - np.median(h0)), 50) * 0.05
best = (0, None)
for _ in range(3000):
    s = low[rng.choice(len(low), 3, replace=False)]
    n = np.cross(s[1]-s[0], s[2]-s[0]); nn = np.linalg.norm(n)
    if nn < 1e-12: continue
    n /= nn
    if abs(n @ up0) < np.cos(np.radians(25)): continue
    k = (np.abs((low - s[0]) @ n) < thr0).sum()
    if k > best[0]: best = (k, (n, s[0]))
n, p0 = best[1]; Q = low[np.abs((low - p0) @ n) < thr0]; c0 = Q.mean(0)
up = np.linalg.svd(Q - c0, full_matrices=False)[2][2]
if up @ up0 < 0: up = -up
hcam = np.median((C - c0) @ up); S = hcam / 0.67
Cc = C - C.mean(0); Cc -= np.outer(Cc @ up, up)
ax1 = np.linalg.svd(Cc, full_matrices=False)[2][0]; ax1 -= (ax1 @ up) * up; ax1 /= np.linalg.norm(ax1); ax2 = np.cross(up, ax1)
def F(p): q = p - c0; return np.stack([q @ ax1, q @ ax2, q @ up], -1)
Xf = F(Xn); Cf = F(C)
cell = 0.04 * S
xy = Xf[np.abs(Xf[:, 2]) < 0.03 * S, :2]
mn = np.minimum(xy.min(0), Cf[:, :2].min(0)) - 0.5 * S; mx = np.maximum(xy.max(0), Cf[:, :2].max(0)) + 0.5 * S
g = np.zeros(np.ceil((mx - mn) / cell).astype(int)[::-1], bool)
ij = ((xy - mn) / cell).astype(int); g[ij[:, 1], ij[:, 0]] = True
cij = ((Cf[:, :2] - mn) / cell).astype(int); g[cij[:, 1], cij[:, 0]] = True
g = ndi.binary_closing(g, np.ones((7, 7)), iterations=2); g = ndi.binary_fill_holes(g)
lab, nl = ndi.label(g); g = lab == (1 + np.argmax(ndi.sum(g, lab, range(1, nl + 1))))
g = ndi.binary_opening(g, np.ones((5, 5)))
gi = np.clip(((Xf[:, :2] - mn) / cell).astype(int), 0, np.array(g.shape[::-1]) - 1)
zin = Xf[g[gi[:, 1], gi[:, 0]] & (Xf[:, 2] > hcam), 2]
H = float(np.clip(np.percentile(zin, 99) if len(zin) > 100 else 2.4 * hcam, 1.6 * hcam, 4.0 * hcam))
wall_dist = ndi.distance_transform_edt(g) * cell          # 평면도 안쪽 칸에서 가장 가까운 벽까지 거리
print(f"camera height {hcam:.3f}, scale {S:.2f}, ceiling {H:.3f}")

# ---------- PLY 읽기 ----------
with open(PIN, "rb") as f:
    head, names, nv = [], [], 0
    while True:
        l = f.readline().decode("ascii").strip(); head.append(l)
        if l.startswith("element vertex"): nv = int(l.split()[-1])
        elif l.startswith("property"): names.append(l.split()[-1])
        elif l == "end_header": break
    need = nv * len(names) * 4; buf = bytearray(); 
    while len(buf) < need:
        ch = f.read(min(32 << 20, need - len(buf)))
        if not ch: break
        buf += ch
A = np.frombuffer(bytes(buf), "<f4").reshape(nv, len(names))
P = F(A[:, [names.index(k) for k in "xyz"]].astype(np.float64))
opa = 1 / (1 + np.exp(-A[:, names.index("opacity")]))
pi = np.clip(((P[:, :2] - mn) / cell).astype(int), 0, np.array(g.shape[::-1]) - 1)
inside = g[pi[:, 1], pi[:, 0]]
wd = np.where(inside, wall_dist[pi[:, 1], pi[:, 0]], 0)

# ---------- 기준 ----------
# A/B: 평면도 안쪽에서 바닥 아래·천장 위 (평면도 밖은 계단·문 너머일 수 있어 그대로 둠)
rA = inside & (P[:, 2] < -0.08 * S)
rB = inside & (P[:, 2] > H + 0.15 * S)
# C: 투과 판정. 사진마다 COLMAP 이 실제로 본 표면까지의 거리를 격자 칸별로 구해 두고,
#    스플랫이 여러 사진에서 '보인 표면보다 확실히 앞(허공)' 에 있으면 부유물.
#    천장처럼 점이 없는 칸은 판정하지 않음 (보강 시작점·바탕층 보호)
W_, H_, fx, fy, cx, cy = CAM
GX, GY = 48, 27
Xw = A[:, [names.index(k) for k in "xyz"]].astype(np.float64)
pid2i = {int(p): i for i, p in enumerate(PID)}
through = np.zeros(nv, np.int16); agree = np.zeros(nv, np.int16)
for vi in range(0, len(ims), 3):
    R, t, uv, pids = ims[vi]
    if len(pids) < 50: continue
    idx = np.array([pid2i.get(int(p), -1) for p in pids]); ok = idx >= 0
    Pc = X[idx[ok]] @ R.T + t; d = Pc[:, 2]; uvk = uv[ok]
    gx = np.clip((uvk[:, 0] / W_ * GX).astype(int), 0, GX - 1); gy = np.clip((uvk[:, 1] / H_ * GY).astype(int), 0, GY - 1)
    cellid = gy * GX + gx
    dmin = np.full(GX * GY, np.inf); cnt = np.zeros(GX * GY, int)
    np.minimum.at(dmin, cellid, d); np.add.at(cnt, cellid, 1)
    dmin[cnt < 3] = np.inf                                  # 점이 적은 칸은 모름
    Sc = Xw @ R.T + t; z = Sc[:, 2]; front = z > 0.05 * S
    u = fx * Sc[:, 0] / np.maximum(z, 1e-9) + cx; v = fy * Sc[:, 1] / np.maximum(z, 1e-9) + cy
    vis = front & (u >= 0) & (u < W_) & (v >= 0) & (v < H_)
    ii = np.nonzero(vis)[0]
    cid = np.clip((v[ii] / H_ * GY).astype(int), 0, GY - 1) * GX + np.clip((u[ii] / W_ * GX).astype(int), 0, GX - 1)
    ds = dmin[cid]; known = np.isfinite(ds)
    th = known & (z[ii] < ds * 0.7) & (ds - z[ii] > 0.1 * S)   # 보인 표면보다 30% 이상 앞
    ag = known & (np.abs(z[ii] - ds) < ds * 0.15)
    through[ii[th]] += 1; agree[ii[ag]] += 1
rC = (through >= 4) & (through > 3 * agree)
if os.environ.get("CLEAN_DEBUG"): np.savez(os.path.splitext(POUT)[0] + "_votes.npz", through=through, agree=agree, opa=opa, P=P, inside=inside)
rem = rA | rB | rC
print(f"splats {nv:,} -> remove {rem.sum():,} ({rem.mean()*100:.1f}%)  "
      f"[below floor {rA.sum():,} / above ceiling {rB.sum():,} / see-through floaters {rC.sum():,}]")

def write(path, M):
    with open(path, "wb") as f:
        h = "ply\nformat binary_little_endian 1.0\n" + "".join(c + "\n" for c in head if c.startswith("comment"))
        h += "element vertex %d\n" % len(M) + "".join("property float %s\n" % k for k in names) + "end_header\n"
        f.write(h.encode("ascii")); f.write(np.ascontiguousarray(M, "<f4").tobytes())
write(POUT, A[~rem]); print("saved", POUT)
if PREM: write(PREM, A[rem]); print("removed ->", PREM)
np.savez(os.path.splitext(POUT)[0] + "_clean_info.npz", P=P[::10], rem=rem[::10], code=(rA + 2*rB + 4*rC)[::10].astype(np.int8),
         Cf=Cf, g=g, mn=mn, cell=cell, H=H, hcam=hcam)
