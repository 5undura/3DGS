"""구간 하나의 빈 곳 채우기: 바탕층 스플랫 PLY + 시작점 보강 sparse 생성

사용: python fill_section.py <Nb\\data 폴더> <data_fill 폴더> <결과 폴더(Nc_fill)> [사진 간격=6]
  - <data_fill>\\sparse\\0 : 원본 sparse 복사 + points3D.bin 에 바닥·천장·벽 샘플 점 추가
  - <결과>\\backdrop.ply    : 사진을 투영해 색을 입힌 불투명 납작 스플랫 (면 바로 뒤)
  - <결과>\\fill_diag.png   : 평면도·높이 진단 그림
COLMAP 축척은 구간마다 달라서, 모든 길이는 '바닥 위 카메라 높이'를 기준으로 자동 조정함.
필요: numpy, scipy, opencv-python (matplotlib 은 있으면 진단 그림)
"""
import sys, os, struct, shutil, glob, time
import numpy as np, cv2
from scipy.spatial import cKDTree
from scipy import ndimage as ndi

SRC, DST, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
STEP = int(sys.argv[4]) if len(sys.argv) > 4 else 6
SP0 = os.path.join(SRC, "sparse", "0")
os.makedirs(os.path.join(DST, "sparse", "0"), exist_ok=True); os.makedirs(OUT, exist_ok=True)
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

# ---------------- COLMAP 읽기/쓰기 ----------------
def qvec2rot(q):
    w, x, y, z = q
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
                     [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
                     [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]])
def read_cameras(p):
    NP = {0:3,1:4,2:4,3:5,4:8,5:8,6:12,7:5,8:4,9:5,10:12}; cams = {}
    with open(p, "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
            cams[cid] = dict(model=model, w=w, h=h, params=np.array(struct.unpack("<%dd" % NP[model], f.read(8*NP[model]))))
    return cams
def read_images(p):
    data = open(p, "rb").read(); off = 8; out = []
    for _ in range(struct.unpack_from("<Q", data, 0)[0]):
        iid, = struct.unpack_from("<i", data, off); off += 4
        q = struct.unpack_from("<4d", data, off); off += 32
        t = struct.unpack_from("<3d", data, off); off += 24
        cid, = struct.unpack_from("<i", data, off); off += 4
        e = data.index(b"\0", off); name = data[off:e].decode(); off = e + 1
        n2, = struct.unpack_from("<Q", data, off); off += 8 + n2 * 24
        out.append(dict(R=qvec2rot(q), t=np.array(t), cam=cid, name=name))
    return out
def read_points3D(p):
    data = open(p, "rb").read(); off = 8; n = struct.unpack_from("<Q", data, 0)[0]
    ids = np.empty(n, np.int64); xyz = np.empty((n, 3)); rgb = np.empty((n, 3), np.uint8); err = np.empty(n); tracks = []
    for i in range(n):
        pid, x, y, z, r, g, b, e, L = struct.unpack_from("<Q3d3BdQ", data, off); off += 51
        ids[i] = pid; xyz[i] = (x, y, z); rgb[i] = (r, g, b); err[i] = e
        tracks.append(data[off:off + 8*L]); off += 8*L
    return ids, xyz, rgb, err, tracks
def write_points3D(p, ids, xyz, rgb, err, tracks):
    with open(p, "wb") as f:
        f.write(struct.pack("<Q", len(ids)))
        for i in range(len(ids)):
            f.write(struct.pack("<Q3d3BdQ", int(ids[i]), *map(float, xyz[i]), *map(int, rgb[i]), float(err[i]), len(tracks[i])//8))
            f.write(tracks[i])

cams = read_cameras(os.path.join(SP0, "cameras.bin"))
ims = read_images(os.path.join(SP0, "images.bin"))
ids, X, RGB, ERR, TR = read_points3D(os.path.join(SP0, "points3D.bin"))
C = np.array([-im["R"].T @ im["t"] for im in ims])
log("cameras", len(ims), "points", len(X))
cam = cams[ims[0]["cam"]]
if cam["model"] not in (0, 1): raise SystemExit("undistort 된 PINHOLE 카메라만 지원 (Nb\\data 를 넣을 것)")
pp = cam["params"]; fxp, fyp, cxp, cyp = (pp[0], pp[0], pp[1], pp[2]) if cam["model"] == 0 else pp

# ---------------- 1) 바닥 평면 + 축척 ----------------
rng = np.random.default_rng(0)
up0 = -np.array([im["R"].T @ [0, 1, 0] for im in ims]).mean(0); up0 /= np.linalg.norm(up0)
good = ERR < 2.0
spread = np.median(np.linalg.norm(C - np.median(C, 0), axis=1))
dist, _ = cKDTree(C).query(X); near = good & (dist < 3 * spread)
Xn = X[near]; h0 = (Xn - np.median(C, 0)) @ up0
low = Xn[h0 < np.percentile(h0, 30)]
best = (0, None)
thr0 = np.percentile(np.abs(h0 - np.median(h0)), 50) * 0.05
for _ in range(3000):
    s = low[rng.choice(len(low), 3, replace=False)]
    n = np.cross(s[1]-s[0], s[2]-s[0]); nn = np.linalg.norm(n)
    if nn < 1e-12: continue
    n /= nn
    if abs(n @ up0) < np.cos(np.radians(25)): continue
    k = (np.abs((low - s[0]) @ n) < thr0).sum()
    if k > best[0]: best = (k, (n, s[0]))
n, p0 = best[1]; inl = np.abs((low - p0) @ n) < thr0
Q = low[inl]; c0 = Q.mean(0); up = np.linalg.svd(Q - c0, full_matrices=False)[2][2]
if up @ up0 < 0: up = -up
hcam = np.median((C - c0) @ up)
S = hcam / 0.67                     # 복도 1 기준 (카메라 높이 0.67) 대비 축척
log(f"floor inliers {inl.mean():.2f}, camera height {hcam:.3f} -> scale {S:.2f}")
Cc = C - C.mean(0); Cc -= np.outer(Cc @ up, up)
ax1 = np.linalg.svd(Cc, full_matrices=False)[2][0]; ax1 -= (ax1 @ up) * up; ax1 /= np.linalg.norm(ax1); ax2 = np.cross(up, ax1)
def F(p): q = p - c0; return np.stack([q @ ax1, q @ ax2, q @ up], -1)
Mw = np.stack([ax1, ax2, up], 1)
def W(p): return c0 + p @ Mw.T
Xf = F(Xn); Cf = F(C)

# ---------------- 2) 평면도 + 천장 높이 ----------------
cell = 0.04 * S
fl = np.abs(Xf[:, 2]) < 0.03 * S
xy = Xf[fl, :2]; mn = np.minimum(xy.min(0), Cf[:, :2].min(0)) - 0.5 * S
mx = np.maximum(xy.max(0), Cf[:, :2].max(0)) + 0.5 * S
Wd = np.ceil((mx - mn) / cell).astype(int)
g = np.zeros(Wd[::-1], bool)
ij = ((xy - mn) / cell).astype(int); g[ij[:, 1], ij[:, 0]] = True
cij = ((Cf[:, :2] - mn) / cell).astype(int); g[cij[:, 1], cij[:, 0]] = True
g = ndi.binary_closing(g, np.ones((7, 7)), iterations=2); g = ndi.binary_fill_holes(g)
lab, nl = ndi.label(g); g = lab == (1 + np.argmax(ndi.sum(g, lab, range(1, nl + 1))))
g = ndi.binary_opening(g, np.ones((5, 5)))
gi = ((Xf[:, :2] - mn) / cell).astype(int); gi = np.clip(gi, 0, np.array(g.shape[::-1]) - 1)
inside = g[gi[:, 1], gi[:, 0]]
zin = Xf[inside & (Xf[:, 2] > hcam), 2]
H = float(np.percentile(zin, 99)) if len(zin) > 100 else 2.4 * hcam
H = float(np.clip(H, 1.6 * hcam, 4.0 * hcam))
log(f"footprint area {g.sum()*cell*cell:.2f}, ceiling {H:.3f} ({H/hcam:.2f} x camera height)")

# ---------------- 3) 샘플 (바닥·천장·벽) ----------------
SP = 0.025 * S
xs = np.arange(mn[0], mn[0] + g.shape[1]*cell, SP); ys = np.arange(mn[1], mn[1] + g.shape[0]*cell, SP)
XX, YY = np.meshgrid(xs, ys)
ins = g[np.clip(((YY-mn[1])/cell).astype(int), 0, g.shape[0]-1), np.clip(((XX-mn[0])/cell).astype(int), 0, g.shape[1]-1)]
fx_, fy_ = XX[ins], YY[ins]
P = [np.stack([fx_, fy_, np.zeros_like(fx_)], 1), np.stack([fx_, fy_, np.full_like(fx_, H)], 1)]
N = [np.tile([0, 0, 1.], (len(fx_), 1)), np.tile([0, 0, -1.], (len(fx_), 1))]
zs = np.arange(SP/2, H, SP); wp = []; wn = []
for di, dj, nrm in [(0, 1, (-1, 0)), (0, -1, (1, 0)), (1, 0, (0, -1)), (-1, 0, (0, 1))]:
    edge = g & ~np.roll(g, (-di, -dj), (0, 1)); r, c = np.nonzero(edge)
    cx = mn[0] + (c+0.5)*cell + dj*cell/2; cy = mn[1] + (r+0.5)*cell + di*cell/2
    for t in np.arange(-cell/2 + SP/4, cell/2, SP):
        ex = cx + (t if di != 0 else 0); ey = cy + (t if dj != 0 else 0)
        for zz in zs:
            wp.append(np.stack([ex, ey, np.full_like(ex, zz)], 1)); wn.append(np.tile([nrm[0], nrm[1], 0.], (len(ex), 1)))
P.append(np.concatenate(wp)); N.append(np.concatenate(wn))
P = np.concatenate(P); N = np.concatenate(N); Pw = W(P); Nw = N @ Mw.T
log("samples", len(P))

# ---------------- 4) 사진 투영으로 색 ----------------
imgdir = os.path.join(SRC, "images")
views = sorted(ims, key=lambda d: d["name"])[::STEP]
bestS = np.full((len(Pw), 6), np.inf); cols = np.zeros((len(Pw), 6, 3), np.float32)
for vi, im in enumerate(views):
    img = cv2.imread(os.path.join(imgdir, im["name"]), cv2.IMREAD_REDUCED_COLOR_4)
    if img is None: continue
    h, w = img.shape[:2]; rs = cam["w"] / w
    Xc = Pw @ im["R"].T + im["t"]; zc = Xc[:, 2]; ok = zc > 0.05 * S
    zsafe = np.maximum(zc, 1e-6)
    u = (fxp * Xc[:, 0] / zsafe + cxp) / rs; v = (fyp * Xc[:, 1] / zsafe + cyp) / rs
    ok &= (u > 2) & (u < w-3) & (v > 2) & (v < h-3)
    Cw = -im["R"].T @ im["t"]; vd = Cw - Pw; dd = np.linalg.norm(vd, axis=1); cs = np.einsum('ij,ij->i', vd, Nw) / dd
    ok &= cs > 0.15; idx = np.nonzero(ok)[0]
    if len(idx) == 0: continue
    sc = dd[idx] / cs[idx]**0.5
    cc = img[v[idx].astype(int), u[idx].astype(int)][:, ::-1].astype(np.float32) / 255
    worst = np.argmax(bestS[idx], 1); bt = sc < bestS[idx, worst]
    bestS[idx[bt], worst[bt]] = sc[bt]; cols[idx[bt], worst[bt]] = cc[bt]
    if vi % 40 == 0: log(f"view {vi}/{len(views)}")
have = np.isfinite(bestS)
col = np.nanmedian(np.where(have[..., None], cols, np.nan), 1)
okc = ~np.isnan(col[:, 0])
if (~okc).any(): col[~okc] = col[okc][cKDTree(Pw[okc]).query(Pw[~okc])[1]]
log(f"no-view samples {(~okc).mean():.3f}")

# ---------------- 5) 바탕층 PLY ----------------
pos = Pw - Nw * 0.012 * S
t1 = np.where(np.abs(Nw @ up)[:, None] > 0.9, np.tile(ax1, (len(Nw), 1)), np.tile(up, (len(Nw), 1)))
t1 -= np.einsum('ij,ij->i', t1, Nw)[:, None] * Nw; t1 /= np.linalg.norm(t1, axis=1)[:, None]
Rm = np.stack([t1, np.cross(Nw, t1), Nw], 2)
qw = np.sqrt(np.maximum(0, 1+Rm[:,0,0]+Rm[:,1,1]+Rm[:,2,2]))/2
qx = np.sqrt(np.maximum(0, 1+Rm[:,0,0]-Rm[:,1,1]-Rm[:,2,2]))/2*np.sign(Rm[:,2,1]-Rm[:,1,2])
qy = np.sqrt(np.maximum(0, 1-Rm[:,0,0]+Rm[:,1,1]-Rm[:,2,2]))/2*np.sign(Rm[:,0,2]-Rm[:,2,0])
qz = np.sqrt(np.maximum(0, 1-Rm[:,0,0]-Rm[:,1,1]+Rm[:,2,2]))/2*np.sign(Rm[:,1,0]-Rm[:,0,1])
q = np.stack([qw, qx, qy, qz], 1); q /= np.linalg.norm(q, axis=1)[:, None]
names = (["f_dc_0", "f_dc_1", "f_dc_2"] + ["f_rest_%d" % i for i in sorted(range(24), key=str)] +
         ["opacity", "rot_0", "rot_1", "rot_2", "rot_3", "scale_0", "scale_1", "scale_2", "x", "y", "z"])
A = np.zeros((len(pos), len(names)), np.float32)
def setc(k, v): A[:, names.index(k)] = v
for i in range(3): setc("f_dc_%d" % i, (col[:, i] - 0.5) / 0.28209479177387814)
setc("opacity", np.log(0.99 / 0.01))
for i in range(4): setc("rot_%d" % i, q[:, i])
setc("scale_0", np.log(SP*0.7)); setc("scale_1", np.log(SP*0.7)); setc("scale_2", np.log(2e-4*S))
for i, k in enumerate("xyz"): setc(k, pos[:, i])
with open(os.path.join(OUT, "backdrop.ply"), "wb") as f:
    f.write(("ply\nformat binary_little_endian 1.0\ncomment Exported from Brush\ncomment Vertical axis: y\ncomment SH degree: 2\n"
             "element vertex %d\n" % len(A) + "".join("property float %s\n" % k for k in names) + "end_header\n").encode())
    f.write(A.astype("<f4").tobytes())
log("backdrop.ply", len(A))

# ---------------- 6) 시작점 보강 sparse ----------------
start = int(ids.max()) + 1
write_points3D(os.path.join(DST, "sparse", "0", "points3D.bin"),
               np.concatenate([ids, np.arange(start, start + len(Pw))]), np.concatenate([X, Pw]),
               np.concatenate([RGB, (np.clip(col, 0, 1)*255).astype(np.uint8)]),
               np.concatenate([ERR, np.ones(len(Pw))]), TR + [b""] * len(Pw))
for fn in os.listdir(SP0):
    if fn != "points3D.bin": shutil.copy(os.path.join(SP0, fn), os.path.join(DST, "sparse", "0", fn))
log("points3D", len(ids), "->", len(ids) + len(Pw))

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    axs[0].imshow(g, origin="lower"); axs[0].plot(cij[:, 0], cij[:, 1], "r.", ms=1); axs[0].set_title("footprint + cameras")
    axs[1].hist(Xf[:, 2], bins=300, range=(-0.5*hcam, 3*hcam)); axs[1].axvline(0, c="g"); axs[1].axvline(H, c="r"); axs[1].axvline(hcam, c="k")
    axs[1].set_yscale("log"); axs[1].set_title("height (green floor, black camera, red ceiling)")
    plt.savefig(os.path.join(OUT, "fill_diag.png"), dpi=70)
except Exception as e:
    print("diag skipped:", e)
log("done")
