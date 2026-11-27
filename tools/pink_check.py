"""분홍 배경 비침 검사: 학습 카메라 몇 곳에서 정면/옆으로 돌려 렌더 -> 비어 있는 곳을 분홍으로 칠한 비교 그림 + 비침 수치
사용: python tools/pink_check.py <splat.ply> <Nb\\data> <출력 폴더> [카메라 수=6] [가로 픽셀=960]
- LichtFeld 헤드리스 렌더(--render-camera-path)를 씀. 배경이 검정 고정이라 2번 렌더함:
  (1) 원래 색  (2) 모든 스플랫을 흰색으로 바꾼 PLY -> 화면 밝기 = 불투명도(알파)
  분홍 합성 = 원래 색 + (1 - 알파) x 분홍
- 출력: pink_grid.png (행 = 카메라, 열 = 정면/45도/70도), pink.txt (비침 수치)
  비침 = 알파 < 0.9 인 픽셀 비율 (벽·바닥이 덜 차서 뒤가 비치는 곳), 구멍 = 알파 < 0.5
- 필요: numpy, opencv-python. config.ps1 의 $LichtFeld 경로를 PowerShell 에서 LICHTFELD 환경 변수로 넘기거나 기본 경로 사용
"""
import json, os, struct, subprocess, sys
import numpy as np, cv2

ply, data, out = sys.argv[1], sys.argv[2], sys.argv[3]
NCAM = int(sys.argv[4]) if len(sys.argv) > 4 else 6
OW = int(sys.argv[5]) if len(sys.argv) > 5 else 960
YAWS = [0, 45, 70]
LFS = os.environ.get("LICHTFELD", r"F:\Sample\tools\lfs\LichtFeld-Studio\build\LichtFeld-Studio.exe")
os.makedirs(out, exist_ok=True)


def qvec2rot(q):
    w, x, y, z = q
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
                     [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
                     [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]])


def rot2quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1) * 2; w = s / 4; x = (R[2, 1]-R[1, 2]) / s; y = (R[0, 2]-R[2, 0]) / s; z = (R[1, 0]-R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; w = (R[2, 1]-R[1, 2]) / s; x = s / 4; y = (R[0, 1]+R[1, 0]) / s; z = (R[0, 2]+R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; w = (R[0, 2]-R[2, 0]) / s; x = (R[0, 1]+R[1, 0]) / s; y = s / 4; z = (R[1, 2]+R[2, 1]) / s
    else:
        s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; w = (R[1, 0]-R[0, 1]) / s; x = (R[0, 2]+R[2, 0]) / s; y = (R[1, 2]+R[2, 1]) / s; z = s / 4
    q = np.array([w, x, y, z]); return q / np.linalg.norm(q)


# ---- COLMAP 카메라 읽기 ----
sp = os.path.join(data, "sparse", "0")
with open(os.path.join(sp, "cameras.bin"), "rb") as f:
    NP = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8}
    struct.unpack("<Q", f.read(8))
    cid, model, W, H = struct.unpack("<iiQQ", f.read(24))
    pp = struct.unpack("<%dd" % NP[model], f.read(8 * NP[model]))
fy = pp[0] if model == 0 else pp[1]
d = open(os.path.join(sp, "images.bin"), "rb").read(); off = 8; ims = []
for _ in range(struct.unpack_from("<Q", d, 0)[0]):
    off += 4; q = struct.unpack_from("<4d", d, off); off += 32; t = struct.unpack_from("<3d", d, off); off += 24; off += 4
    e = d.index(b"\0", off); name = d[off:e].decode(); off = e + 1
    n2, = struct.unpack_from("<Q", d, off); off += 8 + n2 * 24
    R = qvec2rot(q); ims.append((name, R.T, -R.T @ np.array(t)))
ims.sort(key=lambda a: a[0])
sel = [ims[i] for i in np.linspace(0, len(ims) - 1, NCAM + 2).astype(int)[1:-1]]

OH = int(round(OW * H / W / 2)) * 2
f_mm = float(np.clip(24 * fy / H, 10, 200))     # 세로 24mm 센서 기준 35mm 환산 (LichtFeld 규칙)
F = np.diag([1.0, -1.0, -1.0])                   # COLMAP(데이터) <-> LichtFeld 뷰어 좌표
kfs = []
for name, Rc, C in sel:
    for yd in YAWS:
        a = np.deg2rad(yd); Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        Rv = F @ (Rc @ Ry) @ F
        kfs.append({"time": float(len(kfs)), "position": (F @ C).tolist(), "rotation": rot2quat(Rv).tolist(),
                    "focal_length_mm": f_mm, "easing": 0})
path_json = os.path.join(out, "pink_path.json")
json.dump({"version": 4, "clip_duration": float(len(kfs)), "keyframes": kfs}, open(path_json, "w"), indent=1)


# ---- 흰색 PLY ----
def read_ply(p):
    with open(p, "rb") as f:
        head = b""
        while not head.endswith(b"end_header\n"):
            head += f.readline()
        body = f.read()
    names = [l.split()[-1] for l in head.decode().splitlines() if l.startswith("property")]
    return head, names, np.frombuffer(body, "<f4").reshape(-1, len(names)).copy()


head, names, A = read_ply(ply)
for i in range(3): A[:, names.index("f_dc_%d" % i)] = (1.0 - 0.5) / 0.28209479177387814
for k in names:
    if k.startswith("f_rest_"): A[:, names.index(k)] = 0
white = os.path.join(out, "_white.ply")
with open(white, "wb") as f: f.write(head); f.write(A.tobytes())
del A


def render(src, mp4):
    r = subprocess.run([LFS, "--render-camera-path=" + path_json, "--render-load=" + src, "--render-output=" + mp4,
                        "--render-width=%d" % OW, "--render-height=%d" % OH, "--render-fps=1", "--render-crf=0"],
                       capture_output=True, text=True, errors="replace")
    if not os.path.exists(mp4): sys.exit("렌더 실패:\n" + r.stdout[-2000:] + r.stderr[-2000:])
    cap = cv2.VideoCapture(mp4); fr = []
    while True:
        ok, im = cap.read()
        if not ok: break
        fr.append(im.astype(np.float32) / 255)
    return fr


col = render(ply, os.path.join(out, "_color.mp4"))
alp = render(white, os.path.join(out, "_alpha.mp4"))
os.remove(white)
n = min(len(col), len(alp), len(kfs))
pink = np.array([1.0, 0.0, 1.0], np.float32)          # BGR 에서도 (1,0,1) = 분홍
tiles, rows = [], []
for i in range(n):
    a = np.clip(alp[i].mean(2, keepdims=True), 0, 1)
    tiles.append(np.clip(col[i] + (1 - a) * pink, 0, 1))
    rows.append((i // len(YAWS), YAWS[i % len(YAWS)], float((a < 0.9).mean()), float((a < 0.5).mean())))
grid = np.vstack([np.hstack(tiles[r * len(YAWS):(r + 1) * len(YAWS)]) for r in range(n // len(YAWS))])
cv2.imencode(".png", (grid * 255).astype(np.uint8))[1].tofile(os.path.join(out, "pink_grid.png"))
lines = ["PLY: " + ply, f"카메라 {NCAM}곳 x {YAWS}도, {OW}x{OH}", "비침 = 알파<0.9 픽셀 비율, 구멍 = 알파<0.5"]
for yd in YAWS:
    v = [r for r in rows if r[1] == yd]
    lines.append(f"  {yd:>3}도: 비침 {np.mean([r[2] for r in v]) * 100:5.1f}%  구멍 {np.mean([r[3] for r in v]) * 100:5.1f}%")
lines.append(f"  전체: 비침 {np.mean([r[2] for r in rows]) * 100:5.1f}%  구멍 {np.mean([r[3] for r in rows]) * 100:5.1f}%")
open(os.path.join(out, "pink.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
