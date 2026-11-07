"""같은 위치에서 카메라를 옆으로 돌려(벽 쪽) 여러 결과를 나란히 렌더링 -> PNG
사용 (examples/gsplat 폴더에서):
  python nc_compare.py DATA_DIR FACTOR OUT.png 이름1=ckpt1.pt 이름2=ckpt2.pt ...
"""
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw
from gsplat.rendering import rasterization
from datasets.colmap import Parser

data_dir, factor, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
runs = [a.split("=", 1) for a in sys.argv[4:]]
YAWS = [0, 45, 70]                 # 정면(학습 각도), 45도, 70도
N_CAMS = 4

p = Parser(data_dir=data_dir, factor=factor, normalize=False, test_every=0)
K = torch.from_numpy(list(p.Ks_dict.values())[0]).float().cuda()
W, H = list(p.imsize_dict.values())[0]
idx = np.linspace(0, len(p.camtoworlds) - 1, N_CAMS + 2).astype(int)[1:-1]

def yaw(c2w, deg):
    a = np.deg2rad(deg); c, s = np.cos(a), np.sin(a)
    r = c2w.copy(); r[:3, :3] = c2w[:3, :3] @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return r

views = [yaw(p.camtoworlds[i], d) for i in idx for d in YAWS]
cols = []
for name, path in runs:
    s = torch.load(path, map_location="cuda", weights_only=True)["splats"]
    colors = torch.cat([s["sh0"], s["shN"]], 1)
    shd = int(np.sqrt(colors.shape[1])) - 1
    tiles = []
    for v in views:
        vm = torch.linalg.inv(torch.from_numpy(v).float().cuda())[None]
        with torch.no_grad():
            img, alpha, _ = rasterization(s["means"], s["quats"], torch.exp(s["scales"]),
                                          torch.sigmoid(s["opacities"]), colors, vm, K[None],
                                          W, H, sh_degree=shd)
        # 비어 있는 곳을 분홍으로 (Unity 분홍 배경 테스트와 같은 의미)
        pink = torch.tensor([1.0, 0.0, 1.0], device="cuda")
        rgb = img[0].clamp(0, 1) + pink * (1 - alpha[0])
        tiles.append(Image.fromarray((rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)))
    cols.append((name, tiles))
    del s; torch.cuda.empty_cache()

tw, th = W // 2, H // 2
canvas = Image.new("RGB", (tw * len(cols), (th) * len(views) + 40), "white")
d = ImageDraw.Draw(canvas)
for c, (name, tiles) in enumerate(cols):
    d.text((c * tw + 10, 10), name, fill="black")
    for r, t in enumerate(tiles):
        canvas.paste(t.resize((tw, th)), (c * tw, 40 + r * th))
canvas.save(out)
print("saved", out, "| rows = 카메라 x [정면, 45도, 70도], 분홍 = 비어 있는 곳")
