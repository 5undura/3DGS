"""gsplat 체크포인트(.pt) -> 표준 3DGS PLY (Unity / SuperSplat 용)
사용: python nc_export_ply.py ckpt_59999_rank0.pt out.ply
"""
import sys
import numpy as np
import torch

ckpt, out = sys.argv[1], sys.argv[2]
s = torch.load(ckpt, map_location="cpu", weights_only=True)["splats"]
means = s["means"].float().numpy()
scales = s["scales"].float().numpy()                       # log scale
quats = torch.nn.functional.normalize(s["quats"].float(), dim=-1).numpy()  # w x y z
opac = s["opacities"].float().numpy()[:, None]             # logit
sh0 = s["sh0"].float().numpy().reshape(len(means), 3)
shN = s["shN"].float().numpy()                              # [N, K, 3]
f_rest = shN.transpose(0, 2, 1).reshape(len(means), -1)    # channel-major like original 3DGS
normals = np.zeros_like(means)

names = ["x", "y", "z", "nx", "ny", "nz"]
names += [f"f_dc_{i}" for i in range(3)]
names += [f"f_rest_{i}" for i in range(f_rest.shape[1])]
names += ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
data = np.concatenate([means, normals, sh0, f_rest, opac, scales, quats], 1).astype(np.float32)
keep = np.isfinite(data).all(1)
data = data[keep]

with open(out, "wb") as f:
    head = "ply\nformat binary_little_endian 1.0\n"
    head += f"element vertex {len(data)}\n"
    head += "".join(f"property float {n}\n" for n in names)
    head += "end_header\n"
    f.write(head.encode("ascii"))
    f.write(data.tobytes())
print(f"{out}: {len(data):,} splats (SH degree {int(np.sqrt(shN.shape[1] + 1)) - 1})")
