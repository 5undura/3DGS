"""COLMAP 바이너리 모델 읽기·쓰기 (cameras.bin, images.bin, points3D.bin). tools/fill/colmap_io.py 이식, numpy 만 씀.
COLMAP 4.x 도 이 세 파일 형식은 3.x 와 같음 (rigs.bin, frames.bin 은 따로 추가된 파일)."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

# 모델 번호 -> 파라미터 수 (SIMPLE_PINHOLE=0, PINHOLE=1, ...)
NUM_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}
MODEL_ID = {"SIMPLE_PINHOLE": 0, "PINHOLE": 1}


def qvec2rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def read_cameras(path) -> dict:
    cams = {}
    with open(path, "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
            n = NUM_PARAMS[model]
            cams[cid] = dict(model=model, w=w, h=h, params=np.array(struct.unpack("<%dd" % n, f.read(8 * n))))
    return cams


def read_images(path) -> list[dict]:
    """[{id, q, R, t, cam, name}] (2D 점은 건너뜀). 카메라 중심 = -R^T t"""
    data = Path(path).read_bytes()
    off, out = 8, []
    for _ in range(struct.unpack_from("<Q", data, 0)[0]):
        iid, = struct.unpack_from("<i", data, off); off += 4
        q = struct.unpack_from("<4d", data, off); off += 32
        t = struct.unpack_from("<3d", data, off); off += 24
        cid, = struct.unpack_from("<i", data, off); off += 4
        e = data.index(b"\0", off); name = data[off:e].decode("utf-8"); off = e + 1
        n2, = struct.unpack_from("<Q", data, off); off += 8 + n2 * 24
        out.append(dict(id=iid, q=np.array(q), R=qvec2rot(q), t=np.array(t), cam=cid, name=name))
    return out


def read_points3D(path) -> dict:
    data = Path(path).read_bytes()
    off = 8
    n = struct.unpack_from("<Q", data, 0)[0]
    ids = np.empty(n, np.int64); xyz = np.empty((n, 3)); rgb = np.empty((n, 3), np.uint8); err = np.empty(n)
    for i in range(n):
        pid, x, y, z, r, g, b, e, L = struct.unpack_from("<Q3d3BdQ", data, off); off += 51
        ids[i] = pid; xyz[i] = (x, y, z); rgb[i] = (r, g, b); err[i] = e
        off += 8 * L
    return dict(ids=ids, xyz=xyz, rgb=rgb, err=err)


def write_model(folder, cams: dict, images: list[dict], xyz, rgb=None, err=None):
    """테스트·도구용 최소 쓰기 (2D 점·트랙 없음). images 항목: {q, t, cam, name}"""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for cid, c in cams.items():
            f.write(struct.pack("<iiQQ", cid, c["model"], c["w"], c["h"]))
            f.write(struct.pack("<%dd" % len(c["params"]), *map(float, c["params"])))
    with open(folder / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for i, im in enumerate(images, 1):
            f.write(struct.pack("<i4d3di", i, *map(float, im["q"]), *map(float, im["t"]), im["cam"]))
            f.write(im["name"].encode("utf-8") + b"\0")
            f.write(struct.pack("<Q", 0))
    xyz = np.asarray(xyz, float)
    rgb = np.full((len(xyz), 3), 128, np.uint8) if rgb is None else rgb
    err = np.full(len(xyz), 0.5) if err is None else err
    with open(folder / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", len(xyz)))
        for i in range(len(xyz)):
            f.write(struct.pack("<Q3d3BdQ", i + 1, *map(float, xyz[i]), *map(int, rgb[i]), float(err[i]), 0))


def find_model(*folders) -> Path | None:
    """cameras/images/points3D.bin 이 있는 첫 폴더 (폴더 자체 또는 sparse/0)."""
    for d in folders:
        for sub in ("sparse/0", ""):
            p = Path(d) / sub
            if all((p / f).is_file() for f in ("cameras.bin", "images.bin", "points3D.bin")):
                return p
    return None
