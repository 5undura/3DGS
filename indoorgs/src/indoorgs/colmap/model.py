"""COLMAP 모델(sparse/N) 읽기·쓰기 - numpy 만. 저장소 tools/fill/colmap_io.py, tools/diagnose.py 를 옮겨 넓힘.

형식은 COLMAP 4.2.1 소스 src/colmap/scene/reconstruction_io_binary.cc 로 확인 (2026-11-27):
- cameras.bin  : u64 개수, [i32 id, i32 model, u64 w, u64 h, f64 params...]
- images.bin   : u64 개수, [u32 id, f64 qw qx qy qz, f64 tx ty tz, u32 camera_id, name\\0, u64 n, (f64 x, f64 y, u64 point3D_id)*n]
- points3D.bin : u64 개수, [u64 id, f64 xyz, u8 rgb, f64 error, u64 L, (u32 image_id, u32 point2D_idx)*L]
- frames.bin (4.x): u64 개수, [u32 frame_id, u32 rig_id, f64 qw..tz(7), u32 n, (i32 sensor_type, u32 sensor_id, u64 data_id)*n]
- rigs.bin (4.x): 손대지 않고 그대로 복사
txt 모델은 읽기만 (cameras/images/points3D). 쓰기는 항상 bin.
"""
from __future__ import annotations

import shutil
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CAM_NAMES = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 3: "RADIAL", 4: "OPENCV",
             5: "OPENCV_FISHEYE", 6: "FULL_OPENCV", 7: "FOV", 8: "SIMPLE_RADIAL_FISHEYE",
             9: "RADIAL_FISHEYE", 10: "THIN_PRISM_FISHEYE"}
CAM_IDS = {v: k for k, v in CAM_NAMES.items()}
CAM_NP = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}
SENSOR_CAMERA = 0


@dataclass
class Camera:
    id: int
    model: int
    width: int
    height: int
    params: np.ndarray


@dataclass
class Image:
    id: int
    q: np.ndarray          # w x y z (cam_from_world)
    t: np.ndarray
    camera_id: int
    name: str
    xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    pid: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))   # -1 = 3D 점 없음

    @property
    def R(self) -> np.ndarray:
        return qvec2rot(self.q)

    @property
    def center(self) -> np.ndarray:
        """카메라 중심 (월드 좌표) = -R^T t"""
        return -self.R.T @ self.t


@dataclass
class Point3D:
    id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    track: np.ndarray      # (L, 2) uint32: image_id, point2D_idx


@dataclass
class Frame:
    id: int
    rig_id: int
    pose: tuple            # qw qx qy qz tx ty tz
    data: list             # [(sensor_type, sensor_id, data_id)]


@dataclass
class Model:
    cameras: dict[int, Camera]
    images: dict[int, Image]
    points: dict[int, Point3D]
    frames: list[Frame] | None = None
    rigs_bytes: bytes | None = None

    @property
    def num_points(self) -> int:
        return len(self.points)

    def mean_reproj_error(self) -> float | None:
        if not self.points:
            return None
        return float(np.mean([p.error for p in self.points.values()]))


def qvec2rot(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def find_model(folder: Path) -> tuple[Path, str]:
    folder = Path(folder)
    for sub in ("", "sparse/0", "sparse", "0"):
        d = folder / sub
        for ext in ("bin", "txt"):
            if (d / f"images.{ext}").is_file():
                return d, ext
    raise FileNotFoundError(f"COLMAP 모델을 찾지 못함: {folder}")


# ---------------- 읽기 ----------------
def _read_cameras_bin(p: Path) -> dict[int, Camera]:
    data = p.read_bytes()
    off, cams = 8, {}
    for _ in range(struct.unpack_from("<Q", data, 0)[0]):
        cid, model, w, h = struct.unpack_from("<iiQQ", data, off); off += 24
        n = CAM_NP[model]
        params = np.array(struct.unpack_from(f"<{n}d", data, off)); off += 8 * n
        cams[cid] = Camera(cid, model, w, h, params)
    return cams


def _read_images_bin(p: Path) -> dict[int, Image]:
    data = p.read_bytes()
    off, imgs = 8, {}
    pt = np.dtype([("x", "<f8"), ("y", "<f8"), ("p", "<i8")])
    for _ in range(struct.unpack_from("<Q", data, 0)[0]):
        iid, = struct.unpack_from("<I", data, off); off += 4
        q = np.array(struct.unpack_from("<4d", data, off)); off += 32
        t = np.array(struct.unpack_from("<3d", data, off)); off += 24
        cid, = struct.unpack_from("<I", data, off); off += 4
        e = data.index(b"\0", off)
        name = data[off:e].decode("utf-8"); off = e + 1
        n, = struct.unpack_from("<Q", data, off); off += 8
        arr = np.frombuffer(data, pt, count=n, offset=off); off += 24 * n
        imgs[iid] = Image(iid, q, t, cid, name, np.stack([arr["x"], arr["y"]], 1) if n else np.zeros((0, 2)),
                          arr["p"].copy())
    return imgs


def _read_points_bin(p: Path) -> dict[int, Point3D]:
    data = p.read_bytes()
    off, pts = 8, {}
    for _ in range(struct.unpack_from("<Q", data, 0)[0]):
        pid, x, y, z, r, g, b, err, L = struct.unpack_from("<Q3d3BdQ", data, off); off += 51
        tr = np.frombuffer(data, "<u4", count=2 * L, offset=off).reshape(L, 2).copy(); off += 8 * L
        pts[pid] = Point3D(pid, np.array([x, y, z]), np.array([r, g, b], np.uint8), err, tr)
    return pts


def _read_frames_bin(p: Path) -> list[Frame]:
    data = p.read_bytes()
    off, out = 8, []
    for _ in range(struct.unpack_from("<Q", data, 0)[0]):
        fid, rid = struct.unpack_from("<II", data, off); off += 8
        pose = struct.unpack_from("<7d", data, off); off += 56
        n, = struct.unpack_from("<I", data, off); off += 4
        items = []
        for _ in range(n):
            st, sid, did = struct.unpack_from("<iIQ", data, off); off += 16
            items.append((st, sid, did))
        out.append(Frame(fid, rid, pose, items))
    return out


def _read_txt(d: Path) -> tuple[dict, dict, dict]:
    cams, imgs, pts = {}, {}, {}
    for line in (d / "cameras.txt").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            e = line.split()
            cams[int(e[0])] = Camera(int(e[0]), CAM_IDS[e[1]], int(e[2]), int(e[3]), np.array(e[4:], float))
    lines = [ln for ln in (d / "images.txt").read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        e = lines[i].split()
        p2 = lines[i + 1].split() if i + 1 < len(lines) else []
        xy = np.array(p2, float).reshape(-1, 3) if p2 else np.zeros((0, 3))
        iid = int(e[0])
        imgs[iid] = Image(iid, np.array(e[1:5], float), np.array(e[5:8], float), int(e[8]), " ".join(e[9:]),
                          xy[:, :2], xy[:, 2].astype(np.int64))
        i += 2
    pf = d / "points3D.txt"
    if pf.is_file():
        for line in pf.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                e = line.split()
                tr = np.array(e[8:], np.int64).reshape(-1, 2).astype(np.uint32)
                pts[int(e[0])] = Point3D(int(e[0]), np.array(e[1:4], float), np.array(e[4:7], np.uint8),
                                         float(e[7]), tr)
    return cams, imgs, pts


def read_model(folder: Path) -> Model:
    d, ext = find_model(folder)
    if ext == "txt":
        return Model(*_read_txt(d))
    pts = _read_points_bin(d / "points3D.bin") if (d / "points3D.bin").is_file() else {}
    frames = _read_frames_bin(d / "frames.bin") if (d / "frames.bin").is_file() else None
    rigs = (d / "rigs.bin").read_bytes() if (d / "rigs.bin").is_file() else None
    return Model(_read_cameras_bin(d / "cameras.bin"), _read_images_bin(d / "images.bin"), pts, frames, rigs)


def count_images(folder: Path) -> int:
    """등록된 사진 수 (images.bin 앞 8바이트만 읽음)."""
    d, ext = find_model(folder)
    if ext == "bin":
        with (d / "images.bin").open("rb") as f:
            return struct.unpack("<Q", f.read(8))[0]
    return len(_read_txt(d)[1])


# ---------------- 쓰기 (bin) ----------------
def write_model(model: Model, out: Path) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "cameras.bin").open("wb") as f:
        f.write(struct.pack("<Q", len(model.cameras)))
        for cid in sorted(model.cameras):
            c = model.cameras[cid]
            f.write(struct.pack("<iiQQ", c.id, c.model, c.width, c.height))
            f.write(struct.pack(f"<{len(c.params)}d", *map(float, c.params)))
    pt = np.dtype([("x", "<f8"), ("y", "<f8"), ("p", "<i8")])
    with (out / "images.bin").open("wb") as f:
        f.write(struct.pack("<Q", len(model.images)))
        for iid in sorted(model.images):
            im = model.images[iid]
            f.write(struct.pack("<I4d3dI", im.id, *map(float, im.q), *map(float, im.t), im.camera_id))
            f.write(im.name.encode("utf-8") + b"\0")
            f.write(struct.pack("<Q", len(im.pid)))
            arr = np.empty(len(im.pid), pt)
            if len(im.pid):
                arr["x"], arr["y"], arr["p"] = im.xy[:, 0], im.xy[:, 1], im.pid
            f.write(arr.tobytes())
    with (out / "points3D.bin").open("wb") as f:
        f.write(struct.pack("<Q", len(model.points)))
        for pid in sorted(model.points):
            p = model.points[pid]
            f.write(struct.pack("<Q3d3BdQ", p.id, *map(float, p.xyz), *map(int, p.rgb), float(p.error), len(p.track)))
            f.write(np.ascontiguousarray(p.track, "<u4").tobytes())
    if model.frames is not None:
        with (out / "frames.bin").open("wb") as f:
            f.write(struct.pack("<Q", len(model.frames)))
            for fr in model.frames:
                f.write(struct.pack("<II7dI", fr.id, fr.rig_id, *fr.pose, len(fr.data)))
                for st, sid, did in fr.data:
                    f.write(struct.pack("<iIQ", st, sid, did))
    if model.rigs_bytes is not None:
        (out / "rigs.bin").write_bytes(model.rigs_bytes)


def remove_images(model: Model, names: set[str]) -> tuple[Model, int]:
    """사진을 뺀 새 모델 (COLMAP image_deleter 와 같은 결과를 목표로).
    - 사진 삭제, 3D 점 트랙에서 그 사진 관측 삭제, 관측이 2개 미만이 된 점 삭제
    - 삭제된 점을 가리키던 다른 사진의 point3D_id 는 -1
    - frames.bin 에서 그 사진(data_id) 삭제, 빈 프레임은 삭제
    돌려줌: (새 모델, 지운 사진 수)"""
    drop = {iid for iid, im in model.images.items() if im.name in names}
    images = {iid: Image(im.id, im.q, im.t, im.camera_id, im.name, im.xy, im.pid.copy())
              for iid, im in model.images.items() if iid not in drop}
    points, dead = {}, set()
    for pid, p in model.points.items():
        tr = p.track[~np.isin(p.track[:, 0], list(drop))] if drop and len(p.track) else p.track
        if len(tr) < 2:
            dead.add(pid)
            continue
        points[pid] = Point3D(p.id, p.xyz, p.rgb, p.error, tr)
    if dead:
        dead_arr = np.array(sorted(dead), np.int64)
        for im in images.values():
            if len(im.pid):
                im.pid[np.isin(im.pid, dead_arr)] = -1
    frames = None
    if model.frames is not None:
        frames = []
        for fr in model.frames:
            data = [d for d in fr.data if not (d[0] == SENSOR_CAMERA and d[2] in drop)]
            if data:
                frames.append(Frame(fr.id, fr.rig_id, fr.pose, data))
    return Model(dict(model.cameras), images, points, frames, model.rigs_bytes), len(drop)
