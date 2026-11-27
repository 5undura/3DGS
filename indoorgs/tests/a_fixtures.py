"""A 테스트 공용: 합성 COLMAP 모델, 가짜 실행기, 작은 프로젝트."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from indoorgs.colmap import model as cm
from indoorgs.colmap.runner import ToolResult


def make_model(n=30, teleport_at=None, start=(0.0, 0.0, 1.4), step=0.1, frame_step=10) -> cm.Model:
    """x 축으로 일정하게 걷는 카메라 n 대 (회전 없음). teleport_at 번째 사진은 멀리 튐."""
    cams = {1: cm.Camera(1, 1, 64, 48, np.array([50.0, 50.0, 32.0, 24.0]))}
    images, points = {}, {}
    for i in range(n):
        c = np.array([start[0] + step * i, start[1], start[2]])
        if teleport_at is not None and i == teleport_at:
            c = c + np.array([0.0, 50.0, 0.0])
        name = f"f_{i + 1:05d}_{i * frame_step:06d}.jpg"
        # R = I, t = -C
        images[i + 1] = cm.Image(i + 1, np.array([1.0, 0, 0, 0]), -c, 1, name,
                                 np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), np.array([-1, -1, -1], np.int64))
    # 점 하나당 연속한 사진 3장
    pid = 1
    for i in range(1, n - 1):
        tr = np.array([[i, 0], [i + 1, 1], [i + 2, 2]], np.uint32)
        points[pid] = cm.Point3D(pid, np.array([0.1 * i, 2.0, 1.0]), np.array([10, 20, 30], np.uint8), 0.5, tr)
        images[i].pid[0] = pid
        images[i + 1].pid[1] = pid
        images[i + 2].pid[2] = pid
        pid += 1
    frames = [cm.Frame(i, 1, (1.0, 0, 0, 0, 0, 0, 0), [(0, 1, i)]) for i in images]
    rigs = struct.pack("<QIIiI", 1, 1, 1, 0, 1)
    return cm.Model(cams, images, points, frames, rigs)


def write_brush_ply(path: Path, n=100, opacity_logit=0.0, scales=(-3.0, -3.0, -3.0)):
    """Brush 처럼 xyz 가 맨 뒤인 PLY."""
    names = ["f_dc_0", "f_dc_1", "f_dc_2", "opacity", "rot_0", "rot_1", "rot_2", "rot_3",
             "scale_0", "scale_1", "scale_2", "x", "y", "z"]
    a = np.zeros((n, len(names)), np.float32)
    a[:, 3] = opacity_logit
    a[:, 8:11] = scales
    hdr = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n
    hdr += "".join(f"property float {k}\n" for k in names) + "end_header\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(hdr.encode() + a.tobytes())


class FakeRunner:
    """runner.run 대신. handlers[명령 이름](cmd) -> ToolResult|int|None. 부른 명령을 calls 에 모음."""

    def __init__(self, handlers=None, default=0):
        self.calls: list[list[str]] = []
        self.handlers = handlers or {}
        self.default = default

    def key(self, cmd):
        return cmd[1] if len(cmd) > 1 else cmd[0]

    def __call__(self, cmd, log_file=None, echo=None, timeout=None, keep=400):
        cmd = [str(c) for c in cmd]
        self.calls.append(cmd)
        h = self.handlers.get(self.key(cmd))
        r = h(cmd) if h else None
        if isinstance(r, ToolResult):
            return r
        return ToolResult(self.default if r is None else r, [])

    def names(self):
        return [self.key(c) for c in self.calls]


def arg(cmd, name):
    return cmd[cmd.index(name) + 1]


def make_project(root: Path, n_frames=30, colmap=True, trainer_exe=True, extra_yaml="") -> Path:
    """구간 s1 하나, 프레임 n 장(빈 파일)과 done.json 이 있는 프로젝트."""
    import json
    root.mkdir(parents=True, exist_ok=True)
    tools = root / "tools"
    tools.mkdir(exist_ok=True)
    (tools / "colmap.bat").write_text("x")
    (tools / "brush.exe").write_text("x")
    (root / "project.yaml").write_text(
        "tools:\n"
        f"  colmap: {(tools / 'colmap.bat').as_posix() if colmap else 'null'}\n"
        f"  brush: {(tools / 'brush.exe').as_posix() if trainer_exe else 'null'}\n"
        "sections:\n  - name: s1\n    video: " + (root / "v.mp4").as_posix() + "\n" + extra_yaml,
        encoding="utf-8")
    fr = root / "sections" / "s1" / "frames"
    fr.mkdir(parents=True)
    for i in range(n_frames):
        (fr / f"f_{i + 1:05d}_{i * 10:06d}.jpg").write_bytes(b"jpg")
    (fr / "done.json").write_text(json.dumps({"version": 1, "fps": 10.0, "count": n_frames}), encoding="utf-8")
    return root


def colmap_handlers(n_images=30, teleport_at=None, start=(0.0, 0.0, 1.4)):
    """가짜 COLMAP: 각 명령이 실제처럼 파일을 만듦."""
    import shutil
    import sqlite3

    def feat(cmd):
        db, imgs = Path(arg(cmd, "--database_path")), Path(arg(cmd, "--image_path"))
        with sqlite3.connect(str(db)) as c:
            c.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
                      " name TEXT NOT NULL UNIQUE, camera_id INTEGER NOT NULL)")
            for f in sorted(imgs.glob("*.jpg")):
                c.execute("INSERT INTO images (name, camera_id) VALUES (?, 1)", (f.name,))
        c.close()

    def mapper(cmd):
        out = Path(arg(cmd, "--output_path"))
        assert out.is_dir()
        cm.write_model(make_model(n_images, teleport_at, start), out / "0")
        cm.write_model(make_model(5), out / "1")          # 작은 조각 모델

    def undist(cmd):
        src, out = Path(arg(cmd, "--input_path")), Path(arg(cmd, "--output_path"))
        shutil.copytree(arg(cmd, "--image_path"), out / "images")
        cm.write_model(cm.read_model(src), out / "sparse")

    return {"feature_extractor": feat, "mapper": mapper, "pose_prior_mapper": mapper,
            "image_undistorter": undist}
