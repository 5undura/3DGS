import json

import cv2
import numpy as np

from indoorgs.project import Project
from indoorgs.stages.base import StageContext
from indoorgs.stages.extract import Extract, PickArgs, pick_frames


def make_video(path, n=60, fps=10.0, size=(160, 120)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert w.isOpened()
    rng = np.random.default_rng(1)
    base = (rng.random((size[1] * 2, size[0] * 4)) * 255).astype(np.uint8)
    for i in range(n):
        x = i * 4                                    # 옆으로 움직이는 무늬
        fr = cv2.cvtColor(base[:size[1], x:x + size[0]].copy(), cv2.COLOR_GRAY2BGR)
        if i == 30:
            fr = cv2.GaussianBlur(fr, (15, 15), 5)   # 흐린 프레임
        w.write(fr)
    w.release()


def test_pick_frames(tmp_path):
    v = tmp_path / "v.avi"
    make_video(v)
    out = tmp_path / "한글" / "frames"
    done = pick_frames(str(v), out, PickArgs(min_diff=4))
    files = sorted(out.glob("*.jpg"))
    assert done["count"] == len(files) > 5
    assert done["total_frames"] == 60 and abs(done["fps"] - 10.0) < 0.01
    assert files[0].name == "f_00001_000000.jpg"
    assert "f_" in files[-1].name and len(files[-1].stem.split("_")[2]) == 6
    assert not any(f.name.endswith("_000030.jpg") for f in files)    # 흐린 프레임은 안 남김
    assert cv2.imdecode(np.fromfile(str(files[0]), np.uint8), cv2.IMREAD_COLOR) is not None


def test_extract_stage(tmp_path):
    root = tmp_path / "p"
    root.mkdir()
    v = root / "v.avi"
    make_video(v)
    (root / "project.yaml").write_text(f"sections:\n  - name: s1\n    video: {v.as_posix()}\n", encoding="utf-8")
    p = Project(root)
    sec, paths = p.sections()[0]
    paths.frames.mkdir(parents=True)
    (paths.frames / "f_00001_000000.jpg").write_bytes(b"old")       # 끊긴 실행의 흔적
    ctx = StageContext(p, sec, paths, lambda m: None)
    res = Extract().run(ctx)
    assert res.ok, res.message
    d = json.loads(paths.frames_done.read_text(encoding="utf-8"))
    assert {"fps", "total_frames", "count", "args"} <= set(d) and d["args"]["min_diff"] == 4.0
    assert any(x.name.startswith("frames_failed_") for x in paths.root.iterdir())
    assert Extract().is_done(ctx)
