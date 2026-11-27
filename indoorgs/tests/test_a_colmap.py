import sqlite3

import numpy as np
import pytest

from indoorgs.colmap import database, model as cm
from a_fixtures import make_model


def test_model_roundtrip_and_remove(tmp_path):
    m = make_model(12)
    cm.write_model(m, tmp_path / "m")
    r = cm.read_model(tmp_path / "m")
    assert len(r.images) == 12 and len(r.points) == 10 and len(r.frames) == 12
    assert np.allclose(r.images[3].center, [0.2, 0, 1.4])
    assert r.rigs_bytes == m.rigs_bytes
    assert cm.count_images(tmp_path / "m") == 12
    new, n = cm.remove_images(r, {r.images[5].name})
    assert n == 1 and 5 not in new.images and len(new.frames) == 11
    for p in new.points.values():
        assert 5 not in p.track[:, 0] and len(p.track) >= 2
    # 살아남은 점만 가리킴
    alive = set(new.points)
    for im in new.images.values():
        assert all(x == -1 or x in alive for x in im.pid)


def test_model_readable_by_pycolmap(tmp_path):
    pycolmap = pytest.importorskip("pycolmap")
    m = make_model(8)
    new, _ = cm.remove_images(m, {m.images[2].name})
    cm.write_model(new, tmp_path / "m")
    rec = pycolmap.Reconstruction(str(tmp_path / "m"))
    assert rec.num_reg_images() == 7
    assert rec.num_points3D() == len(new.points)


def _db_with_images(path, names):
    with sqlite3.connect(str(path)) as c:
        c.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, name TEXT NOT NULL UNIQUE,"
                  " camera_id INTEGER NOT NULL)")
        for n in names:
            c.execute("INSERT INTO images (name, camera_id) VALUES (?, 1)", (n,))
    c.close()


def test_write_pose_priors(tmp_path):
    db = tmp_path / "db.db"
    _db_with_images(db, ["a.jpg", "b.jpg"])
    pri = {"version": 1, "coordinate_system": "CARTESIAN",
           "frames": [{"image": "a.jpg", "xyz": [1, 2, 3], "sigma": [0.5, 0.5, 0.3]},
                      {"image": "zz.jpg", "xyz": [0, 0, 0]}]}
    n, missing = database.write_pose_priors(db, pri)
    assert n == 1 and missing == ["zz.jpg"]
    n, _ = database.write_pose_priors(db, pri)          # 다시 써도 중복 안 됨
    assert database.count_pose_priors(db) == 1
    with sqlite3.connect(str(db)) as c:
        row = c.execute("SELECT corr_data_id, corr_sensor_id, corr_sensor_type, position, position_covariance,"
                        " coordinate_system FROM pose_priors").fetchone()
    assert row[:3] == (1, 1, 0) and row[5] == 1
    assert np.allclose(np.frombuffer(row[3]), [1, 2, 3])
    assert np.allclose(np.frombuffer(row[4]).reshape(3, 3), np.diag([0.25, 0.25, 0.09]))


def test_pose_priors_match_pycolmap(tmp_path):
    pycolmap = pytest.importorskip("pycolmap")
    db = tmp_path / "db.db"
    d = pycolmap.Database.open(str(db))
    cid = d.write_camera(pycolmap.Camera.create_from_model_name(1, "OPENCV", 500.0, 640, 480))
    d.write_image(pycolmap.Image(name="f_00001_000000.jpg", camera_id=cid))
    d.close()
    database.write_pose_priors(db, {"version": 1, "coordinate_system": "CARTESIAN",
                                    "frames": [{"image": "f_00001_000000.jpg", "xyz": [1, 2, 3],
                                                "sigma": [0.5, 0.5, 0.3]}]})
    d = pycolmap.Database.open(str(db))
    pp = d.read_all_pose_priors()
    d.close()
    assert len(pp) == 1
    assert np.allclose(pp[0].position, [1, 2, 3])
    assert np.allclose(np.asarray(pp[0].position_covariance), np.diag([0.25, 0.25, 0.09]))
    assert "CARTESIAN" in str(pp[0].coordinate_system)
