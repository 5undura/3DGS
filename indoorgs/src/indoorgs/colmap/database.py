"""COLMAP DB(sqlite)에 위치 제약(pose_priors) 쓰기.

스키마 확인: COLMAP 4.2.1 (태그 4.2.1, bd1fcf6) src/colmap/scene/database_sqlite.cc CreatePosePriorTable,
WritePosePrior + pycolmap 4.2.1 으로 실제 DB 를 만들어 표 구조·blob 확인 (2026-11-27).

  pose_priors(pose_prior_id INTEGER PRIMARY KEY NOT NULL, corr_data_id INTEGER NOT NULL,
              corr_sensor_id INTEGER NOT NULL, corr_sensor_type INTEGER NOT NULL,
              position BLOB, position_covariance BLOB, gravity BLOB, coordinate_system INTEGER NOT NULL)
  UNIQUE INDEX pose_prior_data_assignment(corr_data_id, corr_sensor_id, corr_sensor_type)

- 카메라 사진: corr_data_id = images.image_id, corr_sensor_id = images.camera_id, corr_sensor_type = 0 (CAMERA)
- position: float64 x3, position_covariance: float64 3x3 (Eigen 열 우선, 대각이면 순서 무관), gravity: float64 x3 (모르면 NaN)
- coordinate_system: -1 UNDEFINED, 0 WGS84, 1 CARTESIAN
- 3.x 의 pose_priors(image_id, ...) 는 4.x 가 열 때 새 형식으로 옮김 -> 여기선 4.x 형식만 씀
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np

SENSOR_CAMERA = 0
COORD = {"UNDEFINED": -1, "WGS84": 0, "CARTESIAN": 1}

CREATE_POSE_PRIORS = (
    "CREATE TABLE IF NOT EXISTS pose_priors"
    "   (pose_prior_id              INTEGER  PRIMARY KEY  NOT NULL,"
    "    corr_data_id               INTEGER               NOT NULL,"
    "    corr_sensor_id             INTEGER               NOT NULL,"
    "    corr_sensor_type           INTEGER               NOT NULL,"
    "    position                   BLOB,"
    "    position_covariance        BLOB,"
    "    gravity                    BLOB,"
    "    coordinate_system          INTEGER               NOT NULL);"
    "CREATE UNIQUE INDEX IF NOT EXISTS pose_prior_data_assignment ON "
    "   pose_priors(corr_data_id, corr_sensor_id, corr_sensor_type);")


def read_images(db: Path) -> dict[str, tuple[int, int]]:
    """이름 -> (image_id, camera_id)"""
    with closing(sqlite3.connect(str(db))) as c:
        return {n: (i, cid) for i, n, cid in c.execute("SELECT image_id, name, camera_id FROM images")}


def write_pose_priors(db: Path, priors: dict) -> tuple[int, list[str]]:
    """priors.json 내용을 DB 에 씀. 같은 사진의 기존 제약은 바꿔 씀.
    돌려줌: (쓴 수, DB 에 없는 사진 이름 목록)"""
    cs = COORD.get(str(priors.get("coordinate_system", "CARTESIAN")).upper())
    if cs is None:
        raise ValueError(f"coordinate_system {priors.get('coordinate_system')!r} 모름")
    images = read_images(db)
    missing, n = [], 0
    nan3 = np.full(3, np.nan).tobytes()
    # closing: Windows 에서 열린 DB 파일은 이름을 못 바꿈 (실패 시 옆으로 치울 때)
    with closing(sqlite3.connect(str(db))) as c, c:
        c.executescript(CREATE_POSE_PRIORS)
        for fr in priors.get("frames", []):
            name = fr["image"]
            if name not in images:
                missing.append(name)
                continue
            iid, cid = images[name]
            pos = np.asarray(fr["xyz"], np.float64)
            sig = np.asarray(fr.get("sigma", [1.0, 1.0, 1.0]), np.float64)
            cov = np.diag(sig ** 2).astype(np.float64)
            c.execute("DELETE FROM pose_priors WHERE corr_data_id=? AND corr_sensor_id=? AND corr_sensor_type=?",
                      (iid, cid, SENSOR_CAMERA))
            c.execute("INSERT INTO pose_priors (pose_prior_id, corr_data_id, corr_sensor_id, corr_sensor_type, "
                      "position, position_covariance, gravity, coordinate_system) VALUES (NULL,?,?,?,?,?,?,?)",
                      (iid, cid, SENSOR_CAMERA, pos.tobytes(), cov.T.copy().tobytes(), nan3, cs))
            n += 1
    return n, missing


def count_pose_priors(db: Path) -> int:
    with closing(sqlite3.connect(str(db))) as c:
        try:
            return c.execute("SELECT COUNT(*) FROM pose_priors").fetchone()[0]
        except sqlite3.OperationalError:
            return 0
