"""COLMAP 4.2.1 CLI 흉내 (pycolmap 4.2.1 위). stages/register.py 가 부르는 명령만, COLMAP 과 같은 옵션 이름으로.

  python colmap_shim.py <명령> --옵션 값 ...

- 모르는 명령·옵션은 COLMAP 처럼 실패(코드 1) -> register.py 의 옵션 이름이 틀리면 여기서 드러남.
  옵션 이름 출처: COLMAP 4.2.1 controllers/option_manager.cc, exe/feature.cc, exe/sfm.cc, exe/image.cc
  (prior_matching/RESULTS.md 4절, register.py 머리말과 같음).
- 값은 pycolmap 옵션 객체에 그대로 넣음. CPU 4스레드. mapper seed 는 환경 변수 SHIM_SEED (기본 0).
- 끝에 "SHIM <명령> <초>s" 를 찍음 (단계별 시간 측정용).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

THREADS = int(os.environ.get("SHIM_THREADS", "4"))


def _bool(v: str) -> bool:
    s = v.strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"bool 값이 아님: {v}")


# 명령 -> {옵션: 변환 함수}
DB = {"--database_path": str}
COMMANDS = {
    "feature_extractor": {**DB, "--image_path": str, "--ImageReader.single_camera": _bool,
                          "--ImageReader.camera_model": str},
    "sequential_matcher": {**DB, "--SequentialMatching.overlap": int, "--SequentialMatching.quadratic_overlap": _bool,
                           "--SequentialMatching.loop_detection": _bool, "--SequentialMatching.vocab_tree_path": str},
    "spatial_matcher": {**DB, "--SpatialMatching.max_distance": float, "--SpatialMatching.max_num_neighbors": int,
                        "--SpatialMatching.ignore_z": _bool},
    "exhaustive_matcher": dict(DB),
    "mapper": {**DB, "--image_path": str, "--output_path": str, "--input_path": str},
    "pose_prior_mapper": {**DB, "--image_path": str, "--output_path": str, "--input_path": str,
                          "--overwrite_priors_covariance": _bool, "--use_robust_loss_on_prior_position": _bool,
                          "--prior_position_loss_scale": float},
    "image_undistorter": {"--image_path": str, "--input_path": str, "--output_path": str, "--output_type": str},
}
REQUIRED = {
    "feature_extractor": ["--database_path", "--image_path"],
    "sequential_matcher": ["--database_path"], "spatial_matcher": ["--database_path"],
    "exhaustive_matcher": ["--database_path"],
    "mapper": ["--database_path", "--image_path", "--output_path"],
    "pose_prior_mapper": ["--database_path", "--image_path", "--output_path"],
    "image_undistorter": ["--image_path", "--input_path", "--output_path"],
}


def parse(cmd: str, argv: list[str]) -> dict:
    spec = COMMANDS[cmd]
    if len(argv) % 2:
        raise SystemExit(f"옵션 값 짝이 안 맞음: {argv}")
    out = {}
    for k, v in zip(argv[::2], argv[1::2]):
        if k not in spec:
            raise SystemExit(f"unrecognised option '{k}' ({cmd})")
        out[k] = spec[k](v)
    miss = [k for k in REQUIRED[cmd] if k not in out]
    if miss:
        raise SystemExit(f"필수 옵션 없음 {miss} ({cmd})")
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] not in COMMANDS:
        print(f"모르는 명령: {argv[1:2]}  (가능: {sorted(COMMANDS)})", file=sys.stderr)
        return 1
    cmd = argv[1]
    o = parse(cmd, argv[2:])
    print(f"SHIM {cmd} {o}", flush=True)
    import pycolmap
    cpu = pycolmap.Device.cpu
    mo = pycolmap.FeatureMatchingOptions()
    mo.num_threads, mo.use_gpu = THREADS, False
    t0 = time.time()
    if cmd == "feature_extractor":
        eo = pycolmap.FeatureExtractionOptions()
        eo.num_threads, eo.use_gpu = THREADS, False
        ro = pycolmap.ImageReaderOptions()
        if "--ImageReader.camera_model" in o:
            ro.camera_model = o["--ImageReader.camera_model"]
        mode = pycolmap.CameraMode.SINGLE if o.get("--ImageReader.single_camera") else pycolmap.CameraMode.AUTO
        pycolmap.extract_features(o["--database_path"], o["--image_path"], camera_mode=mode, reader_options=ro,
                                  extraction_options=eo, device=cpu)
    elif cmd == "sequential_matcher":
        so = pycolmap.SequentialPairingOptions()
        if "--SequentialMatching.overlap" in o:
            so.overlap = o["--SequentialMatching.overlap"]
        if "--SequentialMatching.quadratic_overlap" in o:
            so.quadratic_overlap = o["--SequentialMatching.quadratic_overlap"]
        if "--SequentialMatching.loop_detection" in o:
            so.loop_detection = o["--SequentialMatching.loop_detection"]
        if "--SequentialMatching.vocab_tree_path" in o:
            so.vocab_tree_path = o["--SequentialMatching.vocab_tree_path"]
        pycolmap.match_sequential(o["--database_path"], matching_options=mo, pairing_options=so, device=cpu)
    elif cmd == "spatial_matcher":
        po = pycolmap.SpatialPairingOptions()
        if "--SpatialMatching.max_distance" in o:
            po.max_distance = o["--SpatialMatching.max_distance"]
        if "--SpatialMatching.max_num_neighbors" in o:
            po.max_num_neighbors = o["--SpatialMatching.max_num_neighbors"]
        if "--SpatialMatching.ignore_z" in o:
            po.ignore_z = o["--SpatialMatching.ignore_z"]
        po.check()
        pycolmap.match_spatial(o["--database_path"], matching_options=mo, pairing_options=po, device=cpu)
    elif cmd == "exhaustive_matcher":
        pycolmap.match_exhaustive(o["--database_path"], matching_options=mo, device=cpu)
    elif cmd in ("mapper", "pose_prior_mapper"):
        out = Path(o["--output_path"])
        if not out.is_dir():                     # COLMAP: output_path 는 있는 폴더여야 함
            print(f"output_path 없음: {out}", file=sys.stderr)
            return 1
        mp = pycolmap.IncrementalPipelineOptions()
        mp.num_threads = THREADS
        mp.random_seed = int(os.environ.get("SHIM_SEED", "0"))
        if cmd == "pose_prior_mapper":
            if o.get("--overwrite_priors_covariance"):
                print("shim: overwrite_priors_covariance 1 은 흉내 안 함", file=sys.stderr)
                return 1
            mp.use_prior_position = True
            if "--use_robust_loss_on_prior_position" in o:
                mp.use_robust_loss_on_prior_position = o["--use_robust_loss_on_prior_position"]
            if "--prior_position_loss_scale" in o:
                mp.prior_position_loss_scale = o["--prior_position_loss_scale"]
        recs = pycolmap.incremental_mapping(o["--database_path"], o["--image_path"], out, mp,
                                            input_path=o.get("--input_path", ""))
        print(f"SHIM models {[r.num_reg_images() for r in recs.values()]}", flush=True)
        if not recs:
            print(f"SHIM {cmd} {time.time() - t0:.1f}s", flush=True)
            return 1                              # COLMAP 도 모델이 없으면 실패로 끝남
    elif cmd == "image_undistorter":
        pycolmap.undistort_images(o["--output_path"], o["--input_path"], o["--image_path"],
                                  output_type=o.get("--output_type", "COLMAP"), num_threads=THREADS)
    print(f"SHIM {cmd} {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    rc = main(sys.argv)
    sys.stdout.flush()
    os._exit(rc)        # pycolmap 이 파이썬 종료 때 죽는 메시지를 내는 문제 회피 (RESULTS.md 실행 방법)
