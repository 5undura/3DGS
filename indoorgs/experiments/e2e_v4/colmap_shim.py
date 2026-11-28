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
    # --Mapper.image_list_path: option_manager.cc AddMapperOptions (mapper·pose_prior_mapper 둘 다 부름)
    # --Mapper.ba_use_gpu, --Mapper.fix_existing_frames: option_manager.cc AddMapperOptions
    "mapper": {**DB, "--image_path": str, "--output_path": str, "--input_path": str, "--Mapper.image_list_path": str,
               "--Mapper.ba_use_gpu": _bool, "--Mapper.fix_existing_frames": _bool},
    "pose_prior_mapper": {**DB, "--image_path": str, "--output_path": str, "--input_path": str,
                          "--overwrite_priors_covariance": _bool, "--use_robust_loss_on_prior_position": _bool,
                          "--prior_position_loss_scale": float, "--Mapper.image_list_path": str,
                          "--Mapper.ba_use_gpu": _bool, "--Mapper.fix_existing_frames": _bool},
    # 동선 고정 정합 (register.mode anchored). 출처: exe/feature.cc RunMatchesImporter (AddDatabaseOptions,
    # match_list_path 필수, match_type {pairs,raw,inliers}), exe/sfm.cc RunPointTriangulator (DB, Image, input_path,
    # output_path, clear_points, refine_intrinsics, AddMapperOptions), RunBundleAdjuster (input_path, output_path,
    # AddBundleAdjustmentOptions), exe/image.cc RunImageRegistrator (DB, input_path, output_path, AddMapperOptions;
    # --image_path 없음). shim 은 register.py 가 쓰는 것만
    "matches_importer": {**DB, "--match_list_path": str, "--match_type": str},
    "point_triangulator": {**DB, "--image_path": str, "--input_path": str, "--output_path": str,
                           "--clear_points": _bool, "--refine_intrinsics": _bool, "--Mapper.ba_use_gpu": _bool},
    "bundle_adjuster": {"--input_path": str, "--output_path": str, "--BundleAdjustmentCeres.max_num_iterations": int,
                        "--BundleAdjustmentCeres.use_gpu": _bool},
    "image_registrator": {**DB, "--input_path": str, "--output_path": str},
    # exe/sfm.cc RunViewGraphCalibrator (AddDatabaseOptions + 선택 옵션), RunGlobalMapper (AddDatabaseOptions,
    # AddImageOptions, output_path, AddGlobalMapperOptions). register.py 가 쓰는 것만
    "view_graph_calibrator": dict(DB),
    "global_mapper": {**DB, "--image_path": str, "--output_path": str, "--GlobalMapper.image_list_path": str},
    "image_undistorter": {"--image_path": str, "--input_path": str, "--output_path": str, "--output_type": str},
}
REQUIRED = {
    "feature_extractor": ["--database_path", "--image_path"],
    "sequential_matcher": ["--database_path"], "spatial_matcher": ["--database_path"],
    "exhaustive_matcher": ["--database_path"],
    "mapper": ["--database_path", "--image_path", "--output_path"],
    "pose_prior_mapper": ["--database_path", "--image_path", "--output_path"],
    "image_undistorter": ["--image_path", "--input_path", "--output_path"],
    "view_graph_calibrator": ["--database_path"],
    "global_mapper": ["--database_path", "--image_path", "--output_path"],
    "matches_importer": ["--database_path", "--match_list_path"],
    "point_triangulator": ["--database_path", "--image_path", "--input_path", "--output_path"],
    "bundle_adjuster": ["--input_path", "--output_path"],
    "image_registrator": ["--database_path", "--input_path", "--output_path"],
}


def _read_lines(path: str) -> list[str]:
    """COLMAP util/file.cc ReadTextFileLines 처럼: 앞뒤 공백 자르고 빈 줄 건너뜀."""
    return [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]


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
        if "--Mapper.image_list_path" in o:
            mp.image_names = _read_lines(o["--Mapper.image_list_path"])
            print(f"SHIM image_names {len(mp.image_names)}", flush=True)
        if "--Mapper.ba_use_gpu" in o:
            mp.ba_use_gpu = o["--Mapper.ba_use_gpu"]             # pycolmap CPU 빌드: 경고 후 CPU
        if "--Mapper.fix_existing_frames" in o:
            mp.fix_existing_frames = o["--Mapper.fix_existing_frames"]
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
    elif cmd == "view_graph_calibrator":
        vo = pycolmap.ViewGraphCalibrationOptions()
        vo.random_seed = int(os.environ.get("SHIM_SEED", "0"))
        if not pycolmap.calibrate_view_graph(o["--database_path"], vo):
            print(f"SHIM {cmd} {time.time() - t0:.1f}s", flush=True)
            return 1
    elif cmd == "global_mapper":
        out = Path(o["--output_path"])
        if not out.is_dir():
            print(f"output_path 없음: {out}", file=sys.stderr)
            return 1
        go = pycolmap.GlobalPipelineOptions()
        go.num_threads = THREADS
        go.random_seed = int(os.environ.get("SHIM_SEED", "0"))
        if "--GlobalMapper.image_list_path" in o:
            go.image_names = _read_lines(o["--GlobalMapper.image_list_path"])
        recs = pycolmap.global_mapping(o["--database_path"], o["--image_path"], out, go)
        print(f"SHIM models {[r.num_reg_images() for r in recs.values()]}", flush=True)
        if not recs:
            print(f"SHIM {cmd} {time.time() - t0:.1f}s", flush=True)
            return 1
    elif cmd == "matches_importer":
        if o.get("--match_type", "pairs") != "pairs":
            print("shim: match_type pairs 만 흉내", file=sys.stderr)
            return 1
        po = pycolmap.ImportedPairingOptions()
        po.match_list_path = o["--match_list_path"]
        pycolmap.match_image_pairs(o["--database_path"], matching_options=mo, pairing_options=po, device=cpu)
    elif cmd in ("point_triangulator", "bundle_adjuster", "image_registrator"):
        inp, out = Path(o["--input_path"]), Path(o["--output_path"])
        if not inp.is_dir() or not out.is_dir():         # COLMAP: 둘 다 있는 폴더여야 함
            print(f"input_path/output_path 없음: {inp} {out}", file=sys.stderr)
            return 1
        rec = pycolmap.Reconstruction(str(inp))
        if cmd == "point_triangulator":
            mp = pycolmap.IncrementalPipelineOptions()
            mp.num_threads = THREADS
            if "--Mapper.ba_use_gpu" in o:
                mp.ba_use_gpu = o["--Mapper.ba_use_gpu"]
            rec = pycolmap.triangulate_points(rec, o["--database_path"], o["--image_path"], out,
                                              o.get("--clear_points", True), mp, o.get("--refine_intrinsics", False))
            print(f"SHIM points {rec.num_points3D()}", flush=True)
        elif cmd == "bundle_adjuster":
            bo = pycolmap.BundleAdjustmentOptions()
            if "--BundleAdjustmentCeres.max_num_iterations" in o:
                bo.ceres.solver_options.max_num_iterations = o["--BundleAdjustmentCeres.max_num_iterations"]
            if "--BundleAdjustmentCeres.use_gpu" in o:
                bo.ceres.use_gpu = o["--BundleAdjustmentCeres.use_gpu"]
            bo.ceres.solver_options.num_threads = THREADS
            pycolmap.bundle_adjustment(rec, bo)
            rec.write(str(out))
        else:
            # exe/image.cc RunImageRegistrator 그대로: DatabaseCache(load_all_images) -> 자세 없는 사진마다 RegisterNextImage
            mp = pycolmap.IncrementalPipelineOptions()
            co = pycolmap.DatabaseCacheOptions()
            co.min_num_matches = mp.min_num_matches
            co.ignore_watermarks = mp.ignore_watermarks
            co.load_all_images = True
            cache = pycolmap.DatabaseCache.create(pycolmap.Database.open(o["--database_path"]), co)
            mapper = pycolmap.IncrementalMapper(cache)
            mapper.begin_reconstruction(rec)
            mopt = mp.get_mapper()
            n0 = rec.num_reg_images()
            for iid in sorted(rec.images):
                if not rec.images[iid].has_pose:
                    mapper.register_next_image(mopt, iid)
            mapper.end_reconstruction(False)
            rec.write(str(out))
            print(f"SHIM registered {n0} -> {rec.num_reg_images()}", flush=True)
    elif cmd == "image_undistorter":
        pycolmap.undistort_images(o["--output_path"], o["--input_path"], o["--image_path"],
                                  output_type=o.get("--output_type", "COLMAP"), num_threads=THREADS)
    print(f"SHIM {cmd} {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    rc = main(sys.argv)
    sys.stdout.flush()
    os._exit(rc)        # pycolmap 이 파이썬 종료 때 죽는 메시지를 내는 문제 회피 (RESULTS.md 실행 방법)
