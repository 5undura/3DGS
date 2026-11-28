"""2. 정합 (담당 A). frames/ (+ colmap/priors.json) -> colmap/db.db, colmap/sparse/<n>, data/

저장소 2_register/register_all.ps1 를 옮김:
  feature_extractor(single_camera=1, camera_model=cfg) -> {sequential|exhaustive}_matcher
  -> mapper (priors 있으면 pose_prior_mapper, 실패하면 mapper 로 자동 전환)
  -> 등록 사진이 가장 많은 모델 -> image_undistorter -> data/ (+ data/sparse/0)
복도 1 실험: global_mapper 는 사진 9장이 수천 단위 밖으로 튀어 못 씀 -> 기본 incremental.

위치 제약(priors.json)이 있고 register.pair_verify 이면 "두 번 정합 + 동선 자세로 짝 검증"
(experiments/prior_matching/RESULTS.md 4·5절. 합성 복도에서 255/255장, 틀린 사진 0, 순간이동 0):
  1) feature_extractor -> sequential_matcher(앞뒤 seq_overlap 장, quadratic_overlap 0, loop_detection 0)
  2) write_pose_priors -> pose_prior_mapper -> colmap/sparse_pass1           (pass1.done)
  3) 같은 db.db 에 후보 짝 추가: spatial_matcher (matcher=exhaustive 면 exhaustive_matcher)  (candidates.done)
  4) db.db -> db_verified.db 복사, 사본에서 1차 자세와 안 맞는 짝 삭제 (colmap/pair_verify.py)  (verify.done = 통계 JSON)
     ※ 사본에는 매처를 다시 돌리지 않는다 (지운 짝이 되살아남: 4.2.1 controllers/feature_matching_utils.cc)
  5) pose_prior_mapper --database_path db_verified.db -> colmap/sparse (실패하면 mapper)
  6) 가장 큰 모델을 수평 맞추고 제약에 다시 맞춤 -> colmap/sparse_aligned/0 -> image_undistorter  (mapper.done)
  기준 자세가 사진의 절반 미만이거나 검증이 실패하면 1차 결과를 그대로 씀 (경고). 1차가 실패하면 db.db 로 mapper.
속도 (experiments/e2e_v4/SPEED.md):
  register.pass1_stride N > 1 (기본 3, v4 seed 3개에서 1 보다 접힘이 적음): 2) 의 1차를 N장마다만 (--Mapper.image_list_path colmap/pass1_images.txt).
    4) 에서 빠진 사진의 기준 자세는 같은 1차 모델 안 앞뒤 이웃으로 보간 (회전 slerp, 중심 직선, 순서 간격 2N 이하만).
  register.pass2_mapper global (실험용: v4 에서 seed 따라 크게 접힘): 5) 를 db_verified.db 사본(db_global.db)에 view_graph_calibrator -> global_mapper.
    global_mapper 는 위치 제약을 안 씀 (명령줄 옵션 없음, 4.2.1 sfm/global_mapper.cc 는 끝에 Normalize) -> 좌표가 제멋대로라
    6) 수평 맞추기(gimbal_level) + 제약에 다시 맞춤이 꼭 필요. 실패하면 pose_prior_mapper -> mapper 로.
동선 고정 정합 register.mode: anchored (실험용, experiments/anchored/RESULTS.md, colmap/anchored.py). 위치 제약이 있을 때만:
  1) feature_extractor -> sequential_matcher (위와 같음)                                  (matching.done = "anchored")
  2) 앞뒤 짝만으로 회전 -> 요. 동선 위치 거리 anchored_max_dist_m·요 차이 anchored_max_yaw_deg 안 짝 목록
     -> colmap/anchored_pairs.txt -> matches_importer --match_type pairs                    (candidates.done)
  3) 자세 그래프 (점 없음): 상대 회전 평균 + 짐벌 수평 + 약한 동선 요, 이동 방향 + 동선 위치 -> colmap/anchored/init
     db.db -> db_anchored.db 사본에서 그래프 회전과 안 맞는 짝 삭제 (anchored_verify_rot_deg)
  4) point_triangulator (자세 고정) -> 관측 적은 사진 빼기 -> 마무리 anchored_finish:
     prior_mapper = pose_prior_mapper --input_path (나머지 사진 등록 + 위치 제약 전역 BA, 기본)
     ba = (bundle_adjuster -> point_triangulator) x anchored_ba_rounds -> image_registrator -> point_triangulator
     windowed = 영상 순서 창마다 bundle_adjuster (창 입력 자세에 닮음 맞춤) -> point_triangulator -> image_registrator -> ...
     none = image_registrator -> point_triangulator
     -> colmap/sparse/0 -> 6) 과 같이 수평·제약 맞춤 -> sparse_aligned/0 -> image_undistorter  (mapper.done)

확인한 버전·출처 (2026-11-27)
- COLMAP 4.2.1 (태그 4.2.1, 커밋 bd1fcf6 = 사용자 PC 버전과 같음) 소스 github.com/colmap/colmap
  - src/colmap/exe/colmap.cc: 명령 feature_extractor, sequential_matcher, exhaustive_matcher, spatial_matcher, mapper,
    pose_prior_mapper, image_undistorter, view_graph_calibrator, global_mapper, image_deleter, model_analyzer
  - src/colmap/exe/sfm.cc RunPosePriorMapper: --database_path --image_path --input_path --output_path(필수, 있는 폴더)
    --overwrite_priors_covariance --prior_position_std_x/y/z --use_robust_loss_on_prior_position
    --prior_position_loss_scale. use_prior_position 은 자동으로 켜짐. 끝에 모델을 priors 좌표로 맞춤
    (controllers/incremental_pipeline.cc AlignReconstructionToPosePriors) -> 결과가 도면 좌표
  - controllers/option_manager.cc AddMapperOptions: --Mapper.image_list_path (PostParse 에서 ReadTextFileLines ->
    mapper->image_names; util/file.cc 는 줄 앞뒤 공백 자르고 빈 줄 건너뜀). RunPosePriorMapper 는 AddMapperOptions 를 부름.
    controllers/incremental_pipeline.cc CreateDatabaseCacheOptions: image_names 의 사진만 DB 에서 읽음
  - exe/sfm.cc RunGlobalMapper: --database_path --image_path --output_path(필수, 있는 폴더) + GlobalMapper.*.
    RunViewGraphCalibrator: --database_path (+ 선택 옵션). controllers/global_pipeline.cc: 초점거리 사전값이 없으면
    view_graph_calibrator 를 먼저 돌리라고 경고
  - controllers/option_manager.cc: --ImageReader.camera_model, --ImageReader.single_camera,
    --SequentialMatching.overlap/quadratic_overlap/loop_detection/vocab_tree_path,
    --SpatialMatching.max_distance/max_num_neighbors/ignore_z (2026-11-28, 4.2.1 소스 사본 + pycolmap 4.2.1 바이너리 문자열로 확인)
  - exe/feature.cc RunSpatialMatcher: AddSpatialPairingOptions + --database_path. controllers/pairing.cc
    SpatialPairingOptions::Check: max_num_neighbors > 0 (0 = 제한 없음이 아님)
  - exe/image.cc RunImageUndistorter: --image_path --input_path --output_path --output_type {COLMAP,...}
  - pose_priors 표 스키마: colmap/database.py 주석 (pycolmap 4.2.1 로 실제 DB 만들어 확인)
- 저장소 3_train/PROGRESS.md "도구 확인": 사용자 PC COLMAP 4.2.1 에서 ImageReader.*, view_graph_calibrator,
  global_mapper 있음 확인
아래 상수 중 PC 에서 직접 -h 로 안 본 것은 TODO 표시.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

from .. import schemas
from ..colmap import anchored, database, model as cmodel, pair_verify, runner
from ..colmap.runner import move_aside
from .base import Stage, StageContext, StageResult

# ---- COLMAP 명령·옵션 이름 (4.2.1 소스로 확인) ----
CMD_FEATURES = "feature_extractor"
CMD_MAPPER = "mapper"
CMD_PRIOR_MAPPER = "pose_prior_mapper"          # TODO(PC에서 -h로 확인): 소스로만 확인
CMD_UNDISTORT = "image_undistorter"
CMD_VGC = "view_graph_calibrator"
CMD_GLOBAL = "global_mapper"                    # TODO(PC에서 -h로 확인): 소스로만 확인 (PROGRESS.md 는 명령 있음만 확인)
CMD_SPATIAL = "spatial_matcher"                 # TODO(PC에서 -h로 확인): 소스로만 확인
OPT_SINGLE_CAMERA = "--ImageReader.single_camera"
OPT_CAMERA_MODEL = "--ImageReader.camera_model"
OPT_SEQ_OVERLAP = "--SequentialMatching.overlap"
# 4.2.1 기본값 quadratic_overlap=1 이면 앞뒤 overlap 장을 다 잇지 않고 1, 2, 4, ... 2^(overlap-1) 장 떨어진 사진만 잇는다
# (controllers/pairing.cc SequentialPairGenerator). 64·128·512 장 떨어진 짝이 가짜 짝이 됨 -> 항상 0.
OPT_SEQ_QUADRATIC = "--SequentialMatching.quadratic_overlap"
OPT_SEQ_LOOP = "--SequentialMatching.loop_detection"
# 어휘 트리: 4.2.1 controllers/pairing.cc 는 vocab_tree_path 가 비면 retrieval/resources.h 의 기본 URI
# (github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_faiss_flickr100K_words256K.bin) 를 받아
# ~/.cache/colmap 에 저장해 씀 (util/file.cc). 오프라인이면 실패 -> register.vocab_tree 로 받아 둔 파일 지정,
# 그래도 실패하면 loop_detection 0 으로 한 번 더 (아래 run).
OPT_SEQ_VOCAB = "--SequentialMatching.vocab_tree_path"
OPT_SPA_MAX_DIST = "--SpatialMatching.max_distance"
OPT_SPA_MAX_NN = "--SpatialMatching.max_num_neighbors"
OPT_SPA_IGNORE_Z = "--SpatialMatching.ignore_z"
CMD_MATCHES_IMPORTER = "matches_importer"       # TODO(PC에서 -h로 확인): 4.2.1 exe/feature.cc RunMatchesImporter
OPT_MATCH_LIST = "--match_list_path"            # 필수. 줄마다 '이름1 이름2' (controllers/pairing.cc ReadImagePairsText)
OPT_MATCH_TYPE = "--match_type"                 # {pairs, raw, inliers}. pairs = 짝만 주고 매칭·기하 검증은 COLMAP
CMD_TRIANGULATOR = "point_triangulator"         # TODO(PC): exe/sfm.cc RunPointTriangulator. 자세 고정(fix_existing_frames) + 점만 BA
CMD_BA = "bundle_adjuster"                      # TODO(PC): exe/sfm.cc RunBundleAdjuster (--input_path --output_path + BundleAdjustment*)
CMD_REGISTRATOR = "image_registrator"           # TODO(PC): exe/image.cc RunImageRegistrator (--database_path --input_path --output_path + Mapper.*, --image_path 없음)
OPT_BA_ITERS = "--BundleAdjustmentCeres.max_num_iterations"   # option_manager.cc AddBundleAdjustmentOptions
OPT_BA_GPU = "--BundleAdjustmentCeres.use_gpu"                 # TODO(PC): CUDA 빌드여야 함 (아니면 CPU 로 경고 후 계속, bundle_adjustment_ceres.cc)
OPT_MAPPER_GPU = "--Mapper.ba_use_gpu"                         # option_manager.cc AddMapperOptions
OPT_MAPPER_IMAGE_LIST = "--Mapper.image_list_path"   # TODO(PC에서 pose_prior_mapper -h 로 확인): 4.2.1 소스로만 확인
# priors.json 의 sigma 를 DB 에 넣으므로 공분산을 덮어쓰지 않음 (기본값도 false 이지만 명시).
# exe/sfm.cc: bool overwrite_priors_covariance 를 AddDefaultOption 으로 등록 -> boost program_options 의 bool 값이라
# 0/1/true/false 모두 받음 (다른 bool 옵션 --SiftExtraction.use_gpu 0 과 같은 방식)
PRIOR_MAPPER_EXTRA = ["--overwrite_priors_covariance", "0"]

MATCH_DONE = "matching.done"      # 내용: 매칭 방식 ("plain" | "pair_verify"). 방식이 바뀌면 매칭부터 다시
PASS1_DONE = "pass1.done"         # 내용: 1차에 쓴 priors.json 의 지문 (바뀌면 매칭부터 다시)
CAND_DONE = "candidates.done"
VERIFY_DONE = "verify.done"       # 내용: 짝 검증 통계 JSON
MAPPER_DONE = "mapper.done"
INFO = "register.json"      # 어떤 mapper 를 썼는지 (diagnose 가 좌표가 도면 좌표인지 판단할 때 씀)
DB_VERIFIED = "db_verified.db"
SPARSE_PASS1 = "sparse_pass1"
SPARSE_ALIGNED = "sparse_aligned"
DB_GLOBAL = "db_global.db"
PASS1_LIST = "pass1_images.txt"
ANCH_DIR = "anchored"            # 동선 고정 정합 중간 결과 (init, tri, kept, ...)
ANCH_PAIRS = "anchored_pairs.txt"
DB_ANCHORED = "db_anchored.db"
MIN_REF_FRAC = 0.5          # 기준 자세가 사진의 이 비율 미만이면 검증 안 하고 1차 결과를 씀


def colmap_exe(ctx: StageContext) -> str | None:
    return ctx.project.cfg.tools.colmap


def best_model(sparse: Path) -> tuple[Path | None, int]:
    """등록 사진이 가장 많은 sparse/<n> (model_analyzer 대신 images.bin 을 직접 셈)."""
    best, n_best = None, -1
    if not sparse.is_dir():
        return None, 0
    for d in sorted(p for p in sparse.iterdir() if p.is_dir()):
        try:
            n = cmodel.count_images(d)
        except (FileNotFoundError, OSError, ValueError):
            continue
        if n > n_best:
            best, n_best = d, n
    return best, max(n_best, 0)


def _has_models(sparse: Path) -> bool:
    return best_model(sparse)[0] is not None


def _priors_tag(path: Path) -> str:
    """priors.json 내용의 지문. 1차 정합·후보 짝·짝 검증·2차 정합이 모두 이 제약으로 만들어짐."""
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()



def _cand_tag(rc) -> str:
    """후보 짝 단계 설정 지문 (바뀌면 후보 짝을 다시 찾아야 함)."""
    return f"{rc.matcher}|{rc.verify_max_distance}|{rc.verify_max_neighbors}"


def _verify_settings(rc) -> dict:
    """짝 검증 설정 (verify.done 에 같이 적어 두고, 바뀌면 검증부터 다시).
    pass1_stride 는 1 이 아닐 때만 넣음 (예전 verify.done 과 지문이 같게)."""
    d = {"seq_keep": rc.verify_seq_keep, "rot_tol_deg": rc.verify_rot_tol_deg,
         "dir_tol_deg": rc.verify_dir_tol_deg, "max_far_m": rc.verify_max_far_m,
         "gimbal_level": rc.gimbal_level}
    if rc.pass1_stride != 1:
        d["pass1_stride"] = rc.pass1_stride
    return d


def _anch_cand_tag(rc) -> str:
    """동선 고정 정합의 짝 후보 설정 지문 (바뀌면 db.db 매칭부터 다시: 짝이 이미 DB 에 들어감)."""
    return f"anchored|{rc.seq_overlap}|{rc.anchored_max_dist_m}|{rc.anchored_max_yaw_deg}|{rc.anchored_pairs_per_image}"


def _anch_settings(rc) -> dict:
    """짝 후보 뒤 단계 설정 (register.json 에 적고, 바뀌면 그 뒤만 다시)."""
    return {k: getattr(rc, k) for k in ("anchored_verify_rot_deg", "anchored_min_obs", "anchored_finish",
                                        "anchored_ba_rounds", "anchored_ba_iters", "anchored_window",
                                        "anchored_max_points_per_image", "anchored_focal", "ba_use_gpu",
                                        "gimbal_level")}


def _pass1_tag(path: Path, rc) -> str:
    """pass1.done 내용: priors.json 지문 (+ 1차 설정, 기본이면 안 붙임 -> 예전 pass1.done 과 같음)."""
    t = _priors_tag(path)
    return t if rc.pass1_stride == 1 else f"{t}|stride={rc.pass1_stride}"


def pass1_image_list(names: list[str], stride: int) -> list[str]:
    """1차에 쓸 사진 (이름 순으로 stride 장마다 + 마지막 사진). stride 1 이면 전부."""
    names = sorted(names)
    if stride <= 1 or not names:
        return names
    out = names[::stride]
    if out[-1] != names[-1]:
        out.append(names[-1])
    return out

def _read_priors(path: Path) -> dict | None:
    try:
        d = schemas.read_json(path)
    except Exception:  # noqa: BLE001
        return None
    return d if d.get("frames") else None


class Register(Stage):
    name = "register"

    def outputs(self, ctx: StageContext) -> list[Path]:
        return [ctx.paths.data_model]

    def prev_targets(self, ctx: StageContext) -> list[Path]:
        # colmap/ 통째로 옮기면 priors.json 까지 빠지므로 register 결과만. 완료 표시 파일은 db 가 없으면 run 이 지움
        p = ctx.paths
        c = p.colmap
        return [p.db, p.sparse, p.data, c / INFO, c / SPARSE_PASS1, c / DB_VERIFIED, c / SPARSE_ALIGNED,
                c / DB_GLOBAL, c / PASS1_LIST, c / ANCH_DIR, c / ANCH_PAIRS, c / DB_ANCHORED]

    def verify_mode(self, ctx: StageContext) -> bool:
        rc, p = ctx.project.cfg.register, ctx.paths
        return bool(rc.pair_verify and rc.use_floorplan_prior and p.priors.exists() and _read_priors(p.priors))

    def anchored_mode(self, ctx: StageContext) -> bool:
        rc, p = ctx.project.cfg.register, ctx.paths
        return bool(rc.mode == "anchored" and rc.use_floorplan_prior and p.priors.exists() and _read_priors(p.priors))

    def run(self, ctx: StageContext) -> StageResult:
        p = ctx.paths
        exe = colmap_exe(ctx)
        if not exe or not Path(exe).exists():
            return StageResult(False, f"COLMAP 없음: {exe} (project.yaml tools.colmap)")
        if not p.frames_done.exists():
            return StageResult(False, "frames/done.json 없음 (extract 먼저)")
        frames = sorted(p.frames.glob("f_*.jpg"))
        if not frames:
            return StageResult(False, "frames/ 에 사진 없음")
        rc = ctx.project.cfg.register
        col, db, sparse, data = p.colmap, p.db, p.sparse, p.data
        col.mkdir(parents=True, exist_ok=True)
        tool_log = ctx.project.logs / f"{ctx.section.name}_register_colmap.log"
        anch = self.anchored_mode(ctx)
        if rc.mode == "anchored" and not anch:
            ctx.log("경고: register.mode anchored 는 위치 제약(priors.json)이 있어야 함 -> 기본 흐름으로")
        verify = self.verify_mode(ctx) and not anch
        mode = "anchored" if anch else "pair_verify" if verify else "plain"

        def colmap(*args) -> runner.ToolResult:
            ctx.log(f"COLMAP {args[0]}")
            return runner.run([exe, *args], log_file=tool_log)

        mk = {n: col / n for n in (MATCH_DONE, PASS1_DONE, CAND_DONE, VERIFY_DONE, MAPPER_DONE)}
        match_done, mapper_done = mk[MATCH_DONE], mk[MAPPER_DONE]
        pass1_dir, dbv, aligned_dir = col / SPARSE_PASS1, col / DB_VERIFIED, col / SPARSE_ALIGNED
        # --rerun 으로 다시 돌리는 중이면 이어서 하지 않음 (앞 단계가 다시 돌아 사진·제약이 바뀌었을 수 있음.
        # 끝난 결과는 run_all 이 _prev_ 로 옮겼고, 끝나지 않은 중간 결과는 아래에서 옆으로 치움)
        if ctx.rerun and any(f.exists() for f in mk.values()):
            ctx.log("--rerun: 끝나지 않은 이전 실행은 이어서 하지 않고 처음부터")
            for f in mk.values():
                f.unlink(missing_ok=True)
        # 1차 정합 뒤에 위치 제약(priors.json)이 바뀌었으면 1차 자세·후보 짝·DB 의 제약이 모두 낡음 -> 매칭부터 다시
        if verify and mk[PASS1_DONE].exists():
            tag = mk[PASS1_DONE].read_text(encoding="utf-8").strip()
            if tag and tag.split("|")[0] != _priors_tag(p.priors):
                ctx.log("위치 제약(priors.json)이 1차 정합 뒤에 바뀜 -> 매칭부터 다시")
                match_done.unlink(missing_ok=True)
            elif tag and tag != _pass1_tag(p.priors, rc) and not mapper_done.exists():
                # 1차는 앞뒤 짝만 있는 db.db 로 돌아야 함. 후보 짝을 이미 넣었으면 매칭부터 다시
                if mk[CAND_DONE].exists():
                    ctx.log(f"1차 설정(pass1_stride {rc.pass1_stride})이 바뀜 -> 후보 짝이 이미 들어가 매칭부터 다시")
                    match_done.unlink(missing_ok=True)
                else:
                    ctx.log(f"1차 설정(pass1_stride {rc.pass1_stride})이 바뀜 -> 1차부터 다시")
                    mk[PASS1_DONE].unlink()
        if anch and mk[CAND_DONE].exists() and mk[CAND_DONE].read_text(encoding="utf-8").strip() != _anch_cand_tag(rc):
            ctx.log("동선 짝 후보 설정이 바뀜 -> 매칭부터 다시 (짝이 이미 db.db 에 들어감)")
            match_done.unlink(missing_ok=True)
        # 후보 짝·짝 검증 설정이 바뀌었으면 그 단계부터 다시 (끝난 register 는 mapper.done 으로 건너뜀)
        if verify and not mapper_done.exists():
            if mk[CAND_DONE].exists() and mk[CAND_DONE].read_text(encoding="utf-8").strip() not in (
                    "", _cand_tag(rc)):
                ctx.log("후보 짝 설정이 바뀜 -> 후보 짝부터 다시 (db.db 는 매칭부터 다시)")
                match_done.unlink(missing_ok=True)
            if mk[VERIFY_DONE].exists():
                try:
                    old_set = json.loads(mk[VERIFY_DONE].read_text(encoding="utf-8")).get("settings")
                except ValueError:
                    old_set = None
                if old_set is not None and old_set != _verify_settings(rc):
                    ctx.log("짝 검증 설정이 바뀜 -> 짝 검증부터 다시")
                    mk[VERIFY_DONE].unlink()
        # 완료 표시 파일 정리: 앞 단계가 안 끝났으면 뒤 단계 표시도 무효
        if match_done.exists() and match_done.read_text(encoding="utf-8").strip() not in (
                (mode,) if (verify or anch) else ("", mode)):
            ctx.log(f"매칭 방식이 바뀜 (-> {mode}) -> 매칭부터 다시")
            match_done.unlink()
        if not db.exists():
            match_done.unlink(missing_ok=True)
        if not match_done.exists():
            for f in mk.values():
                f.unlink(missing_ok=True)
        # pair_verify 중간 단계: 앞 단계가 안 끝났으면 뒤 단계 표시도 무효
        steps = [MATCH_DONE, CAND_DONE] if anch else [MATCH_DONE, PASS1_DONE, CAND_DONE, VERIFY_DONE]
        for a, b in zip(steps, steps[1:]):
            if not mk[a].exists():
                mk[b].unlink(missing_ok=True)
        # mapper.done 은 register.json 이 같은 방식으로 끝났다고 할 때만 (1차 실패로 mapper 만 돈 경우도 끝난 것)
        info_path = col / INFO
        if mapper_done.exists() and not (info_path.exists()
                                         and schemas.read_json(info_path).get("mode", "plain") == mode):
            mapper_done.unlink()
        # 2차 정합 방식이 바뀌었으면 2차부터 다시 (1차·후보 짝·검증은 그대로)
        if verify and mapper_done.exists() and \
                schemas.read_json(info_path).get("pass2_mapper", "incremental") != rc.pass2_mapper:
            ctx.log(f"2차 정합 방식이 바뀜 (-> {rc.pass2_mapper}) -> 2차 정합부터 다시")
            mapper_done.unlink()
        if anch and mapper_done.exists() and schemas.read_json(info_path).get("anchored_settings") != _anch_settings(rc):
            ctx.log("동선 고정 정합 설정이 바뀜 -> 자세 그래프부터 다시 (매칭은 그대로)")
            mapper_done.unlink()
        # 표시 파일 없이 남은 결과 = 끊긴 실행 -> 옆으로 치움 (삭제 안 함)
        if db.exists() and not match_done.exists():
            ctx.log(f"끝나지 않은 db.db -> {move_aside(db).name}")
        if verify and pass1_dir.exists() and not mk[PASS1_DONE].exists():
            ctx.log(f"끝나지 않은 {SPARSE_PASS1} -> {move_aside(pass1_dir).name}")
        if verify and dbv.exists() and not mk[VERIFY_DONE].exists():
            ctx.log(f"끝나지 않은 {DB_VERIFIED} -> {move_aside(dbv).name}")
        if verify and (col / DB_GLOBAL).exists() and not mapper_done.exists():
            ctx.log(f"끝나지 않은 {DB_GLOBAL} -> {move_aside(col / DB_GLOBAL).name}")
        for nm in (ANCH_DIR, DB_ANCHORED):
            if anch and (col / nm).exists() and not mapper_done.exists():
                ctx.log(f"끝나지 않은 {nm} -> {move_aside(col / nm).name}")
        if sparse.exists() and not mapper_done.exists():
            ctx.log(f"끝나지 않은 sparse -> {move_aside(sparse).name}")
        if aligned_dir.exists() and not mapper_done.exists():
            ctx.log(f"끝나지 않은 {SPARSE_ALIGNED} -> {move_aside(aligned_dir).name}")
        if data.exists():
            ctx.log(f"끝나지 않은 data -> {move_aside(data).name}")

        def fail(msg: str) -> StageResult:
            return StageResult(False, f"{msg} (COLMAP 출력: {tool_log})")

        # 1) 특징점 + 매칭 (pair_verify 면 1차용 sequential 만)
        if not match_done.exists():
            r = colmap(CMD_FEATURES, "--database_path", db, "--image_path", p.frames,
                       OPT_SINGLE_CAMERA, "1", OPT_CAMERA_MODEL, rc.camera_model)
            if not r.ok:
                move_aside(db)
                return fail(f"{CMD_FEATURES} 실패 (코드 {r.code})")
            matcher = "sequential" if (verify or anch) else rc.matcher
            loop = matcher == "sequential" and rc.loop_detection and not (verify or anch)

            def seq_args(lp: bool) -> list:
                # 옵션 이름: COLMAP 4.2.1 controllers/option_manager.cc
                a = [OPT_SEQ_OVERLAP, str(rc.seq_overlap), OPT_SEQ_QUADRATIC, "0", OPT_SEQ_LOOP, "1" if lp else "0"]
                if lp and rc.vocab_tree:
                    a += [OPT_SEQ_VOCAB, rc.vocab_tree]
                return a
            margs = ["--database_path", db] + (seq_args(loop) if matcher == "sequential" else [])
            r = colmap(f"{matcher}_matcher", *margs)
            if not r.ok and loop:
                # 어휘 트리 다운로드 실패(오프라인) 등 -> 루프 검출 없이 다시. 매칭은 DB 에 이어 쓰므로 그대로 다시 돌림
                ctx.log(f"경고: loop_detection 매칭 실패 (코드 {r.code}) -> loop_detection 0 으로 다시 "
                        "(어휘 트리를 받아 두고 project.yaml register.vocab_tree 에 경로를 적으면 루프 검출 가능)")
                r = colmap(f"{matcher}_matcher", "--database_path", db, *seq_args(False))
            if not r.ok:
                move_aside(db)
                return fail(f"{matcher}_matcher 실패 (코드 {r.code})")
            match_done.write_text(mode, encoding="utf-8")
            ctx.log("매칭 끝")

        # 2) 정합
        info: dict = {}
        if mapper_done.exists():
            info = schemas.read_json(col / INFO) if (col / INFO).exists() else {}
        else:
            if anch:
                res = self._map_anchored(ctx, colmap, mk, frames)
                if isinstance(res, StageResult):
                    return fail(res.message)
                info = res
            elif verify:
                res = self._map_verified(ctx, colmap, mk, frames)
                if isinstance(res, StageResult):
                    return fail(res.message)
                info = res
            else:
                sparse.mkdir(parents=True, exist_ok=True)
                used = self._map(ctx, exe, colmap, db, sparse)
                if used is None:
                    old = move_aside(sparse)
                    return fail(f"정합 실패: 모델 없음 (중간 결과 {old.name if old else '-'})")
                info = {"mapper": used, "pair_verify": None, "aligned": False}
            info = {"version": schemas.VERSION, **info, "matcher": rc.matcher, "camera_model": rc.camera_model,
                    "mode": mode}
            mapper_done.touch()
            schemas.write_json(col / INFO, info)
        used = info.get("mapper")

        # 3) 사용할 모델 -> undistort
        if info.get("model"):
            best = col / info["model"]
            try:
                n = cmodel.count_images(best)
            except (FileNotFoundError, OSError, ValueError):
                return fail(f"{info['model']} 에 모델 없음")
        else:
            best, n = best_model(sparse)
        if best is None:
            return fail("sparse 에 모델 없음")
        ctx.log(f"사용할 모델: {best.relative_to(col)} ({n} / {len(frames)} 장), 정합 {used}"
                + (", 제약에 직접 맞춤" if info.get("aligned") else ""))
        r = colmap(CMD_UNDISTORT, "--image_path", p.frames, "--input_path", best,
                   "--output_path", data, "--output_type", "COLMAP")
        dsp = data / "sparse"
        if not r.ok or not any((dsp / f"images.{e}").exists() for e in ("bin", "txt")):
            old = move_aside(data)
            return fail(f"{CMD_UNDISTORT} 실패 (코드 {r.code}, 중간 결과 {old.name if old else '-'})")
        # image_undistorter 는 모델을 data/sparse/ 바로 아래에 씀 -> 학습기가 찾는 data/sparse/0/ 으로
        tmp = data / "sparse_0_tmp"
        tmp.mkdir()
        for f in [f for f in dsp.iterdir() if f.is_file()]:
            shutil.move(str(f), str(tmp / f.name))
        tmp.rename(p.data_model)
        pv = info.get("pair_verify") or {}
        extra = f", 짝 검증 {pv['dropped']}개 삭제" if "dropped" in pv else ""
        return StageResult(True, f"{n}/{len(frames)} 장 등록 ({used}{extra})", [p.data_model])

    # ---------------------------------------------------------------- 두 번 정합 + 짝 검증
    def _map_verified(self, ctx, colmap, mk: dict, frames: list) -> dict | StageResult:
        """pair_verify 흐름의 2)~6). 돌려줌: register.json 에 넣을 dict, 실패하면 StageResult(False)."""
        p, rc = ctx.paths, ctx.project.cfg.register
        col, db, sparse = p.colmap, p.db, p.sparse
        pass1_dir, dbv = col / SPARSE_PASS1, col / DB_VERIFIED
        priors = _read_priors(p.priors)

        # 2) 1차: 앞뒤 짝만 + 위치 제약
        if not mk[PASS1_DONE].exists():
            try:
                n, missing = database.write_pose_priors(db, priors)
                ctx.log(f"위치 제약 {n}장 DB 에 씀" + (f", DB 에 없는 사진 {len(missing)}장" if missing else ""))
            except Exception as e:  # noqa: BLE001
                n = 0
                ctx.log(f"경고: 위치 제약 쓰기 실패 {e}")
            ok = False
            if n:
                pass1_dir.mkdir(parents=True, exist_ok=True)
                extra = []
                if rc.pass1_stride > 1:
                    sub = pass1_image_list([f.name for f in frames], rc.pass1_stride)
                    (col / PASS1_LIST).write_text("\n".join(sub) + "\n", encoding="utf-8")
                    extra = [OPT_MAPPER_IMAGE_LIST, col / PASS1_LIST]
                    ctx.log(f"1차는 {rc.pass1_stride}장마다: {len(sub)}/{len(frames)}장 ({PASS1_LIST})")
                r = colmap(CMD_PRIOR_MAPPER, "--database_path", db, "--image_path", p.frames,
                           "--output_path", pass1_dir, *PRIOR_MAPPER_EXTRA, *extra)
                ok = r.ok and _has_models(pass1_dir)
            if not ok:
                old = move_aside(pass1_dir)
                ctx.log(f"경고: 1차 정합 실패 -> 짝 검증 없이 db.db 로 {CMD_MAPPER}. 중간 결과 {old.name if old else '-'}")
                sparse.mkdir(parents=True, exist_ok=True)
                r = colmap(CMD_MAPPER, "--database_path", db, "--image_path", p.frames, "--output_path", sparse)
                if not (r.ok and _has_models(sparse)):
                    old = move_aside(sparse)
                    return StageResult(False, f"정합 실패: 모델 없음 (중간 결과 {old.name if old else '-'})")
                return {"mapper": CMD_MAPPER, "pair_verify": {"skipped": "1차 정합 실패"}, "aligned": False,
                        "pass1_stride": rc.pass1_stride, "pass2_mapper": rc.pass2_mapper}
            mk[PASS1_DONE].write_text(_pass1_tag(p.priors, rc), encoding="utf-8")
            ctx.log(f"1차 정합 끝: {SPARSE_PASS1} 가장 큰 모델 {best_model(pass1_dir)[1]}장")

        # 3) 후보 짝 추가 (같은 db.db. 이미 매칭한 짝은 COLMAP 이 건너뜀)
        if not mk[CAND_DONE].exists():
            if rc.matcher == "exhaustive":
                r = colmap("exhaustive_matcher", "--database_path", db)
            else:
                r = colmap(CMD_SPATIAL, "--database_path", db, OPT_SPA_MAX_DIST, str(rc.verify_max_distance),
                           OPT_SPA_MAX_NN, str(rc.verify_max_neighbors), OPT_SPA_IGNORE_Z, "1")
            if not r.ok:
                return StageResult(False, f"후보 짝 매칭 실패 (코드 {r.code}). 다시 실행하면 이어서 함")
            mk[CAND_DONE].write_text(_cand_tag(rc), encoding="utf-8")

        # 4) 사본에서 짝 검증
        if not mk[VERIFY_DONE].exists():
            stats = self._verify(ctx, priors, [f.name for f in frames])
            stats["settings"] = _verify_settings(rc)
            mk[VERIFY_DONE].write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")
        try:
            stats = json.loads(mk[VERIFY_DONE].read_text(encoding="utf-8"))
        except ValueError:
            stats = {"skipped": "verify.done 을 못 읽음"}

        # 5) 2차 정합 (검증한 사본으로. 1차 결과를 그대로 쓰는 경우는 건너뜀)
        if stats.get("skipped"):
            src, used = pass1_dir, CMD_PRIOR_MAPPER
        else:
            src, used = sparse, None
            sparse.mkdir(parents=True, exist_ok=True)
            if rc.pass2_mapper == "global":
                # view_graph_calibrator 가 DB 를 고치므로 사본에서 (db_verified.db 는 실패 시 incremental 이 그대로 씀)
                dbg = col / DB_GLOBAL
                if dbg.exists():
                    move_aside(dbg)
                shutil.copy2(dbv, dbg)
                r = colmap(CMD_VGC, "--database_path", dbg)
                if r.ok:
                    r = colmap(CMD_GLOBAL, "--database_path", dbg, "--image_path", p.frames, "--output_path", sparse)
                if r.ok and _has_models(sparse):
                    used = CMD_GLOBAL
                else:
                    old = move_aside(sparse, "global_failed")
                    sparse.mkdir(parents=True, exist_ok=True)
                    ctx.log(f"경고: 2차 {CMD_GLOBAL} 실패 (코드 {r.code}) -> {CMD_PRIOR_MAPPER} 로 다시. "
                            f"중간 결과 {old.name if old else '-'}")
            r = None
            if used is None:
                r = colmap(CMD_PRIOR_MAPPER, "--database_path", dbv, "--image_path", p.frames,
                           "--output_path", sparse, *PRIOR_MAPPER_EXTRA)
                used = CMD_PRIOR_MAPPER
            if r is not None and not (r.ok and _has_models(sparse)):
                old = move_aside(sparse, "prior_failed")
                sparse.mkdir(parents=True, exist_ok=True)
                ctx.log(f"경고: 2차 {CMD_PRIOR_MAPPER} 실패 (코드 {r.code}) -> {CMD_MAPPER} 로 다시. "
                        f"중간 결과 {old.name if old else '-'}")
                r = colmap(CMD_MAPPER, "--database_path", dbv, "--image_path", p.frames, "--output_path", sparse)
                used = CMD_MAPPER
                if not (r.ok and _has_models(sparse)):
                    old = move_aside(sparse)
                    return StageResult(False, f"2차 정합 실패: 모델 없음 (중간 결과 {old.name if old else '-'})")

        # 6) 가장 큰 모델을 수평 맞추고 제약에 다시 맞춤
        best, n = best_model(src)
        out = {"mapper": used, "pair_verify": stats, "aligned": False, "model": str(best.relative_to(col).as_posix()),
               "pass1_stride": rc.pass1_stride, "pass2_mapper": rc.pass2_mapper}
        return self._align(ctx, best, priors, used, out)

    def _align(self, ctx, best: Path, priors: dict, used: str, out: dict) -> dict:
        """best 모델을 수평 맞추고 제약에 다시 맞춰 sparse_aligned/0 에 씀. out 에 결과를 더해 돌려줌 (실패하면 그대로)."""
        p, rc = ctx.paths, ctx.project.cfg.register
        col = p.colmap
        try:
            if used != CMD_PRIOR_MAPPER and not rc.gimbal_level:
                # 일반 mapper·global_mapper 결과는 좌표가 제멋대로라 위가 어디인지 모름. 수평 맞추기 없이 수직축 회전만으로는 못 맞춤
                raise ValueError(f"{used} 결과인데 gimbal_level 이 꺼져 있어 위 방향을 모름")
            new, ast = pair_verify.align_model_to_priors(cmodel.read_model(best), priors, level=rc.gimbal_level)
            tgt = col / SPARSE_ALIGNED / "0"
            tmp = col / SPARSE_ALIGNED / "0_tmp"
            cmodel.write_model(new, tmp)
            tmp.rename(tgt)
            out.update({"aligned": True, "align": ast, "model": f"{SPARSE_ALIGNED}/0"})
            ctx.log(f"제약에 직접 맞춤: 수평 {ast['leveled_deg']}°, 축척 {ast['scale']:.3f}, "
                    f"오차 중앙 {ast['resid_median_m']:.2f} m, 1 m 넘는 사진 {ast['far_1m']}")
        except Exception as e:  # noqa: BLE001
            old = move_aside(col / SPARSE_ALIGNED)
            ctx.log(f"경고: 제약에 직접 맞추기 실패 {e} -> {best.relative_to(col)} 그대로 씀"
                    + (f" (중간 결과 {old.name})" if old else ""))
        return out

    # ---------------------------------------------------------------- 동선 고정 정합 (실험용)
    def _map_anchored(self, ctx, colmap, mk: dict, frames: list) -> dict | StageResult:
        """register.mode anchored 의 2)~4) + 수평·제약 맞춤. 돌려줌: register.json 에 넣을 dict, 실패하면 StageResult(False)."""
        p, rc = ctx.paths, ctx.project.cfg.register
        col, db, sparse = p.colmap, p.db, p.sparse
        priors = _read_priors(p.priors)
        names = [f.name for f in frames]
        prm = anchored.GraphParams(focal=rc.anchored_focal)
        secs: dict[str, float] = {}
        gpu_m = [OPT_MAPPER_GPU, "1"] if rc.ba_use_gpu else []
        gpu_b = [OPT_BA_GPU, "1"] if rc.ba_use_gpu else []
        try:
            n_pri, _ = database.write_pose_priors(db, priors)      # pose_prior_mapper 마무리용 (다른 마무리는 안 씀)
        except Exception as e:  # noqa: BLE001
            n_pri = 0
            ctx.log(f"경고: 위치 제약 쓰기 실패 {e}")

        # 2) 동선 근처 짝 후보 -> matches_importer
        pst: dict = {}
        if not mk[CAND_DONE].exists():
            t0 = time.time()
            order, P, _, _ = anchored.route_order(priors, names)
            try:
                g0 = anchored.solve_graph(db, priors, names, prm)
                yaw, src = anchored.graph_yaw(g0), "graph"
            except Exception as e:  # noqa: BLE001
                ctx.log(f"경고: 앞뒤 짝만으로 회전 못 구함 {e} -> 동선 진행 방향을 요로")
                yaw, src = anchored.route_headings(P, anchored.route_order(priors, names)[2])["yaw"], "route"
            pairs = [(i, j) for i, j in anchored.select_pairs(P, yaw, seq_window=rc.seq_overlap,
                                                                 max_dist_m=rc.anchored_max_dist_m,
                                                                 max_yaw_deg=rc.anchored_max_yaw_deg,
                                                                 max_per_image=rc.anchored_pairs_per_image)
                     if j - i > rc.seq_overlap]                 # 앞뒤 seq_overlap 장은 sequential 이 이미 매칭
            anchored.write_pairs_file(col / ANCH_PAIRS, order, pairs)
            secs["pairs_select"] = round(time.time() - t0, 2)
            pst = {"pairs": len(pairs), "yaw_source": src}
            ctx.log(f"동선 짝 후보 {len(pairs)}개 ({ANCH_PAIRS}, 요 {src})")
            if pairs:
                r = colmap(CMD_MATCHES_IMPORTER, "--database_path", db, OPT_MATCH_LIST, col / ANCH_PAIRS,
                           OPT_MATCH_TYPE, "pairs")
                if not r.ok:
                    return StageResult(False, f"{CMD_MATCHES_IMPORTER} 실패 (코드 {r.code}). 다시 실행하면 이어서 함")
            mk[CAND_DONE].write_text(_anch_cand_tag(rc), encoding="utf-8")

        # 3) 자세 그래프 -> 초기 모델, 짝 정리
        work = col / ANCH_DIR
        if work.exists():
            move_aside(work)
        work.mkdir(parents=True)
        t0 = time.time()
        try:
            g = anchored.solve_graph(db, priors, names, prm)
        except Exception as e:  # noqa: BLE001
            return StageResult(False, f"동선 고정 정합: 자세 그래프 실패 {e}")
        secs["graph"] = round(time.time() - t0, 2)
        gst = g["stats"]
        ctx.log(f"자세 그래프: {gst['used']}/{gst['frames']}장 (연결 {gst['connected']}, 동선에서 먼 것 {gst['dropped_far']}), "
                f"초점 {gst['focal']}, 상대 회전 잔차 중앙 {gst['rotation'].get('rel_res_median_deg')}°, "
                f"동선 표시(멈춤·급회전) {gst['route_flagged']}장")
        dbx = col / DB_ANCHORED
        if dbx.exists():
            move_aside(dbx)
        shutil.copy2(db, dbx)
        vst = None
        if rc.anchored_verify_rot_deg > 0:
            t0 = time.time()
            vst = pair_verify.verify_pairs(dbx, anchored.reference_from_graph(g), seq_keep=prm.seq_graph,
                                           rot_tol_deg=rc.anchored_verify_rot_deg, dir_tol_deg=180.0,
                                           drop_unknown=False)
            secs["pair_check"] = round(time.time() - t0, 2)
            ctx.log(f"짝 정리: {vst['pairs']}개 중 {vst['dropped']}개 삭제 (회전 {vst['drop_rot']}, 자세 못 구함 {vst['drop_no_pose']})")
        init = anchored.initial_model(g["dbd"], g["order"], g["R"], g["C"], g["use"], g["focal"])
        if len(init.images) < 2:
            return StageResult(False, f"동선 고정 정합: 초기 자세 {len(init.images)}장")
        cmodel.write_model(init, work / "init")

        def tri(src: Path, dst: Path) -> bool:
            dst.mkdir(parents=True, exist_ok=True)
            r = colmap(CMD_TRIANGULATOR, "--database_path", dbx, "--image_path", p.frames, "--input_path", src,
                       "--output_path", dst, *gpu_m)
            return r.ok and (dst / "images.bin").exists()

        def ba(src: Path, dst: Path) -> bool:
            dst.mkdir(parents=True, exist_ok=True)
            r = colmap(CMD_BA, "--input_path", src, "--output_path", dst, OPT_BA_ITERS, str(rc.anchored_ba_iters), *gpu_b)
            return r.ok and (dst / "images.bin").exists()

        def register_rest(src: Path, dst: Path) -> bool:
            reg = work / "registered"
            reg.mkdir(parents=True, exist_ok=True)
            r = colmap(CMD_REGISTRATOR, "--database_path", dbx, "--input_path", src, "--output_path", reg)
            return r.ok and (reg / "images.bin").exists() and tri(reg, dst)

        if not tri(work / "init", work / "tri"):
            return StageResult(False, f"{CMD_TRIANGULATOR} 실패 (초기 자세 {len(init.images)}장)")
        m = cmodel.read_model(work / "tri")
        weak = anchored.weak_images(m, rc.anchored_min_obs)
        m, _ = cmodel.remove_images(m, weak)
        thin = None
        if rc.anchored_max_points_per_image > 0:
            m, thin = anchored.thin_points(m, rc.anchored_max_points_per_image)
        cmodel.write_model(m, work / "kept")
        ctx.log(f"자세 고정 삼각측량: 점 {len(m.points)}개, 관측 {rc.anchored_min_obs} 미만 사진 {len(weak)}장 뺌"
                + (f", 점 줄임 {thin}" if thin else ""))

        # 4) 마무리
        fin, out0 = rc.anchored_finish, sparse / "0"
        sparse.mkdir(parents=True, exist_ok=True)
        ok, wst = False, None
        if fin == "prior_mapper":
            if not n_pri:
                return StageResult(False, "동선 고정 정합: 위치 제약을 DB 에 못 써서 pose_prior_mapper 마무리 불가")
            out0.mkdir(parents=True, exist_ok=True)
            r = colmap(CMD_PRIOR_MAPPER, "--database_path", dbx, "--image_path", p.frames, "--input_path", work / "kept",
                       "--output_path", out0, *PRIOR_MAPPER_EXTRA, *gpu_m)
            ok = r.ok and (out0 / "images.bin").exists()
        else:
            cur = work / "kept"
            if fin == "ba":
                for k in range(max(1, rc.anchored_ba_rounds)):
                    if not ba(cur, work / f"ba{k}") or not tri(work / f"ba{k}", work / f"ba{k}_tri"):
                        return StageResult(False, f"{CMD_BA}/{CMD_TRIANGULATOR} 실패 ({k + 1}번째)")
                    cur = work / f"ba{k}_tri"
            elif fin == "windowed":
                base = cmodel.read_model(cur)
                ordn = sorted(im.name for im in base.images.values())
                res = []
                for w, (win, core) in enumerate(anchored.windows(ordn, rc.anchored_window)):
                    wd = work / "win" / f"{w:03d}"
                    cmodel.write_model(anchored.submodel(base, set(win)), wd / "in")
                    if not ba(wd / "in", wd / "out"):
                        return StageResult(False, f"{CMD_BA} 실패 (창 {w})")
                    res.append((core, cmodel.read_model(wd / "out")))
                merged, wst = anchored.merge_windows(base, res)
                cmodel.write_model(merged, work / "win_merged")
                if not tri(work / "win_merged", work / "win_tri"):
                    return StageResult(False, f"{CMD_TRIANGULATOR} 실패 (창 BA 뒤)")
                cur = work / "win_tri"
            ok = register_rest(cur, out0)
        if not ok:
            old = move_aside(sparse)
            return StageResult(False, f"동선 고정 정합 마무리({fin}) 실패 (중간 결과 {old.name if old else '-'})")
        used = CMD_PRIOR_MAPPER if fin == "prior_mapper" else "anchored"
        st = {"graph": gst, "pairs": pst or None, "pair_check": vst, "weak_removed": len(weak), "thin": thin,
              "windows": wst, "python_seconds": secs}
        schemas.write_json(work / "anchored.json", st)
        out = {"mapper": used, "pair_verify": None, "aligned": False, "model": f"{p.sparse.name}/0",
               "anchored": st, "anchored_finish": fin, "anchored_settings": _anch_settings(rc)}
        return self._align(ctx, out0, priors, used, out)

    def _verify(self, ctx, priors: dict, names: list[str]) -> dict:
        """load_reference + db_verified.db 복사 + verify_pairs. 못 하면 {"skipped": 이유, ...}.
        names: 전체 사진 이름 (시간 순). pass1_stride > 1 이면 1차에 빠진 사진을 보간 (간격 2 x stride 이하)."""
        p, rc = ctx.paths, ctx.project.cfg.register
        col, db = p.colmap, p.db
        dbv = col / DB_VERIFIED
        n_frames = len(names)
        kw = {}
        if rc.pass1_stride > 1:
            kw = {"order": sorted(names), "interp_max_gap": 2 * rc.pass1_stride}
        try:
            ref = pair_verify.load_reference(col / SPARSE_PASS1, priors, max_far_m=rc.verify_max_far_m,
                                             level=rc.gimbal_level, **kw)
        except Exception as e:  # noqa: BLE001
            ctx.log(f"경고: 1차 기준 자세 읽기 실패 {e} -> 1차 결과를 그대로 씀")
            return {"skipped": f"기준 자세 실패: {e}"}
        ri = {k: ref.info[k] for k in ("models", "poses", "far_dropped", "unfit_models", "leveled_deg",
                                       "interpolated", "interp_far_dropped") if k in ref.info}
        if len(ref.poses) < MIN_REF_FRAC * n_frames:
            ctx.log(f"경고: 1차 기준 자세 {len(ref.poses)}/{n_frames}장 (절반 미만) -> 짝 검증 안 하고 1차 결과를 그대로 씀")
            return {"skipped": "기준 자세가 절반 미만", "reference": ri}
        shutil.copy2(db, dbv)
        try:
            st = pair_verify.verify_pairs(dbv, ref, seq_keep=rc.verify_seq_keep, rot_tol_deg=rc.verify_rot_tol_deg,
                                          dir_tol_deg=rc.verify_dir_tol_deg)
        except Exception as e:  # noqa: BLE001
            old = move_aside(dbv)
            ctx.log(f"경고: 짝 검증 실패 {e} -> 1차 결과를 그대로 씀 (사본 {old.name if old else '-'})")
            return {"skipped": f"짝 검증 실패: {e}", "reference": ri}
        st["reference"] = ri
        ctx.log(f"짝 검증: 짝 {st['pairs']}개 중 {st['dropped']}개 삭제 (회전 {st['drop_rot']}, 방향 {st['drop_dir']}, "
                f"기준 없음 {st['drop_unknown']}, 자세 못 구함 {st['drop_no_pose']}), 기준 자세 {len(ref.poses)}장"
                + (f" (보간 {ref.info['interpolated']}장)" if ref.info.get("interpolated") else ""))
        return st

    # ---------------------------------------------------------------- 한 번 정합 (예전 흐름)
    def _map(self, ctx, exe, colmap, db: Path, sparse: Path) -> str | None:
        """정합을 돌리고 쓴 방식 이름을 돌려줌. 실패하면 None."""
        p, rc = ctx.paths, ctx.project.cfg.register
        if p.priors.exists() and rc.use_floorplan_prior:
            try:
                n, missing = database.write_pose_priors(db, schemas.read_json(p.priors))
                ctx.log(f"위치 제약 {n}장 DB 에 씀" + (f", DB 에 없는 사진 {len(missing)}장" if missing else ""))
            except Exception as e:  # noqa: BLE001
                n = 0
                ctx.log(f"경고: 위치 제약 쓰기 실패 {e} -> mapper 로")
            if n:
                r = colmap(CMD_PRIOR_MAPPER, "--database_path", db, "--image_path", p.frames,
                           "--output_path", sparse, *PRIOR_MAPPER_EXTRA)
                if r.ok and _has_models(sparse):
                    return CMD_PRIOR_MAPPER
                old = move_aside(sparse, "prior_failed")
                sparse.mkdir(parents=True, exist_ok=True)
                ctx.log(f"경고: {CMD_PRIOR_MAPPER} 실패 (코드 {r.code}) -> mapper 로 다시. 중간 결과 {old.name if old else '-'}")
        if rc.mapper == "global":
            # GLOMAP 은 초점거리 추정이 필요 (영상 프레임엔 EXIF 없음) -> view_graph_calibrator.
            # DB 를 직접 고치므로 사본에서 (incremental 로 돌아갈 때 원본 DB 를 그대로 씀)
            dbg = db.with_name("db_global.db")
            shutil.copy2(db, dbg)
            r = colmap(CMD_VGC, "--database_path", dbg)
            if r.ok:
                r = colmap(CMD_GLOBAL, "--database_path", dbg, "--image_path", p.frames, "--output_path", sparse)
            if r.ok and _has_models(sparse):
                return CMD_GLOBAL
            old = move_aside(sparse, "global_failed")
            sparse.mkdir(parents=True, exist_ok=True)
            ctx.log(f"경고: global 정합 실패 -> incremental 로 다시. 중간 결과 {old.name if old else '-'}")
        r = colmap(CMD_MAPPER, "--database_path", db, "--image_path", p.frames, "--output_path", sparse)
        if r.ok and _has_models(sparse):
            return CMD_MAPPER
        return None
