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

확인한 버전·출처 (2026-11-27)
- COLMAP 4.2.1 (태그 4.2.1, 커밋 bd1fcf6 = 사용자 PC 버전과 같음) 소스 github.com/colmap/colmap
  - src/colmap/exe/colmap.cc: 명령 feature_extractor, sequential_matcher, exhaustive_matcher, spatial_matcher, mapper,
    pose_prior_mapper, image_undistorter, view_graph_calibrator, global_mapper, image_deleter, model_analyzer
  - src/colmap/exe/sfm.cc RunPosePriorMapper: --database_path --image_path --input_path --output_path(필수, 있는 폴더)
    --overwrite_priors_covariance --prior_position_std_x/y/z --use_robust_loss_on_prior_position
    --prior_position_loss_scale. use_prior_position 은 자동으로 켜짐. 끝에 모델을 priors 좌표로 맞춤
    (controllers/incremental_pipeline.cc AlignReconstructionToPosePriors) -> 결과가 도면 좌표
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

import json
import shutil
from pathlib import Path

from .. import schemas
from ..colmap import database, model as cmodel, pair_verify, runner
from ..colmap.runner import move_aside
from .base import Stage, StageContext, StageResult

# ---- COLMAP 명령·옵션 이름 (4.2.1 소스로 확인) ----
CMD_FEATURES = "feature_extractor"
CMD_MAPPER = "mapper"
CMD_PRIOR_MAPPER = "pose_prior_mapper"          # TODO(PC에서 -h로 확인): 소스로만 확인
CMD_UNDISTORT = "image_undistorter"
CMD_VGC = "view_graph_calibrator"
CMD_GLOBAL = "global_mapper"
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
# priors.json 의 sigma 를 DB 에 넣으므로 공분산을 덮어쓰지 않음 (기본값도 false 이지만 명시).
# exe/sfm.cc: bool overwrite_priors_covariance 를 AddDefaultOption 으로 등록 -> boost program_options 의 bool 값이라
# 0/1/true/false 모두 받음 (다른 bool 옵션 --SiftExtraction.use_gpu 0 과 같은 방식)
PRIOR_MAPPER_EXTRA = ["--overwrite_priors_covariance", "0"]

MATCH_DONE = "matching.done"      # 내용: 매칭 방식 ("plain" | "pair_verify"). 방식이 바뀌면 매칭부터 다시
PASS1_DONE = "pass1.done"
CAND_DONE = "candidates.done"
VERIFY_DONE = "verify.done"       # 내용: 짝 검증 통계 JSON
MAPPER_DONE = "mapper.done"
INFO = "register.json"      # 어떤 mapper 를 썼는지 (diagnose 가 좌표가 도면 좌표인지 판단할 때 씀)
DB_VERIFIED = "db_verified.db"
SPARSE_PASS1 = "sparse_pass1"
SPARSE_ALIGNED = "sparse_aligned"
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
        return [p.db, p.sparse, p.data, c / INFO, c / SPARSE_PASS1, c / DB_VERIFIED, c / SPARSE_ALIGNED]

    def verify_mode(self, ctx: StageContext) -> bool:
        rc, p = ctx.project.cfg.register, ctx.paths
        return bool(rc.pair_verify and rc.use_floorplan_prior and p.priors.exists() and _read_priors(p.priors))

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
        verify = self.verify_mode(ctx)
        mode = "pair_verify" if verify else "plain"

        def colmap(*args) -> runner.ToolResult:
            ctx.log(f"COLMAP {args[0]}")
            return runner.run([exe, *args], log_file=tool_log)

        mk = {n: col / n for n in (MATCH_DONE, PASS1_DONE, CAND_DONE, VERIFY_DONE, MAPPER_DONE)}
        match_done, mapper_done = mk[MATCH_DONE], mk[MAPPER_DONE]
        pass1_dir, dbv, aligned_dir = col / SPARSE_PASS1, col / DB_VERIFIED, col / SPARSE_ALIGNED
        # 완료 표시 파일 정리: 앞 단계가 안 끝났으면 뒤 단계 표시도 무효
        if match_done.exists() and match_done.read_text(encoding="utf-8").strip() not in (
                (mode,) if verify else ("", mode)):
            ctx.log(f"매칭 방식이 바뀜 (-> {mode}) -> 매칭부터 다시")
            match_done.unlink()
        if not db.exists():
            match_done.unlink(missing_ok=True)
        if not match_done.exists():
            for f in mk.values():
                f.unlink(missing_ok=True)
        # pair_verify 중간 단계: 앞 단계가 안 끝났으면 뒤 단계 표시도 무효
        steps = [MATCH_DONE, PASS1_DONE, CAND_DONE, VERIFY_DONE]
        for a, b in zip(steps, steps[1:]):
            if not mk[a].exists():
                mk[b].unlink(missing_ok=True)
        # mapper.done 은 register.json 이 같은 방식으로 끝났다고 할 때만 (1차 실패로 mapper 만 돈 경우도 끝난 것)
        info_path = col / INFO
        if mapper_done.exists() and not (info_path.exists()
                                         and schemas.read_json(info_path).get("mode", "plain") == mode):
            mapper_done.unlink()
        # 표시 파일 없이 남은 결과 = 끊긴 실행 -> 옆으로 치움 (삭제 안 함)
        if db.exists() and not match_done.exists():
            ctx.log(f"끝나지 않은 db.db -> {move_aside(db).name}")
        if verify and pass1_dir.exists() and not mk[PASS1_DONE].exists():
            ctx.log(f"끝나지 않은 {SPARSE_PASS1} -> {move_aside(pass1_dir).name}")
        if verify and dbv.exists() and not mk[VERIFY_DONE].exists():
            ctx.log(f"끝나지 않은 {DB_VERIFIED} -> {move_aside(dbv).name}")
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
            matcher = "sequential" if verify else rc.matcher
            loop = matcher == "sequential" and rc.loop_detection and not verify

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
            if verify:
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
                r = colmap(CMD_PRIOR_MAPPER, "--database_path", db, "--image_path", p.frames,
                           "--output_path", pass1_dir, *PRIOR_MAPPER_EXTRA)
                ok = r.ok and _has_models(pass1_dir)
            if not ok:
                old = move_aside(pass1_dir)
                ctx.log(f"경고: 1차 정합 실패 -> 짝 검증 없이 db.db 로 {CMD_MAPPER}. 중간 결과 {old.name if old else '-'}")
                sparse.mkdir(parents=True, exist_ok=True)
                r = colmap(CMD_MAPPER, "--database_path", db, "--image_path", p.frames, "--output_path", sparse)
                if not (r.ok and _has_models(sparse)):
                    old = move_aside(sparse)
                    return StageResult(False, f"정합 실패: 모델 없음 (중간 결과 {old.name if old else '-'})")
                return {"mapper": CMD_MAPPER, "pair_verify": {"skipped": "1차 정합 실패"}, "aligned": False}
            mk[PASS1_DONE].touch()
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
            mk[CAND_DONE].touch()

        # 4) 사본에서 짝 검증
        if not mk[VERIFY_DONE].exists():
            stats = self._verify(ctx, priors, len(frames))
            mk[VERIFY_DONE].write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")
        try:
            stats = json.loads(mk[VERIFY_DONE].read_text(encoding="utf-8"))
        except ValueError:
            stats = {"skipped": "verify.done 을 못 읽음"}

        # 5) 2차 정합 (검증한 사본으로. 1차 결과를 그대로 쓰는 경우는 건너뜀)
        if stats.get("skipped"):
            src, used = pass1_dir, CMD_PRIOR_MAPPER
        else:
            src = sparse
            sparse.mkdir(parents=True, exist_ok=True)
            r = colmap(CMD_PRIOR_MAPPER, "--database_path", dbv, "--image_path", p.frames,
                       "--output_path", sparse, *PRIOR_MAPPER_EXTRA)
            used = CMD_PRIOR_MAPPER
            if not (r.ok and _has_models(sparse)):
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
        out = {"mapper": used, "pair_verify": stats, "aligned": False, "model": str(best.relative_to(col).as_posix())}
        try:
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

    def _verify(self, ctx, priors: dict, n_frames: int) -> dict:
        """load_reference + db_verified.db 복사 + verify_pairs. 못 하면 {"skipped": 이유, ...}."""
        p, rc = ctx.paths, ctx.project.cfg.register
        col, db = p.colmap, p.db
        dbv = col / DB_VERIFIED
        try:
            ref = pair_verify.load_reference(col / SPARSE_PASS1, priors, max_far_m=rc.verify_max_far_m,
                                             level=rc.gimbal_level)
        except Exception as e:  # noqa: BLE001
            ctx.log(f"경고: 1차 기준 자세 읽기 실패 {e} -> 1차 결과를 그대로 씀")
            return {"skipped": f"기준 자세 실패: {e}"}
        ri = {k: ref.info[k] for k in ("models", "poses", "far_dropped", "unfit_models", "leveled_deg")}
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
                f"기준 없음 {st['drop_unknown']}, 자세 못 구함 {st['drop_no_pose']}), 기준 자세 {len(ref.poses)}장")
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
