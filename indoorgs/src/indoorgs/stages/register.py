"""2. 정합 (담당 A). frames/ (+ colmap/priors.json) -> colmap/db.db, colmap/sparse/<n>, data/

저장소 2_register/register_all.ps1 를 옮김:
  feature_extractor(single_camera=1, camera_model=cfg) -> {sequential|exhaustive}_matcher
  -> mapper (priors 있으면 pose_prior_mapper, 실패하면 mapper 로 자동 전환)
  -> 등록 사진이 가장 많은 모델 -> image_undistorter -> data/ (+ data/sparse/0)
복도 1 실험: global_mapper 는 사진 9장이 수천 단위 밖으로 튀어 못 씀 -> 기본 incremental.

확인한 버전·출처 (2026-11-27)
- COLMAP 4.2.1 (태그 4.2.1, 커밋 bd1fcf6 = 사용자 PC 버전과 같음) 소스 github.com/colmap/colmap
  - src/colmap/exe/colmap.cc: 명령 feature_extractor, sequential_matcher, exhaustive_matcher, mapper,
    pose_prior_mapper, image_undistorter, view_graph_calibrator, global_mapper, image_deleter, model_analyzer
  - src/colmap/exe/sfm.cc RunPosePriorMapper: --database_path --image_path --input_path --output_path(필수, 있는 폴더)
    --overwrite_priors_covariance --prior_position_std_x/y/z --use_robust_loss_on_prior_position
    --prior_position_loss_scale. use_prior_position 은 자동으로 켜짐. 끝에 모델을 priors 좌표로 맞춤
    (controllers/incremental_pipeline.cc AlignReconstructionToPosePriors) -> 결과가 도면 좌표
  - controllers/option_manager.cc: --ImageReader.camera_model, --ImageReader.single_camera
  - exe/image.cc RunImageUndistorter: --image_path --input_path --output_path --output_type {COLMAP,...}
  - pose_priors 표 스키마: colmap/database.py 주석 (pycolmap 4.2.1 로 실제 DB 만들어 확인)
- 저장소 3_train/PROGRESS.md "도구 확인": 사용자 PC COLMAP 4.2.1 에서 ImageReader.*, view_graph_calibrator,
  global_mapper 있음 확인
아래 상수 중 PC 에서 직접 -h 로 안 본 것은 TODO 표시.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .. import schemas
from ..colmap import database, model as cmodel, runner
from ..colmap.runner import move_aside
from .base import Stage, StageContext, StageResult

# ---- COLMAP 명령·옵션 이름 (4.2.1 소스로 확인) ----
CMD_FEATURES = "feature_extractor"
CMD_MAPPER = "mapper"
CMD_PRIOR_MAPPER = "pose_prior_mapper"          # TODO(PC에서 -h로 확인): 소스로만 확인
CMD_UNDISTORT = "image_undistorter"
CMD_VGC = "view_graph_calibrator"
CMD_GLOBAL = "global_mapper"
OPT_SINGLE_CAMERA = "--ImageReader.single_camera"
OPT_CAMERA_MODEL = "--ImageReader.camera_model"
OPT_SEQ_OVERLAP = "--SequentialMatching.overlap"
OPT_SEQ_LOOP = "--SequentialMatching.loop_detection"   # 어휘 트리 경로 기본값 = 자동 다운로드 (TODO(PC): 오프라인이면 실패)
# priors.json 의 sigma 를 DB 에 넣으므로 공분산을 덮어쓰지 않음 (기본값도 0 이지만 명시)
PRIOR_MAPPER_EXTRA = ["--overwrite_priors_covariance", "0"]   # TODO(PC에서 -h로 확인): bool 을 0/1 로 받는지

MATCH_DONE = "matching.done"
MAPPER_DONE = "mapper.done"
INFO = "register.json"      # 어떤 mapper 를 썼는지 (diagnose 가 좌표가 도면 좌표인지 판단할 때 씀)


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


class Register(Stage):
    name = "register"

    def outputs(self, ctx: StageContext) -> list[Path]:
        return [ctx.paths.data_model]

    def prev_targets(self, ctx: StageContext) -> list[Path]:
        # colmap/ 통째로 옮기면 priors.json 까지 빠지므로 register 결과만. 완료 표시 파일은 db 가 없으면 run 이 지움
        p = ctx.paths
        return [p.db, p.sparse, p.data, p.colmap / INFO]

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

        def colmap(*args) -> runner.ToolResult:
            ctx.log(f"COLMAP {args[0]}")
            return runner.run([exe, *args], log_file=tool_log)

        match_done, mapper_done = col / MATCH_DONE, col / MAPPER_DONE
        # 표시 파일 없이 남은 결과 = 끊긴 실행 -> 옆으로 치움 (삭제 안 함)
        if not db.exists():
            match_done.unlink(missing_ok=True)
        if not match_done.exists():
            mapper_done.unlink(missing_ok=True)
        if db.exists() and not match_done.exists():
            ctx.log(f"끝나지 않은 db.db -> {move_aside(db).name}")
        if sparse.exists() and not mapper_done.exists():
            ctx.log(f"끝나지 않은 sparse -> {move_aside(sparse).name}")
        if data.exists():
            ctx.log(f"끝나지 않은 data -> {move_aside(data).name}")

        def fail(msg: str) -> StageResult:
            return StageResult(False, f"{msg} (COLMAP 출력: {tool_log})")

        # 1) 특징점 + 매칭
        if not match_done.exists():
            r = colmap(CMD_FEATURES, "--database_path", db, "--image_path", p.frames,
                       OPT_SINGLE_CAMERA, "1", OPT_CAMERA_MODEL, rc.camera_model)
            if not r.ok:
                move_aside(db)
                return fail(f"{CMD_FEATURES} 실패 (코드 {r.code})")
            margs = ["--database_path", db]
            if rc.matcher == "sequential":
                # 옵션 이름: COLMAP 4.2.1 controllers/option_manager.cc
                margs += [OPT_SEQ_OVERLAP, str(rc.seq_overlap), OPT_SEQ_LOOP, "1" if rc.loop_detection else "0"]
            r = colmap(f"{rc.matcher}_matcher", *margs)
            if not r.ok:
                move_aside(db)
                return fail(f"{rc.matcher}_matcher 실패 (코드 {r.code})")
            match_done.touch()
            ctx.log("매칭 끝")

        # 2) 정합
        used = None
        if mapper_done.exists():
            used = (schemas.read_json(col / INFO).get("mapper") if (col / INFO).exists() else None)
        else:
            sparse.mkdir(parents=True, exist_ok=True)
            used = self._map(ctx, exe, colmap, db, sparse)
            if used is None:
                old = move_aside(sparse)
                return fail(f"정합 실패: 모델 없음 (중간 결과 {old.name if old else '-'})")
            mapper_done.touch()
            schemas.write_json(col / INFO, {"version": schemas.VERSION, "mapper": used,
                                            "matcher": rc.matcher, "camera_model": rc.camera_model})

        # 3) 가장 큰 모델 -> undistort
        best, n = best_model(sparse)
        if best is None:
            return fail("sparse 에 모델 없음")
        ctx.log(f"사용할 모델: {best.name} ({n} / {len(frames)} 장), 정합 {used}")
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
        return StageResult(True, f"{n}/{len(frames)} 장 등록 ({used})", [p.data_model])

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
