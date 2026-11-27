"""3. 진단·정리 (담당 A). data/ -> diagnose.json + data_clean/

- 순간이동: 저장소 tools/diagnose.py report_jumps 를 함수로. 촬영 순서(파일명의 원본 프레임 번호)대로
  이동 속도를 재서 중앙값의 5배 넘게(그리고 상위 10%) 튄 곳. 가장 빠른 15곳의 앞뒤 사진을 뺌
  (복도 1: 상위 15곳 25장 제외 -> PSNR +0.25)
- outside_space: 공간(plan.json)이 있고 모델이 도면 좌표일 때(pose_prior_mapper 로 정합)만. 어느 방에도 없는 카메라
- far_from_prior: priors.json 이 있고 모델이 도면 좌표일 때만. 제약 위치에서 1m 넘게 떨어진 카메라
- excluded = 위 셋의 합집합 -> data_clean/sparse/0 (원본 data/ 는 그대로)

data_clean/images 는 사진을 복사하지 않고 data/images 를 가리킴:
  Windows: 디렉터리 정션(junction). 관리자 권한·개발자 모드가 필요 없고(심볼릭 링크는 필요), 다른 프로그램
  (Brush, LichtFeld)에는 보통 폴더로 보임. 정션이 안 되면 파일별 하드 링크(같은 드라이브, 권한 불필요),
  그것도 안 되면 복사.  리눅스·맥: 심볼릭 링크.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np

from .. import schemas
from ..colmap import model as cmodel
from ..colmap.runner import move_aside
from ..pathing import FRAME_RE
from .base import Stage, StageContext, StageResult

JUMP_FACTOR = 5.0      # 중앙 속도의 몇 배부터 순간이동
JUMP_TOP = 15          # 가장 빠른 몇 곳을 뺄지 (복도 1 실험 값)
FAR_FROM_PRIOR_M = 1.0
OUTSIDE_MAX_FRAC = 0.5  # 이보다 많이 밖이면 좌표가 안 맞는 것으로 보고 빼지 않음 (경고만)


def frame_no(name: str) -> int | None:
    m = FRAME_RE.match(Path(name).name)
    return int(m.group(2)) if m else None


def find_teleports(names: list[str], C: np.ndarray, k: float = JUMP_FACTOR, top: int = JUMP_TOP) -> list[dict]:
    """촬영 순서대로 이동 속도가 중앙값의 k 배 넘게 튄 곳 (빠른 순, 최대 top 개).
    C: 카메라 중심 (N,3). 돌려줌: [{"from", "to", "speed_x_median", "frames"}]"""
    fr = np.array([frame_no(n) if frame_no(n) is not None else -1 for n in names])
    if np.sum(fr >= 0) < 10:
        return []
    o = np.argsort(fr, kind="stable")
    o = o[fr[o] >= 0]
    C, fr = C[o], fr[o]
    nm = [names[i] for i in o]
    df = np.maximum(np.diff(fr), 1)
    v = np.linalg.norm(np.diff(C, axis=0), axis=1) / df
    mv = max(float(np.median(v)), 1e-12)
    bad = np.where((v > k * mv) & (v >= np.percentile(v, 90)))[0]
    bad = bad[np.argsort(-v[bad])][:top]
    return [{"from": nm[i], "to": nm[i + 1], "speed_x_median": round(float(v[i] / mv), 1), "frames": int(df[i])}
            for i in bad]


def link_images(src: Path, dst: Path) -> str:
    """dst 가 src 사진 폴더를 가리키게. 돌려줌: 쓴 방법 이름."""
    src = Path(src).resolve()
    if sys.platform == "win32":
        try:
            import _winapi  # 표준 라이브러리 (문서엔 없지만 3.5 부터 있음)
            _winapi.CreateJunction(str(src), str(dst))
            return "junction"
        except Exception:  # noqa: BLE001
            pass
        dst.mkdir(parents=True, exist_ok=True)
        try:
            for f in src.iterdir():
                os.link(f, dst / f.name)
            return "hardlink"
        except OSError:
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst)
            return "copy"
    os.symlink(src, dst, target_is_directory=True)
    return "symlink"


def model_in_plan_coords(ctx: StageContext) -> bool:
    info = ctx.paths.colmap / "register.json"
    return info.exists() and schemas.read_json(info).get("mapper") == "pose_prior_mapper"


def diagnose_model(m: cmodel.Model, plan=None, priors: dict | None = None) -> dict:
    ims = list(m.images.values())
    names = [im.name for im in ims]
    C = np.array([im.center for im in ims]) if ims else np.zeros((0, 3))
    tele = find_teleports(names, C)
    outside, far = [], []
    if plan is not None:
        outside = sorted(n for n, c in zip(names, C) if plan.room_at(float(c[0]), float(c[1])) is None)
    if priors:
        pri = {f["image"]: np.asarray(f["xyz"], float) for f in priors.get("frames", [])}
        far = sorted(n for n, c in zip(names, C) if n in pri and np.linalg.norm(c - pri[n]) > FAR_FROM_PRIOR_M)
    return {"teleports": tele, "outside_space": outside, "far_from_prior": far}


class Diagnose(Stage):
    name = "diagnose"

    def outputs(self, ctx: StageContext) -> list[Path]:
        return [ctx.paths.diagnose, ctx.paths.data_clean / "sparse" / "0"]

    def run(self, ctx: StageContext) -> StageResult:
        p = ctx.paths
        if not p.data_model.exists():
            return StageResult(False, "data/sparse/0 없음 (register 먼저)")
        m = cmodel.read_model(p.data_model)
        total = len(list(p.frames.glob("f_*.jpg"))) if p.frames.exists() else len(m.images)
        in_plan = model_in_plan_coords(ctx)
        plan = schemas.load_plan(ctx.project.plan_json) if (in_plan and ctx.project.has_space) else None
        priors = schemas.read_json(p.priors) if (in_plan and p.priors.exists()) else None
        if ctx.project.has_space and not in_plan:
            ctx.log("참고: 공간은 있지만 모델이 도면 좌표가 아님 (pose_prior_mapper 안 씀) -> 공간 검사 건너뜀")
        r = diagnose_model(m, plan, priors)
        warn = []
        if r["outside_space"] and len(r["outside_space"]) > OUTSIDE_MAX_FRAC * len(m.images):
            warn.append(f"방 밖 카메라 {len(r['outside_space'])}장: 절반 넘음 -> 좌표가 안 맞는 것으로 보고 빼지 않음")
            outside_ex = []
        else:
            outside_ex = r["outside_space"]
        excluded = set(outside_ex) | set(r["far_from_prior"])
        for t in r["teleports"]:
            excluded |= {t["from"], t["to"]}
        out = {"version": schemas.VERSION, "registered": len(m.images), "total": total,
               "reproj_error_px": None if m.mean_reproj_error() is None else round(m.mean_reproj_error(), 3),
               "teleports": r["teleports"], "outside_space": r["outside_space"],
               "far_from_prior": r["far_from_prior"], "excluded": sorted(excluded), "warnings": warn}
        for w in warn:
            ctx.log("경고: " + w)

        clean = p.data_clean
        if clean.exists():
            ctx.log(f"끝나지 않은 data_clean -> {move_aside(clean).name}")
        try:
            new, n = cmodel.remove_images(m, excluded)
            cmodel.write_model(new, clean / "sparse" / "0_tmp")
            how = link_images(p.data / "images", clean / "images")
            (clean / "sparse" / "0_tmp").rename(clean / "sparse" / "0")
        except Exception as e:  # noqa: BLE001
            old = move_aside(clean)
            return StageResult(False, f"data_clean 만들기 실패: {e} (중간 결과 {old.name if old else '-'})")
        schemas.write_json(p.diagnose, out)
        ctx.log(f"사진 {how} 로 연결")
        return StageResult(True, f"등록 {len(m.images)}/{total}, 순간이동 {len(r['teleports'])}곳, "
                                 f"방 밖 {len(r['outside_space'])}, 제약에서 멂 {len(r['far_from_prior'])} -> {n}장 뺌",
                           [p.diagnose, clean / "sparse" / "0"])
