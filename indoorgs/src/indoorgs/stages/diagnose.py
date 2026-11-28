"""3. 진단·정리 (담당 A). data/ -> diagnose.json + data_clean/

- 순간이동: 저장소 tools/diagnose.py report_jumps 를 함수로. 촬영 순서(파일명의 원본 프레임 번호)대로
  이동 속도를 재서 중앙값의 5배 넘게(그리고 상위 10%) 튄 곳. 가장 빠른 15곳의 앞뒤 사진을 뺌
  (복도 1: 상위 15곳 25장 제외 -> PSNR +0.25)
- outside_space: 공간(plan.json)이 있고 모델이 도면 좌표일 때(pose_prior_mapper 로 정합, 또는 register 가 제약에 직접 맞춤)만.
  어느 방에도 없는 카메라
- far_from_prior: priors.json 이 있고 모델이 도면 좌표일 때만. 제약 위치에서 문턱 넘게 떨어진 카메라.
  문턱: 기본 1 m. register 가 제약에 직접 맞춘(aligned) 모델이면 맞춘 뒤 좌표로 재고, 문턱은 그 맞춤의 잔차 중앙값
  (register.json align.resid_median_m) 에 비례: clip(FAR_K x 잔차, 1 m, FAR_CAP_M) (e2e_v4: 잔차 0.28 m 인데 1 m 는
  맞는 사진 18장(7%)을 뺌, 진짜 틀린 사진은 3 m 넘게 틀림). 쓴 문턱은 diagnose.json far_from_prior_threshold.
  COLMAP 배치 그대로면 맞는 사진도 걸림 (prior_matching RESULTS 5절)
- orientation (도면 좌표일 때만): 짐벌 영상이라 카메라 위 방향(-R 의 둘째 행)이 +z 근처여야 함. 중앙값 z 가
  FLIP_UP_Z 미만이면 모델이 뒤집힘(위아래·좌우 거울로 놓임, e2e_v4 한 번 정합 -0.83) -> data_clean 을 만들지 않고 실패
  (뒤집힌 모델을 train 에 넘기지 않음; 남아 있던 data_clean 도 _prev_ 로 옮김). register.json align 회전 행렬식이 음수(거울)여도 실패.
  TILT_UP_Z 미만이면 경고만.
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
FAR_FROM_PRIOR_M = 1.0   # 문턱 아래 한계 (직접 맞춤 잔차가 없으면 이 값)
FAR_K = 5.0              # 직접 맞춤(aligned) 잔차 중앙값의 몇 배까지 '제약 근처'로 볼지
FAR_CAP_M = 2.0          # 문턱 위 한계 (잔차가 크면 모델이 접힌 것일 수 있어 너무 느슨해지지 않게)
FLIP_UP_Z = -0.5         # 카메라 위 방향 z 중앙값이 이보다 작으면 뒤집힌 모델 (실패)
TILT_UP_Z = 0.7          # 이보다 작으면 (위 방향이 45도 넘게 기욺) 경고
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


def paths_in_plan_coords(paths) -> bool:
    """구간 모델이 도면 좌표인지 (register.json mapper == pose_prior_mapper, 또는 aligned = 제약에 직접 맞춤).
    모든 단계가 이 판정 하나만 씀 (diagnose 공간 검사, walls·snap, fill 바닥·천장, merge 배치).
    priors.json 이 있어도 mapper 로 바뀌었고 직접 맞추지도 않았으면 아님."""
    info = paths.colmap / "register.json"
    if not info.exists():
        return False
    d = schemas.read_json(info)
    return bool(d.get("aligned")) or d.get("mapper") == "pose_prior_mapper"


def model_in_plan_coords(ctx: StageContext) -> bool:
    return paths_in_plan_coords(ctx.paths)


def far_threshold(reg: dict | None) -> dict:
    """far_from_prior 문턱. register.json 이 직접 맞춤(aligned)이고 align.resid_median_m 이 있으면
    clip(FAR_K x 잔차, FAR_FROM_PRIOR_M, FAR_CAP_M), 아니면 FAR_FROM_PRIOR_M. 돌려줌 {"m", "basis", ...}."""
    al = (reg or {}).get("align") or {}
    r = al.get("resid_median_m")
    if (reg or {}).get("aligned") and isinstance(r, (int, float)) and r > 0:
        thr = min(FAR_CAP_M, max(FAR_FROM_PRIOR_M, FAR_K * float(r)))
        return {"m": round(thr, 3), "basis": "align_resid", "resid_median_m": round(float(r), 4), "k": FAR_K,
                "min_m": FAR_FROM_PRIOR_M, "cap_m": FAR_CAP_M}
    return {"m": FAR_FROM_PRIOR_M, "basis": "default"}


def orientation_check(m: cmodel.Model, reg: dict | None = None) -> dict:
    """도면 좌표 모델이 뒤집히지 않았는지 (짐벌 영상: 카메라 위 방향 ≈ +z).
    up = 카메라 -y 축의 월드 방향 = -R[1] (R = cam_from_world). 돌려줌 {"up_z_median", "up_down_frac", "flipped",
    "mirrored", "tilted", "message"}. mirrored = register.json align 회전 행렬식 < 0."""
    ims = list(m.images.values())
    out = {"n": len(ims), "up_z_median": None, "up_down_frac": None, "flipped": False, "mirrored": False,
           "tilted": False, "message": ""}
    if ims:
        uz = np.array([-im.R[1, 2] for im in ims])
        out["up_z_median"] = round(float(np.median(uz)), 3)
        out["up_down_frac"] = round(float((uz < 0).mean()), 3)
        out["flipped"] = bool(np.median(uz) < FLIP_UP_Z)
        out["tilted"] = bool(not out["flipped"] and np.median(uz) < TILT_UP_Z)
    rot = (((reg or {}).get("align") or {}).get("transform") or {}).get("rotation")
    if rot is not None:
        try:
            out["mirrored"] = bool(np.linalg.det(np.asarray(rot, float)) < 0)
        except Exception:  # noqa: BLE001
            pass
    if out["flipped"]:
        out["message"] = (f"모델이 도면 좌표에서 뒤집힘: 카메라 위 방향 z 중앙값 {out['up_z_median']:+.2f} "
                          f"(사진 {out['up_down_frac'] * 100:.0f}% 가 아래를 위로) -> 접히거나 뒤집힌 정합. "
                          "register pair_verify 를 켜고 다시 정합 (짐벌 수평 맞춤)")
    elif out["mirrored"]:
        out["message"] = "register.json align 변환이 거울(행렬식 < 0) -> 다시 정합"
    elif out["tilted"]:
        out["message"] = f"카메라 위 방향이 +z 에서 45도 넘게 기욺 (z 중앙값 {out['up_z_median']:+.2f}): 수평 맞춤 확인"
    return out


def diagnose_model(m: cmodel.Model, plan=None, priors: dict | None = None, far_m: float = FAR_FROM_PRIOR_M) -> dict:
    ims = list(m.images.values())
    names = [im.name for im in ims]
    C = np.array([im.center for im in ims]) if ims else np.zeros((0, 3))
    tele = find_teleports(names, C)
    outside, far = [], []
    if plan is not None:
        outside = sorted(n for n, c in zip(names, C) if plan.room_at(float(c[0]), float(c[1])) is None)
    if priors:
        pri = {f["image"]: np.asarray(f["xyz"], float) for f in priors.get("frames", [])}
        far = sorted(n for n, c in zip(names, C) if n in pri and np.linalg.norm(c - pri[n]) > far_m)
    return {"teleports": tele, "outside_space": outside, "far_from_prior": far}


class Diagnose(Stage):
    name = "diagnose"

    def outputs(self, ctx: StageContext) -> list[Path]:
        return [ctx.paths.diagnose, ctx.paths.data_clean / "sparse" / "0"]

    def prev_targets(self, ctx: StageContext) -> list[Path]:
        return [ctx.paths.diagnose, ctx.paths.data_clean]

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
        reg = None
        if in_plan and (p.colmap / "register.json").exists():
            try:
                reg = schemas.read_json(p.colmap / "register.json")
            except Exception:  # noqa: BLE001
                reg = None
        far = far_threshold(reg)
        r = diagnose_model(m, plan, priors, far_m=far["m"])
        warn = []
        orient = orientation_check(m, reg) if in_plan else None
        if priors and far["basis"] != "default":
            ctx.log(f"far_from_prior 문턱 {far['m']:.2f} m (직접 맞춤 잔차 중앙 {far['resid_median_m']:.2f} m x {FAR_K:g})")
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
               "far_from_prior": r["far_from_prior"],
               "far_from_prior_threshold": far if priors else None,
               "orientation": orient, "excluded": sorted(excluded), "warnings": warn}
        if orient and orient["tilted"]:
            warn.append(orient["message"])
        for w in warn:
            ctx.log("경고: " + w)
        if orient and (orient["flipped"] or orient["mirrored"]):
            out["failed"] = orient["message"]
            clean = p.data_clean
            if clean.exists():   # 예전 data_clean 으로 train 이 돌지 않게 (지우지 않고 옮김)
                ctx.log(f"이전 data_clean -> {move_aside(clean).name} (뒤집힌 모델이라 새로 안 만듦)")
            schemas.write_json(p.diagnose, out)
            return StageResult(False, "뒤집힌 모델: " + orient["message"], [p.diagnose])

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
