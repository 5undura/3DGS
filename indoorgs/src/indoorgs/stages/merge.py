"""6. 합치기 (담당 C, 프로젝트 범위). 구간별 최종 PLY -> merged/building.ply + layout.json

구간 PLY: fill/holes_backdrop.ply 가 있으면 그것, 없으면 최신 train/<학습기>_<프리셋>[_N]/splat.ply.
구간 변환(모델 좌표 -> 건물 좌표, 4x4):
1. 공간 정합 구간 (plan.json + 구간 colmap/priors.json): 이미 도면 좌표 -> 단위 행렬.
   plan.json 의 문 근처에서 겹치는 구간끼리 작은 강체 정렬(점-점 ICP, 회전 3도·이동 0.3m 이하,
   나아지지 않으면 변환 안 함). 첫 구간이 기준, 뒤 구간을 앞 구간에 맞춤.
2. 아니면 사용자가 둔 merged/layout.json:
   - IndoorGS 형식 {"version":1, "sections": {"<구간>": {"matrix_4x4": [[..]]}}}
   - 또는 tools/layout_v4.json 형식 {"<키>": {"matrix_4x4": [[..]], ...}}. 키가 구간 이름과 다르면
     최상위 "section_keys": {"<구간>": "<키>"} 로 이어 줌. p_건물 = M[:3,:3] p + M[:3,3] (realign_layout.py 와 같은 수식).
     unity_fix_4x4 (Unity 좌표에서 나중에 곱하던 보정)는 적용하지 않고 layout.json notes 에 적음.
3. 구간이 하나뿐이면 단위 행렬.
쓰기 전에 사용자가 둔 layout.json 은 merged/layout_input_<시각>.json 으로 옮겨 보존.
"""
from __future__ import annotations

import datetime as dt
import shutil
import warnings
from pathlib import Path

import numpy as np

from .. import plyio, schemas
from ..fillcore import align
from .base import Stage, StageContext, StageResult

DOOR_RADIUS = 1.5      # 문 중심에서 이 수평 거리 안 스플랫을 ICP 에 씀 (m)
ICP_MAX_PTS = 20000
ICP_MIN_PTS = 200


def section_ply(project, sec, paths) -> Path:
    if paths.fill_ply.exists():
        return paths.fill_ply
    t = project.cfg.train
    run = paths.latest_train_run(t.trainer, t.preset)                      # 최신 학습 (_2, _3 ...)
    from .snap import SNAP_PLY
    return run / SNAP_PLY if (run / SNAP_PLY).exists() else run / "splat.ply"   # 벽에 붙인 것 우선


def read_layout(path: Path, names: list[str]) -> tuple[dict[str, np.ndarray], list[str]]:
    """{구간: 4x4}, 메모. 형식은 모듈 주석."""
    d = schemas.read_json(path)
    notes = []
    if isinstance(d.get("sections"), dict):
        table, keys = d["sections"], {n: n for n in names}
        fmt = "indoorgs"
    else:
        table = {k: v for k, v in d.items() if isinstance(v, dict) and "matrix_4x4" in v}
        keys = {n: str(d.get("section_keys", {}).get(n, n)) for n in names}
        fmt = "layout_v4"
    out = {}
    for n in names:
        e = table.get(keys[n])
        if e is None:
            continue
        M = np.array(e["matrix_4x4"], float)
        plyio.similarity_parts(M)             # 닮음 변환인지 검사 (아니면 ValueError)
        out[n] = M
        if "unity_fix_4x4" in e:
            notes.append(f"{n}: unity_fix_4x4 는 적용 안 함 (Unity 좌표에서 따로 곱하던 보정)")
    notes.insert(0, f"입력 layout 형식: {fmt}")
    return out, notes


def door_points(ply: plyio.GaussianPly, door, plan, seed: int = 0) -> np.ndarray:
    """문 중심 수평 DOOR_RADIUS 안, 바닥~문 높이+0.5m, 불투명도 > 0.5 인 스플랫 중심 (최대 ICP_MAX_PTS)."""
    rooms = [r for r in plan.rooms if r.id in door.rooms]
    fz = min((r.floor_z for r in rooms), default=0.0)
    P = ply.xyz
    m = (np.hypot(P[:, 0] - door.center[0], P[:, 1] - door.center[1]) < DOOR_RADIUS) \
        & (P[:, 2] > fz - 0.2) & (P[:, 2] < fz + door.height + 0.5) & (ply.opacity() > 0.5)
    Q = P[m]
    if len(Q) > ICP_MAX_PTS:
        Q = Q[np.random.default_rng(seed).choice(len(Q), ICP_MAX_PTS, replace=False)]
    return Q


def refine_at_doors(names: list[str], door_pts: dict[str, dict[str, np.ndarray]], log=lambda m: None,
                    **icp_kw) -> dict[str, tuple[np.ndarray, dict]]:
    """door_pts[구간][문 id] = 점. 첫 구간 기준, 뒤 구간을 이미 정한 앞 구간에 맞춤.
    반환 {구간: (보정 4x4, 기록)}"""
    out: dict[str, tuple[np.ndarray, dict]] = {}
    for i, n in enumerate(names):
        if i == 0:
            out[n] = (np.eye(4), {"icp": "기준 구간"})
            continue
        best = None
        for m in names[:i]:
            for did, P in door_pts.get(n, {}).items():
                Q = door_pts.get(m, {}).get(did)
                if Q is None or len(P) < ICP_MIN_PTS or len(Q) < ICP_MIN_PTS:
                    continue
                k = min(len(P), len(Q))
                if best is None or k > best[0]:
                    best = (k, m, did)
        if best is None:
            out[n] = (np.eye(4), {"icp": "겹치는 문 없음"})
            continue
        _, m, did = best
        T, info = align.icp(door_pts[n][did], align.apply(out[m][0], door_pts[m][did]), **icp_kw)
        rec = {"icp_door": did, "icp_to": m, **info}
        if T is None:
            log(f"{n}: 문 {did} 에서 {m} 에 맞추기 안 함 ({info.get('reason')})")
            out[n] = (np.eye(4), rec)
        else:
            log(f"{n}: 문 {did} 에서 {m} 에 맞춤 (회전 {info['rot_deg']}도, 이동 {info['trans']}m, "
                f"거리 중앙값 {info['median_before']} -> {info['median_after']})")
            out[n] = (T, rec)
    return out


class Merge(Stage):
    name = "merge"
    scope = "project"

    def outputs(self, ctx: StageContext) -> list[Path]:
        return [ctx.project.building_ply, ctx.project.layout_json]

    def prev_targets(self, ctx: StageContext) -> list[Path]:
        # layout.json 은 다음 merge 의 입력이기도 함 -> 그대로 두고 run 이 layout_input_<시각>.json 으로 보존
        return [ctx.project.building_ply]

    def run(self, ctx: StageContext) -> StageResult:
        pr = ctx.project
        secs = pr.sections()
        names = [s.name for s, _ in secs]
        plys = {s.name: section_ply(pr, s, p) for s, p in secs}
        missing = [f"{n} ({p})" for n, p in plys.items() if not p.exists()]
        if missing:
            return StageResult(False, "구간 PLY 없음: " + ", ".join(missing))
        on_plan = [s.name for s, p in secs if pr.has_space and p.priors.exists()]
        user_layout, notes = {}, []
        if pr.layout_json.exists():
            try:
                user_layout, notes = read_layout(pr.layout_json, [n for n in names if n not in on_plan])
            except (ValueError, KeyError) as e:
                return StageResult(False, f"merged/layout.json 읽기 실패: {e}")
        layout: dict[str, dict] = {}
        for n in names:
            if n in on_plan:
                layout[n] = {"matrix": np.eye(4), "source": "floorplan"}
            elif n in user_layout:
                layout[n] = {"matrix": user_layout[n], "source": "layout.json"}
            elif len(names) == 1:
                layout[n] = {"matrix": np.eye(4), "source": "single"}
        lost = [n for n in names if n not in layout]
        if lost:
            return StageResult(False, f"배치가 없는 구간 {lost}: 도면 정합(plan.json + priors.json)이나 "
                                      "merged/layout.json 에 구간별 matrix_4x4 를 넣을 것")
        # 문 근처 미세 정렬 (도면 좌표 구간끼리)
        if len(on_plan) > 1:
            plan = schemas.load_plan(pr.plan_json)
            if plan.doors:
                pts = {}
                for n in on_plan:
                    ply = plyio.read(plys[n])
                    pts[n] = {d.id: door_points(ply, d, plan) for d in plan.doors}
                    del ply
                for n, (T, rec) in refine_at_doors(on_plan, pts, ctx.log).items():
                    layout[n]["matrix"] = T @ layout[n]["matrix"]
                    layout[n].update(rec)
                    if not np.allclose(T, np.eye(4)):
                        layout[n]["source"] = "floorplan+icp"
        # 합치기
        items = [(plys[n], None if np.allclose(layout[n]["matrix"], np.eye(4)) else layout[n]["matrix"]) for n in names]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", plyio.ShNotRotatedWarning)
            res = plyio.merge_files(pr.building_ply, items,
                                    comments=["comment IndoorGS merged building", "comment Vertical axis: see layout.json"])
        for w in {str(w.message) for w in caught}:
            ctx.log("경고: " + w)
            notes.append(w)
        if pr.layout_json.exists():
            keep = pr.merged / f"layout_input_{dt.datetime.now():%Y%m%d_%H%M%S}.json"
            shutil.move(str(pr.layout_json), str(keep))
            notes.append(f"입력 layout.json 은 {keep.name} 로 옮겨 보존")
        out = {"version": 1, "count": res["count"], "notes": notes, "sections": {}}
        for n, k in zip(names, res["parts"]):
            e = layout[n]
            out["sections"][n] = {"matrix_4x4": [[round(float(v), 8) for v in row] for row in e["matrix"]],
                                  "source": e["source"], "ply": str(plys[n]), "count": k,
                                  **{a: b for a, b in e.items() if a not in ("matrix", "source")}}
        schemas.write_json(pr.layout_json, out)
        return StageResult(True, f"{len(names)}개 구간 {res['count']:,}개 스플랫", [pr.building_ply, pr.layout_json])
