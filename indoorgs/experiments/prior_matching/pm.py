"""동선으로 매칭 짝 고르기·검증 실험 (pycolmap 4.2.1, CPU).

  python pm.py --scene <dir> --runs <이름,...> [--stages features,map,eval] [--table]

<dir>: images/, gt_poses.json, priors/priors_<kind>.json (render_v4.py 또는 synth_corridor/run.py 가 만든 것).
실행 이름과 설정은 아래 RUNS. 모든 짝 변형은 같은 exhaustive DB 의 two_view_geometries 에서 짝을 지워 만든다
(같은 특징·같은 매칭 -> 짝 집합만 다름). 결과: <dir>/pm/<run>/{db.db, sparse/<n>, done.json, eval.json}
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parent / "synth_corridor"))

import pairlib as pl  # noqa: E402
from indoorgs import schemas  # noqa: E402
from indoorgs.colmap import database  # noqa: E402
from indoorgs.stages.diagnose import find_teleports  # noqa: E402

SEQ_KEEP = 10            # 앞뒤 이 장수 안의 짝은 어떤 검사로도 지우지 않음
FOCAL_ERR = 1.15         # 한 번에 거르는 검사(T_rot, T_depth)가 쓰는 초점거리 = 참값 x 이 값 (카메라 화각을 대충 안다고 가정)
PLAN_DIAG = 30.0         # 도면에서 가장 먼 두 점 거리쯤 (24 m 복도) x 여유

# 이름 -> 설정. pairs: 차례로 적용하는 거르기 목록
#   "seqonly:N"      앞뒤 N 장 짝만 (= sequential_matcher overlap N, quadratic_overlap 0)
#   "seqquad:N"      2^k (k < N) 장 떨어진 짝만 (= sequential_matcher 기본값 quadratic_overlap 1. 4.2.1 은 이때 앞뒤 N 장을 다 잇지 않음)
#   "spatial:D:K"    제약 위치 거리 D m 안, 가까운 K 장 (0 = 제한 없음) + 앞뒤 SEQ_KEEP 장  (= spatial_matcher)
#   "rot:T"          T_rot (진행 방향 차이 vs 상대 회전각, 허용 T 도)
#   "headall:T"      예전 heading_filter (진행 방향이 T 도 넘게 다르면 무조건 삭제)
#   "depth:M"        T_depth (도면 크기 M m)
#   "pose:<run>"     T_pose: <run> 의 모델(도면 좌표) 자세와 비교. "pose:<run>:keepunk" = 자세 없는 사진의 짝은 둠
#   "oracle"         정답으로 가짜 짝 삭제 (진단용)
RUNS: dict[str, dict] = {
    # 기준
    "exh": {"pairs": []},
    "exh_prior": {"pairs": [], "prior": "good"},
    "seqquad_prior": {"pairs": ["seqquad:20"], "prior": "good"},
    "oracle": {"pairs": ["oracle"]},
    "oracle_prior": {"pairs": ["oracle"], "prior": "good"},
    # (a) 짝 고르기
    "spa6_prior": {"pairs": ["spatial:6:0"], "prior": "good"},
    "spa6k80_prior": {"pairs": ["spatial:6:80"], "prior": "good"},
    "spa3_prior": {"pairs": ["spatial:3:0"], "prior": "good"},
    "seq20_prior": {"pairs": ["seqonly:20"], "prior": "good"},        # 1차 정합 후보
    "seq20": {"pairs": ["seqonly:20"]},
    "seq10_prior": {"pairs": ["seqonly:10"], "prior": "good"},
    # (b) 짝 검증 (한 번에)
    "headall_prior": {"pairs": ["headall:120"], "prior": "good"},
    "rot_prior": {"pairs": ["rot:80"], "prior": "good"},
    "rot": {"pairs": ["rot:80"]},
    "rot_depth_prior": {"pairs": ["rot:80", "depth:30"], "prior": "good"},
    "spa6_rot_prior": {"pairs": ["spatial:6:0", "rot:80"], "prior": "good"},
    "spa6_rot_depth_prior": {"pairs": ["spatial:6:0", "rot:80", "depth:30"], "prior": "good"},
    # (b) 두 번에: 1차 = 앞뒤 짝만 + 제약, 2차 = 1차 자세로 검증한 짝 + 제약
    "pose_seq20_prior": {"pairs": ["pose:seq20_prior"], "prior": "good"},
    "pose_seq20": {"pairs": ["pose:seq20_prior"]},
    "pose_seq20_keepunk_prior": {"pairs": ["pose:seq20_prior:keepunk"], "prior": "good"},
    "pose_rot_prior": {"pairs": ["pose:rot_prior"], "prior": "good"},
    "pose2_prior": {"pairs": ["pose:pose_seq20_prior"], "prior": "good"},          # 한 번 더 되풀이
    "spa6_pose_seq20_prior": {"pairs": ["spatial:6:0", "pose:seq20_prior"], "prior": "good"},
    "pose_seq20_level_prior": {"pairs": ["pose:seq20_prior:level"], "prior": "good"},
    "pose_seq20_level_prior_t2s": {"pairs": ["pose:seq20_prior_t2s:level"], "prior": "t2s"},
    "pose_seq10_prior": {"pairs": ["pose:seq10_prior"], "prior": "good"},
    # 제약을 mapper 에 넣지 않는 길: 1차 = 앞뒤 짝 + 일반 mapper, 조각을 직접 제약에 맞춤 -> 검증 -> 2차 = 일반 mapper
    "np_pose": {"pairs": ["pose:seq20:align"]},
    "np_pose_prior": {"pairs": ["pose:seq20:align"], "prior": "good"},
    "np_pose_t2s": {"pairs": ["pose:seq20:align"], "prior_for_filter": "t2s"},
    "np_pose_x1m": {"pairs": ["pose:seq20:align"], "prior_for_filter": "x1m"},
    "np_spa6k80_pose": {"pairs": ["spatial:6:80", "pose:seq20:align"]},
    "np_spa6k80_pose_prior": {"pairs": ["spatial:6:80", "pose:seq20:align"], "prior": "good"},
    "spa6k80_pose_seq20_prior": {"pairs": ["spatial:6:80", "pose:seq20_prior"], "prior": "good"},
    "spa6k80_pose_seq20": {"pairs": ["spatial:6:80", "pose:seq20_prior"]},
    # (d) 되풀이: 일반 exhaustive + 제약 결과에서 제약 가까운 사진의 자세로 검증
    "pose_exh_prior": {"pairs": ["pose:exh_prior"], "prior": "good"},
    # (c) 제약 세기·mapper 옵션 (짝은 그대로)
    "exh_prior_s02": {"pairs": [], "prior": "good", "sigma": (0.2, 0.2)},
    "exh_prior_s005": {"pairs": [], "prior": "good", "sigma": (0.05, 0.05)},
    "exh_prior_robust": {"pairs": [], "prior": "good", "robust": True},
    "exh_prior_strict": {"pairs": [], "prior": "good",
                         "opts": {"min_num_matches": 50, "mapper.abs_pose_min_num_inliers": 60,
                                  "mapper.abs_pose_min_inlier_ratio": 0.5, "mapper.abs_pose_max_error": 6.0}},
    "pose_seq20_prior_s02": {"pairs": ["pose:seq20_prior"], "prior": "good", "sigma": (0.2, 0.2)},
    "pose_seq20_prior_robust": {"pairs": ["pose:seq20_prior"], "prior": "good", "robust": True},
    # 초점거리를 미리 줌 (참값의 1.05배)
    "exh_cam": {"pairs": [], "cam": 1.05},
    "exh_prior_cam": {"pairs": [], "prior": "good", "cam": 1.05},
    "oracle_cam": {"pairs": ["oracle"], "cam": 1.05},
    "seq20_prior_cam": {"pairs": ["seqonly:20"], "prior": "good", "cam": 1.05},
    "pose_seq20_prior_cam": {"pairs": ["pose:seq20_prior_cam"], "prior": "good", "cam": 1.05},
    # 틀린 동선
    "seq20_prior_t2s": {"pairs": ["seqonly:20"], "prior": "t2s"},
    "pose_seq20_prior_t2s": {"pairs": ["pose:seq20_prior_t2s"], "prior": "t2s"},
    "rot_prior_t2s": {"pairs": ["rot:80"], "prior": "t2s"},
    "seq20_prior_x1m": {"pairs": ["seqonly:20"], "prior": "x1m"},
    "pose_seq20_prior_x1m": {"pairs": ["pose:seq20_prior_x1m"], "prior": "x1m"},
}


def split_seed(run: str) -> tuple[str, int]:
    head, _, tail = run.rpartition("_s")
    if head and tail.isdigit() and head in RUNS:
        return head, int(tail)
    return run, 0


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- 장면 정보
class Scene:
    def __init__(self, root: Path):
        self.root = root
        g = schemas.read_json(root / "gt_poses.json")
        self.K = np.array(g["K"])
        self.size = g["size"]
        self.gt = {f["image"]: (np.array(f["R_world_from_cam"]), np.array(f["center"])) for f in g["frames"]}
        self.time = {f["image"]: f["t"] for f in g["frames"]}
        self.names = sorted(self.gt)
        self.px = 4.0 * self.size[0] / 640          # 가짜 짝 판정 문턱 (v3 640px 에서 4px)
        self._d = self._lab = None

    def priors(self, kind: str) -> dict:
        return schemas.read_json(self.root / "priors" / f"priors_{kind}.json")

    @property
    def exh(self) -> Path:
        return self.root / "db" / "exhaustive.db"

    def db(self) -> dict:
        if self._d is None:
            self._d = pl.read_db(self.exh)
        return self._d

    def labels(self) -> dict[int, bool]:
        if self._lab is None:
            f = self.root / "db" / "exhaustive.labels.pkl"
            if f.exists():
                self._lab = pickle.loads(f.read_bytes())
            else:
                self._lab = pl.label_pairs(self.db(), self.gt, self.K, self.px)
                f.write_bytes(pickle.dumps(self._lab))
        return self._lab

    def rel(self, focal: float, pids) -> dict:
        """상대 자세 (초점거리별 캐시)."""
        f = self.root / "db" / f"exhaustive.rel_{focal:.0f}.pkl"
        cache = pickle.loads(f.read_bytes()) if f.exists() else {}
        todo = [p for p in pids if p not in cache]
        if todo:
            t0 = time.time()
            new = pl.relative_poses(self.db(), focal, todo)
            for p in todo:
                cache[p] = new.get(p)
            f.write_bytes(pickle.dumps(cache))
            log(f"  상대 자세 {len(todo)}짝 (f={focal:.0f}) {time.time() - t0:.0f}s")
        return {p: cache[p] for p in pids if cache.get(p) is not None}


# ---------------------------------------------------------------- 특징·매칭 (exhaustive 한 번)
def stage_features(sc: Scene):
    import pycolmap
    db = sc.root / "db" / "features.db"
    db.parent.mkdir(exist_ok=True)
    times = {}
    if not db.exists():
        t0 = time.time()
        eo = pycolmap.FeatureExtractionOptions()
        eo.num_threads = 4
        eo.use_gpu = False
        ro = pycolmap.ImageReaderOptions()
        ro.camera_model = "SIMPLE_RADIAL"
        tmp = db.with_suffix(".tmp")
        tmp.unlink(missing_ok=True)
        pycolmap.extract_features(tmp, sc.root / "images", camera_mode=pycolmap.CameraMode.SINGLE,
                                  reader_options=ro, extraction_options=eo, device=pycolmap.Device.cpu)
        tmp.rename(db)
        times["features"] = time.time() - t0
        log(f"features {times['features']:.0f}s")
    if not sc.exh.exists():
        tmp = sc.exh.with_suffix(".tmp")
        shutil.copy(db, tmp)
        mo = pycolmap.FeatureMatchingOptions()
        mo.num_threads = 4
        mo.use_gpu = False
        t0 = time.time()
        pycolmap.match_exhaustive(tmp, matching_options=mo, device=pycolmap.Device.cpu)
        tmp.rename(sc.exh)
        times["exhaustive"] = time.time() - t0
        log(f"match exhaustive {times['exhaustive']:.0f}s")
    if times:
        f = sc.root / "db" / "match_times.json"
        old = schemas.read_json(f) if f.exists() else {"version": 1}
        old.update(times)
        schemas.write_json(f, old)


# ---------------------------------------------------------------- 1차 모델 자세 읽기
def load_models(rd: Path, min_size: int = 10) -> list[dict]:
    import pycolmap
    out = []
    for m in sorted((p for p in (rd / "sparse").iterdir() if p.is_dir()), key=lambda p: p.name):
        rec = pycolmap.Reconstruction(m)
        names, C, R = [], [], []
        for im in rec.images.values():
            if im.has_pose:
                names.append(im.name)
                C.append(im.projection_center())
                R.append(im.cam_from_world().rotation.matrix().T)
        o = np.argsort(names)
        cam = next(iter(rec.cameras.values()))
        out.append({"dir": m, "names": [names[i] for i in o], "C": np.array(C)[o].reshape(-1, 3),
                    "R": np.array(R)[o].reshape(-1, 3, 3), "focal": float(cam.params[0]),
                    "xyz": np.array([p.xyz for p in rec.points3D.values()]).reshape(-1, 3)})
    out.sort(key=lambda m: -len(m["names"]))
    return [m for i, m in enumerate(out) if i == 0 or len(m["names"]) >= min_size]


def level_model(m: dict) -> float:
    """짐벌 카메라는 수평을 유지한다 = 카메라 오른쪽 축(R 의 첫 열)이 모두 수평. 그 축들과 가장 수직인 방향을 '위'로 보고
    모델을 카메라 중심 평균 둘레로 돌려 위를 +z 에 맞춤 (제자리에서 m 을 고침). 돌린 각도(도)를 돌려줌.
    일직선 동선에서는 위치 제약만으로 복도 축 둘레 기울기가 안 정해지는 것을 보완."""
    rx = m["R"][:, :, 0]
    w, v = np.linalg.eigh(rx.T @ rx)
    g = v[:, 0]
    if g[2] < 0:
        g = -g
    ax = np.cross(g, [0, 0, 1.0])
    sn, cs = np.linalg.norm(ax), float(g[2])
    if sn < 1e-9:
        return 0.0
    ax /= sn
    Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    Q = np.eye(3) + sn * Kx + (1 - cs) * Kx @ Kx
    c0 = m["C"].mean(0)
    m["C"] = (m["C"] - c0) @ Q.T + c0
    m["R"] = np.einsum("ij,njk->nik", Q, m["R"])
    return math.degrees(math.atan2(sn, cs))


def align_to_priors(m: dict, pri: dict, iters: int = 5, scale_m: float = 1.0) -> dict:
    """모델을 제약 위치에 직접 맞춤 (제자리에서 m 을 고침): ① 수평 맞추기(level_model) ② 수직축 회전 + 축척 + 이동을
    강건 최소제곱(Cauchy 가중)으로. pose_prior_mapper 의 정렬 대신 쓰는 것. 돌려줌: 통계 (span = 제약 위치가 퍼진 길이 m)."""
    tilt = level_model(m)
    P = np.array([pri[n] for n in m["names"]])
    C = m["C"]
    w = np.ones(len(C))
    for _ in range(iters):
        ca, pa = (w[:, None] * C).sum(0) / w.sum(), (w[:, None] * P).sum(0) / w.sum()
        za = (C[:, 0] - ca[0]) + 1j * (C[:, 1] - ca[1])
        zb = (P[:, 0] - pa[0]) + 1j * (P[:, 1] - pa[1])
        k = (w * np.conj(za)) @ zb / max(1e-12, (w * np.conj(za)) @ za)
        sc_, th = abs(k), float(np.angle(k))
        Rz = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1.0]])
        C2 = sc_ * (C - ca) @ Rz.T + pa
        r = np.linalg.norm(C2 - P, axis=1)
        w = 1.0 / (1.0 + (r / scale_m) ** 2)
    m["C"] = C2
    m["R"] = np.einsum("ij,njk->nik", Rz, m["R"])
    if len(m.get("xyz", [])):
        m["xyz"] = sc_ * (m["xyz"] - ca) @ Rz.T + pa      # (수평 맞추기 전 좌표라 근사. 평가에는 안 씀)
    ev = np.linalg.eigvalsh(np.cov(P[:, :2].T)) if len(P) > 2 else np.zeros(2)
    return {"n": len(C), "tilt_deg": tilt, "scale": float(sc_), "resid_median": float(np.median(r)),
            "span_m": float(4 * math.sqrt(max(ev.max(), 0)))}


def reference_poses(sc: Scene, run: str, prior_kind: str, max_far: float = 1.5, level: bool = False,
                    align: bool = False, min_span: float = 3.0) -> tuple[dict, float, dict]:
    """<run> 의 모든 모델(도면 좌표)에서 제약 가까이(max_far m 안) 놓인 사진의 자세. 겹치면 큰 모델 것.
    돌려줌: (시간 순 번호 -> (R_wc, C), 초점거리, 통계)"""
    pri = {f["image"]: np.array(f["xyz"]) for f in sc.priors(prior_kind)["frames"]}
    idx = {n: k for k, n in enumerate(sc.names)}
    poses, n_far = {}, 0
    models = load_models(sc.root / "pm" / run)
    leveled = [level_model(m) for m in models] if level and not align else []
    aligned = []
    if align:      # 제약 없이 만든 모델 조각들을 직접 제약에 맞춤. 너무 짧은 조각(제약이 퍼진 길이 < min_span)은 방향을 못 믿어 뺌
        keep = []
        for m in models:
            st = align_to_priors(m, pri)
            aligned.append(st)
            if st["span_m"] >= min_span:
                keep.append(m)
        models_used = keep
    else:
        models_used = models
    for m in models_used:
        for n, C, R in zip(m["names"], m["C"], m["R"]):
            k = idx[n]
            if k in poses:
                continue
            if np.linalg.norm(C - pri[n]) > max_far:
                n_far += 1
                continue
            poses[k] = (R, C)
    return poses, models[0]["focal"], {"ref_run": run, "ref_poses": len(poses), "ref_far_dropped": n_far,
                                       "ref_focal": models[0]["focal"], "ref_models": [len(m["names"]) for m in models],
                                       "leveled_deg": leveled, "aligned": aligned}


# ---------------------------------------------------------------- 짝 거르기
def apply_filters(sc: Scene, spec: dict, db: Path) -> dict:
    d = sc.db()
    order = d["order"]
    lab = sc.labels()
    alive = set(d["pairs"])
    stats = {"steps": [], "pairs_exhaustive": len(alive), "false_exhaustive": int(sum(lab.values()))}
    kind = spec.get("prior") or spec.get("prior_for_filter") or "good"
    inv = {k: i for i, k in order.items()}
    t_filter = 0.0

    def P_t():
        pos = pl.prior_arrays(sc.priors(kind), d["images"])
        P = np.array([pos[inv[k]] for k in range(len(inv))])
        t = np.array([sc.time[d["images"][inv[k]]] for k in range(len(inv))])
        return pos, P, t

    for f in spec["pairs"]:
        t0 = time.time()
        a = f.split(":")
        extra = {}
        if a[0] in ("seqonly", "seqquad"):
            n = int(a[1])
            drop = set()
            for p in alive:
                i, j = (order[x] for x in pl.id_pair(p))
                gap = abs(i - j)
                if a[0] == "seqquad":      # 4.2.1 pairing.cc: quadratic_overlap 이면 i + 2^k (k < overlap) 짝만 만든다
                    ok = (gap & (gap - 1)) == 0 and gap <= (1 << (n - 1))
                else:
                    ok = gap <= n
                if not ok:
                    drop.add(p)
        elif a[0] == "spatial":
            pos, _, _ = P_t()
            keep = pl.select_pairs(pos, order, float(a[1]), int(a[2]), SEQ_KEEP)
            drop = alive - keep
        elif a[0] == "oracle":
            drop = {p for p in alive if lab[p]}
        elif a[0] in ("rot", "headall", "depth"):
            _, P, t = P_t()
            far = [p for p in alive if abs(order[pl.id_pair(p)[0]] - order[pl.id_pair(p)[1]]) > SEQ_KEEP]
            if a[0] == "headall":
                head = pl.path_headings(P, t, turn_window_s=0.0)
                drop = set()
                for p in far:
                    i, j = (order[x] for x in pl.id_pair(p))
                    if not (np.isnan(head[i, 0]) or np.isnan(head[j, 0])) and \
                            head[i] @ head[j] < math.cos(math.radians(float(a[1]))):
                        drop.add(p)
            else:
                rel = sc.rel(sc.K[0, 0] * spec.get("focal_err", FOCAL_ERR), far)
                if a[0] == "rot":
                    drop = pl.check_heading(rel, order, pl.path_headings(P, t), float(a[1]), SEQ_KEEP)
                else:
                    drop = pl.check_depth(rel, order, P, float(a[1]), seq_keep=SEQ_KEEP)
        elif a[0] == "pose":
            poses, focal, extra = reference_poses(sc, a[1], kind, level="level" in a[2:], align="align" in a[2:])
            far = [p for p in alive if abs(order[pl.id_pair(p)[0]] - order[pl.id_pair(p)[1]]) > SEQ_KEEP]
            rel = sc.rel(round(focal / 5) * 5.0, far)
            drop, st = pl.check_poses(rel, order, poses, seq_keep=SEQ_KEEP, drop_unknown="keepunk" not in a[2:],
                                      rot_tol=spec.get("rot_tol", 30.0), dir_tol=spec.get("dir_tol", 45.0))
            norel = {p for p in far if p not in rel}      # 상대 자세를 못 구한 짝은 지움
            drop |= norel
            extra.update(st)
            extra["no_rel"] = len(norel)
        else:
            raise ValueError(f)
        drop &= alive
        alive -= drop
        dt = time.time() - t0
        t_filter += dt
        stats["steps"].append({"filter": f, "dropped": len(drop), "dropped_false": int(sum(lab[p] for p in drop)),
                               "dropped_true": int(sum(not lab[p] for p in drop)), "left": len(alive),
                               "left_false": int(sum(lab[p] for p in alive)), "seconds": dt, **extra})
    pl.delete_pairs(db, set(d["pairs"]) - alive)
    # 15 미만 inlier 짝(read_db 가 안 읽은 것)은 mapper 의 min_num_matches 15 에 걸려 어차피 안 쓰임
    stats["pairs"] = len(alive)
    stats["false"] = int(sum(lab[p] for p in alive))
    stats["filter_seconds"] = t_filter
    # 갈 때·올 때를 잇는 진짜 짝이 얼마나 남았나 (모델이 하나로 이어지려면 필요)
    half = len(sc.names) // 2
    stats["true_far"] = int(sum(1 for p in alive if not lab[p]
                                and abs(order[pl.id_pair(p)[0]] - order[pl.id_pair(p)[1]]) > 40))
    return stats


# ---------------------------------------------------------------- 정합
def stage_map(sc: Scene, run: str):
    import pycolmap
    base, seed = split_seed(run)
    spec = RUNS[base]
    rd = sc.root / "pm" / run
    if (rd / "done.json").exists():
        log(f"map {run}: 있음")
        return
    for f in spec["pairs"]:
        if f.startswith("pose:"):
            ref = f.split(":")[1]
            if not (sc.root / "pm" / ref / "done.json").exists():
                stage_map(sc, ref)
    if rd.exists():
        shutil.rmtree(rd)        # 이 스크립트가 만든 실패한 중간 산출물
    (rd / "sparse").mkdir(parents=True)
    db = rd / "db.db"
    shutil.copy(sc.exh, db)
    fst = apply_filters(sc, spec, db)
    prior = spec.get("prior")
    if prior:
        pj = sc.priors(prior)
        if spec.get("sigma"):
            sx, sz = spec["sigma"]
            for fr in pj["frames"]:
                fr["sigma"] = [sx, sx, sz]
        n, missing = database.write_pose_priors(db, pj)
        assert n and not missing
    if spec.get("cam"):      # 초점거리를 미리 넣음 (참값 x cam). 기본은 COLMAP 추정값 1.2 x 폭
        import sqlite3
        from contextlib import closing
        prm = np.array([sc.K[0, 0] * float(spec["cam"]), sc.K[0, 2], sc.K[1, 2], 0.0])
        with closing(sqlite3.connect(str(db))) as c, c:
            c.execute("UPDATE cameras SET params=?, prior_focal_length=1", (prm.tobytes(),))
    o = pycolmap.IncrementalPipelineOptions()
    o.num_threads = 4
    o.random_seed = seed
    o.use_prior_position = bool(prior)
    o.use_robust_loss_on_prior_position = bool(spec.get("robust", False))
    for k, v in spec.get("opts", {}).items():
        tgt = o
        *path, leaf = k.split(".")
        for p in path:
            tgt = getattr(tgt, p)
        setattr(tgt, leaf, v)
    t0 = time.time()
    recs = pycolmap.incremental_mapping(db, sc.root / "images", rd / "sparse", o)
    dt = time.time() - t0
    schemas.write_json(rd / "done.json", {"version": 1, "spec": spec, "seed": seed, "seconds": dt,
                                          "models": len(recs), "filter": fst})
    log(f"map {run}: 짝 {fst['pairs']} (가짜 {fst['false']}) 모델 {[r.num_reg_images() for r in recs.values()]} {dt:.0f}s")


# ---------------------------------------------------------------- 평가
def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / xs.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def robust_sim3(src, dst, thr=0.5, iters=2000, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        idx = rng.choice(len(src), 3, replace=False)
        try:
            s, R, t = umeyama(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(s):
            continue
        inl = np.linalg.norm((s * src @ R.T + t) - dst, axis=1) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    for _ in range(3):
        if best.sum() < 3:
            break
        s, R, t = umeyama(src[best], dst[best])
        best = np.linalg.norm((s * src @ R.T + t) - dst, axis=1) < thr
    s, R, t = umeyama(src[best], dst[best]) if best.sum() >= 3 else (1.0, np.eye(3), np.zeros(3))
    return s, R, t, best


def eval_run(sc: Scene, run: str) -> dict:
    rd = sc.root / "pm" / run
    done = schemas.read_json(rd / "done.json")
    spec = done["spec"]
    res = {"run": run, "pairs": done["filter"]["pairs"], "false_pairs": done["filter"]["false"],
           "true_far_pairs": done["filter"]["true_far"], "prior": spec.get("prior"), "seconds": done["seconds"],
           "filter_seconds": done["filter"]["filter_seconds"], "total": len(sc.names), "steps": done["filter"]["steps"]}
    models = load_models(rd) if any((rd / "sparse").iterdir()) else []
    if not models:
        res.update({"model_sizes": [], "registered": 0})
        schemas.write_json(rd / "eval.json", res)
        return res
    res["model_sizes"] = [len(m["names"]) for m in models]
    big = models[0]
    res["registered"] = len(big["names"])
    # 모델별 Sim3 정렬 (모델 안에서 맞게 놓였나)
    ok_all = bad_all = 0
    for k, m in enumerate(models):
        Cg = np.array([sc.gt[n][1] for n in m["names"]])
        if len(m["names"]) < 4:
            continue
        s, R, t, inl = robust_sim3(m["C"], Cg)
        e = np.linalg.norm(s * m["C"] @ R.T + t - Cg, axis=1)
        ok_all += int((e <= 0.5).sum())
        bad_all += int((e > 1.0).sum())
        if k == 0:
            res["sim3_ok"], res["sim3_bad"] = int((e <= 0.5).sum()), int((e > 1.0).sum())
            Rg = np.array([sc.gt[n][0] for n in m["names"]])
            re = [math.degrees(math.acos(max(-1, min(1, (np.trace((R @ m["R"][i]).T @ Rg[i]) - 1) / 2))))
                  for i in range(len(Rg))]
            res["sim3_rot_median"] = float(np.median(re))
    res["sim3_ok_all"], res["sim3_bad_all"] = ok_all, bad_all
    res["teleports"] = len(find_teleports(big["names"], big["C"]))
    res["focal"] = big["focal"]
    kind = spec.get("prior") or spec.get("prior_for_filter") or "good"
    pri = {f["image"]: np.array(f["xyz"]) for f in sc.priors(kind)["frames"]}
    # 큰 모델을 직접 제약에 맞춘 뒤(수평 + 수직축 회전·축척·이동)의 도면 좌표 지표: mapper 에 제약을 넣었든 안 넣었든 계산
    import copy
    mb = copy.deepcopy(big)
    ast = align_to_priors(mb, pri)
    ea = np.linalg.norm(mb["C"] - np.array([sc.gt[n][1] for n in mb["names"]]), axis=1)
    upa = np.array([R @ sc.gt[n][0].T @ np.array([0, 0, 1.0]) for n, R in zip(mb["names"], mb["R"])])
    res.update({"own_ok": int((ea <= 0.5).sum()), "own_bad": int((ea > 1.0).sum()), "own_median": float(np.median(ea)),
                "own_tilt": float(np.median(np.degrees(np.arccos(np.clip(upa[:, 2], -1, 1))))),
                "own_far": int((np.linalg.norm(mb["C"] - np.array([pri[n] for n in mb["names"]]), axis=1) > 1.0).sum())})
    if spec.get("prior"):
        seen, seenR = {}, {}
        for m in models:
            for n, C, R in zip(m["names"], m["C"], m["R"]):
                if n not in seen:
                    seen[n], seenR[n] = C, R
        an = sorted(seen)
        AC = np.array([seen[n] for n in an])
        # 도면 좌표에서의 방향 오차 (정렬 없이). 일직선 동선이면 복도 축 둘레 기울기를 위치 제약이 못 잡는다
        rr = [math.degrees(math.acos(max(-1, min(1, (np.trace(seenR[n].T @ sc.gt[n][0]) - 1) / 2)))) for n in an]
        res["plan_rot_median"] = float(np.median(rr))
        up = np.array([seenR[n] @ sc.gt[n][0].T @ np.array([0, 0, 1.0]) for n in an])   # 정답의 위쪽이 모델에서 어디로
        res["plan_tilt_median"] = float(np.median(np.degrees(np.arccos(np.clip(up[:, 2], -1, 1)))))
        e = np.linalg.norm(AC - np.array([sc.gt[n][1] for n in an]), axis=1)
        far = np.linalg.norm(AC - np.array([pri[n] for n in an]), axis=1) > 1.0
        eb = np.linalg.norm(big["C"] - np.array([sc.gt[n][1] for n in big["names"]]), axis=1)
        res.update({"plan_registered": len(an), "plan_ok": int((e <= 0.5).sum()), "plan_bad": int((e > 1.0).sum()),
                    "plan_median": float(np.median(e)), "plan_ok_big": int((eb <= 0.5).sum()),
                    "plan_bad_big": int((eb > 1.0).sum()),
                    "far_flagged": int(far.sum()), "far_caught": int((far & (e > 1)).sum()),
                    "plan_teleports": len(find_teleports(an, AC))})
    schemas.write_json(rd / "eval.json", res)
    return res


def fmt_table(rows: list[dict]) -> str:
    hdr = ("| run | 짝 (가짜) | 갈때-올때 진짜 짝 | 모델 크기 | 큰 모델: 맞음/틀림 (Sim3) | 모든 모델: 맞음/틀림 (Sim3, 모델별) |"
           " 도면좌표: 등록/맞음/틀림 (모든 모델) | 도면 오차 중앙 (m) | 도면 기울기 중앙 (°) | 순간이동 | 제약에서 >1m (그중 진짜 틀림) | 시간 정합+거르기 (s) |")
    lines = [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    for r in rows:
        if not r.get("registered"):
            lines.append(f"| {r['run']} | {r['pairs']} ({r['false_pairs']}) | {r['true_far_pairs']} | 없음 | | | | | | | | | {r['seconds']:.0f} |")
            continue
        if r.get("prior"):
            plan = f"{r['plan_registered']}/{r['plan_ok']}/{r['plan_bad']}"
            med = f"{r['plan_median']:.2f}"
            tilt = f"{r['plan_tilt_median']:.1f}"
            far = f"{r['far_flagged']} ({r['far_caught']})"
        else:
            plan = med = far = tilt = "-"
        lines.append(f"| {r['run']} | {r['pairs']} ({r['false_pairs']}) | {r['true_far_pairs']} | {r['model_sizes']} | "
                     f"{r['sim3_ok']}/{r['sim3_bad']} | {r['sim3_ok_all']}/{r['sim3_bad_all']} | {plan} | {med} | {tilt} | "
                     f"{r['teleports']} | {far} | {r['own_ok']}/{r['own_bad']}, {r['own_median']:.2f}, {r['own_tilt']:.1f} | "
                     f"{r['seconds']:.0f}+{r['filter_seconds']:.0f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, type=Path)
    ap.add_argument("--runs", default="")
    ap.add_argument("--stages", default="features,map,eval")
    a = ap.parse_args()
    sc = Scene(a.scene.resolve())
    st = a.stages.split(",")
    runs = [r for r in a.runs.split(",") if r]
    if "features" in st:
        stage_features(sc)
    if "map" in st:
        for r in runs:
            stage_map(sc, r)
    if "eval" in st:
        rows = [eval_run(sc, r) for r in runs if (sc.root / "pm" / r / "done.json").exists()]
        print(fmt_table(rows))


if __name__ == "__main__":
    main()
