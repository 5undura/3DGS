"""project.yaml 읽기·검사. 필드와 기본값은 이 파일이 기준 (CONTRACT.md 4절)."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


@dataclass
class Tools:
    colmap: str | None = None       # COLMAP 4.x (Windows CUDA 빌드의 COLMAP.bat)
    brush: str | None = None        # brush_app.exe (기본 학습기)
    lichtfeld: str | None = None    # LichtFeld-Studio.exe (있으면 고화질 옵션)
    python: str = "python"


@dataclass
class Section:
    name: str
    video: str


@dataclass
class ExtractCfg:
    min_diff: float = 4.0
    sharp_ratio: float = 0.8


@dataclass
class RegisterCfg:
    matcher: str = "sequential"      # sequential | exhaustive
    mapper: str = "incremental"      # incremental | global  (복도 1 실험: global 은 못 씀)
    use_floorplan_prior: bool = True  # plan.json + path.json 이 있을 때만 의미 있음
    camera_model: str = "OPENCV"
    seq_overlap: int = 20             # sequential 일 때 앞뒤 몇 장과 매칭
    loop_detection: bool = True       # sequential 일 때 같은 곳 다시 지나면 연결 (어휘 트리 필요)
    vocab_tree: str | None = None     # 어휘 트리 파일 (없으면 COLMAP 이 인터넷에서 받음, 오프라인이면 루프 검출 없이 다시)
    # 두 번 정합 + 동선 자세로 짝 검증 (위치 제약이 있을 때만. experiments/prior_matching/RESULTS.md).
    # 1차: 앞뒤 seq_overlap 장 짝 + pose_prior_mapper -> 후보 짝 추가(spatial, matcher=exhaustive 면 exhaustive)
    # -> DB 사본에서 1차 자세와 안 맞는 짝 삭제 -> 2차 정합 -> 수평 맞추고 제약에 다시 맞춤. 끄면 예전 한 번 정합
    pair_verify: bool = True
    verify_max_distance: float = 12.0  # 후보 짝: 제약 위치 이 거리(m) 안 (6 m·80장은 부족했음, 8 m 까지는 됨)
    verify_max_neighbors: int = 300    # 후보 짝: 사진마다 가까운 순 최대 장수 (COLMAP 은 0 을 못 받음)
    verify_seq_keep: int = 10          # 이름 순 앞뒤 이 장수 안의 짝은 검사 안 하고 둠
    verify_rot_tol_deg: float = 30.0   # 상대 회전이 1차 자세와 이보다 다르면 삭제
    verify_dir_tol_deg: float = 45.0   # 이동 방향이 1차 자세와 이보다 다르면 삭제
    verify_max_far_m: float = 1.5      # 1차에서 제약과 이보다 먼 사진은 기준에서 뺌
    gimbal_level: bool = True          # 화면 수평 유지(짐벌) 가정으로 모델을 수평 맞춤. 손으로 기울여 찍은 영상이면 끔
    # 속도 (experiments/e2e_v4/SPEED.md). 1차는 짝 검증용 기준 자세만 있으면 되므로 N장마다만 정합하고 사이는 보간 가능
    pass1_stride: int = 3              # 1차 pose_prior_mapper 를 N장마다 (1 = 모두, 예전). 사이 사진은 같은 모델 안 앞뒤로 보간(간격 2N 이하만)
    pass2_mapper: str = "incremental"  # incremental | global (2차를 global_mapper 로. 끝나면 수평 맞추고 제약에 다시 맞춤)


@dataclass
class TrainCfg:
    trainer: str = "brush"           # brush | lichtfeld
    preset: str = "default"


@dataclass
class FillCfg:
    holes_only: bool = True          # 벽 바탕층은 빼고 바닥·천장 구멍만 (복도 1 결론)


@dataclass
class WallsCfg:
    """벽 평탄화 (도면 좌표로 정합됐을 때만, = pose_prior_mapper). stages/walls.py, stages/snap.py"""
    enabled: bool = True             # 정합 점으로 도면 면 위치 다시 맞추기 (walls/planes.json)
    seed: bool = True                # 면 위에 초기 점 심기 -> data_seed/ 로 학습 (흰 벽 구멍·안개)
    seed_spacing: float = 0.08       # 심는 점 간격 (m)
    snap: bool = False               # 학습 후 벽 근처 스플랫을 면에 붙임 -> splat_walls.ply. 안 적으면 parse 가 finetune 켜짐(Brush)+seed 일 때만 켬 (RESULTS2: 재학습 없이 붙이면 손해)
    band: float = 0.03               # 면에서 이 거리(m) 안의 스플랫만 붙임 (액자 등 입체감 보호, 지운 의견 ±3 cm)
    color_tol: float = 0.12          # 그 자리 벽 색과 채널 최대 차이가 이보다 크면 안 붙임 (0..1, 벽에 붙은 물체 보호)
    flatten: bool = False            # snap 때 법선 방향으로 납작하게도 (False = 옮기기만). 안 적으면 snap 과 같은 규칙 (재학습과 함께면 납작하게가 이득, RESULTS2 E)
    remove_behind: bool = False      # snap 때 벽 뒤(문 구멍 아닌 곳) 부유물 지우기 (RESULTS2: 학습된 벽 일부를 지워 비침 -> 기본 끔)
    search: float = 0.30             # 도면 면에서 이 거리(m) 안의 정합 점으로 실제 면을 찾음
    trust_plan_dims: bool = False    # 줄자로 잰 도면 치수를 믿음: 점이 없어 못 맞춘 벽을 맞춘 평행 벽 + 도면 폭으로 추정 ("infer")
    scale_search: bool = True        # 못 맞춘 먼 면(끝 벽)을 맞춘 면 축척으로 옮긴 뒤 거리에 비례해 넓혀 다시 찾음 ("fit_scaled", e2e_v4)
    seed_scaled: bool = False        # "fit_scaled" 면에도 초기 점 심기 (넓힌 탐색이라 다른 면을 잡을 위험 -> 기본 끔)
    repair: bool = False             # snap 뒤 면에서 비치거나 빈 칸을 둘레 스플랫 복사로 메움 -> splat_repaired.ply (snap 필요, 기본 끔)
    repair_min_cover: float = 0.5    # 덮임(0..1)이 이보다 낮은 칸을 메움
    repair_max_hole_m2: float = 4.0  # 이보다 큰 구멍은 안 메움 (못 본 넓은 면을 지어내지 않도록)
    texture: bool = True             # 메울 색을 면 텍스처에서: 실제 사진 투영 > PatchMatch 메우기 (walls/tex/*.png, 고쳐도 됨)


CLEANUP_STEPS = ("floaters", "huge", "needles", "sh")   # walls/cleanup.STEPS 와 같은 이름


@dataclass
class CleanupCfg:
    """학습 후 자동 정리 (모든 구간, snap 뒤·repair 앞). stages/snap.Cleanup, walls/cleanup.py.
    바늘 쪼개기로 스플랫 수가 늘어남 (기본값에서 바늘 1개 -> 보통 7~8 조각). 증가는 clean.json·보고서에 기록만."""
    enabled: bool = True             # splat_clean.ply 를 만들고 repair·fill·merge 가 그것을 씀
    steps: list[str] = field(default_factory=lambda: ["floaters"])   # 차례대로 (CLEANUP_STEPS 중). RESULTS2: floaters 만 이득, 나머지는 영향 없음


FINETUNE_TRAINERS = ("brush",)   # finetune 을 지원하는 학습기 (trainers/*.finetune_unsupported == "")


@dataclass
class FinetuneCfg:
    """후처리(snap·cleanup·repair·fill) 뒤 짧은 재학습. stages/finetune.py. Brush 만 (LichtFeld 는 건너뜀).
    처리한 PLY 를 init.ply 로 두고 원래 반복 번호에서 이어 iters 반복, 밀도 조절 없음, 위치·크기 학습률 고정.
    wall_ab RESULTS2 E: 후처리 + 1000 반복 > raw + 같은 재학습 (600 반복 이상에서)."""
    enabled: bool = True             # 켜면 merge 가 train/<run>/finetune/splat_ft.ply 를 씀 (유효할 때)
    iters: int = 1000                # 300 은 후처리 손해를 다 못 메움 (RESULTS2 E)
    lr_mean: float = 1e-6            # 위치 학습률 (시작 = 끝 = 이 값). Brush 기본 일정의 끝 값 (--lr-mean-end 기본)
    lr_scale: float = 6e-3           # 크기 학습률 (시작 = 끝). Brush 기본 일정의 끝 값 (--lr-scale-end 기본)
    refine: bool = False             # True 면 죽은 스플랫 지우고 다시 뽑기(Brush refine)는 둠. 늘리기는 어느 쪽이든 없음


@dataclass
class ProjectConfig:
    sections: list[Section]
    tools: Tools = field(default_factory=Tools)
    gpu_profile: str = "8gb"         # 8gb | 12gb
    extract: ExtractCfg = field(default_factory=ExtractCfg)
    register: RegisterCfg = field(default_factory=RegisterCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    fill: FillCfg = field(default_factory=FillCfg)
    walls: WallsCfg = field(default_factory=WallsCfg)
    cleanup: CleanupCfg = field(default_factory=CleanupCfg)
    finetune: FinetuneCfg = field(default_factory=FinetuneCfg)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_ENUMS = {
    ("gpu_profile",): {"8gb", "12gb"},
    ("register", "matcher"): {"sequential", "exhaustive"},
    ("register", "mapper"): {"incremental", "global"},
    ("register", "pass2_mapper"): {"incremental", "global"},
    ("train", "trainer"): {"brush", "lichtfeld"},
}


def _build(cls, data: dict | None, where: str):
    data = dict(data or {})
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"{where}: 모르는 항목 {sorted(unknown)} (가능: {sorted(known)})")
    return cls(**data)


def parse(data: dict) -> ProjectConfig:
    if not isinstance(data, dict):
        raise ConfigError("project.yaml 최상위는 표(매핑)여야 함")
    raw_sections = data.get("sections") or []
    if not raw_sections:
        raise ConfigError("sections 가 비어 있음: 구간을 하나 이상 적어야 함")
    sections, names = [], set()
    for i, s in enumerate(raw_sections):
        sec = _build(Section, s, f"sections[{i}]")
        if not sec.name.replace("_", "").replace("-", "").isalnum() or not sec.name.isascii():
            raise ConfigError(f"sections[{i}].name '{sec.name}': 영문·숫자·_·- 만 (폴더 이름이 됨)")
        if sec.name in names:
            raise ConfigError(f"구간 이름 중복: {sec.name}")
        names.add(sec.name)
        sections.append(sec)
    cfg = ProjectConfig(
        sections=sections,
        tools=_build(Tools, data.get("tools"), "tools"),
        gpu_profile=data.get("gpu_profile", "8gb"),
        extract=_build(ExtractCfg, data.get("extract"), "extract"),
        register=_build(RegisterCfg, data.get("register"), "register"),
        train=_build(TrainCfg, data.get("train"), "train"),
        fill=_build(FillCfg, data.get("fill"), "fill"),
        walls=_build(WallsCfg, data.get("walls"), "walls"),
        cleanup=_build(CleanupCfg, data.get("cleanup"), "cleanup"),
        finetune=_build(FinetuneCfg, data.get("finetune"), "finetune"),
    )
    extra = set(data) - {"sections", "tools", "gpu_profile", "extract", "register", "train", "fill", "walls", "cleanup",
                         "finetune"}
    if extra:
        raise ConfigError(f"모르는 최상위 항목 {sorted(extra)}")
    ft = cfg.finetune
    for k in ("lr_mean", "lr_scale"):          # PyYAML 은 1e-6 (점 없음) 을 글자로 읽음
        v = getattr(ft, k)
        if isinstance(v, str):
            try:
                setattr(ft, k, float(v))
            except ValueError:
                pass
    if not isinstance(ft.iters, int) or ft.iters < 1 or not all(
            isinstance(v, (int, float)) and v > 0 for v in (ft.lr_mean, ft.lr_scale)):
        raise ConfigError(f"finetune: iters 는 1 이상 정수, lr_mean·lr_scale 은 0 보다 큰 수 ({ft})")
    # walls.snap·flatten 을 project.yaml 에 안 적었으면 재학습 여부로 정함 (RESULTS2 E: snap+납작 은 재학습 없이 −4.4 dB,
    # seed + 재학습과 함께면 raw+재학습보다 테스트 +0.24, 스침 +0.92 dB. seed 없으면 손해 -> seed 도 켜져 있어야).
    # remove_behind·repair 는 그대로 끔 (적어 넣은 값은 항상 그대로)
    ft_on = bool(ft.enabled and cfg.train.trainer in FINETUNE_TRAINERS and cfg.walls.seed)
    given = data.get("walls") or {}
    for k in ("snap", "flatten"):
        if k not in given:
            setattr(cfg.walls, k, ft_on)
    st = cfg.cleanup.steps
    if not isinstance(st, list) or any(x not in CLEANUP_STEPS for x in st) or len(set(st)) != len(st):
        raise ConfigError(f"cleanup.steps = {st!r}: {list(CLEANUP_STEPS)} 중에서 겹치지 않게 (목록)")
    for path, allowed in _ENUMS.items():
        v = cfg
        for p in path:
            v = getattr(v, p)
        if v not in allowed:
            raise ConfigError(f"{'.'.join(path)} = {v!r}: {sorted(allowed)} 중 하나")
    st = cfg.register.pass1_stride
    if isinstance(st, bool) or not isinstance(st, int) or st < 1:
        raise ConfigError(f"register.pass1_stride = {st!r}: 1 이상 정수")
    return cfg


def load(path: str | Path) -> ProjectConfig:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        return parse(yaml.safe_load(f) or {})
