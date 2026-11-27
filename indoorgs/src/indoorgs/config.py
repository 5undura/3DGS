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
    snap: bool = True                # 학습 후 벽 근처 스플랫을 면에 붙이고 납작하게 -> splat_walls.ply
    band: float = 0.03               # 면에서 이 거리(m) 안의 스플랫만 붙임 (액자 등 입체감 보호, 지운 의견 ±3 cm)
    color_tol: float = 0.12          # 그 자리 벽 색과 채널 최대 차이가 이보다 크면 안 붙임 (0..1, 벽에 붙은 물체 보호)
    remove_behind: bool = True       # 벽 뒤(문 구멍 아닌 곳) 부유물 지우기
    search: float = 0.30             # 도면 면에서 이 거리(m) 안의 정합 점으로 실제 면을 찾음
    repair: bool = True              # snap 뒤 면에서 비치거나 빈 칸을 둘레 스플랫 복사로 메움 -> splat_repaired.ply
    repair_min_cover: float = 0.5    # 덮임(0..1)이 이보다 낮은 칸을 메움
    repair_max_hole_m2: float = 4.0  # 이보다 큰 구멍은 안 메움 (못 본 넓은 면을 지어내지 않도록)
    texture: bool = True             # 메울 색을 면 텍스처에서: 실제 사진 투영 > PatchMatch 메우기 (walls/tex/*.png, 고쳐도 됨)


CLEANUP_STEPS = ("floaters", "huge", "needles", "sh")   # walls/cleanup.STEPS 와 같은 이름


@dataclass
class CleanupCfg:
    """학습 후 자동 정리 (모든 구간, snap 뒤·repair 앞). stages/snap.Cleanup, walls/cleanup.py.
    바늘 쪼개기로 스플랫 수가 늘어남 (기본값에서 바늘 1개 -> 보통 7~8 조각). 증가는 clean.json·보고서에 기록만."""
    enabled: bool = True             # splat_clean.ply 를 만들고 repair·fill·merge 가 그것을 씀
    steps: list[str] = field(default_factory=lambda: list(CLEANUP_STEPS))   # 차례대로. 빼면 그 단계 안 함


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_ENUMS = {
    ("gpu_profile",): {"8gb", "12gb"},
    ("register", "matcher"): {"sequential", "exhaustive"},
    ("register", "mapper"): {"incremental", "global"},
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
    )
    extra = set(data) - {"sections", "tools", "gpu_profile", "extract", "register", "train", "fill", "walls", "cleanup"}
    if extra:
        raise ConfigError(f"모르는 최상위 항목 {sorted(extra)}")
    st = cfg.cleanup.steps
    if not isinstance(st, list) or any(x not in CLEANUP_STEPS for x in st) or len(set(st)) != len(st):
        raise ConfigError(f"cleanup.steps = {st!r}: {list(CLEANUP_STEPS)} 중에서 겹치지 않게 (목록)")
    for path, allowed in _ENUMS.items():
        v = cfg
        for p in path:
            v = getattr(v, p)
        if v not in allowed:
            raise ConfigError(f"{'.'.join(path)} = {v!r}: {sorted(allowed)} 중 하나")
    return cfg


def load(path: str | Path) -> ProjectConfig:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        return parse(yaml.safe_load(f) or {})
