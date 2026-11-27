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


@dataclass
class TrainCfg:
    trainer: str = "brush"           # brush | lichtfeld
    preset: str = "default"


@dataclass
class FillCfg:
    holes_only: bool = True          # 벽 바탕층은 빼고 바닥·천장 구멍만 (복도 1 결론)


@dataclass
class ProjectConfig:
    sections: list[Section]
    tools: Tools = field(default_factory=Tools)
    gpu_profile: str = "8gb"         # 8gb | 12gb
    extract: ExtractCfg = field(default_factory=ExtractCfg)
    register: RegisterCfg = field(default_factory=RegisterCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    fill: FillCfg = field(default_factory=FillCfg)

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
    )
    extra = set(data) - {"sections", "tools", "gpu_profile", "extract", "register", "train", "fill"}
    if extra:
        raise ConfigError(f"모르는 최상위 항목 {sorted(extra)}")
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
