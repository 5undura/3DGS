"""프로젝트·구간 폴더 경로 (CONTRACT.md 2절). 경로는 여기서만 만든다."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import config as config_mod


@dataclass(frozen=True)
class SectionPaths:
    root: Path

    @property
    def path_json(self) -> Path: return self.root / "path.json"
    @property
    def frames(self) -> Path: return self.root / "frames"
    @property
    def frames_done(self) -> Path: return self.frames / "done.json"
    @property
    def colmap(self) -> Path: return self.root / "colmap"
    @property
    def priors(self) -> Path: return self.colmap / "priors.json"
    @property
    def db(self) -> Path: return self.colmap / "db.db"
    @property
    def sparse(self) -> Path: return self.colmap / "sparse"
    @property
    def data(self) -> Path: return self.root / "data"
    @property
    def data_model(self) -> Path: return self.data / "sparse" / "0"
    @property
    def diagnose(self) -> Path: return self.root / "diagnose.json"
    @property
    def data_clean(self) -> Path: return self.root / "data_clean"
    @property
    def train(self) -> Path: return self.root / "train"
    def train_run(self, trainer: str, preset: str) -> Path: return self.train / f"{trainer}_{preset}"
    @property
    def fill(self) -> Path: return self.root / "fill"
    @property
    def fill_ply(self) -> Path: return self.fill / "holes_backdrop.ply"


class Project:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.cfg = config_mod.load(self.config_path)

    @property
    def config_path(self) -> Path: return self.root / "project.yaml"
    @property
    def plan_json(self) -> Path: return self.root / "floorplan" / "plan.json"
    @property
    def plan_image(self) -> Path: return self.root / "floorplan" / "plan.png"
    @property
    def merged(self) -> Path: return self.root / "merged"
    @property
    def building_ply(self) -> Path: return self.merged / "building.ply"
    @property
    def layout_json(self) -> Path: return self.merged / "layout.json"
    @property
    def report(self) -> Path: return self.root / "report.html"
    @property
    def logs(self) -> Path: return self.root / "logs"

    @property
    def has_space(self) -> bool:
        return self.plan_json.exists()

    def section(self, name: str) -> SectionPaths:
        names = [s.name for s in self.cfg.sections]
        if name not in names:
            raise KeyError(f"구간 없음: {name} (있는 것: {names})")
        return SectionPaths(self.root / "sections" / name)

    def sections(self) -> list[tuple[config_mod.Section, SectionPaths]]:
        return [(s, self.section(s.name)) for s in self.cfg.sections]

    def warnings(self) -> list[str]:
        out = []
        if not str(self.root).isascii():
            out.append(f"프로젝트 경로에 영문이 아닌 글자가 있음: {self.root} (COLMAP·OpenCV 가 실패할 수 있음)")
        return out
