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

    def train_runs(self, trainer: str, preset: str) -> list[tuple[int, Path]]:
        """있는 학습 폴더 [(번호, 경로)] 번호순. <trainer>_<preset> = 1, _2, _3 ...
        실패(_failed_<시각>)·이전(_prev_<시각>) 폴더는 이름이 숫자로 안 끝나므로 빠짐."""
        base = self.train_run(trainer, preset)
        out = []
        if self.train.is_dir():
            for d in self.train.iterdir():
                if not d.is_dir():
                    continue
                if d.name == base.name:
                    out.append((1, d))
                elif d.name.startswith(base.name + "_") and d.name[len(base.name) + 1:].isdigit():
                    n = int(d.name[len(base.name) + 1:])
                    if n >= 2:
                        out.append((n, d))
        return sorted(out)

    def latest_train_run(self, trainer: str, preset: str) -> Path:
        """최신 학습 폴더 (fill·merge·report·train 이 같이 씀): splat.ply 가 있는 것 중 번호가 가장 큰 것.
        없으면 번호가 가장 큰 폴더(끝나지 않은 것), 그것도 없으면 기본 폴더 (아직 없을 수 있음)."""
        runs = self.train_runs(trainer, preset)
        done = [d for _, d in runs if (d / "splat.ply").exists()]
        if done:
            return done[-1]
        return runs[-1][1] if runs else self.train_run(trainer, preset)

    def new_train_run(self, trainer: str, preset: str) -> Path:
        """다시 학습할 새 폴더: 기존 가장 큰 번호 + 1 (하나도 없으면 기본 폴더). 기존 폴더는 안 건드림."""
        runs = self.train_runs(trainer, preset)
        if not runs:
            return self.train_run(trainer, preset)
        return self.train / f"{trainer}_{preset}_{runs[-1][0] + 1}"
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
