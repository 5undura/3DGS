"""동선(path.json) -> 프레임별 대략 위치 (priors.json). CONTRACT.md 4절.

선(stroke)을 따라 걸은 거리 s 로 위치를 정하고, 고정점(anchor)의 (시간 t, 거리 s) 사이를
일정한 속도로 이어 s(t) 를 만든다. 첫 고정점 전·마지막 뒤는 그 자리에 멈춘 것으로 본다.
"""
from __future__ import annotations

import bisect
import math
import re
from dataclasses import dataclass
from pathlib import Path

from .schemas import CameraPath, SchemaError, VERSION

FRAME_RE = re.compile(r"^f_(\d+)_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


@dataclass
class Polyline:
    pts: list[tuple[float, float]]

    def __post_init__(self):
        self.cum = [0.0]
        for (x1, y1), (x2, y2) in zip(self.pts, self.pts[1:]):
            self.cum.append(self.cum[-1] + math.hypot(x2 - x1, y2 - y1))
        if self.length <= 0:
            raise SchemaError("동선 길이가 0")

    @property
    def length(self) -> float:
        return self.cum[-1]

    def at(self, s: float) -> tuple[float, float]:
        s = min(max(s, 0.0), self.length)
        i = max(1, bisect.bisect_left(self.cum, s))
        i = min(i, len(self.pts) - 1)
        seg = self.cum[i] - self.cum[i - 1]
        u = 0.0 if seg == 0 else (s - self.cum[i - 1]) / seg
        (x1, y1), (x2, y2) = self.pts[i - 1], self.pts[i]
        return x1 + u * (x2 - x1), y1 + u * (y2 - y1)

    def project(self, p: tuple[float, float]) -> tuple[float, float]:
        """점에서 선까지 가장 가까운 곳의 (거리 s, 떨어진 거리)."""
        best = (0.0, math.inf)
        px, py = p
        for i in range(len(self.pts) - 1):
            (x1, y1), (x2, y2) = self.pts[i], self.pts[i + 1]
            dx, dy = x2 - x1, y2 - y1
            L2 = dx * dx + dy * dy
            u = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L2))
            qx, qy = x1 + u * dx, y1 + u * dy
            dist = math.hypot(px - qx, py - qy)
            if dist < best[1]:
                best = (self.cum[i] + u * math.sqrt(L2), dist)
        return best


class PathModel:
    """시간 t(초) -> 도면 좌표 (x, y, z)."""

    def __init__(self, path: CameraPath, snap_warn_m: float = 1.0):
        self.path = path
        pts = path.stroke if path.stroke is not None else [a.point for a in path.anchors]
        self.line = Polyline(list(pts))
        self.warnings: list[str] = []
        knots = []
        for a in sorted(path.anchors, key=lambda a: a.t):
            if a.s is not None:
                s = a.s
                if not 0 <= s <= self.line.length + 1e-6:
                    raise SchemaError(f"고정점 t={a.t}: s={s} 가 선 길이 {self.line.length:.2f} 밖")
            else:
                s, dist = self.line.project(a.point)
                if dist > snap_warn_m:
                    self.warnings.append(f"고정점 t={a.t}: 선에서 {dist:.2f}m 떨어져 있음 (가까운 곳으로 붙임)")
            knots.append((a.t, s))
        for (t1, s1), (t2, s2) in zip(knots, knots[1:]):
            if t2 == t1 and s2 != s1:
                raise SchemaError(f"같은 시간 t={t1} 에 다른 위치 두 개")
            if s2 < s1 - 1e-6:
                self.warnings.append(f"t={t1}->{t2}: 선을 거꾸로 감 ({s1:.2f}->{s2:.2f}m). 선을 걸은 순서대로 그렸는지 확인")
        self.ts = [k[0] for k in knots]
        self.ss = [k[1] for k in knots]

    def s_at(self, t: float) -> float:
        ts, ss = self.ts, self.ss
        if t <= ts[0]:
            return ss[0]
        if t >= ts[-1]:
            return ss[-1]
        i = bisect.bisect_right(ts, t)
        t1, t2, s1, s2 = ts[i - 1], ts[i], ss[i - 1], ss[i]
        return s1 if t2 == t1 else s1 + (t - t1) / (t2 - t1) * (s2 - s1)

    def position(self, t: float) -> tuple[float, float, float]:
        x, y = self.line.at(self.s_at(t))
        return x, y, self.path.camera_height


def frame_time(filename: str, fps: float) -> float:
    """f_<순번>_<원본 프레임 번호>.jpg -> 초."""
    m = FRAME_RE.match(Path(filename).name)
    if not m:
        raise ValueError(f"프레임 이름 형식이 아님: {filename} (f_00001_000123.jpg)")
    if fps <= 0:
        raise ValueError("fps 가 0 이하")
    return int(m.group(2)) / fps


def build_priors(path: CameraPath, frame_names: list[str], fps: float) -> tuple[dict, list[str]]:
    """priors.json 내용과 경고 목록."""
    model = PathModel(path)
    t_end = model.ts[-1]
    frames, late = [], 0
    for name in sorted(frame_names):
        t = frame_time(name, fps)
        if t > t_end + 1.0:
            late += 1
        x, y, z = model.position(t)
        frames.append({"image": Path(name).name, "t": round(t, 3),
                       "xyz": [round(x, 4), round(y, 4), round(z, 4)],
                       "sigma": [path.sigma_xy, path.sigma_xy, path.sigma_z]})
    warnings = list(model.warnings)
    if late:
        warnings.append(f"마지막 고정점(t={t_end}) 뒤 프레임 {late}장: 끝 위치에 멈춘 것으로 둠. 끝 고정점을 더 찍을 것")
    return {"version": VERSION, "coordinate_system": "CARTESIAN", "frames": frames}, warnings
