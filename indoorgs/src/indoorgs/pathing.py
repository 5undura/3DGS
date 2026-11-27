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

FORWARD_TOL = 0.5      # m. 고정점을 선에 붙일 때 앞으로 가는 후보를 이만큼 더 멀어도 고름
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

    def project(self, p: tuple[float, float], s_min: float | None = None) -> tuple[float, float]:
        """점에서 선까지 가장 가까운 곳의 (거리 s, 떨어진 거리). s_min 을 주면 s >= s_min 인 부분에서만."""
        best = (0.0, math.inf)
        px, py = p
        for i in range(len(self.pts) - 1):
            if s_min is not None and self.cum[i + 1] < s_min:
                continue
            (x1, y1), (x2, y2) = self.pts[i], self.pts[i + 1]
            dx, dy = x2 - x1, y2 - y1
            L2 = dx * dx + dy * dy
            lo = 0.0
            if s_min is not None and L2 > 0 and s_min > self.cum[i]:
                lo = min(1.0, (s_min - self.cum[i]) / math.sqrt(L2))
            u = lo if L2 == 0 else max(lo, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L2))
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
                if knots and s < knots[-1][1]:
                    # 방을 한 바퀴 돌아 문으로 나오기·왕복처럼 선이 제 자신 가까이 지나가면 가장 가까운 곳이
                    # 앞서 지난 부분일 수 있음 -> 앞 고정점 뒤쪽에 거의 같은 거리(FORWARD_TOL) 후보가 있으면 그쪽
                    s2, d2 = self.line.project(a.point, s_min=knots[-1][1])
                    if d2 <= dist + FORWARD_TOL:
                        s, dist = s2, d2
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


# ---------------- 브러시 선 정리 (GUI 용) ----------------
def simplify_rdp(pts: list[tuple[float, float]], eps: float = 0.05) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker. eps(m) 보다 덜 벗어난 점을 없앰. 양 끝은 남김."""
    pts = [(float(x), float(y)) for x, y in pts]
    if len(pts) < 3:
        return pts
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        (x1, y1), (x2, y2) = pts[i], pts[j]
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy)
        best, k = -1.0, -1
        for m in range(i + 1, j):
            px, py = pts[m]
            d = math.hypot(px - x1, py - y1) if L == 0 else abs(dy * px - dx * py + x2 * y1 - y2 * x1) / L
            if d > best:
                best, k = d, m
        if k >= 0 and best > eps:
            keep[k] = True
            stack += [(i, k), (k, j)]
    return [p for p, f in zip(pts, keep) if f]


def resample(pts: list[tuple[float, float]], step: float = 0.25) -> list[tuple[float, float]]:
    """선을 따라 step(m) 간격으로 다시 찍음. 꺾인 점(원래 꼭짓점)은 남기지 않고 끝점은 남김."""
    if step <= 0:
        raise ValueError("step 은 양수")
    line = Polyline(dedupe(pts))
    n = max(1, int(math.ceil(line.length / step - 1e-9)))
    return [line.at(line.length * i / n) for i in range(n + 1)]


def dedupe(pts: list[tuple[float, float]], tol: float = 1e-6) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for x, y in pts:
        p = (float(x), float(y))
        if not out or math.dist(out[-1], p) > tol:
            out.append(p)
    return out


def clean_stroke(pts: list[tuple[float, float]], eps: float = 0.05, step: float | None = None,
                 min_gap: float = 0.02) -> list[tuple[float, float]]:
    """브러시 원본 점 -> 저장용 선. 너무 붙은 점 제거 -> RDP 단순화 -> (step 이 있으면) 재표본."""
    p = dedupe(pts, min_gap)
    if len(p) < 2:
        raise SchemaError("동선은 서로 다른 점 2개 이상")
    p = simplify_rdp(p, eps)
    return resample(p, step) if step else p


# ---------------- 고정점 편집 (GUI 용). 새 CameraPath 를 돌려줌 ----------------
def _copy(path: CameraPath, anchors) -> CameraPath:
    return CameraPath(sorted(anchors, key=lambda a: a.t), None if path.stroke is None else list(path.stroke),
                      path.camera_height, path.sigma_xy, path.sigma_z)


def add_anchor(path: CameraPath, t: float, point: tuple[float, float] | None = None, s: float | None = None,
               snap: bool = True) -> CameraPath:
    """고정점 추가 (t 순으로 정렬). snap 이고 선이 있으면 point 를 선 위 가장 가까운 점으로 옮겨 저장."""
    from .schemas import Anchor
    if (point is None) == (s is None):
        raise SchemaError("point 와 s 중 하나만")
    if point is not None and snap and path.stroke is not None:
        line = Polyline(list(path.stroke))
        point = line.at(line.project(point)[0])
    a = Anchor(float(t), None if point is None else (float(point[0]), float(point[1])), None if s is None else float(s))
    return _copy(path, list(path.anchors) + [a])


def move_anchor(path: CameraPath, index: int, t: float | None = None, point: tuple[float, float] | None = None,
                s: float | None = None, snap: bool = True) -> CameraPath:
    """index 번째(t 순) 고정점의 시간·위치 바꾸기."""
    from .schemas import Anchor
    anchors = sorted(path.anchors, key=lambda a: a.t)
    old = anchors[index]
    if point is not None and s is not None:
        raise SchemaError("point 와 s 중 하나만")
    new_point, new_s = old.point, old.s
    if point is not None:
        if snap and path.stroke is not None:
            line = Polyline(list(path.stroke))
            point = line.at(line.project(point)[0])
        new_point, new_s = (float(point[0]), float(point[1])), None
    elif s is not None:
        new_point, new_s = None, float(s)
    anchors[index] = Anchor(old.t if t is None else float(t), new_point, new_s)
    return _copy(path, anchors)


def remove_anchor(path: CameraPath, index: int) -> CameraPath:
    anchors = sorted(path.anchors, key=lambda a: a.t)
    anchors.pop(index)
    return _copy(path, anchors)


def path_to_json(path: CameraPath) -> dict:
    """CameraPath -> path.json 내용 (schemas.parse_path 로 다시 읽힘)."""
    d = {"version": VERSION, "camera_height": path.camera_height, "sigma_xy": path.sigma_xy, "sigma_z": path.sigma_z}
    if path.stroke is not None:
        d["stroke"] = [[round(x, 4), round(y, 4)] for x, y in path.stroke]
    anchors = []
    for a in sorted(path.anchors, key=lambda a: a.t):
        e = {"t": a.t}
        if a.point is not None:
            e["point"] = [round(a.point[0], 4), round(a.point[1], 4)]
        else:
            e["s"] = round(a.s, 4)
        anchors.append(e)
    d["anchors"] = anchors
    return d


def save_path(path: CameraPath, file) -> Path:
    from .schemas import parse_path, write_json
    data = path_to_json(path)
    parse_path(data)
    write_json(file, data)
    return Path(file)


# ---------------- 검사 ----------------
def _inside_plan(plan, x: float, y: float, tol: float) -> bool:
    from .space import geom
    for r in plan.rooms:
        if r.contains(x, y) or geom.dist_to_boundary((x, y), r.floor) <= tol:
            return True
    return False


def outside_segments(path: CameraPath, plan, step: float = 0.1, tol: float = 0.05,
                     min_len: float = 0.0) -> list[dict]:
    """동선이 어느 방에도 안 들어가는 구간. [{"s0", "s1", "start": [x,y], "end": [x,y], "length"}].
    방 경계에서 tol(m) 안은 안으로 봄 (문 지나는 곳)."""
    pts = path.stroke if path.stroke is not None else [a.point for a in path.anchors]
    line = Polyline(list(pts))
    n = max(1, int(math.ceil(line.length / step)))
    out, cur = [], None
    for i in range(n + 1):
        s = line.length * i / n
        x, y = line.at(s)
        bad = not _inside_plan(plan, x, y, tol)
        if bad and cur is None:
            cur = [s, s]
        elif bad:
            cur[1] = s
        if (not bad or i == n) and cur is not None:
            out.append(cur)
            cur = None
    res = []
    for s0, s1 in out:
        if s1 - s0 + 1e-9 >= min_len:
            a, b = line.at(s0), line.at(s1)
            res.append({"s0": round(s0, 3), "s1": round(s1, 3), "start": [round(a[0], 3), round(a[1], 3)],
                        "end": [round(b[0], 3), round(b[1], 3)], "length": round(s1 - s0, 3)})
    return res


def check_duration(path: CameraPath, video_duration: float, tol: float = 2.0) -> list[str]:
    """영상 길이(초)와 고정점 시간 비교 경고."""
    ts = sorted(a.t for a in path.anchors)
    w = []
    if ts[0] > tol:
        w.append(f"첫 고정점이 t={ts[0]}초: 영상 시작({tol}초 안)에 고정점을 찍을 것 (그 전은 첫 위치에 멈춘 것으로 봄)")
    if ts[-1] < video_duration - tol:
        w.append(f"마지막 고정점 t={ts[-1]}초 < 영상 길이 {video_duration:.1f}초: 끝에 고정점을 더 찍을 것")
    if ts[-1] > video_duration + tol:
        w.append(f"마지막 고정점 t={ts[-1]}초 > 영상 길이 {video_duration:.1f}초: 시간 확인")
    return w


def path_warnings(path: CameraPath, plan=None, video_duration: float | None = None,
                  outside_min_len: float = 0.3) -> list[str]:
    """GUI·prior 공용 경고 묶음: 고정점(PathModel), 방 밖 구간, 영상 길이."""
    w = list(PathModel(path).warnings)
    if plan is not None:
        for seg in outside_segments(path, plan, min_len=outside_min_len):
            w.append(f"동선이 방 밖으로 나감: s={seg['s0']}~{seg['s1']}m ({seg['start']} -> {seg['end']}, {seg['length']}m)")
    if video_duration is not None:
        w += check_duration(path, video_duration)
    return w
