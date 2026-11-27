"""1. 추출 (담당 A). 영상 -> frames/*.jpg + frames/done.json

저장소 1_extract/pick_best.py 를 함수로 옮김 (동작 같음):
선명하고(국소 중앙값의 sharp_ratio 배 이상) 마지막으로 남긴 프레임과 충분히 다른(min_diff) 프레임만 남김.
max_gap 초 동안 아무것도 안 남으면 그 구간에서 가장 선명한 프레임을 강제로 남김(빈틈 방지).
- 저장은 cv2.imencode + tofile (cv2.imwrite 는 Windows 한글 경로에서 조용히 실패)
- 파일명 f_<순번 5자리>_<원본 프레임 번호 6자리>.jpg, done.json 에 fps·total_frames·count·args
- done.json 은 맨 마지막에 씀. done.json 없이 frames/ 가 있으면 끊긴 실행 -> frames_failed_<시각> 으로 치우고 다시
"""
from __future__ import annotations

import statistics
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .. import schemas
from ..colmap.runner import move_aside
from .base import Stage, StageContext, StageResult


@dataclass
class PickArgs:
    min_diff: float = 6.0        # 마지막으로 남긴 프레임과 평균 밝기 차 (0-255)
    sharp_ratio: float = 0.75    # 국소(약 2초) 선명도 중앙값의 이 배 미만은 버림
    max_gap: float = 1.0         # 초. 이만큼 아무것도 안 남으면 가장 선명한 것 강제로
    min_gap: int = 3             # 프레임. 이보다 가깝게는 안 남김
    abs_floor_pct: float = 0.0   # 영상 전체 선명도의 이 백분위 미만은 버림 (0 = 끔, 영상을 두 번 읽음)
    sharp_res: int = 960         # 선명도 잴 때 긴 변 (0 = 원본)
    jpeg_quality: int = 95


def _sharpness(frame, sharp_res: int):
    """(라플라시안 분산, 960px 회색 이미지). pick_best.sharpness 와 같음."""
    import cv2
    h, w = frame.shape[:2]
    s = 960.0 / max(h, w)
    gray = cv2.cvtColor(cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2GRAY)
    if sharp_res == 960:
        return cv2.Laplacian(gray, cv2.CV_64F).var(), gray
    if sharp_res <= 0 or sharp_res >= max(h, w):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        r = sharp_res / max(h, w)
        g = cv2.cvtColor(cv2.resize(frame, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var(), gray


def _open(video: str):
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없음: {video}")
    return cap


def _floor(video: str, pct: float, sharp_res: int) -> float:
    import numpy as np
    cap = _open(video)
    vals = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vals.append(_sharpness(frame, sharp_res)[0])
    cap.release()
    if not vals:
        raise RuntimeError(f"프레임 없음: {video}")
    return float(np.percentile(np.array(vals), pct))


def pick_frames(video: str, out_dir: Path, a: PickArgs,
                log: Callable[[str], None] = lambda m: None) -> dict:
    """영상에서 쓸 만한 프레임을 골라 out_dir 에 저장. done.json 에 들어갈 내용(dict)을 돌려줌."""
    import cv2
    import numpy as np
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = _open(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_gap = max(1, int(round(a.max_gap * fps)))
    hist: deque = deque(maxlen=max(1, int(2 * fps)))
    log(f"{video}: {fps:.3f} fps, {total} 프레임")
    floor = _floor(video, a.abs_floor_pct, a.sharp_res) if a.abs_floor_pct > 0 else None

    last_thumb, last_keep = None, -10 ** 9
    cand = None   # (선명도, 번호, 프레임, 썸네일): 마지막으로 남긴 뒤 가장 선명한 것
    idx = saved = forced = 0
    t0 = time.time()

    def save(frame, i):
        nonlocal saved
        saved += 1
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, a.jpeg_quality])
        if not ok:
            raise RuntimeError(f"JPEG 인코딩 실패 (프레임 {i})")
        buf.tofile(str(out_dir / f"f_{saved:05d}_{i:06d}.jpg"))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        sharp, gray = _sharpness(frame, a.sharp_res)
        hist.append(sharp)
        thumb = cv2.resize(gray, (160, int(160 * gray.shape[0] / gray.shape[1])),
                           interpolation=cv2.INTER_AREA).astype(np.int16)
        if cand is None or sharp > cand[0]:
            cand = (sharp, idx, frame, thumb)
        med = statistics.median(hist)
        sharp_ok = sharp >= a.sharp_ratio * med
        far_enough = idx - last_keep >= a.min_gap
        diff = 255.0 if last_thumb is None else float(np.abs(thumb - last_thumb).mean())
        above = floor is None or sharp >= floor
        if sharp_ok and far_enough and diff >= a.min_diff and above:
            save(frame, idx)
            last_thumb, last_keep, cand = thumb, idx, None
        elif idx - last_keep >= max_gap and cand is not None and (floor is None or cand[0] >= floor):
            save(cand[2], cand[1])
            forced += 1
            last_thumb, last_keep, cand = cand[3], cand[1], None
        idx += 1
        if idx % 3000 == 0:
            log(f"  {idx}/{total} 프레임, {saved}장 (강제 {forced}), {idx / (time.time() - t0):.1f} f/s")
    cap.release()
    if idx == 0:
        raise RuntimeError(f"프레임을 하나도 못 읽음: {video}")
    log(f"끝: {saved}장 (빈틈 메우려 강제 {forced}), {time.time() - t0:.0f}초")
    if saved > 3000:
        log("경고: 3000장 넘음 -> 정합이 느림. extract.min_diff 를 키울 것 (예: 9)")
    elif saved < 800:
        log("참고: 800장 미만. 정합이 끊기면 extract.min_diff 를 줄일 것 (예: 4)")
    return {"version": schemas.VERSION, "video": str(video), "fps": fps, "total_frames": idx,
            "count": saved, "forced": forced, "args": asdict(a)}


class Extract(Stage):
    name = "extract"

    def outputs(self, ctx: StageContext) -> list[Path]:
        return [ctx.paths.frames_done]

    def prev_targets(self, ctx: StageContext) -> list[Path]:
        return [ctx.paths.frames]

    def run(self, ctx: StageContext) -> StageResult:
        video = ctx.section.video
        if not Path(video).exists():
            return StageResult(False, f"영상 없음: {video}")
        frames = ctx.paths.frames
        if frames.exists() and any(frames.iterdir()):
            old = move_aside(frames)
            ctx.log(f"끝나지 않은 frames -> {old.name} 로 옮기고 다시")
        e = ctx.project.cfg.extract
        args = PickArgs(min_diff=e.min_diff, sharp_ratio=e.sharp_ratio)
        try:
            done = pick_frames(video, frames, args, ctx.log)
        except Exception as ex:  # noqa: BLE001
            old = move_aside(frames)
            return StageResult(False, f"추출 실패: {ex} (중간 결과 {old.name if old else '-'})")
        if done["count"] == 0:
            old = move_aside(frames)
            return StageResult(False, f"남긴 프레임 0장 (중간 결과 {old.name if old else '-'})")
        schemas.write_json(ctx.paths.frames_done, done)
        return StageResult(True, f"{done['count']}장 / {done['total_frames']} 프레임 ({done['fps']:.3f} fps)",
                           [ctx.paths.frames_done])
