"""도면 사진(피난 안내도 등) -> 벽 선분 + 방 후보 초안 (반자동, OpenCV 4.x).

1. 네 점 원근 보정: 사진 속 도면 네 모서리(왼위, 오른위, 오른아래, 왼아래)와 실제 크기(m) -> 위에서 본 이미지
2. 벽 선분: 이진화 -> Canny -> HoughLinesP -> 축 정렬 스냅 -> 같은 줄 합치기 -> 짧은 선 버림
3. 방 후보: 벽 마스크에서 글자·기호(작은 덩어리)를 지우고, 문 폭만큼 닫아 방을 나눈 뒤
   바깥과 이어지지 않은 빈 공간 덩어리 -> 다각형 (벽 안쪽 치수)

완전 자동이 목표가 아니다. 결과는 사람이 박스 편집기에서 고칠 초안이다.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .draft import DraftOptions, DraftResult, build_draft
from .geom import Pt, Seg


def read_image(path: str | Path) -> np.ndarray:
    """한글 경로에서도 읽히게 imdecode."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"이미지를 못 읽음: {path}")
    return img


def order_corners(pts) -> np.ndarray:
    """아무 순서 네 점 -> 왼위, 오른위, 오른아래, 왼아래 (이미지 좌표)."""
    p = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s, d = p.sum(1), np.diff(p, axis=1).ravel()
    return np.array([p[np.argmin(s)], p[np.argmin(d)], p[np.argmax(s)], p[np.argmax(d)]], dtype=np.float32)


def rectify(img: np.ndarray, corners, size_m: tuple[float, float], px_per_m: float = 50.0,
            reorder: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """네 점 원근 보정. corners = 사진 속 도면 네 모서리(px), size_m = 그 사각형의 실제 (가로, 세로) m.
    돌려줌: (보정 이미지, 3x3 호모그래피 사진->보정)."""
    src = order_corners(corners) if reorder else np.asarray(corners, dtype=np.float32).reshape(4, 2)
    w, h = int(round(size_m[0] * px_per_m)), int(round(size_m[1] * px_per_m))
    if w < 2 or h < 2:
        raise ValueError("보정 크기가 너무 작음 (size_m·px_per_m 확인)")
    # 네 점 = 보정 이미지의 바깥 테두리(픽셀 가장자리, OpenCV 픽셀 중심 좌표로 -0.5 와 w-0.5): 픽셀 i 의 중심이
    # (i + 0.5) / px_per_m m 가 되어 _to_m 과 맞음. 예전 (0, w-1) 은 y 로 1px(0.02m) 어긋났음
    dst = np.array([[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, h - 0.5], [-0.5, h - 0.5]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return out, H


def wall_mask(img: np.ndarray, block_m: float | None = None, px_per_m: float = 50.0) -> np.ndarray:
    """어두운 선 = 255. block_m 을 주면 적응형 이진화(조명 고르지 않은 사진), 아니면 Otsu."""
    # 밝기(GRAY) 대신 세 채널 중 최댓값(HSV 의 V): 초록 대피 화살표·빨간 기호는 GRAY 로는 어두워(110, 87)
    # Otsu 에서 벽으로 잡혀 복도를 막았음. V 로는 160·220 이라 밝은 쪽 (experiments/accuracy/plan_eval.py)
    gray = img if img.ndim == 2 else img.max(axis=2)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    if block_m:
        b = max(3, int(block_m * px_per_m) | 1)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, b, 10)
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return m


def _drop_small_blobs(mask: np.ndarray, min_px: float) -> np.ndarray:
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = np.maximum(st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]) >= min_px
    return np.where(keep[lab], 255, 0).astype(np.uint8)


def _to_m(p, h_px: int, ppm: float) -> Pt:
    return float(p[0]) / ppm, float(h_px - p[1]) / ppm


def merge_axis_segments(segs: list[Seg], line_tol: float, gap: float) -> list[Seg]:
    """가로·세로 선분을 같은 줄(line_tol 안)끼리 묶고, gap 안으로 이어지면 하나로."""
    hor, ver, other = [], [], []
    for a, b in segs:
        if abs(a[1] - b[1]) < 1e-9:
            hor.append((a[1], min(a[0], b[0]), max(a[0], b[0])))
        elif abs(a[0] - b[0]) < 1e-9:
            ver.append((a[0], min(a[1], b[1]), max(a[1], b[1])))
        else:
            other.append((a, b))

    def merge(items):
        # 긴 선부터, 그 선에서 line_tol 안의 남은 선을 한 묶음으로. 예전엔 좌표순으로 바로 앞 선과 line_tol 안이면
        # 이어 붙여서, 계단 빗금처럼 촘촘한 평행선이 사슬로 이어져 바깥벽까지 한 줄로 합쳐져 수 m 어긋났음
        groups: list[list] = []
        left = sorted(items, key=lambda it: -(it[2] - it[1]))
        while left:
            c0 = left[0][0]
            groups.append([it for it in left if abs(it[0] - c0) <= line_tol])
            left = [it for it in left if abs(it[0] - c0) > line_tol]
        out = []
        for g in groups:
            ivs = sorted((lo, hi, c) for c, lo, hi in g)
            cur = None
            for lo, hi, c in ivs:
                if cur and lo <= cur[1] + gap:
                    w0, w1 = cur[1] - cur[0], hi - lo
                    cur = [cur[0], max(cur[1], hi), (cur[2] * w0 + c * w1) / max(w0 + w1, 1e-9)]
                else:
                    if cur:
                        out.append(cur)
                    cur = [lo, hi, c]
            if cur:
                out.append(cur)
        return out

    res = [((lo, c), (hi, c)) for lo, hi, c in merge(hor)]
    res += [((c, lo), (c, hi)) for lo, hi, c in merge(ver)]
    return res + other


def detect_segments(img: np.ndarray, px_per_m: float, *, min_len_m: float = 0.5, axis_snap_deg: float = 5.0,
                    merge_tol_m: float = 0.25, merge_gap_m: float = 0.2, hough_threshold: int = 30,
                    canny: tuple[int, int] = (50, 150), mask: np.ndarray | None = None) -> list[Seg]:
    """벽 선분 후보 (m, 도면 좌표 y 위). 벽 두께 양쪽 선은 merge_tol_m 안이면 한 줄로 합쳐짐."""
    if mask is None:
        mask = _drop_small_blobs(wall_mask(img, px_per_m=px_per_m), min_len_m * px_per_m)
    # 테두리를 0 으로 2px 넓혀 Canny: 네 모서리를 바깥벽에 찍으면 바깥벽이 이미지 테두리에 붙어 바깥 면 선이
    # 안 나오고 안쪽 면만 남아 벽 반 두께(0.1m)만큼 안으로 들어왔음
    pad = 2
    edges = cv2.Canny(cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0), *canny)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, hough_threshold,
                            minLineLength=max(5, int(min_len_m * px_per_m * 0.5)),
                            maxLineGap=max(2, int(merge_gap_m * px_per_m)))
    h = mask.shape[0]
    segs: list[Seg] = []
    for x1, y1, x2, y2 in (lines.reshape(-1, 4) - pad if lines is not None else []):
        a, b = _to_m((x1 + 0.5, y1 + 0.5), h, px_per_m), _to_m((x2 + 0.5, y2 + 0.5), h, px_per_m)
        ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180
        if min(ang, 180 - ang) <= axis_snap_deg:
            y = (a[1] + b[1]) / 2
            a, b = (a[0], y), (b[0], y)
        elif abs(ang - 90) <= axis_snap_deg:
            x = (a[0] + b[0]) / 2
            a, b = (x, a[1]), (x, b[1])
        segs.append((a, b))
    segs = merge_axis_segments(segs, merge_tol_m, merge_gap_m)
    return [s for s in segs if math.dist(*s) >= min_len_m]


def detect_rooms(img: np.ndarray, px_per_m: float, *, door_close_m: float = 1.1, min_blob_m: float = 0.5,
                 min_area: float = 1.0, simplify_m: float = 0.1, mask: np.ndarray | None = None) -> list[list[Pt]]:
    """빈 공간 덩어리 -> 방 다각형 (m, 벽 안쪽 치수). 바깥(이미지 테두리에 닿은 공간)은 뺌."""
    if mask is None:
        mask = _drop_small_blobs(wall_mask(img, px_per_m=px_per_m), min_blob_m * px_per_m)
    k = max(3, int(door_close_m * px_per_m))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    free = cv2.bitwise_not(closed)
    n, lab, st, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    H, W = free.shape
    polys = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        if a / px_per_m ** 2 < min_area:
            continue
        comp = np.where(lab == i, 255, 0).astype(np.uint8)
        cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, max(1.0, simplify_m * px_per_m), True).reshape(-1, 2)
        if len(c) < 3:
            continue
        # 윤곽은 픽셀 중심이라 한 픽셀 작게 나옴: 바깥 경계 쪽으로 0.5px 보정은 무시 (오차 1/ppm m)
        polys.append([_to_m((px + 0.5, py + 0.5), H, px_per_m) for px, py in c])
    return polys


def import_image(image, opts: DraftOptions | None = None, *, corners=None, size_m: tuple[float, float] | None = None,
                 px_per_m: float = 50.0, door_close_m: float = 1.1, min_len_m: float = 0.5,
                 adaptive_block_m: float | None = None) -> DraftResult:
    """도면 사진(경로 또는 BGR 배열) -> DraftResult.

    corners·size_m 을 주면 원근 보정 후 px_per_m 로 다시 그림. 안 주면 이미지가 이미 위에서 본 것이고
    px_per_m 이 그 이미지 축척이어야 함. extra: rectified(보정 이미지), homography, segments_px 없음.
    """
    opts = opts or DraftOptions(boxes=True)
    img = read_image(image) if isinstance(image, (str, Path)) else image
    warnings: list[str] = []
    Hm = None
    if corners is not None:
        if size_m is None:
            raise ValueError("corners 를 주면 size_m(실제 가로, 세로 m)도 필요")
        img, Hm = rectify(img, corners, size_m, px_per_m)
    mask = wall_mask(img, adaptive_block_m, px_per_m)
    mask = _drop_small_blobs(mask, min_len_m * px_per_m)
    segs = detect_segments(img, px_per_m, min_len_m=min_len_m, mask=mask)
    rooms = detect_rooms(img, px_per_m, door_close_m=door_close_m, min_area=opts.min_area, mask=mask)
    if not segs:
        warnings.append("벽 선분을 못 찾음 (이진화·축척 확인)")
    warnings.append("사진 초안: 방 크기는 벽 안쪽 치수, 글자·기호가 벽으로 잡힐 수 있으니 확인할 것")
    source = {"kind": "image", "px_per_m": px_per_m}
    if corners is not None:
        source["corners"] = [[float(x), float(y)] for x, y in order_corners(corners)]
        source["size_m"] = [float(size_m[0]), float(size_m[1])]
    # 방은 빈 공간 덩어리에서 이미 닫힌 다각형으로 나옴. 선분은 밑그림·수정용으로만 돌려줌
    draft_opts = DraftOptions(**{**opts.__dict__, "drop_outer": False, "recenter": False})
    res = build_draft(rooms, [], draft_opts, source, warnings)
    res.segments = segs
    res.extra.update({"rectified": img, "homography": Hm, "wall_mask": mask})
    return res
