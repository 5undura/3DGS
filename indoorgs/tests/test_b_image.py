import math
import cv2
import numpy as np
import pytest

from indoorgs.space import import_image
from indoorgs.space.image_import import order_corners

PPM = 40  # 도면 그림 축척 px/m


def draw_plan():
    """12x8m 도면: 복도(아래 12x2.5), 방 두 개(위 6x5.5, 6x5.5), 문 틈, 글자. 벽 두께 0.2m."""
    W, H = 12 * PPM + 80, 8 * PPM + 80
    img = np.full((H, W, 3), 255, np.uint8)
    o = 40
    t = int(0.2 * PPM)

    def P(x, y):  # 도면 m -> 픽셀 (y 위)
        return int(o + x * PPM), int(H - o - y * PPM)

    def wall(a, b):
        cv2.line(img, P(*a), P(*b), (0, 0, 0), t)

    wall((0, 0), (12, 0)); wall((12, 0), (12, 8)); wall((12, 8), (0, 8)); wall((0, 8), (0, 0))
    # 복도 윗벽 y=2.5, 문 틈 두 곳 (x 2~3, 8~9)
    wall((0, 2.5), (2, 2.5)); wall((3, 2.5), (8, 2.5)); wall((9, 2.5), (12, 2.5))
    wall((6, 2.5), (6, 8))
    cv2.putText(img, "B117", P(1.5, 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return img


def test_image_topdown():
    res = import_image(draw_plan(), px_per_m=PPM)
    rooms = sorted(res.model.rooms, key=lambda r: r.area)
    assert len(rooms) == 3, [r.area for r in rooms]
    # 벽 안쪽 치수: 방 (6-0.2)x(5.5-0.2) ≈ 30.7, 복도 (12-0.2)x(2.5-0.2) ≈ 27.1
    assert sorted(round(r.area) for r in rooms) == pytest.approx([27, 31, 31], abs=2)
    assert len(res.segments) >= 5
    hv = [s for s in res.segments if s[0][0] == s[1][0] or s[0][1] == s[1][1]]
    assert len(hv) == len(res.segments)                      # 모두 축 정렬


def test_image_perspective():
    plan = draw_plan()
    h, w = plan.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[60, 40], [w + 20, 90], [w - 10, h + 60], [30, h + 10]])
    photo = cv2.warpPerspective(plan, cv2.getPerspectiveTransform(src, dst), (w + 80, h + 120),
                                borderValue=(200, 200, 200))
    # 사진 속 도면 바깥벽 모서리(=도면 0,8 / 12,8 / 12,0 / 0,0) 를 찍었다고 봄
    o = 40
    corners_plan = np.float32([[o, o], [o + 12 * PPM, o], [o + 12 * PPM, o + 8 * PPM], [o, o + 8 * PPM]])
    corners_photo = cv2.perspectiveTransform(corners_plan[None], cv2.getPerspectiveTransform(src, dst))[0]
    shuffled = corners_photo[[2, 0, 3, 1]]
    assert np.allclose(order_corners(shuffled), corners_photo)
    # 바깥벽 중심선 사각형 12x8m 를 0.5m 여백 두고 보정하려면 여백만큼 키운 네 점이 필요 -> 여기선 벽 중심 그대로
    res = import_image(photo, corners=shuffled, size_m=(12, 8), px_per_m=50)
    # 바깥벽이 보정 이미지 테두리에 걸리므로 방은 테두리 안쪽이어야 함: 테두리에 닿는 덩어리는 빠짐
    areas = sorted(round(r.area) for r in res.model.rooms)
    assert res.model.source["kind"] == "image" and len(res.model.source["corners"]) == 4
    assert len(areas) == 3, areas
    assert areas == pytest.approx([27, 31, 31], abs=2.5)
    assert res.extra["rectified"].shape[:2] == (400, 600)


# ---- experiments/accuracy/plan_eval.py 에서 찾은 것 ----
def _box_plan(ppm=50, arrow=False, hatch=False):
    """10x4m 바깥벽(두께 0.2) 하나, 바깥 면이 그림 테두리 = 네 점을 바깥 모서리에 찍은 것과 같음."""
    W, H = int(10.2 * ppm), int(4.2 * ppm)
    img = np.full((H, W, 3), 250, np.uint8)
    t = int(0.2 * ppm)
    cv2.rectangle(img, (0, 0), (W - 1, H - 1), (30, 30, 30), -1)
    cv2.rectangle(img, (t, t), (W - 1 - t, H - 1 - t), (205, 245, 205), -1)   # 옅은 초록 방
    if arrow:   # 초록 대피 화살표 (밝기 GRAY 110): 방 가운데, 벽에서 1m 안
        cv2.arrowedLine(img, (int(8 * ppm), H // 2), (int(2 * ppm), H // 2), (40, 160, 40), 9, tipLength=0.2)
    if hatch:   # 계단 빗금: 윗벽 아래로 0.2m 간격 가는 선 8줄
        for k in range(1, 9):
            y = t + int(k * 0.2 * ppm)
            cv2.line(img, (int(7 * ppm), y), (int(9.5 * ppm), y), (60, 60, 60), 2)
    return img


def test_image_colored_arrow_is_not_wall():
    res = import_image(_box_plan(arrow=True), px_per_m=50)
    areas = [r.area for r in res.model.rooms]
    assert len(areas) == 1 and areas[0] == pytest.approx(9.8 * 3.8, rel=0.03), areas


def test_image_outer_wall_on_border_is_centered():
    # 바깥벽 중심선 = 0.1m 안쪽. 예전엔 안쪽 면(0.2m)만 잡혀 0.1m 어긋났음
    res = import_image(_box_plan(), px_per_m=50)
    ver = sorted(s[0][0] for s in res.segments if s[0][0] == s[1][0])
    hor = sorted(s[0][1] for s in res.segments if s[0][1] == s[1][1])
    assert ver[0] == pytest.approx(0.1, abs=0.03) and ver[-1] == pytest.approx(10.1, abs=0.03), ver
    assert hor[0] == pytest.approx(0.1, abs=0.03) and hor[-1] == pytest.approx(4.1, abs=0.03), hor


def test_image_hatching_does_not_drag_wall():
    # 촘촘한 평행선이 사슬로 묶여 윗벽(y=4.1)이 빗금 쪽으로 끌려 내려갔음
    res = import_image(_box_plan(hatch=True), px_per_m=50)
    top = [s for s in res.segments if s[0][1] == s[1][1] and math.dist(*s) > 8]
    assert any(abs(s[0][1] - 4.1) < 0.03 for s in top), top
