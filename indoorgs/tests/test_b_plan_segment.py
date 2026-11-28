"""B: 대피도 사진 영역 분할 (plan_segment) · 합성 대피도 (plan_synth) · 마법봉 API."""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from indoorgs.space import import_image, plan_segment as ps
from indoorgs.space.plan_synth import layout, photograph, render

PPM = 20.0


def _iou(poster, floors):
    FW, FH = poster.size_m
    shape = next(iter(poster.truth.values()))[1].shape
    m = np.zeros(shape, np.uint8)
    for f in floors:
        q = np.array([[x / poster.truth_res, (FH - y) / poster.truth_res] for x, y in f])
        cv2.fillPoly(m, [np.round(q).astype(np.int32)], 1)
    return m.astype(bool)


def _best(poster, res):
    rooms = {r.id: r for r in res.model.rooms}
    det = [_iou(poster, [rooms[i].floor for i in it["rooms"]]) for it in res.extra["items"]]
    out = {}
    for gid, (kind, g) in poster.truth.items():
        if kind in ("room", "corridor"):
            out[gid] = max(((g & d).sum() / (g | d).sum() for d in det), default=0.0)
    return out


@pytest.fixture(scope="module")
def t_thin():
    p = render("T", "thin", seed=0)
    ph, c = photograph(p, seed=0)
    return p, ph, c


def test_synth_truth():
    p = render("L", "thick", seed=1)
    kinds = [k for k, _ in p.truth.values()]
    assert kinds.count("corridor") == 1 and kinds.count("room") == 9
    corr = p.truth["corr"][1]
    # ㄱ자 복도 = 가로 28 x 2.5 + 세로 2.5 x 13.5 에서 벽 두께만큼 줄어든 넓이 (0.05m 칸)
    assert corr.sum() * p.truth_res ** 2 == pytest.approx(27.7 * 2.2 + 2.2 * 13.5, rel=0.06)
    assert len(layout("hall").doors) == 17


def test_segment_photo_T_thin(t_thin):
    p, ph, c = t_thin
    res = import_image(ph, corners=c + 2.0, size_m=p.size_m, px_per_m=PPM)
    items = res.extra["items"]
    assert sum(1 for x in items if x["kind"] == "corridor") == 1
    assert sum(1 for x in items if x["kind"] == "room") == 9
    corr = next(x for x in items if x["kind"] == "corridor")
    assert len(corr["rooms"]) == 2                                   # T 자 -> 박스 2개
    best = _best(p, res)
    assert min(best.values()) > 0.8, best
    assert all(0 <= x["conf"] <= 1 for x in items)
    # 문: 정답 문 중심 0.8m 안에 대부분
    found = [d.center for d in res.model.doors]
    hit = sum(1 for q in p.doors_m if any(np.hypot(q[0] - f[0], q[1] - f[1]) < 0.8 for f in found))
    assert hit >= 0.8 * len(p.doors_m)
    # 한 복도 박스끼리는 통로로 이어짐
    assert any(set(d.rooms) == set(corr["rooms"]) for d in res.model.doors)
    assert res.model.source["method"] == "segment"


def test_segment_colour_corridor_and_legend():
    p = render("L", "thick", seed=1)
    ph, c = photograph(p, seed=1)
    res = import_image(ph, corners=c, size_m=p.size_m, px_per_m=PPM)
    seg = res.extra["seg"]
    assert seg["colour_mode"]                                         # 색칠한 복도
    best = _best(p, res)
    assert best["corr"] > 0.85 and min(best.values()) > 0.85, best
    # 범례·행동요령 상자는 건물 밖이라 방이 아님: 찾은 항목 수 = 정답 10
    assert len(res.extra["items"]) == 10


def test_segment_hatch_dropped():
    p = render("hall", "thin", seed=0)
    ph, c = photograph(p, seed=0)
    res = import_image(ph, corners=c, size_m=p.size_m, px_per_m=PPM)
    assert res.extra["seg"]["dropped"]["hatch"] >= 2
    assert any("빗금" in w for w in res.warnings)


def test_binary_method_kept(t_thin):
    p, ph, c = t_thin
    res = import_image(ph, corners=c, size_m=p.size_m, px_per_m=PPM, method="binary")
    assert "items" not in res.extra and res.model.rooms
    with pytest.raises(ValueError):
        import_image(ph, corners=c, size_m=p.size_m, px_per_m=PPM, method="nope")


def test_boxes_from_mask_L():
    m = np.zeros((200, 300), bool)
    m[20:60, 20:280] = True          # 가로
    m[60:180, 230:280] = True        # 세로
    boxes = ps.boxes_from_mask(m, 20.0)
    assert len(boxes) == 2
    cov = np.zeros_like(m, np.uint8)
    for c0, r0, c1, r1 in boxes:
        cov[r0:r1, c0:c1] += 1
    assert cov.max() == 1                                             # 겹치지 않음
    assert (cov.astype(bool) & m).sum() / m.sum() > 0.97


def test_magic_wand(t_thin):
    p, ph, c = t_thin
    res = import_image(ph, corners=c, size_m=p.size_m, px_per_m=PPM)
    img = res.extra["rectified"]
    prep = res.extra["seg"]["prep"]
    H = img.shape[0]
    # 방 r6 (B106) 가운데: 글자 위를 찍어도 방 전체
    g = p.truth["r6"][1]
    ys, xs = np.nonzero(g)
    xm, ym = xs.mean() * p.truth_res, p.size_m[1] - ys.mean() * p.truth_res
    it = ps.magic_wand(prep, (xm * PPM, H - ym * PPM))
    assert it.kind == "room" and len(it.boxes) == 1
    x0, y0, x1, y1 = ps.px_box_to_m(it.boxes[0], H, PPM)
    iou = (_iou(p, [[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]) & g).sum() / (_iou(p, [[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]) | g).sum()
    assert iou > 0.85
    with pytest.raises(ValueError):                                   # 건물 밖 = 닫히지 않음
        ps.magic_wand(prep, (5, 5))


# ---------------- GUI API ----------------
fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient  # noqa: E402

from indoorgs.gui.server import create_app  # noqa: E402

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo"


def test_api_import_and_wand(tmp_path, t_thin):
    p, ph, c = t_thin
    root = tmp_path / "demo"
    shutil.copytree(DEMO, root)
    cl = TestClient(create_app(root))
    ok, buf = cv2.imencode(".jpg", ph)
    opt = {"corners": c.tolist(), "size_m": list(p.size_m), "px_per_m": PPM}
    r = cl.post("/api/plan/import", files={"file": ("대피도.jpg", buf.tobytes())}, data={"options": json.dumps(opt)})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["plan"]["source"]["method"] == "segment" and len(j["items"]) == 10
    assert {"id", "kind", "conf", "rooms"} <= set(j["items"][0])
    g = p.truth["r1"][1]
    ys, xs = np.nonzero(g)
    pt = [float(xs.mean() * p.truth_res), float(p.size_m[1] - ys.mean() * p.truth_res)]
    w = cl.post("/api/plan/wand", json={**opt, "source_file": j["source_file"], "point": pt})
    assert w.status_code == 200, w.text
    wj = w.json()
    assert wj["kind"] == "room" and len(wj["rooms"]) == 1 and 0 < wj["conf"] <= 1
    (cx, cy), (sw, sh) = wj["rooms"][0]["box"]["center"], wj["rooms"][0]["box"]["size"]
    assert abs(cx - pt[0]) < 0.6 and abs(cy - pt[1]) < 0.6 and sw * sh > 20
    bad = cl.post("/api/plan/wand", json={**opt, "source_file": "없음.jpg", "point": pt})
    assert bad.status_code == 400
    old = cl.post("/api/plan/import", files={"file": ("대피도.jpg", buf.tobytes())},
                  data={"options": json.dumps({**opt, "method": "binary"})})
    assert old.status_code == 200 and "items" not in old.json()
