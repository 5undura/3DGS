"""도면 사진 불러오기 정확도: 합성 비상대피도 사진 -> 방·복도 IoU, 예전(binary) vs 새(segment).

python experiments/plan_import/run.py [출력폴더]   (기본: experiments/plan_import/out)
정답은 indoorgs.space.plan_synth (안쪽 테두리 왼아래 = 0,0). 네 점 = 사진 속 안쪽 테두리 모서리 + 클릭 오차 ±3px.
실제 사진(학생회관 지하)은 REAL 경로가 있으면 오버레이만 만듦 (정답 없음).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from indoorgs.space import import_image                      # noqa: E402
from indoorgs.space.plan_synth import LAYOUTS, photograph, render   # noqa: E402

PPM = 20.0


def raster(polys, shape, res, FH):
    m = np.zeros(shape, np.uint8)
    for p in polys:
        q = np.array([[x / res, (FH - y) / res] for x, y in p])
        cv2.fillPoly(m, [np.round(q).astype(np.int32)], 1)
    return m.astype(bool)


def detected_items(res):
    """[(kind, [floor 다각형...])]: segment 면 항목별, binary 면 방 하나 = 항목 하나."""
    rooms = {r.id: r for r in res.model.rooms}
    if "items" in res.extra:
        return [(x["kind"], [rooms[i].floor for i in x["rooms"]]) for x in res.extra["items"]]
    return [("room", [r.floor]) for r in res.model.rooms]


def score(poster, res):
    FW, FH = poster.size_m
    shape = next(iter(poster.truth.values()))[1].shape
    det = [(k, raster(ps, shape, poster.truth_res, FH)) for k, ps in detected_items(res)]
    gts = [(i, k, m) for i, (k, m) in poster.truth.items() if k in ("room", "corridor")]
    pairs = []
    for gi, (i, k, g) in enumerate(gts):
        for di, (_, d) in enumerate(det):
            inter = (g & d).sum()
            if inter:
                pairs.append((inter / (g | d).sum(), gi, di))
    pairs.sort(reverse=True)
    best, used = {}, set()
    for v, gi, di in pairs:
        if gi in best or di in used:
            continue
        best[gi] = v
        used.add(di)
    per = {gts[gi][0]: round(float(best.get(gi, 0.0)), 3) for gi in range(len(gts))}
    room_iou = [best.get(gi, 0.0) for gi, (_, k, _) in enumerate(gts) if k == "room"]
    corr_iou = [best.get(gi, 0.0) for gi, (_, k, _) in enumerate(gts) if k == "corridor"]
    # 문: 정답 문 중심 0.8m 안에 찾은 문
    dr = None
    if res.model.doors or "items" in res.extra:
        found = [d.center for d in res.model.doors]
        hit = sum(1 for p in poster.doors_m if any(np.hypot(p[0] - q[0], p[1] - q[1]) < 0.8 for q in found))
        dr = round(hit / max(1, len(poster.doors_m)), 3)
    return {"room_iou": round(float(np.mean(room_iou)), 3), "corr_iou": round(float(np.mean(corr_iou)), 3),
            "rooms_ok": sum(1 for v in room_iou if v >= 0.7), "n_rooms": len(room_iou),
            "corr_ok": sum(1 for v in corr_iou if v >= 0.7), "n_corr": len(corr_iou),
            "extra": len(det) - len(used), "door_recall": dr, "per": per}


def overlay(img, res, path, title=""):
    """보정 이미지 위에 찾은 박스 (복도 주황, 방 파랑, 믿음 낮음 빨강) + 문 (초록 점)."""
    ppm = res.model.source["px_per_m"]
    out = img.copy()
    h = img.shape[0]
    meta = {r: x for x in res.extra.get("items", []) for r in x["rooms"]}
    lay = out.copy()
    for r in res.model.rooms:
        x = meta.get(r.id, {"kind": "room", "conf": 1.0})
        col = (0, 140, 255) if x["kind"] == "corridor" else (220, 120, 30)
        if x["conf"] < 0.6:
            col = (40, 40, 230)
        q = np.array([[px * ppm, h - py * ppm] for px, py in r.floor])
        cv2.fillPoly(lay, [np.round(q).astype(np.int32)], col)
        cv2.polylines(out, [np.round(q).astype(np.int32)], True, col, 2)
    out = cv2.addWeighted(lay, 0.35, out, 0.65, 0)
    for r in res.model.rooms:
        x = meta.get(r.id)
        c = r.center
        txt = r.id + (f" {int(x['conf'] * 100)}%" if x else "")
        cv2.putText(out, txt, (int(c[0] * ppm) - 30, int(h - c[1] * ppm)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(out, txt, (int(c[0] * ppm) - 30, int(h - c[1] * ppm)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    for d in res.model.doors:
        cv2.circle(out, (int(d.center[0] * ppm), int(h - d.center[1] * ppm)), 5, (0, 200, 0), -1)
    if title:
        cv2.putText(out, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imwrite(str(path), out)


def run(out_dir: Path, seeds=(0, 1)):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for style in ("thin", "thick"):
        for name in LAYOUTS:
            for sd in seeds:
                poster = render(name, style, seed=sd)
                ph, c = photograph(poster, seed=sd)
                click = c + np.random.default_rng(100 + sd).uniform(-3, 3, c.shape).astype(np.float32)
                row = {"case": f"{name}/{style}/s{sd}"}
                for meth in ("binary", "segment"):
                    t = time.time()
                    res = import_image(ph, corners=click, size_m=poster.size_m, px_per_m=PPM, method=meth)
                    row[meth] = score(poster, res) | {"sec": round(time.time() - t, 2)}
                    if sd == seeds[0]:
                        overlay(res.extra["rectified"], res, out_dir / f"{name}_{style}_{meth}.jpg", f"{name} {style} {meth}")
                if sd == seeds[0]:
                    cv2.imwrite(str(out_dir / f"{name}_{style}_photo.jpg"), ph)
                b, s = row["binary"], row["segment"]
                print(f"{row['case']:16s} before room {b['room_iou']:.2f} corr {b['corr_iou']:.2f} extra {b['extra']:2d} | "
                      f"after room {s['room_iou']:.2f} corr {s['corr_iou']:.2f} ok {s['rooms_ok']}/{s['n_rooms']} "
                      f"extra {s['extra']:2d} door {s['door_recall']} {s['sec']}s", flush=True)
                rows.append(row)
    (out_dir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return rows


REAL = Path("/tmp/claude-0/-home-user-3DGS/5698f89c-b779-591a-9900-84c73f83daf3/scratchpad/plan_import_real/student_hall_b1_evac.jpg")
# 안쪽 테두리선 네 모서리 (사진 px, 눈으로 찍음) + 추정 실제 크기 (20m/25m 글자로 대략 10px/m -> 88 x 56 m, 축척은 추정)
REAL_CORNERS = [[615, 506], [1490, 516], [1521, 1039], [588, 1060]]
REAL_SIZE = (88.0, 56.0)


def run_real(out_dir: Path, path: Path = REAL):
    if not path.exists():
        print("실제 사진 없음:", path)
        return None
    out = {}
    for meth in ("binary", "segment"):
        res = import_image(path, corners=REAL_CORNERS, size_m=REAL_SIZE, px_per_m=PPM, method=meth)
        overlay(res.extra["rectified"], res, out_dir / f"real_{meth}.jpg", f"student hall B1 {meth}")
        items = res.extra.get("items", [{"kind": "room"} for _ in res.model.rooms])
        out[meth] = {"rooms": sum(1 for x in items if x["kind"] == "room"),
                     "corridors": sum(1 for x in items if x["kind"] == "corridor"),
                     "boxes": len(res.model.rooms), "doors": len(res.model.doors), "warnings": res.warnings}
        print("real", meth, out[meth])
    cv2.imwrite(str(out_dir / "real_rectified.jpg"), res.extra["rectified"])
    cv2.imwrite(str(out_dir / "real_wallmask.png"), res.extra["wall_mask"])
    (out_dir / "real.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    od = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "out"
    od.mkdir(parents=True, exist_ok=True)
    run_real(od)
    if "--real-only" not in sys.argv:
        run(od)
