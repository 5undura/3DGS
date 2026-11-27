"""두 실험의 평가 점수를 '같은 사진'끼리만 비교 (Brush eval.csv / LichtFeld per_image_metrics.json)
사용: python tools/compare_eval.py <결과폴더A> <결과폴더B> [...]
- Brush: <폴더>\\eval.csv (tools/eval_renders.py 결과)
- LichtFeld: <폴더>\\per_image_metrics.json 의 마지막 step (psnr, psnr_aligned)
"""
import csv
import json
import os
import sys

import numpy as np


def load(d):
    out = {}
    j = os.path.join(d, "per_image_metrics.json")
    c = os.path.join(d, "eval.csv")
    if os.path.exists(j):
        for name, v in json.load(open(j, encoding="utf-8")).items():
            ev = [e for e in v["evaluations"] if e.get("split") == "test"]
            if ev:
                e = max(ev, key=lambda e: e["step"])
                out[os.path.splitext(name)[0]] = (e["psnr"], e.get("psnr_aligned", float("nan")))
    elif os.path.exists(c):
        for r in csv.DictReader(open(c, encoding="utf-8")):
            out[r["image"]] = (float(r["psnr"]), float("nan"))
    else:
        sys.exit(f"평가 파일 없음: {d}")
    return out


runs = [(d, load(d)) for d in sys.argv[1:]]
common = sorted(set.intersection(*[set(r) for _, r in runs]))
print(f"공통 평가 사진 {len(common)}장")
for d, r in runs:
    p = np.array([r[k][0] for k in common]); a = np.array([r[k][1] for k in common])
    extra = "" if np.isnan(a).all() else f", 정렬 후 {np.nanmean(a):.2f}"
    print(f"  {os.path.basename(d.rstrip(os.sep))}: PSNR 평균 {p.mean():.2f} (중앙 {np.median(p):.2f}, 하위 10% {np.percentile(p, 10):.2f}){extra}")
if len(runs) == 2:
    diff = np.array([runs[1][1][k][0] - runs[0][1][k][0] for k in common])
    print(f"  B-A 차이: 평균 {diff.mean():+.2f}, B 가 나은 사진 {np.mean(diff > 0) * 100:.0f}%")
