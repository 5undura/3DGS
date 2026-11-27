"""정확도 측정 공용: src 경로 추가, 통계 출력."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = Path(__file__).resolve().parent / "out"


def stats(a) -> dict:
    a = np.asarray(a, float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "median": round(float(np.median(a)), 3), "p95": round(float(np.percentile(a, 95)), 3),
            "max": round(float(a.max()), 3), "mean": round(float(a.mean()), 3)}


def fmt(d: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in d.items())
