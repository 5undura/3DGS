"""평가 렌더와 원본 사진 비교 (저장소 tools/eval_renders.py 와 같은 정의).
원본을 렌더 크기로 줄여(INTER_AREA) 비교. SSIM: 가우시안 11x11, sigma 1.5, 채널 평균. 한글 경로 지원."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def imread(p: Path):
    import cv2
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def ssim(a, b) -> float:
    import cv2
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def blur(x):
        return cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_a, mu_b = blur(a), blur(b)
    va = blur(a * a) - mu_a ** 2
    vb = blur(b * b) - mu_b ** 2
    cov = blur(a * b) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return float(m.mean())


def psnr(a, b) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float(10 * np.log10(255 ** 2 / max(mse, 1e-10)))


def compare_dir(render_dir: Path, images_dir: Path) -> list[tuple[str, float, float]]:
    """render_dir/*.png 를 images_dir 의 같은 이름(확장자 무관) 사진과 비교 -> [(이름, psnr, ssim)]"""
    import cv2
    rows = []
    for r in sorted(Path(render_dir).glob("*.png")):
        gt = next((images_dir / (r.stem + e) for e in (".jpg", ".JPG", ".jpeg", ".png")
                   if (images_dir / (r.stem + e)).exists()), None)
        if gt is None:
            continue
        img, ref = imread(r), imread(gt)
        if img is None or ref is None:
            continue
        ref = cv2.resize(ref, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
        rows.append((r.stem, psnr(img, ref), ssim(img, ref)))
    return rows
