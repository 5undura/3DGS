"""Brush 평가 렌더(--eval-save-to-disk)로 PSNR / SSIM 계산
Brush CLI 는 평가 점수를 출력하지 않음 -> eval_<iter>\\*.png 와 원본 사진(data\\images)을 비교.

사용:
  python tools/eval_renders.py <export_path>\\eval_30000 <data 폴더> [--csv out.csv]
- 원본 사진을 렌더 크기로 줄여서(INTER_AREA) 비교. 한글 경로 지원(np.fromfile + imdecode)
- SSIM: 가우시안 11x11, sigma 1.5, 채널 평균 (일반 3DGS 논문 방식과 같은 정의)
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np


def imread(p):
    a = np.fromfile(p, np.uint8)
    return cv2.imdecode(a, cv2.IMREAD_COLOR)


def ssim(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_a, mu_b = blur(a), blur(b)
    va = blur(a * a) - mu_a ** 2
    vb = blur(b * b) - mu_b ** 2
    cov = blur(a * b) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return float(m.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    ap.add_argument("data")
    ap.add_argument("--csv")
    a = ap.parse_args()
    renders = sorted(glob.glob(os.path.join(a.eval_dir, "*.png")))
    if not renders:
        sys.exit(f"렌더 없음: {a.eval_dir}")
    rows = []
    for r in renders:
        stem = os.path.splitext(os.path.basename(r))[0]
        gt_path = None
        for ext in (".jpg", ".JPG", ".png", ".jpeg"):
            p = os.path.join(a.data, "images", stem + ext)
            if os.path.exists(p):
                gt_path = p
                break
        if gt_path is None:
            print(f"원본 없음: {stem}")
            continue
        img = imread(r)
        gt = imread(gt_path)
        gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
        mse = np.mean((img.astype(np.float64) - gt.astype(np.float64)) ** 2)
        psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-10))
        rows.append((stem, psnr, ssim(img, gt)))
    ps = np.array([x[1] for x in rows])
    ss = np.array([x[2] for x in rows])
    print(f"평가 {len(rows)}장, 크기 {img.shape[1]}x{img.shape[0]}")
    print(f"PSNR 평균 {ps.mean():.2f} (중앙 {np.median(ps):.2f}, 하위 10% {np.percentile(ps, 10):.2f})")
    print(f"SSIM 평균 {ss.mean():.4f}")
    worst = sorted(rows, key=lambda x: x[1])[:8]
    print("가장 낮은 사진: " + ", ".join(f"{s}({p:.1f})" for s, p, _ in worst))
    if a.csv:
        with open(a.csv, "w", encoding="utf-8") as f:
            f.write("image,psnr,ssim\n")
            for s, p, q in rows:
                f.write(f"{s},{p:.3f},{q:.4f}\n")


if __name__ == "__main__":
    main()
