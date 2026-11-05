"""
Keep every *useful* frame of a video: sharp enough AND different enough
from the last kept frame. No fixed frames-per-second.

Usage:
    python pick_best.py 4_sdr.mp4 frames4_best
    python pick_best.py 4_sdr.mp4 frames4_best --min-diff 20      (fewer images)
    python pick_best.py 4_sdr.mp4 frames4_best --min-diff 4      (more images)

Options:
    --sharp-ratio R  drop frames whose sharpness < R x local median (default 0.75)
    --min-diff D     keep a frame only if it differs from the last kept frame
                     by at least D (mean absolute difference, 0-255 scale; default 6)
    --max-gap S      if nothing was kept for S seconds, force-keep the sharpest
                     frame of that stretch so there are no holes (default 1.0)
    --min-gap N      never keep two frames closer than N frames apart (default 3)
"""
import argparse
import os
import statistics
import time
from collections import deque

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out_dir")
    ap.add_argument("--sharp-ratio", type=float, default=0.75)
    ap.add_argument("--min-diff", type=float, default=6.0)
    ap.add_argument("--max-gap", type=float, default=1.0)
    ap.add_argument("--min-gap", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_gap = max(1, int(round(a.max_gap * fps)))
    hist = deque(maxlen=int(2 * fps))  # ~2 s of sharpness values
    print(f"{a.video}: {fps:.2f} fps, {total} frames")

    last_thumb, last_keep = None, -10**9
    cand = None  # (score, idx, frame, thumb) best frame since last keep
    idx = saved = forced = 0
    t0 = time.time()

    def save(frame, i):
        nonlocal saved
        saved += 1
        cv2.imwrite(os.path.join(a.out_dir, f"f_{saved:05d}_{i:06d}.jpg"),
                    frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        s = 960.0 / max(h, w)
        gray = cv2.cvtColor(cv2.resize(frame, (int(w * s), int(h * s)),
                                       interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
        hist.append(sharp)
        thumb = cv2.resize(gray, (160, int(160 * gray.shape[0] / gray.shape[1])),
                           interpolation=cv2.INTER_AREA).astype(np.int16)

        if cand is None or sharp > cand[0]:
            cand = (sharp, idx, frame, thumb)

        med = statistics.median(hist)
        sharp_ok = sharp >= a.sharp_ratio * med
        far_enough = idx - last_keep >= a.min_gap
        diff = 255.0 if last_thumb is None else float(np.abs(thumb - last_thumb).mean())

        if sharp_ok and far_enough and diff >= a.min_diff:
            save(frame, idx)
            last_thumb, last_keep, cand = thumb, idx, None
        elif idx - last_keep >= max_gap and cand is not None:
            save(cand[2], cand[1])
            forced += 1
            last_thumb, last_keep, cand = cand[3], cand[1], None

        idx += 1
        if idx % 3000 == 0:
            el = time.time() - t0
            print(f"  frame {idx}/{total}  kept {saved} (forced {forced})  {idx / el:.1f} f/s")

    print(f"Done: {saved} images ({forced} forced to fill gaps) in {a.out_dir}, "
          f"{time.time() - t0:.0f}s")
    if saved > 3000:
        print("Over 3000 images: alignment will be slow. Re-run with a larger --min-diff (e.g. 9).")
    elif saved < 800:
        print("Under 800 images: consider a smaller --min-diff (e.g. 4).")


if __name__ == "__main__":
    main()
