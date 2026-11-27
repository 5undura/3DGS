"""매칭 DB 의 검증된 짝(two_view_geometries) 중 '가짜 짝' 세기 (정답 자세로 만든 F 와 안 맞는 짝).

  python pairs_diag.py <out> <db.db>

가짜 짝 = inlier 매칭의 Sampson 오차 중앙값 > 4px. 같은 구간(갈 때끼리/올 때끼리) vs 다른 구간, 시선 차이 90도 기준으로 나눠 셈.
"""
import json
import sqlite3
import sys

import numpy as np


def main(out, db):
    g = json.load(open(f"{out}/gt_poses.json", encoding="utf-8"))
    Ki = np.linalg.inv(np.array(g["K"]))
    gt = {f["image"]: (np.array(f["R_world_from_cam"]), np.array(f["center"]), f["t"]) for f in g["frames"]}
    c = sqlite3.connect(db)
    ims = dict(c.execute("SELECT image_id, name FROM images"))
    kp = {i: np.frombuffer(b, np.float32).reshape(r, cc)[:, :2]
          for i, r, cc, b in c.execute("SELECT image_id, rows, cols, data FROM keypoints")}

    def skew(t):
        return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    st, dist = {}, []
    for pid, r, b in c.execute("SELECT pair_id, rows, data FROM two_view_geometries WHERE rows >= 15"):
        i2 = pid % 2147483647
        i1 = (pid - i2) // 2147483647
        m = np.frombuffer(b, np.uint32).reshape(r, 2)
        (R1, C1, t1), (R2, C2, t2) = gt[ims[i1]], gt[ims[i2]]
        F = Ki.T @ skew(R2.T @ (C1 - C2)) @ (R2.T @ R1) @ Ki
        x1 = np.c_[kp[i1][m[:, 0]], np.ones(r)]
        x2 = np.c_[kp[i2][m[:, 1]], np.ones(r)]
        Fx1, Ftx2 = x1 @ F.T, x2 @ F
        e = np.sqrt(np.sum(x2 * Fx1, 1) ** 2 / (Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2))
        false = bool(np.median(e) > 4)
        leg = "같은 구간" if (t1 > 26.5) == (t2 > 26.5) else "갈때-올때"
        ang = np.degrees(np.arccos(np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1)))
        k = (leg, "시선차>90°" if ang > 90 else "시선차<90°")
        s = st.setdefault(k, [0, 0])
        s[0] += 1
        s[1] += false
        dist.append((np.linalg.norm(C1 - C2), false))
    tot = sum(v[0] for v in st.values())
    print(f"검증된 짝 {tot}, 가짜 {sum(v[1] for v in st.values())}")
    for k, v in sorted(st.items()):
        print(f"  {k[0]} {k[1]}: {v[0]} 짝, 가짜 {v[1]}")
    d = np.array(dist)
    for lo, hi in [(0, 2), (2, 4), (4, 8), (8, 14), (14, 30)]:
        mm = (d[:, 0] >= lo) & (d[:, 0] < hi)
        print(f"  정답 거리 {lo}-{hi} m: {int(mm.sum())} 짝, 가짜 {int(d[mm, 1].sum())}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
