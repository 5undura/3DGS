"""fill (도면 방 바닥·천장 구멍 메우기) 정확도. fillcore.backdrop 의 samples_from_plan -> to_splats -> hole_mask
(make_fill 에서 색 입히기만 뺀 것)를 합성 방 스플랫에 돌린다.

방 6 x 5 m (floor_z, ceiling_z 는 plan.json 값), 학습 스플랫: 바닥·천장·벽 면 위 격자 + 면 수직 잡음 1cm.
바닥 구멍 1.5 x 1.0 m, 천장 구멍 1.0 x 1.0 m. 카메라 1.4m 높이로 방을 가로지름.
재는 것: 넣은 스플랫 높이 - 바닥(z), 구멍 안에 든 비율, 구멍 밖(멀쩡한 곳)에 넣은 넓이, 구멍을 덮은 비율.
변수: 학습 스플랫 간격(흰 바닥은 성김), 불투명도, 실제 바닥과 plan floor_z 차이(카메라 높이 가정 오차).

python experiments/accuracy/fill_eval.py
"""
from __future__ import annotations

import json

import numpy as np

from common import OUT
from indoorgs import plyio, schemas
from indoorgs.fillcore import backdrop as bd

ROOM = [[0, 0], [6, 0], [6, 5], [0, 5]]
FHOLE = (2.0, 3.5, 2.0, 3.0)       # 바닥 구멍 x0 x1 y0 y1
CHOLE = (4.0, 5.0, 1.0, 2.0)       # 천장 구멍


def in_rect(P, r, pad=0.0):
    return (P[:, 0] > r[0] - pad) & (P[:, 0] < r[1] + pad) & (P[:, 1] > r[2] - pad) & (P[:, 1] < r[3] + pad)


def train_splats(rng, sp, fz_true, cz_true, op_range=(0.5, 1.0)):
    g = np.arange(sp / 2, 6, sp); h = np.arange(sp / 2, 5, sp)
    X, Y = np.meshgrid(g, h); X, Y = X.ravel(), Y.ravel()
    F = np.stack([X, Y, np.full_like(X, fz_true)], 1); F = F[~in_rect(F, FHOLE)]
    C = np.stack([X, Y, np.full_like(X, cz_true)], 1); C = C[~in_rect(C, CHOLE)]
    W = []
    zs = np.arange(fz_true + sp / 2, cz_true, sp)
    for x0, y0, x1, y1 in ((0, 0, 6, 0), (6, 0, 6, 5), (6, 5, 0, 5), (0, 5, 0, 0)):
        L = np.hypot(x1 - x0, y1 - y0); t = np.arange(sp / 2, L, sp)
        T, Z = np.meshgrid(t, zs)
        W.append(np.stack([x0 + (x1 - x0) * T.ravel() / L, y0 + (y1 - y0) * T.ravel() / L, Z.ravel()], 1))
    P = np.concatenate([F, C] + W)
    P = P + rng.normal(0, 0.01, P.shape)
    ply = plyio.make(len(P), plyio.brush_names(0))
    for i, c in enumerate("xyz"):
        ply.data[c] = P[:, i]
    op = rng.uniform(*op_range, len(P))
    ply.data["opacity"] = np.log(op / (1 - op))
    return ply


def run(sp=0.04, dz=0.0, op_range=(0.5, 1.0), fz=0.0, cz=2.6, seed=0):
    rng = np.random.default_rng(seed)
    plan = schemas.parse_plan({"version": 1, "rooms": [{"id": "r", "floor": ROOM, "floor_z": fz, "ceiling_z": cz}]})
    fz_true = fz + dz                        # 실제(정합된) 바닥 = 도면 바닥 + dz
    train = train_splats(rng, sp, fz_true, cz + dz, op_range)
    t = np.linspace(0.5, 5.5, 200)
    Cam = np.stack([t, 2.5 + 0.8 * np.sin(t), np.full_like(t, fz_true + 1.4)], 1)
    sm = bd.samples_from_plan(bd.rooms_of_cameras(plan.rooms, Cam), Cam, train.xyz)
    back = bd.to_splats(sm, np.full((len(sm.P), 3), 0.5, np.float32), train.names)
    hole = bd.hole_mask(train, back.xyz, sm.kind, 1.0, 3.0)
    A = back.xyz[hole]; k = sm.kind[hole]
    fl, ce = A[k == bd.FLOOR], A[k == bd.CEILING]
    cell = sm.SP
    def cover(Pxy, r):
        g = np.arange(r[0] + cell / 2, r[1], cell); h = np.arange(r[2] + cell / 2, r[3], cell)
        GX, GY = np.meshgrid(g, h); G = np.stack([GX.ravel(), GY.ravel()], 1)
        if len(Pxy) == 0:
            return 0.0
        from scipy.spatial import cKDTree
        d = cKDTree(Pxy[:, :2]).query(G)[0]
        return float((d < cell).mean())
    pad = 3.0 * cell   # hole_mask 반경 (간격 x 3) 만큼 가장자리 허용
    res = {
        "train_spacing": sp, "opacity": op_range, "dz_real_minus_plan": dz, "floor_z": fz,
        "S": round(sm.S, 3), "sample_spacing": round(sm.SP, 4), "offset_behind": round(bd.OFFSET * sm.S, 4),
        "added_floor": int(len(fl)), "added_ceiling": int(len(ce)), "added_wall": int((k == bd.WALL).sum()),
        "floor_z_minus_plan": round(float(np.median(fl[:, 2]) - fz), 4) if len(fl) else None,
        "floor_z_minus_real": round(float(np.median(fl[:, 2]) - fz_true), 4) if len(fl) else None,
        "ceil_z_minus_plan": round(float(np.median(ce[:, 2]) - cz), 4) if len(ce) else None,
        "floor_in_hole_frac": round(float(in_rect(fl, FHOLE, pad).mean()), 3) if len(fl) else None,
        "floor_outside_hole_m2": round(float((~in_rect(fl, FHOLE, pad)).sum() * cell ** 2), 3),
        "ceil_outside_hole_m2": round(float((~in_rect(ce, CHOLE, pad)).sum() * cell ** 2), 3),
        "floor_hole_covered": round(cover(fl, FHOLE), 3), "ceil_hole_covered": round(cover(ce, CHOLE), 3)}
    return res


def main():
    out = []
    print("간격  불투명도    dz   | 바닥z-도면 바닥z-실제 | 구멍안비율 구멍밖m2(바닥/천장) | 구멍덮음(바닥/천장) | 벽")
    cases = [dict(sp=s) for s in (0.04, 0.08, 0.12, 0.16, 0.2)]
    cases += [dict(sp=0.08, op_range=(0.2, 0.5)), dict(sp=0.08, op_range=(0.05, 0.3))]
    cases += [dict(sp=0.08, dz=d) for d in (0.05, 0.1, 0.2, -0.1)]
    cases += [dict(sp=0.08, fz=-0.35, cz=2.3)]
    for c in cases:
        r = run(**c)
        out.append(r)
        print(f"{r['train_spacing']:.2f} {str(r['opacity']):11s} {r['dz_real_minus_plan']:+.2f} | "
              f"{r['floor_z_minus_plan']!s:>8} {r['floor_z_minus_real']!s:>8} | {r['floor_in_hole_frac']!s:>6} "
              f"{r['floor_outside_hole_m2']:>6}/{r['ceil_outside_hole_m2']:<6} | {r['floor_hole_covered']}/{r['ceil_hole_covered']} | "
              f"{r['added_wall']}  (floor_z={r['floor_z']}, 천장z-도면 {r['ceil_z_minus_plan']})")
    OUT.mkdir(exist_ok=True)
    (OUT / "fill.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
