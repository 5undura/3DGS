"""
flatten2.py - 3DGS PLY 후처리 (국소 평면 버전)
벽이 완벽한 평면이 아니어도(긴 복도처럼 살짝 휘어도) 동작하도록,
작은 구역(복셀)마다 주변 스플랫으로 국소 평면을 구해 그 평면 근처 스플랫을 납작하게 만든다.

사용법:
  python flatten2.py 입력.ply 출력.ply [--voxel 0.008] [--band 3] [--thin 0.1] [--planar 0.12] [--preview 0.3]
  --voxel  구역 크기 (장면 크기 대비 비율, 기본 0.008)
  --band   국소 평면 두께의 몇 배까지 처리할지 (기본 3)
  --thin   벽 방향 두께를 이 비율로 줄임 (기본 0.1)
  --planar 이 값보다 납작한 구역만 '벽'으로 봄 (작을수록 엄격, 기본 0.12)
  --preview 확인용으로 무작위 일부를 뽑은 작은 PLY를 같이 저장 (예: 0.3)
필요: numpy
"""
import argparse, time
import numpy as np
from flatten import read_ply, write_ply, quat_to_mat, mat_to_quat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inp'); ap.add_argument('out')
    ap.add_argument('--voxel', type=float, default=0.008)
    ap.add_argument('--band', type=float, default=3.0)
    ap.add_argument('--thin', type=float, default=0.1)
    ap.add_argument('--planar', type=float, default=0.12)
    ap.add_argument('--minpts', type=int, default=30)
    ap.add_argument('--preview', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    header, props, D = read_ply(a.inp)
    ix = {p: i for i, p in enumerate(props)}
    N = len(D)
    print('스플랫 %d개 읽음 (%.0fs)' % (N, time.time() - t0))
    xyz = D[:, [ix['x'], ix['y'], ix['z']]].astype(np.float64)
    sc_i = [ix['scale_0'], ix['scale_1'], ix['scale_2']]
    rt_i = [ix['rot_0'], ix['rot_1'], ix['rot_2'], ix['rot_3']]
    op = 1 / (1 + np.exp(-D[:, ix['opacity']].astype(np.float64)))
    lo, hi = np.percentile(xyz, 1, 0), np.percentile(xyz, 99, 0)
    extent = np.linalg.norm(hi - lo)
    v = a.voxel * extent
    print('장면 크기 %.3f, 구역 크기 %.4f' % (extent, v))

    # 복셀 키
    g = np.floor((xyz - lo) / v).astype(np.int64)
    g -= g.min(0)
    gm = g.max(0) + 3
    def key(gg):
        return (gg[:, 0] + 1) + gm[0] * ((gg[:, 1] + 1) + gm[1] * (gg[:, 2] + 1))
    k_all = key(g)
    # 불투명 스플랫 중심으로 복셀별 통계
    m = op > 0.3
    P = xyz[m]; kP = k_all[m]
    uk, inv = np.unique(kP, return_inverse=True)
    M = len(uk)
    cnt = np.bincount(inv, minlength=M).astype(np.float64)
    S1 = np.stack([np.bincount(inv, P[:, i], M) for i in range(3)], 1)
    S2 = np.zeros((M, 3, 3))
    for i in range(3):
        for j in range(i, 3):
            S2[:, i, j] = S2[:, j, i] = np.bincount(inv, P[:, i] * P[:, j], M)
    # 3x3x3 이웃 합산
    Cn = np.zeros(M); C1 = np.zeros((M, 3)); C2 = np.zeros((M, 3, 3))
    offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    for dx, dy, dz in offs:
        nk = uk + dx + gm[0] * (dy + gm[1] * dz)
        pos = np.searchsorted(uk, nk)
        pos[pos >= M] = 0
        ok = uk[pos] == nk
        Cn[ok] += cnt[pos[ok]]; C1[ok] += S1[pos[ok]]; C2[ok] += S2[pos[ok]]
    good = Cn >= a.minpts
    mu = C1 / np.maximum(Cn, 1)[:, None]
    Cov = C2 / np.maximum(Cn, 1)[:, None, None] - mu[:, :, None] * mu[:, None, :]
    ev, evec = np.linalg.eigh(Cov)
    ev = np.maximum(ev, 1e-14)
    planar = good & (np.sqrt(ev[:, 0] / ev[:, 1]) < a.planar)
    normal = evec[:, :, 0]
    thick = np.sqrt(ev[:, 0])
    print('복셀 %d개 중 벽처럼 납작한 복셀 %d개 (%.0fs)' % (M, planar.sum(), time.time() - t0))

    # 각 스플랫 -> 자기 복셀의 국소 평면
    pos = np.searchsorted(uk, k_all); pos[pos >= M] = 0
    has = (uk[pos] == k_all)
    has[has] = planar[pos[has]]
    idx = np.where(has)[0]
    pv = pos[idx]
    n = normal[pv]; c = mu[pv]
    dist = np.einsum('ij,ij->i', xyz[idx] - c, n)
    band = a.band * np.maximum(thick[pv], 0.05 * v)
    inb = np.abs(dist) < band
    idx = idx[inb]; n = n[inb]; dist = dist[inb]
    print('납작하게 만들 스플랫 %d개 (%.1f%%)' % (len(idx), 100 * len(idx) / N))

    s = np.exp(D[idx][:, sc_i].astype(np.float64))
    R = quat_to_mat(D[idx][:, rt_i].astype(np.float64))
    RS = R * s[:, None, :]
    C = RS @ RS.transpose(0, 2, 1)
    beta = a.thin
    Mx = np.eye(3)[None] - (1 - beta) * n[:, :, None] * n[:, None, :]
    Cs = Mx @ C @ Mx
    e2, v2 = np.linalg.eigh(Cs)
    e2 = np.maximum(e2, 1e-14)
    det = np.linalg.det(v2); v2[det < 0, :, 0] *= -1
    newD = D.copy()
    newD[np.ix_(idx, rt_i)] = mat_to_quat(v2).astype(np.float32)
    newD[np.ix_(idx, sc_i)] = (0.5 * np.log(e2)).astype(np.float32)
    newxyz = xyz[idx] - (1 - beta) * dist[:, None] * n
    for j, nm in enumerate(['x', 'y', 'z']):
        newD[idx, ix[nm]] = newxyz[:, j]
    tb = np.sqrt(np.einsum('ni,nij,nj->n', n, C, n))
    print('벽 방향 두께 중앙값 %.5f -> %.5f, 위치 이동 중앙값 %.5f' % (np.median(tb), np.median(tb) * beta, np.median(np.abs(dist)) * (1 - beta)))
    write_ply(a.out, header, newD)
    print('저장: %s (%.0fs)' % (a.out, time.time() - t0))
    if a.preview > 0:
        rng = np.random.default_rng(a.seed)
        mm = rng.random(N) < a.preview
        write_ply(a.inp[:-4] + '_preview.ply', header, D[mm])
        write_ply(a.out[:-4] + '_preview.ply', header, newD[mm])
        print('확인용 미리보기 저장 (%d개)' % mm.sum())


if __name__ == '__main__':
    main()
