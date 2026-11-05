"""
flatten.py - 3DGS PLY 후처리: 큰 평면(벽/바닥/천장) 근처 스플랫을 평면에 붙이고 납작하게 만든다.

사용법:
  python flatten.py 입력.ply 출력.ply [--planes 8] [--band 2.0] [--aniso 6] [--thin 0.05] [--preview 0.33]

  --planes  찾을 평면 최대 개수 (기본 8)
  --band    평면 두께(자동 추정)의 몇 배까지를 '벽 위'로 볼지 (기본 2.0)
  --aniso   벽을 따라 길쭉한 비율 상한 (기본 6)
  --thin    납작하게 만든 뒤 두께 = 벽 방향 짧은 축 x 이 값 (기본 0.05)
  --preview 0보다 크면, 확인용으로 이 비율만큼 무작위로 뽑은 작은 PLY(입력/출력 둘 다)를 같이 저장
필요: numpy
"""
import sys, argparse, time
import numpy as np


def read_ply(path):
    with open(path, 'rb') as f:
        raw = f.read()
    h = raw.index(b'end_header\n') + len(b'end_header\n')
    header = raw[:h].decode('latin1')
    lines = header.splitlines()
    n = int([l for l in lines if l.startswith('element vertex')][0].split()[-1])
    props = [l.split()[-1] for l in lines if l.startswith('property')]
    for l in lines:
        if l.startswith('property') and l.split()[1] != 'float':
            raise SystemExit('float 속성만 지원: ' + l)
    data = np.frombuffer(raw, dtype='<f4', count=n * len(props), offset=h).reshape(n, len(props)).copy()
    return header, props, data


def write_ply(path, header, data):
    lines = header.splitlines()
    lines = [('element vertex %d' % len(data)) if l.startswith('element vertex') else l for l in lines]
    with open(path, 'wb') as f:
        f.write(('\n'.join(lines) + '\n').encode('latin1'))
        f.write(np.ascontiguousarray(data, dtype='<f4').tobytes())


def quat_to_mat(q):  # q: (N,4) w,x,y,z
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    R = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                  2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                  2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)
    return R


def mat_to_quat(R):  # (N,3,3) -> (N,4) w,x,y,z
    m = R
    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = np.zeros((len(R), 4))
    c0 = t > 0
    s = np.sqrt(np.maximum(t[c0] + 1, 1e-12)) * 2
    q[c0, 0] = 0.25 * s
    q[c0, 1] = (m[c0, 2, 1] - m[c0, 1, 2]) / s
    q[c0, 2] = (m[c0, 0, 2] - m[c0, 2, 0]) / s
    q[c0, 3] = (m[c0, 1, 0] - m[c0, 0, 1]) / s
    r = ~c0
    i = np.argmax(np.stack([m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]], 1), 1)
    for k in range(3):
        c = r & (i == k)
        if not c.any():
            continue
        a, b, d = k, (k + 1) % 3, (k + 2) % 3
        s = np.sqrt(np.maximum(1 + m[c, a, a] - m[c, b, b] - m[c, d, d], 1e-12)) * 2
        q[c, 0] = (m[c, d, b] - m[c, b, d]) / s
        q[c, 1 + a] = 0.25 * s
        q[c, 1 + b] = (m[c, b, a] + m[c, a, b]) / s
        q[c, 1 + d] = (m[c, d, a] + m[c, a, d]) / s
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def find_planes(P, n_planes, extent, rng):
    """반복 RANSAC. P: 불투명 스플랫 중심 (M,3). 반환: [(normal, d, thickness, count)]"""
    planes = []
    pts = P.copy()
    thr = 0.004 * extent
    total = len(P)
    for _ in range(n_planes):
        if len(pts) < 1000:
            break
        best = (0, None)
        for _ in range(600):
            s = pts[rng.choice(len(pts), 3, replace=False)]
            nrm = np.cross(s[1] - s[0], s[2] - s[0])
            ln = np.linalg.norm(nrm)
            if ln < 1e-12:
                continue
            nrm /= ln
            sub = pts[rng.choice(len(pts), min(20000, len(pts)), replace=False)]
            c = (np.abs((sub - s[0]) @ nrm) < thr).sum()
            if c > best[0]:
                best = (c, (nrm, -nrm @ s[0]))
        nrm, d = best[1]
        dist = pts @ nrm + d
        inl = np.abs(dist) < thr
        # 최소제곱 재추정
        Q = pts[inl]
        cen = Q.mean(0)
        u, sv, vt = np.linalg.svd(Q - cen, full_matrices=False)
        nrm = vt[2]
        d = -nrm @ cen
        dist = pts @ nrm + d
        inl = np.abs(dist) < thr
        cnt = inl.sum()
        if cnt < 0.02 * total:
            break
        thick = 1.4826 * np.median(np.abs(dist[inl])) * 3  # 견고한 두께 추정
        planes.append((nrm, d, max(thick, 0.5 * thr), cnt))
        pts = pts[~inl]
    return planes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inp'); ap.add_argument('out')
    ap.add_argument('--planes', type=int, default=8)
    ap.add_argument('--band', type=float, default=1.0)
    ap.add_argument('--aniso', type=float, default=10.0)
    ap.add_argument('--thin', type=float, default=0.1)
    ap.add_argument('--preview', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    header, props, D = read_ply(a.inp)
    ix = {p: i for i, p in enumerate(props)}
    N = len(D)
    print('스플랫 %d개 읽음 (%.0fs)' % (N, time.time() - t0))

    xyz = D[:, [ix['x'], ix['y'], ix['z']]].astype(np.float64)
    sc_i = [ix['scale_0'], ix['scale_1'], ix['scale_2']]
    rt_i = [ix['rot_0'], ix['rot_1'], ix['rot_2'], ix['rot_3']]
    s = np.exp(D[:, sc_i].astype(np.float64))
    op = 1 / (1 + np.exp(-D[:, ix['opacity']].astype(np.float64)))

    lo, hi = np.percentile(xyz, 1, 0), np.percentile(xyz, 99, 0)
    extent = np.linalg.norm(hi - lo)
    sel = op > 0.3
    P = xyz[sel]
    if len(P) > 400000:
        P = P[rng.choice(len(P), 400000, replace=False)]
    planes = find_planes(P, a.planes, extent, rng)
    print('평면 %d개 찾음:' % len(planes))
    for k, (n, d, th, c) in enumerate(planes):
        print('  #%d 법선 %s 두께 %.4f 지지점 %d' % (k, np.round(n, 3), th, c))

    thr = 0.004 * extent
    band = a.band * thr
    # 중복 평면 제거
    keep = []
    for n, d, th, c in planes:
        if any(abs(n @ m) > 0.985 and abs(d - dm * np.sign(n @ m)) < 3 * thr for m, dm, _, _ in keep):
            continue
        keep.append((n, d, th, c))
    planes = keep
    print('중복 제거 후 평면 %d개, 밴드 %.4f' % (len(planes), band))
    cell = 0.015 * extent
    best_k = np.full(N, -1); best_r = np.full(N, np.inf)
    bases = []
    for k, (n, d, th, c) in enumerate(planes):
        a1 = np.cross(n, [1, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1, 0])
        a1 /= np.linalg.norm(a1); a2 = np.cross(n, a1); bases.append((a1, a2))
        # 평면이 실제로 있는 영역(점유 격자)
        inl = xyz[sel][np.abs(xyz[sel] @ n + d) < thr]
        g = np.floor(np.stack([inl @ a1, inl @ a2], 1) / cell).astype(np.int64)
        g0 = g.min(0) - 2; gs = g.max(0) - g0 + 3
        occ = np.zeros(gs, np.int32); np.add.at(occ, (g[:, 0] - g0[0], g[:, 1] - g0[1]), 1)
        occ = occ >= 3
        o2 = occ.copy()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                o2 |= np.roll(np.roll(occ, dx, 0), dy, 1)
        dist = np.abs(xyz @ n + d)
        cand = dist < band
        gi = np.floor(np.stack([xyz @ a1, xyz @ a2], 1) / cell).astype(np.int64) - g0
        inside = (gi[:, 0] >= 0) & (gi[:, 1] >= 0) & (gi[:, 0] < gs[0]) & (gi[:, 1] < gs[1])
        ok = np.zeros(N, bool)
        ii = np.where(cand & inside)[0]
        ok[ii] = o2[gi[ii, 0], gi[ii, 1]]
        r = dist / band
        m = ok & (r < best_r)
        best_k[m] = k; best_r[m] = r[m]
    tgt = best_k >= 0
    print('평면에 붙일 스플랫: %d개 (%.1f%%)' % (tgt.sum(), 100 * tgt.mean()))

    Rm = quat_to_mat(D[:, rt_i].astype(np.float64))
    newD = D.copy()
    beta = a.thin
    for k, (n, d, th, c) in enumerate(planes):
        idx = np.where(best_k == k)[0]
        if len(idx) == 0:
            continue
        R = Rm[idx]; S = s[idx]
        RS = R * S[:, None, :]
        C = RS @ RS.transpose(0, 2, 1)
        M = np.eye(3) - (1 - beta) * np.outer(n, n)
        C2 = M[None] @ C @ M[None]
        ev, evec = np.linalg.eigh(C2)
        ev = np.maximum(ev, 1e-14)
        sc = np.sqrt(ev)  # 오름차순: [두께, 작은축, 큰축]
        sc[:, 2] = np.minimum(sc[:, 2], a.aniso * sc[:, 1])
        det = np.linalg.det(evec)
        evec[det < 0, :, 0] *= -1
        q = mat_to_quat(evec)
        newD[np.ix_(idx, rt_i)] = q.astype(np.float32)
        newD[np.ix_(idx, sc_i)] = np.log(sc).astype(np.float32)
        dist = xyz[idx] @ n + d
        newxyz = xyz[idx] - (1 - beta) * dist[:, None] * n[None]
        newD[idx, ix['x']] = newxyz[:, 0]; newD[idx, ix['y']] = newxyz[:, 1]; newD[idx, ix['z']] = newxyz[:, 2]
        tb = np.sqrt(np.einsum('i,mij,j->m', n, C, n))
        print('  평면#%d: %d개, 벽 방향 두께 중앙값 %.4f -> %.4f' % (k, len(idx), np.median(tb), np.median(tb) * beta))

    write_ply(a.out, header, newD)
    print('저장: %s (%.0fs)' % (a.out, time.time() - t0))
    if a.preview > 0:
        m = rng.random(N) < a.preview
        write_ply(a.inp[:-4] + '_preview.ply', header, D[m])
        write_ply(a.out[:-4] + '_preview.ply', header, newD[m])
        print('확인용 미리보기 저장 (%d개)' % m.sum())


if __name__ == '__main__':
    main()
