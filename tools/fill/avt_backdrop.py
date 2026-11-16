"""4차 보강 스캔(placed_fill_1..6) 용: 스캔 위치 보정 + 벽을 따라가는 배경 면(바닥/천장/벽) + 공간별 자르기

왜 필요한가
  - 3DGS 벽/천장은 반투명한 곳이 많아 뒤가 비침. 뒤에 다른 스캔(옆 방/복도)의 스플랫이 있으면 겹쳐 보임.
  - 팀원 배경(StudentHallBackdrop.kSpaces 상자)은 2차 촬영 벽에 맞춘 평평한 면이라 4차 데이터와 10~40cm 어긋남.
하는 일
  1) 스캔 위치 보정(ADJUST): 벽 지도 겹쳐보기로 확인된 스캔만 살짝 돌리고 옮김 (33B117 방 스캔).
  2) 각 공간의 각 면(벽/바닥/천장)마다, 그 공간을 맡은 스캔을 안쪽에서 바깥으로 바라본 정사영 '사진'을 찍어
     칸(5cm)마다 처음 불투명해지는 깊이 = 보이는 벽 표면을 구함. 면은 그 표면을 따라가는 굴곡 있는 메쉬로,
     표면 바로 뒤(3cm)에 놓음 -> 벽을 가리지 않고, 옆 공간도 침범하지 않음. (relief 함수 설명 참고)
  3) 두 공간의 면이 서로 엇갈린 곳(스캔끼리 어긋나 겹친 곳)은 두 면을 가운데로 모음.
  4) 면 텍스처 = 같은 '사진'의 벽 표면 층. 벽이 안 찍힌 곳은 그 앞에 보이는 것, 그래도 없으면 주변 색을 부드럽게 이어 붙임.
     -> 반투명 스플랫 뒤에 같은 그림이 같은 자리에 있으므로 비쳐도 얼룩이 지지 않음.
  5) 자르기: 각 스캔은 자기 공간(면 안쪽 + 0.3m) 안만 남기고, 다른 공간 안에 들어간 스플랫은 지움.
     옆 공간 안으로 삐져나온 면 조각도 잘라냄.
출력
  out\\mesh.json (면 목록: 꼭짓점/UV/삼각형/텍스처 이름), out\\Baked\\*.png, ply (합본)
  Unity: StudentHallBackdrop.CreateFitted("v4_fill") 가 mesh.json 을 읽어 한쪽 면만 보이는 불투명 메쉬를 만듦.

사용: python avt_backdrop.py src=<placed_fill_1..6 폴더> cs=<AVT\\Assets\\Editor\\StudentHallBackdrop.cs>
                             out=<AVT\\Assets\\Backdrop\\v4_fill> ply=<합본 출력 .ply> [tag=_fill] [preview=1]
"""
import json, os, re, sys
import numpy as np
from PIL import Image

OWN = {1: ['West stair hall', 'West corridor (west)', 'West corridor (33B116)', 'West corridor (east)'],
       2: ['Junction hall', 'East stair hall', 'East corridor'],
       3: ['33B102-103'], 4: ['33B101'], 5: ['33B117'], 6: ['33B115']}
OWNER = {n: i for i, ns in OWN.items() for n in ns}
# 스캔 위치 보정: (cx,cz) 를 중심으로 yaw(도) 돌리고 (dx,dz) 옮김. x' = c·x - s·z, z' = s·x + c·z
# 33B117(스캔 5): 방 동쪽 벽이 교차로 벽보다 20~30cm 동쪽에 있었음(서로 뚫고 지나감) -> 복도 스캔과 겹쳐보기로 맞춤
ADJUST = {5: dict(yaw=1.25, dx=-0.08, dz=-0.28, cx=-2.975, cz=5.165)}
FLOOR_Y = -0.03
CORR_CEIL, FULL = 3.02, 99.0
BEHIND = 0.02        # 면을 보이는 벽 표면보다 이만큼(+1cm) 뒤에
MARGIN = 0.30        # 자르기: 자기 공간 밖으로 남기는 폭
TEX = 0.015          # 텍스처 1픽셀 = 1.5cm
DOOR_MAX = 2.5       # 이보다 낮은 열린 칸 = 문 (그 위는 공간끼리 크게 트인 곳)
DRES = 0.05          # 벽 표면 찾기용 해상도
SH0 = 0.28209479177387814
PREVIEW = False
DT = np.dtype([(n, '<f4') for n in ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
                                    'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']])
SIDES = {'N': (0, 'x0', +1), 'S': (0, 'x1', -1), 'W': (2, 'z0', +1), 'E': (2, 'z1', -1)}   # (법선 축, 상자 키, 안쪽 부호)
AX = {'y': (0, 2, 1), 'x': (2, 1, 0), 'z': (0, 1, 2)}   # 평면 종류 -> (u 축, v 축, 법선 축)
PLANE = {'N': 'x', 'S': 'x', 'W': 'z', 'E': 'z'}


def parse_spaces(cs):
    src = open(cs, encoding='utf-8-sig').read().replace('\r\n', '\n')
    body = src.split('kSpaces =')[1].split('};\n\n')[0]
    num = r'(-?[\d.]+f|kFull|kCorridorCeil)'
    val = lambda s: FULL if s == 'kFull' else CORR_CEIL if s == 'kCorridorCeil' else float(s.rstrip('f'))
    spaces = []
    for blk in re.split(r'\n\s*new\(\) \{', body)[1:]:
        g = lambda k: val(re.search(k + r' = ' + num, blk).group(1))
        s = dict(name=re.search(r'name = "(.*?)"', blk).group(1), look=re.search(r'Look\.(\w+)', blk).group(1),
                 x0=g('x0'), x1=g('x1'), z0=g('z0'), z1=g('z1'), ceil=g('ceil'), open=[], recess=[])
        if 'open = ' in blk:
            seg = blk.split('open = ')[1].split('}')[0]
            s['open'] = [[c, val(a), val(b), val(t)] for c, a, b, t in
                         re.findall(r"\('(\w)', " + num + ', ' + num + ', ' + num + r'\)', seg)]
        if 'recess = ' in blk:
            seg = blk.split('recess = ')[1].split('}')[0]
            s['recess'] = [[c, val(a), val(b)] for c, a, b in re.findall(r"\('(\w)', " + num + ', ' + num + r'\)', seg)]
        s['owner'] = OWNER[s['name']]
        spaces.append(s)
    return spaces


def quat_mat(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)], -1),
                     np.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)], -1),
                     np.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)], -1)], 1)


def load(src, i, tag='_fill'):
    """placed_<i>.ply 를 읽고 ADJUST 보정(위치 + 회전)을 적용"""
    with open(os.path.join(src, f'placed{tag}_{i}.ply'), 'rb') as f:
        h = b''
        while not h.endswith(b'end_header\n'):
            h += f.readline()
        a = np.fromfile(f, DT, int(re.search(rb'element vertex (\d+)', h).group(1)))
    if i in ADJUST:
        t = ADJUST[i]; th = np.radians(t['yaw']); c, s = np.cos(th), np.sin(th)
        x = a['x'].astype(np.float64) - t['cx']; z = a['z'].astype(np.float64) - t['cz']
        a['x'] = c * x - s * z + t['cx'] + t['dx']; a['z'] = s * x + c * z + t['cz'] + t['dz']
        # 같은 회전을 스플랫 방향에도: 행렬 [[c,0,-s],[0,1,0],[s,0,c]] = y축 -yaw 회전 -> 쿼터니언 (cos(-θ/2), 0, sin(-θ/2), 0)
        qw, qy = np.cos(-th / 2), np.sin(-th / 2)
        w0, x0, y0, z0 = [a[f'rot_{k}'].astype(np.float64) for k in range(4)]
        a['rot_0'] = qw * w0 - qy * y0; a['rot_1'] = qw * x0 + qy * z0
        a['rot_2'] = qw * y0 + qy * w0; a['rot_3'] = qw * z0 - qy * x0
    return a


def splats(a):
    o = 1 / (1 + np.exp(-a['opacity'])); m = o > 0.05
    return dict(P=np.stack([a['x'], a['y'], a['z']], 1)[m].astype(np.float64),
                col=np.clip(0.5 + SH0 * np.stack([a['f_dc_0'], a['f_dc_1'], a['f_dc_2']], 1)[m], 0, 1),
                alpha=o[m],
                R=quat_mat(np.stack([a['rot_0'], a['rot_1'], a['rot_2'], a['rot_3']], 1)[m].astype(np.float64)),
                S2=np.exp(2 * np.stack([a['scale_0'], a['scale_1'], a['scale_2']], 1)[m].astype(np.float64)))


def raster(U, V, cov, alpha, k, K, u0, v0, res, W_, H_, col=None):
    """스플랫을 평면에 정사영으로 찍음(가우시안 모양 그대로), 깊이 층(k)별로 모음 -> 덮음(K,H,W) [, 색 합]"""
    cv_ = np.zeros((K, H_, W_), np.float32); acc = np.zeros((K, H_, W_, 3), np.float32) if col is not None else None
    a, b, c = cov[:, 0, 0] + (res / 2) ** 2, cov[:, 0, 1], cov[:, 1, 1] + (res / 2) ** 2
    det = a * c - b * b
    ia, ib, ic = c / det, -b / det, a / det
    r = np.clip(np.ceil(3 * np.sqrt(np.maximum(a, c)) / res).astype(int), 1, 30)
    cu = (U - u0) / res; cv = (V - v0) / res
    for R_ in np.unique(r):
        m = np.nonzero(r == R_)[0]
        o = np.arange(-R_, R_ + 1); du, dv = np.meshgrid(o, o); du = du.ravel(); dv = dv.ravel()
        step = max(1, 300000 // len(du))
        for s0 in range(0, len(m), step):
            idx = m[s0:s0 + step]
            iu = np.floor(cu[idx])[:, None] + du; iv = np.floor(cv[idx])[:, None] + dv
            x = (iu + 0.5 - cu[idx][:, None]) * res; y = (iv + 0.5 - cv[idx][:, None]) * res
            g = alpha[idx][:, None] * np.exp(-0.5 * (ia[idx][:, None] * x * x + 2 * ib[idx][:, None] * x * y + ic[idx][:, None] * y * y))
            ok = (iu >= 0) & (iu < W_) & (iv >= 0) & (iv < H_) & (g > 2e-3)
            kk = np.broadcast_to(k[idx][:, None], ok.shape)[ok]
            iu = iu[ok].astype(int); iv = iv[ok].astype(int); g = g[ok].astype(np.float32)
            np.add.at(cv_, (kk, iv, iu), g)
            if col is not None:
                cc = np.broadcast_to(col[idx][:, None, :], ok.shape + (3,))[ok].astype(np.float32)
                for ch in range(3): np.add.at(acc[..., ch], (kk, iv, iu), g * cc[:, ch])
    return cv_, acc


def view(S, pl, A, B, sg, rect, dmin, dmax, res, step=0.02, color=False, keep=None, near=None, far=None, slabs=False):
    """평면(법선좌표 = A + B·u) 을 안쪽(+sg)에서 바깥쪽으로 바라본 정사영 '사진'. rect=(u0,u1,v0,v1).
    안쪽 거리 d 가 dmax..dmin 인 스플랫을 앞에서부터 겹쳐 그림.
    -> 처음 불투명(투과<0.5)해지는 거리 맵, 색, 투과율 [, near 보다 가까운 층만의 색, 투과율]"""
    ua, va, na = AX[pl]; u0, u1, v0, v1 = rect
    P = S['P']; d = (P[:, na] - (A + B * P[:, ua])) * sg
    m = (d > dmin) & (d < dmax) & (P[:, ua] > u0 - 0.3) & (P[:, ua] < u1 + 0.3) & (P[:, va] > v0 - 0.3) & (P[:, va] < v1 + 0.3)
    if keep is not None: m &= keep
    W_ = int(np.ceil((u1 - u0) / res)) + 1; H_ = int(np.ceil((v1 - v0) / res)) + 1
    K = int(np.ceil((dmax - dmin) / step))
    k = np.clip(((dmax - d[m]) / step).astype(int), 0, K - 1)        # 0 = 가장 안쪽(앞)
    Rk = S['R'][m][:, [ua, va], :]
    cov = np.einsum('nik,nk,njk->nij', Rk, S['S2'][m], Rk)
    cv_, acc = raster(P[m, ua], P[m, va], cov, S['alpha'][m], k, K, u0, v0, res, W_, H_, S['col'][m] if color else None)
    T = np.ones((H_, W_)); depth = np.full((H_, W_), np.nan); C = np.zeros((H_, W_, 3))
    Tn = np.ones((H_, W_)); Cn = np.zeros((H_, W_, 3))
    for kk in range(K):
        al = 1 - np.exp(-cv_[kk])
        dk = dmax - (kk + 0.5) * step
        if color:
            c = acc[kk] / np.maximum(cv_[kk], 1e-9)[..., None]
            if far is not None: al = al * (dk > far)          # 면보다 뒤(바깥)는 안 그림
            C += (T * al)[..., None] * c
            if near is not None:
                aln = al * (dk < near)
                Cn += (Tn * aln)[..., None] * c; Tn = Tn * (1 - aln)
        T = T * (1 - al)
        hit = np.isnan(depth) & (T < 0.5)
        depth[hit] = dk
    if slabs: return depth, 1 - np.exp(-cv_), dmax - (np.arange(K) + 0.5) * step
    return depth, C, T, Cn, Tn


def blur(x, sig):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(x, sigma=(sig, sig) + (0,) * (x.ndim - 2), mode='nearest')


def inpaint(C, A):
    """C: 미리 곱한 색(H,W,3), A: 덮인 정도(H,W). 덮인 곳은 그대로, 빈 곳은 주변 색이 경계 없이 부드럽게 번지도록 채움
    (해상도를 반씩 줄여 가며 평균 -> 다시 키우며 빈 만큼만 섞는 방식)"""
    if A.sum() < 1e-6: return np.zeros_like(C) + 0.35
    lv = [(C, np.clip(A, 0, 1))]
    while min(lv[-1][1].shape) > 1:
        c, a = lv[-1]
        c = np.pad(c, ((0, c.shape[0] % 2), (0, c.shape[1] % 2), (0, 0)), mode='edge')
        a = np.pad(a, ((0, a.shape[0] % 2), (0, a.shape[1] % 2)), mode='edge')
        lv.append(((c[0::2, 0::2] + c[1::2, 0::2] + c[0::2, 1::2] + c[1::2, 1::2]) / 4,
                   (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2]) / 4))
    c, a = lv[-1]
    img = np.broadcast_to(c.reshape(-1, 3).sum(0) / max(a.sum(), 1e-9), c.shape).copy()
    from scipy.ndimage import zoom as ndzoom
    for c, a in lv[-2::-1]:
        up = ndzoom(img, (c.shape[0] / img.shape[0], c.shape[1] / img.shape[1], 1), order=1, mode='nearest')[:c.shape[0], :c.shape[1]]
        # 평균을 내며 줄였으니 a 는 '덮인 비율': 덮인 만큼은 자기 색, 나머지는 더 거친 단계의 색
        cov = np.clip(a * 4, 0, 1)[..., None] if c is not lv[0][0] else a[..., None]
        own = c / np.maximum(a, 1e-9)[..., None]
        img = cov * own + (1 - cov) * up
    return img


def is_full_open(s, side):
    ax = SIDES[side][0]
    a0, a1 = (s['z0'], s['z1']) if ax == 0 else (s['x0'], s['x1'])
    return any(c == side and ot >= FULL - 1 and oa <= a0 + 0.01 and ob >= a1 - 0.01 for c, oa, ob, ot in s['open'])


def open_mask(s, side, u, v, doors=True, pad=0.0):
    """(u,v) 가 그 면의 열린 곳(공간끼리 통하는 곳 / 문) 안인지"""
    m = np.zeros(np.shape(u), bool)
    for c, oa, ob, top in s['open']:
        if c != side or (not doors and top < FULL - 1): continue
        m |= (u > oa - pad) & (u < ob + pad) & (v < top + pad)
    return m


def faces_of(s, k, ext=None):
    """key -> [dict(pl, p, u0,u1,v0,v1, sg)] : StudentHallBackdrop.BuildSide 와 같은 구성(문/오목부), 상자 평면 기준.
    ext[side] = 그 면 쪽으로 더 늘릴 길이 (벽이 상자보다 바깥에 있을 때 모서리가 비지 않게)"""
    ext = ext or {}
    e = lambda side: ext.get(side, 0.0)
    F = {}
    def add(key, pl, p, u0, u1, v0, v1, sg):
        if u1 - u0 > 0.005 and v1 - v0 > 0.005:
            F.setdefault(f"{s['name']}|{key}", []).append(dict(pl=pl, p=p, u0=u0, u1=u1, v0=v0, v1=v1, sg=sg))
    fy = FLOOR_Y - 0.0007 * k                      # 공간마다 살짝 다른 높이 (겹치는 가장자리에서 깜빡임 방지)
    rec = lambda side, face: next((r[2] for r in s['recess'] if r[0] == side), face)
    add('floor', 'y', fy, s['x0'] - e('N'), s['x1'] + e('S'), s['z0'] - e('W'), s['z1'] + e('E'), +1)
    add('ceiling', 'y', s['ceil'], min(rec('N', s['x0']), s['x0'] - e('N')), max(rec('S', s['x1']), s['x1'] + e('S')),
        min(rec('W', s['z0']), s['z0'] - e('W')), max(rec('E', s['z1']), s['z1'] + e('E')), -1)
    for side in 'NSWE':
        ax, key, sg = SIDES[side]
        alongZ = ax == 0
        p = s[key]; a0, a1 = (s['z0'], s['z1']) if alongZ else (s['x0'], s['x1'])
        e0, e1 = (a0 - e('W'), a1 + e('E')) if alongZ else (a0 - e('N'), a1 + e('S'))
        op = [o for o in s['open'] if o[0] == side and o[3] >= DOOR_MAX]   # 문은 여기서 안 자름: 실제 뚫린 곳을 데이터로 찾음
        r = next((r for r in s['recess'] if r[0] == side), None)
        sill, q = (r[1], r[2]) if r else (s['ceil'] + 1, p)
        cuts = sorted(set([min(max(c, e0), e1) for o in op for c in (o[1], o[2])] + [e0, e1]))
        pl = PLANE[side]
        for u, w in zip(cuts, cuts[1:]):
            mid = (u + w) / 2
            top = max([o[3] for o in op if o[1] <= mid <= o[2]], default=FLOOR_Y - 0.05)
            add(side, pl, p, u, w, top, min(sill, s['ceil'] + 0.7), sg)       # 천장 면이 올라갈 수 있으니 넉넉히 (나중에 천장 높이로 자름)
            if not r: continue
            add(side + '|upper', pl, q, u, w, max(top, sill), s['ceil'] + 0.7, sg)
            if top < sill:
                if alongZ: add(side + '|sill', 'y', sill, min(p, q), max(p, q), u, w, +1)
                else:      add(side + '|sill', 'y', sill, u, w, min(p, q), max(p, q), +1)
        if r:
            lo, hi = min(p, q), max(p, q)
            if alongZ:
                add('W', 'z', a0, lo, hi, sill, s['ceil'] + 0.7, +1); add('E', 'z', a1, lo, hi, sill, s['ceil'] + 0.7, -1)
            else:
                add('N', 'x', a0, lo, hi, sill, s['ceil'] + 0.7, +1); add('S', 'x', a1, lo, hi, sill, s['ceil'] + 0.7, -1)
    return F


def relief(dep, al, dks, ref_pct=30, bump=0.12, back=0.25):
    """면의 깊이 맵(안쪽 +) = 칸마다 '안에서 보이는 벽 표면'(처음 불투명해지는 깊이) 바로 뒤(3cm).
    - 표면보다 앞으로는 절대 안 나옴 (주변 3x3 칸 중 가장 뒤쪽 기준) -> 스플랫 벽을 가리지 않음
    - 표면보다 한참 뒤로도 안 감 -> 옆 공간 벽을 침범하지 않음
    - 벽 기준(ref)보다 back(25cm) 넘게 뚫려 보이는 곳 / 안 보이는 곳 / 벽 앞 물건(기준 + bump 넘게 나온 것)
      = 벽 깊이를 모르는 곳 -> 가장 가까운 벽 칸의 깊이
    - 반투명해서 생긴 좁은 구멍(0.55m 이하)은 메움. 문처럼 넓게 들어간 곳은 따라감"""
    from scipy.ndimage import grey_closing, grey_erosion, gaussian_filter
    ok = np.isfinite(dep)
    if ok.sum() < 20: return None, None
    ref = float(np.percentile(dep[ok], ref_pct))
    # 벽으로 볼 칸: 기준에서 뒤로 back, 앞으로 bump 안. 나머지(벽 앞 물건, 뚫려 보이는 곳, 안 보이는 곳)는
    # 벽 깊이를 모르므로 가장 가까운 벽 칸의 깊이를 이어받음 (벽이 살짝 기울어 있어도 따라감)
    known = ok & (dep >= ref - back) & (dep <= ref + bump)
    if known.mean() < 0.03:
        d = np.full(dep.shape, ref)
    else:
        from scipy.ndimage import distance_transform_edt
        idx = distance_transform_edt(~known, return_distances=False, return_indices=True)
        d = np.where(known, dep, 0.0)[idx[0], idx[1]]
    d = grey_closing(d, size=(11, 11), mode='nearest')
    f = grey_erosion(d, size=(3, 3), mode='nearest') - BEHIND - 0.01
    return np.minimum(gaussian_filter(f, 1.0, mode='nearest'), f), ref


def sample(D, u, v, u0, v0, res):
    """깊이 맵 D[v,u] 를 (u,v) 위치에서 읽음 (가장자리 밖은 가장자리 값)"""
    from scipy.ndimage import map_coordinates
    return map_coordinates(D, [np.clip((v - v0) / res - 0.5, 0, D.shape[0] - 1), np.clip((u - u0) / res - 0.5, 0, D.shape[1] - 1)], order=1, mode='nearest')


def scan_keep(S, spaces, i):
    """다른 공간 깊숙이(30cm 넘게) 있는 스플랫 = 문 너머로 보인 것 -> 벽 찾기에서 뺌"""
    P = S['P']; keep = np.ones(len(P), bool)
    for s in spaces:
        if s['owner'] != i:
            keep &= ~((P[:, 0] > s['x0'] + .3) & (P[:, 0] < s['x1'] - .3) & (P[:, 2] > s['z0'] + .3) & (P[:, 2] < s['z1'] - .3))
    return keep


def build(src, spaces, out_dir, tag):
    """벽 표면 찾기 -> 옆 공간과 겹치는 곳 정리 -> 텍스처 굽기 + 메쉬. -> maps[key] = dict(pl,p,sg,u0,v0,D) (자르기용)"""
    from scipy.ndimage import zoom as ndzoom
    recs = []
    # ---- 1) 면마다 '안에서 보이는 벽 표면' 깊이 맵
    for i in range(1, 7):
        S = splats(load(src, i, tag)); keep = scan_keep(S, spaces, i)
        for k, s in enumerate(spaces):
            if s['owner'] != i: continue
            ext = {}                       # 벽이 상자보다 얼마나 바깥에 있는지 (모서리/바닥/천장 늘림 폭)
            for side in 'NSWE':
                if is_full_open(s, side): continue
                ax, key, sg = SIDES[side]
                a0, a1 = (s['z0'], s['z1']) if ax == 0 else (s['x0'], s['x1'])
                top_ = min(2.2, s['ceil'] - 0.3)
                D, ref = relief(*view(S, PLANE[side], s[key], 0.0, sg, (a0, a1, 0.3, top_), -0.8, 0.9, DRES, keep=keep, slabs=True))
                if D is not None:
                    uu = a0 + (np.arange(D.shape[1]) + 0.5) * DRES; vv = 0.3 + (np.arange(D.shape[0]) + 0.5) * DRES
                    U_, V_ = np.meshgrid(uu, vv); wall = ~open_mask(s, side, U_, V_, pad=0.15)   # 열린 곳 깊이는 벽이 아님
                    if wall.any(): ext[side] = float(np.clip(-D[wall].min() + 0.05, 0.0, 0.85))
            cD, cref = relief(*view(S, 'y', s['ceil'], 0.0, -1, (s['x0'] + .3, s['x1'] - .3, s['z0'] + .3, s['z1'] - .3), -0.8, 0.9, DRES, keep=keep, slabs=True), ref_pct=25)
            ceil_y = s['ceil'] - (float(cD.min()) if cD is not None else 0.0)      # 천장 면의 가장 높은 곳
            for key, qs in faces_of(s, k, ext).items():
                pl = qs[0]['pl']; part = key.split('|', 1)[1]
                u0 = min(q['u0'] for q in qs); u1 = max(q['u1'] for q in qs)
                v0 = min(q['v0'] for q in qs); v1 = max(q['v1'] for q in qs)
                if pl != 'y': v1 = min(v1, ceil_y + 0.05)                      # 벽은 천장 면 바로 위까지만
                groups = {}
                for q in qs: groups.setdefault((q['p'], q['sg']), []).append(q)
                for gi, ((p, sg), gq) in enumerate(groups.items()):
                    dep, al, dks = view(S, pl, p, 0.0, sg, (u0, u1, v0, v1), -0.8, 0.9, DRES, keep=keep, slabs=True)
                    D, ref = relief(dep, al, dks, ref_pct=25 if part == 'ceiling' else 30, bump=0.03 if part == 'floor' else 0.12)
                    if D is None: D = np.zeros(dep.shape) - BEHIND
                    if part == 'floor': D = np.clip(D, -0.10, 0.02)             # 바닥은 거의 평평하게
                    # 문: kSpaces 의 문 칸 안에서 실제로 뚫려 보이는 곳(안 보이거나 벽 뒤로 35cm 넘게 보이는 곳)만 구멍
                    hole = None
                    if part in SIDES and any(o[0] == part and o[3] < DOOR_MAX for o in s['open']):
                        from scipy.ndimage import binary_opening, binary_closing, label
                        uu = u0 + (np.arange(dep.shape[1]) + 0.5) * DRES; vv = v0 + (np.arange(dep.shape[0]) + 0.5) * DRES
                        U_, V_ = np.meshgrid(uu, vv)
                        door = np.zeros(U_.shape, bool)
                        for c, oa, ob, top in s['open']:
                            if c == part and top < DOOR_MAX: door |= (U_ > oa - 0.10) & (U_ < ob + 0.10) & (V_ < top + 0.10)
                        deep = ~np.isfinite(dep) | (dep < ref - 0.35)
                        hole = binary_closing(binary_opening(door & deep, np.ones((3, 3))), np.ones((3, 3))) & door
                        lab, n = label(hole)                                   # 바닥에 닿는 덩어리만 (문은 바닥부터 뚫림)
                        keepl = [l for l in range(1, n + 1) if V_[lab == l].min() < 0.25]
                        hole = np.isin(lab, keepl)
                    recs.append(dict(i=i, k=k, key=key, pl=pl, part=part, p=p, sg=sg, u0=u0, u1=u1, v0=v0, v1=v1, D0=D, D=D.copy(), hole=hole,
                                     main=gi == 0 and part in ('N', 'S', 'W', 'E', 'floor', 'ceiling'),
                                     rects=[(q['u0'], q['u1'], q['v0'], min(q['v1'], v1)) for q in gq if min(q['v1'], v1) - q['v0'] > 0.01]))
            print(f"  [{i}] {s['name']}: 벽 바깥으로 " + ' '.join(f'{sd}{ext[sd]*100:.0f}cm' for sd in ext) + f", 천장 {s['ceil']:.2f}->{ceil_y:.2f}", flush=True)
        del S
    # ---- 1b) 문 구멍을 반듯한 사각형으로: 문마다 양쪽 면(복도 쪽, 방 쪽)에서 찾은 구멍을 모아 실제 문 폭/높이를 정하고
    #          두 면 모두 같은 사각형으로 뚫음. 아무 쪽에서도 못 찾으면 kSpaces 의 문 칸 그대로
    OPP_ = {'N': 'S', 'S': 'N', 'W': 'E', 'E': 'W'}
    def grid(r):
        h_, w_ = r['D0'].shape
        return np.meshgrid(r['u0'] + (np.arange(w_) + 0.5) * DRES, r['v0'] + (np.arange(h_) + 0.5) * DRES)
    done_doors = set()
    for r in [x for x in recs if x['hole'] is not None or x['part'] in OPP_]:
        if not r['main'] or r['part'] not in OPP_: continue
        sp = spaces[r['k']]
        for c, oa, ob, top in sp['open']:
            if c != r['part'] or top >= DOOR_MAX: continue
            dk = (round(r['p'], 1), round((oa + ob) / 2, 1))
            if dk in done_doors: continue
            done_doors.add(dk)
            mates = [r] + [x for x in recs if x['main'] and x['k'] != r['k'] and x['part'] == OPP_[r['part']]
                           and abs(x['p'] - r['p']) < 0.3 and x['u0'] < ob and x['u1'] > oa]
            us, vt = [], []
            for x in mates:
                if x['hole'] is None: continue
                U_, V_ = grid(x)
                inr = (U_[0] > oa - 0.12) & (U_[0] < ob + 0.12)
                colf = x['hole'][:, inr].mean(0)
                if (colf > 0.25).sum() >= 4:
                    uu = U_[0][inr][colf > 0.25]; us += [uu.min() - DRES / 2, uu.max() + DRES / 2]
                    vt.append(V_[:, 0][x['hole'][:, inr].any(1)].max() + DRES / 2)
            a_, b_ = (min(us), max(us)) if us else (oa, ob)
            t_ = min(max(vt), top + 0.05) if vt else top
            if t_ < 1.7: t_ = top
            for x in mates:
                U_, V_ = grid(x)
                rect = (U_ > a_) & (U_ < b_) & (V_ < t_)
                if x['hole'] is None: x['hole'] = np.zeros(x['D0'].shape, bool)
                near_door = (U_ > oa - 0.3) & (U_ < ob + 0.3)
                x['hole'] = np.where(near_door, rect, x['hole'])
            print(f"  문 {sp['name']} {c} {oa:.2f}~{ob:.2f}: 실제 {a_:.2f}~{b_:.2f}, 높이 {t_:.2f}" + ('' if us else ' (못 찾음: 상자 값)'))
            sp.setdefault('_doors', []).append((c, a_, b_, t_))
    # ---- 2) 마주 보는 두 공간의 벽 면이 서로 엇갈린 곳(스캔끼리 위치가 어긋나 두 공간이 겹친 곳): 두 면을 가운데로.
    #         겹친 부분의 스플랫은 어차피 양쪽 다 잘리므로(서로 상대 공간 안) 면을 당겨도 가리는 것이 없음.
    OPP = {'N': 'S', 'S': 'N', 'W': 'E', 'E': 'W'}
    for A in recs:
        if not A['main'] or A['part'] not in OPP: continue
        ua, va, na = AX[A['pl']]
        hh, ww = A['D0'].shape
        uu = A['u0'] + (np.arange(ww) + 0.5) * DRES; vv = A['v0'] + (np.arange(hh) + 0.5) * DRES
        U, V = np.meshgrid(uu, vv)
        fA = A['p'] + A['sg'] * A['D0']
        for Bm in recs:
            if not Bm['main'] or Bm['i'] == A['i'] or Bm['part'] != OPP[A['part']]: continue      # 다른 스캔끼리만
            sb = spaces[Bm['k']]
            cen = ((sb['x0'] + sb['x1']) / 2, 0, (sb['z0'] + sb['z1']) / 2)[na]
            if (cen - A['p']) * A['sg'] > 0: continue                 # B 는 A 의 이 벽 바깥쪽에 있는 공간이어야 함
            inB = np.zeros(U.shape, bool)
            for (r0, r1, s0, s1) in Bm['rects']: inB |= (U >= r0) & (U <= r1) & (V >= s0) & (V <= s1)
            if not inB.any(): continue
            fB = Bm['p'] + Bm['sg'] * sample(Bm['D0'], U.ravel(), V.ravel(), Bm['u0'], Bm['v0'], DRES).reshape(U.shape)
            o = (fB - fA) * A['sg']                                    # >0: B 의 면이 A 공간 안쪽에 있음 = 겹침
            m = inB & (o > 0) & (o < 0.35)
            if m.any():
                A['D'] = np.where(m, np.maximum(A['D'], A['D0'] + o / 2 + 0.005), A['D'])
                print(f"  겹침 정리: {A['key']} <-> {Bm['key']}: {m.mean()*100:.0f}% 칸, 최대 {o[m].max()*100:.0f}cm")
    maps = {r['key']: r for r in recs if r['main']}
    # ---- 3) 텍스처 + 메쉬
    mesh = []; dbg = []
    bykey = {}
    for r in recs: bykey.setdefault(r['key'], []).append(r)
    for i in range(1, 7):
        S = splats(load(src, i, tag))
        for key, rs in bykey.items():
            if rs[0]['i'] != i: continue
            r0 = rs[0]; pl = r0['pl']; ua, va, na = AX[pl]
            u0, u1, v0, v1 = r0['u0'], r0['u1'], r0['v0'], r0['v1']
            W_ = int(np.ceil((u1 - u0) / TEX)) + 1; H_ = int(np.ceil((v1 - v0) / TEX)) + 1
            C = np.zeros((H_, W_, 6)); A = np.zeros((H_, W_, 2))
            V = []; UV = []; TRI = []
            for r in rs:
                p, sg, D = r['p'], r['sg'], r['D']
                # 텍스처: 벽 표면 층(원래 면 10cm 뒤 ~ 20cm 앞). 그 층이 비면 그 앞에 보이는 것까지
                Dt = ndzoom(r['D0'], (H_ / D.shape[0], W_ / D.shape[1]), order=1, mode='nearest')[:H_, :W_]
                cd, Td, cn, Tn = view(S, pl, p, 0.0, sg, (u0, u1, v0, v1), float(r['D0'].min()) - 0.12, 0.9, TEX,
                                       color=True, near=Dt + 0.20, far=Dt - 0.10)[1:]
                c = np.concatenate([cn, cd], -1); a_ = np.stack([1 - Tn, 1 - Td], -1)     # (벽 층, 앞쪽까지) 를 같이 들고 감
                for (q0, q1, w0, w1) in r['rects']:                    # 이 평면에 속한 조각 범위에만 그림
                    j0, j1 = int((q0 - u0) / TEX), int(np.ceil((q1 - u0) / TEX)) + 1
                    i0, i1 = int((w0 - v0) / TEX), int(np.ceil((w1 - v0) / TEX)) + 1
                    C[i0:i1, j0:j1] = c[i0:i1, j0:j1]; A[i0:i1, j0:j1] = a_[i0:i1, j0:j1]
                # 메쉬: 조각마다 10cm 격자, 꼭짓점을 면 깊이만큼 법선 방향으로 옮김
                for (q0, q1, w0, w1) in r['rects']:
                    nu = max(1, int(np.ceil((q1 - q0) / 0.10))); nv = max(1, int(np.ceil((w1 - w0) / 0.10)))
                    gu, gv = np.meshgrid(np.linspace(q0, q1, nu + 1), np.linspace(w0, w1, nv + 1))
                    gd = sample(D, gu.ravel(), gv.ravel(), u0, v0, DRES)
                    pts = np.zeros((gu.size, 3)); pts[:, ua] = gu.ravel(); pts[:, va] = gv.ravel(); pts[:, na] = p + sg * gd
                    base = len(V) // 3
                    V += pts.ravel().tolist()
                    UV += np.stack([(gu.ravel() - u0) / (u1 - u0), (gv.ravel() - v0) / (v1 - v0)], 1).ravel().tolist()
                    ii, jj = np.meshgrid(np.arange(nv), np.arange(nu), indexing='ij')
                    a00 = (ii * (nu + 1) + jj).ravel() + base; a01 = a00 + 1; a10 = a00 + nu + 1; a11 = a10 + 1
                    n = np.zeros(3); n[na] = sg
                    eu = np.zeros(3); eu[ua] = 1; ev = np.zeros(3); ev[va] = 1
                    flip = np.dot(np.cross(eu, ev), n) < 0          # Unity: cross(b-a, c-a) 가 보이는 쪽
                    t = np.stack([a00, a01, a11, a00, a11, a10], 1) if not flip else np.stack([a00, a11, a01, a00, a10, a11], 1)
                    if r.get('hole') is not None:                          # 문 구멍 칸의 삼각형은 뺌
                        cu_ = (gu[:-1, :-1] + gu[1:, 1:]).ravel() / 2; cv_ = (gv[:-1, :-1] + gv[1:, 1:]).ravel() / 2
                        hj = np.clip(((cu_ - u0) / DRES).astype(int), 0, r['hole'].shape[1] - 1)
                        hi = np.clip(((cv_ - v0) / DRES).astype(int), 0, r['hole'].shape[0] - 1)
                        t = t[~np.repeat(r['hole'][hi, hj], 1)]
                    TRI += t.ravel().tolist()
                dbg.append(dict(k=r['k'], key=key, pl=pl, p=p, sg=sg, u0=u0, u1=u1, v0=v0, v1=v1, D=D, rects=r['rects'], hole=r.get('hole')))
            # 벽 층이 있는 곳은 그 층, 덜 덮인 만큼은 (앞쪽에 보이는 것 -> 그래도 없으면 주변에서 부드럽게 이어 붙인 색)
            under = inpaint(C[..., 3:], A[..., 1])
            img = C[..., :3] + (1 - A[..., :1]) * under
            name = re.sub(r'[^A-Za-z0-9]+', '_', key).strip('_')
            Image.fromarray((np.clip(img[::-1], 0, 1) * 255 + 0.5).astype(np.uint8)).save(os.path.join(out_dir, 'Baked', name + '.png'))
            nn = [0.0, 0.0, 0.0]; nn[na] = float(r0['sg'])
            mesh.append(dict(key=key, tex=name, n=nn, v=V, uv=UV, t=TRI, k=r0['k']))
        print(f'  [{i}] 텍스처/메쉬 완료', flush=True)
        del S
    # 다른 공간 안으로 파고든 면 조각(모서리/바닥/천장을 늘린 부분)은 잘라냄:
    # 한쪽 면만 보이는 판이라도 옆 공간 안에서는 정면으로 보일 수 있음
    for f in mesh:
        V = np.array(f['v']).reshape(-1, 3); T = np.array(f['t']).reshape(-1, 3)
        bad = np.zeros(len(V), bool)
        for k2, s2 in enumerate(spaces):
            if k2 != f['k']: bad |= inside(V, s2, maps, -0.01)
        drop = bad[T].any(1)                                  # 꼭짓점 하나라도 옆 공간 안이면 그 삼각형은 뺌
        f['t'] = T[~drop].ravel().tolist(); f['v'] = [round(x, 4) for x in f['v']]; f['uv'] = [round(x, 5) for x in f['uv']]
        del f['k']
    mesh = [f for f in mesh if f['t']]
    json.dump({'faces': mesh}, open(os.path.join(out_dir, 'mesh.json'), 'w'))
    if PREVIEW:                                   # 미리보기 렌더러(확인용)에서 쓰는 자료
        import pickle
        pickle.dump(dict(faces=dbg, spaces=spaces, maps=maps), open(os.path.join(out_dir, 'preview_faces.pkl'), 'wb'))
    made = {f['tex'] + '.png' for f in mesh}      # 이번에 안 쓰는 예전 텍스처는 치움 (이 스크립트가 만든 것)
    for fn in os.listdir(os.path.join(out_dir, 'Baked')):
        if fn.endswith('.png') and fn not in made: os.remove(os.path.join(out_dir, 'Baked', fn))
    return maps


def inside(P, s, maps, grow):
    """공간 s 의 면(벽·천장) 안쪽인지. grow>0 이면 그만큼 넓게"""
    m = (P[:, 1] > -grow)
    cm = maps.get(f"{s['name']}|ceiling")
    top = s['ceil'] - sample(cm['D'], P[:, 0], P[:, 2], cm['u0'], cm['v0'], DRES) if cm else s['ceil']
    m &= P[:, 1] < top + grow
    for side in 'NSWE':
        ax, key, sg = SIDES[side]
        d = (P[:, ax] - s[key]) * sg
        mp = maps.get(f"{s['name']}|{side}")
        if mp is None or is_full_open(s, side):
            face = 0.0
        else:
            u = P[:, 2] if ax == 0 else P[:, 0]
            face = sample(mp['D'], u, P[:, 1], mp['u0'], mp['v0'], DRES)
            r = next((r for r in s['recess'] if r[0] == side), None)
            if r is not None:                                   # 창문 오목부: 창턱 위로는 더 깊음
                face = np.where(P[:, 1] > r[1], np.minimum(face, (r[2] - s[key]) * sg - 0.3), face)
            face = np.where(open_mask(s, side, u, P[:, 1]), 0.0, face)    # 열린 곳/문: 경계는 상자 평면
        m &= d > face - grow
    return m


def crop_merge(src, spaces, maps, ply, tag):
    parts = []
    for i in range(1, 7):
        a = load(src, i, tag)
        P = np.stack([a['x'], a['y'], a['z']], 1).astype(np.float64)
        own = np.zeros(len(a), bool)
        for s in spaces:
            if s['owner'] == i: own |= inside(P, s, maps, MARGIN)
        # 다른 공간 안 = 지움. 중심만 보지 않고, 스플랫이 퍼진 끝(가장 긴 두 축으로 ±2σ)이 다른 공간 안으로 들어가도 지움:
        # 벽/문틀 근처의 큰 스플랫은 중심이 벽 뒤에 있어도 옆 공간 쪽으로 번져 보임 (방 초록색이 복도 벽에 비치는 것)
        R = quat_mat(np.stack([a['rot_0'], a['rot_1'], a['rot_2'], a['rot_3']], 1).astype(np.float64))
        sc = np.exp(np.stack([a['scale_0'], a['scale_1'], a['scale_2']], 1).astype(np.float64))
        order = np.argsort(-sc, 1)
        tips = [P]
        for r in range(2):
            ax_ = order[:, r]
            v = R[np.arange(len(a)), :, ax_] * (2 * sc[np.arange(len(a)), ax_])[:, None]
            tips += [P + v, P - v]
        for s in spaces:
            if s['owner'] != i:
                for T in tips: own &= ~inside(T, s, maps, 0.0)
        del R
        # 문(방-복도 사이 출입구): 문틀 평면 너머(상대 공간 쪽)에 있는 이 스캔의 스플랫은 지움. 문 너머로 멀리 찍힌 것이라
        # 흐릿한 덩어리(부유물)가 되어 문을 지날 때 화면을 가림. 그쪽은 상대 스캔이 가까이서 찍은 것으로 보임
        for s in spaces:
            if s['owner'] != i: continue
            for side, oa, ob, top in s['open']:
                if top >= DOOR_MAX: continue
                ax, key, sg = SIDES[side]
                u = P[:, 2] if ax == 0 else P[:, 0]
                d = (P[:, ax] - s[key]) * sg
                ps = (u > oa - 0.10) & (u < ob + 0.10) & (P[:, 1] < top + 0.10) & (d > -0.9) & (d < 0.9)
                own &= ~(ps & (d < 0))
                # 문 칸 안의 스플랫이 문틀 밖(옆 벽/문 위)으로 번지면 벽 위에 얼룩으로 보임 -> 지움
                for T in tips[1:]:
                    tu = T[:, 2] if ax == 0 else T[:, 0]
                    own &= ~(ps & ((tu < oa - 0.06) | (tu > ob + 0.06) | (T[:, 1] > top + 0.06)))
        parts.append(a[own])
        print(f'  [{i}] {len(a):,} -> {own.sum():,} ({own.mean()*100:.0f}%)')
    m = np.concatenate(parts)
    with open(ply, 'wb') as fh:
        fh.write(('ply\nformat binary_little_endian 1.0\n' + f'element vertex {len(m)}\n' +
                  ''.join(f'property float {p}\n' for p in DT.names) + 'end_header\n').encode())
        m.tofile(fh)
    print(f'{ply}: {len(m):,} splats')


if __name__ == '__main__':
    args = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
    SRC, CS, OUT, PLY = args['src'], args['cs'], args['out'], args['ply']
    TAG = args.get('tag', '_fill')
    PREVIEW = args.get('preview') == '1'
    os.makedirs(os.path.join(OUT, 'Baked'), exist_ok=True)
    spaces = parse_spaces(CS)
    print('1/2 벽 표면 찾기 + 텍스처 + 메쉬')
    maps = build(SRC, spaces, OUT, TAG)
    print('2/2 자르고 합치기')
    crop_merge(SRC, spaces, maps, PLY, TAG)
