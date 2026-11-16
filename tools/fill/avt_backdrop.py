"""4차 보강 스캔(placed_fill_1..6) 에 맞춘 배경 메쉬(바탕 벽) + 공간별 자르기

팀원 배경(StudentHallBackdrop.kSpaces)은 2차 촬영 벽에 맞춰져 있어서 4차 데이터에선 꺼져 있음.
-> 같은 공간 상자 구조(문, 창문 오목부)는 그대로 쓰고, 벽/천장 면 위치만 우리 스플랫에 다시 맞춤:
   각 면은 그 공간을 맡은 스캔의 벽 표면 바로 뒤(바깥쪽 3cm)에 놓임. 한쪽 면만 보이는 불투명 면이라
   - 얇은 벽/천장/바닥 뒤로 분홍 배경이 비치지 않고 벽 색이 보임
   - 옆 공간(다른 스캔)의 벽 뒷면이 겹쳐 보이지 않음 (면이 가림)
면 색은 그 면 앞쪽 스플랫 색을 구워서 텍스처로 만듦 (팀원 bake.py 와 같은 방식/형식).

두 공간이 같은 벽을 사이에 둘 때 두 면이 엇갈리면(스캔끼리 위치가 몇 cm 어긋난 경우) 둘 다 두 표면의 가운데로.
자르기: 각 스캔은 자기 공간(다시 맞춘 상자 + 0.3m) 안만 남기고, 다른 공간 상자 안(2cm 이상)은 지움.

사용: python avt_backdrop.py src=<placed_fill_1..6 폴더> cs=<AVT\\Assets\\Editor\\StudentHallBackdrop.cs>
                             out=<AVT\\Assets\\Backdrop\\v4_fill> ply=<합본 출력 .ply> [tag=_fill]
"""
import json, os, re, sys
import numpy as np
from PIL import Image

args = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
SRC, CS, OUT, PLY = args['src'], args['cs'], args['out'], args['ply']
TAG = args.get('tag', '_fill')
os.makedirs(os.path.join(OUT, 'Baked'), exist_ok=True)

OWN = {1: ['West stair hall', 'West corridor (west)', 'West corridor (33B116)', 'West corridor (east)'],
       2: ['Junction hall', 'East stair hall', 'East corridor'],
       3: ['33B102-103'], 4: ['33B101'], 5: ['33B117'], 6: ['33B115']}
OWNER = {n: i for i, ns in OWN.items() for n in ns}
FLOOR_Y = -0.03
CORR_CEIL, FULL = 3.02, 99.0
BEHIND = 0.03        # 면을 벽 표면보다 이만큼 바깥에
MARGIN = 0.30        # 자르기: 자기 상자 밖으로 남기는 폭
INWARD = 0.10       # 면을 공간 안쪽으로는 최대 이만큼만 (실제 벽을 가리지 않게)
TEX = 0.015          # 텍스처 1픽셀 = 1.5cm

# ---------------------------------------------------------------- kSpaces 읽기
src = open(CS, encoding='utf-8-sig').read().replace('\r\n', '\n')
body = src.split('kSpaces =')[1].split('};\n\n')[0]
num = r'(-?[\d.]+f|kFull|kCorridorCeil)'
def val(s):
    return FULL if s == 'kFull' else CORR_CEIL if s == 'kCorridorCeil' else float(s.rstrip('f'))
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
orig = json.loads(json.dumps(spaces))

# ---------------------------------------------------------------- 스캔 읽기
DT = np.dtype([(n, '<f4') for n in ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
                                    'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']])
def load(i):
    p = os.path.join(SRC, f'placed{TAG}_{i}.ply')
    with open(p, 'rb') as f:
        h = b''
        while not h.endswith(b'end_header\n'):
            h += f.readline()
        n = int(re.search(rb'element vertex (\d+)', h).group(1))
        return np.fromfile(f, DT, n)

SIDES = {  # side: (축, 상자 키, 안쪽 방향 부호)
    'N': (0, 'x0', +1), 'S': (0, 'x1', -1), 'W': (2, 'z0', +1), 'E': (2, 'z1', -1)}
SH0 = 0.28209479177387814

def quat_mat(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)], -1),
                     np.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)], -1),
                     np.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)], -1)], 1)

def splats(i):
    """스캔 i 의 스플랫: 위치, 색, 불투명도, 회전행렬, 크기^2 (불투명도 0.05 이상)"""
    a = load(i)
    o = 1 / (1 + np.exp(-a['opacity'])); m = o > 0.05
    S = dict(P=np.stack([a['x'], a['y'], a['z']], 1)[m].astype(np.float64),
             col=np.clip(0.5 + SH0 * np.stack([a['f_dc_0'], a['f_dc_1'], a['f_dc_2']], 1)[m], 0, 1),
             alpha=o[m],
             R=quat_mat(np.stack([a['rot_0'], a['rot_1'], a['rot_2'], a['rot_3']], 1)[m].astype(np.float64)),
             S2=np.exp(2 * np.stack([a['scale_0'], a['scale_1'], a['scale_2']], 1)[m].astype(np.float64)))
    return S

def raster(U, V, cov, alpha, k, K, u0, v0, res, W_, H_, col=None):
    """스플랫을 평면에 정사영으로 찍음(가우시안 모양 그대로), 깊이 층(k)별로 모음.
    -> 덮음(K,H,W) [, 색 합(K,H,W,3)]"""
    cv_ = np.zeros((K, H_, W_)); acc = np.zeros((K, H_, W_, 3)) if col is not None else None
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
            iu = iu[ok].astype(int); iv = iv[ok].astype(int); g = g[ok]
            np.add.at(cv_, (kk, iv, iu), g)
            if col is not None:
                cc = np.broadcast_to(col[idx][:, None, :], ok.shape + (3,))[ok]
                for ch in range(3): np.add.at(acc[..., ch], (kk, iv, iu), g * cc[:, ch])
    return cv_, acc

AX = {'y': (0, 2, 1), 'x': (2, 1, 0), 'z': (0, 1, 2)}   # 평면 종류 -> (u 축, v 축, 법선 축)
PLANE = {'N': 'x', 'S': 'x', 'W': 'z', 'E': 'z'}

def view(S, pl, pv, sg, rect, dmin, dmax, res, step=0.02, color=False, keep=None):
    """평면(pl=pv) 을 안쪽(+sg)에서 바깥쪽으로 바라본 정사영 '사진'.
    rect=(u0,u1,v0,v1). 안쪽 거리 d 가 dmax..dmin 인 스플랫을 앞에서부터 겹쳐 그림.
    -> 처음 불투명(투과<0.5)해지는 거리 맵, [색, 투과율]"""
    ua, va, na = AX[pl]; u0, u1, v0, v1 = rect
    P = S['P']; d = (P[:, na] - pv) * sg
    m = (d > dmin) & (d < dmax) & (P[:, ua] > u0 - 0.3) & (P[:, ua] < u1 + 0.3) & (P[:, va] > v0 - 0.3) & (P[:, va] < v1 + 0.3)
    if keep is not None: m &= keep
    W_ = int(np.ceil((u1 - u0) / res)) + 1; H_ = int(np.ceil((v1 - v0) / res)) + 1
    K = int(np.ceil((dmax - dmin) / step))
    k = np.clip(((dmax - d[m]) / step).astype(int), 0, K - 1)        # 0 = 가장 안쪽(앞)
    Rk = S['R'][m][:, [ua, va], :]
    cov = np.einsum('nik,nk,njk->nij', Rk, S['S2'][m], Rk)
    cv_, acc = raster(P[m, ua], P[m, va], cov, S['alpha'][m], k, K, u0, v0, res, W_, H_, S['col'][m] if color else None)
    T = np.ones((H_, W_)); depth = np.full((H_, W_), np.nan); C = np.zeros((H_, W_, 3))
    for kk in range(K):
        al = 1 - np.exp(-cv_[kk])
        if color:
            c = acc[kk] / np.maximum(cv_[kk], 1e-9)[..., None]
            C += (T * al)[..., None] * c
        T = T * (1 - al)
        hit = np.isnan(depth) & (T < 0.5)
        depth[hit] = dmax - (kk + 0.5) * step
    return depth, C, T

# ---------------------------------------------------------------- 면 다시 맞추기
# 각 면 자리에서 공간 안쪽 -> 바깥쪽으로 바라본 정사영 '사진'을 찍어, 칸마다 처음 불투명해지는 깊이(= 사람이 보는
# 벽 표면)를 구함. 면은 그 깊이들의 97% 보다 뒤(바깥)에 둠 -> 보이는 벽을 가리지 않음. 벽 뒤에 숨은 부스러기는 무시됨.
DRES = 0.05
fits = {}  # (space idx, side) -> (표면 거리(중앙값), 면 거리)
ceil_fit = {}
def deep_in_others(P, i, g=0.3):
    m = np.zeros(len(P), bool)
    for s in orig:
        if s['owner'] != i:
            m |= (P[:, 0] > s['x0'] + g) & (P[:, 0] < s['x1'] - g) & (P[:, 2] > s['z0'] + g) & (P[:, 2] < s['z1'] - g)
    return m

def fit_face(S, keep, pl, pv, sg, rects, lim, mask_fn=None):
    ds = []
    for rect in rects:
        dep, _, _ = view(S, pl, pv, sg, rect, -0.6, 0.8, DRES, keep=keep)
        if mask_fn is not None: dep = np.where(mask_fn(rect, dep.shape), dep, np.nan)
        ds.append(dep[np.isfinite(dep)])
    d = np.concatenate(ds) if ds else np.zeros(0)
    if len(d) < 40: return None
    face = np.percentile(d, 3) - BEHIND
    return float(np.median(d)), float(np.clip(face, -lim, INWARD)), len(d)

for i in range(1, 7):
    S = splats(i)
    keep = ~deep_in_others(S['P'], i)      # 문 너머로 보인 다른 공간 (30cm 넘게 안쪽) 은 뺌
    mine = [k for k, s in enumerate(spaces) if s['owner'] == i]
    groups = {}
    for k in mine:
        s = spaces[k]
        for side in 'NSWE':
            full = any(c == side and ot >= FULL - 1 and oa <= (s['z0'] if SIDES[side][0] == 0 else s['x0']) + 0.01
                       and ob >= (s['z1'] if SIDES[side][0] == 0 else s['x1']) - 0.01 for c, oa, ob, ot in s['open'])
            if full: continue
            groups.setdefault((side, round(s[SIDES[side][1]], 2)), []).append(k)
    for (side, p), ks in groups.items():
        ax, key, sg = SIDES[side]
        rects = []
        for k in ks:
            s = spaces[k]
            a0, a1 = (s['z0'], s['z1']) if ax == 0 else (s['x0'], s['x1'])
            top = min(2.2, s['ceil'] - 0.3)
            rec = [r for r in s['recess'] if r[0] == side]
            if rec: top = min(top, rec[0][1] - 0.1)
            cuts = [a0 + 0.3]
            segs = []
            ops = sorted([(oa - 0.15, ob + 0.15) for c, oa, ob, ot in s['open'] if c == side and ot > 0.5])
            cur = a0 + 0.3
            for oa, ob in ops:
                if oa > cur: segs.append((cur, min(oa, a1 - 0.3)))
                cur = max(cur, ob)
            if cur < a1 - 0.3: segs.append((cur, a1 - 0.3))
            rects += [(u, w, 0.3, top) for u, w in segs if w - u > 0.2]
        r = fit_face(S, keep, PLANE[side], p, sg, rects, 0.6) if rects else None
        if r is None:
            print(f'  [{i}] {side} {p}: 보이는 벽 부족 - 그대로'); continue
        surf, face, n = r
        for k in ks: fits[(k, side)] = (surf, face)
        print(f'  [{i}] {",".join(spaces[k]["name"] for k in ks)} {side}: 면 {face*100:+.1f}cm (보이는 표면 중앙 {surf*100:+.1f}cm, {n}칸)')
    cg = {}
    for k in mine: cg.setdefault(round(spaces[k]['ceil'], 2), []).append(k)
    for c0, ks in cg.items():
        rects = [(spaces[k]['x0'] + 0.3, spaces[k]['x1'] - 0.3, spaces[k]['z0'] + 0.3, spaces[k]['z1'] - 0.3) for k in ks]
        r = fit_face(S, keep, 'y', c0, -1, [q for q in rects if q[1] > q[0] and q[3] > q[2]], 0.7)
        if r is None: print(f'  [{i}] 천장 {c0}: 보이는 천장 부족'); continue
        surf, face, n = r
        for k in ks: ceil_fit[k] = c0 - face; spaces[k]['_bake_ceiling'] = surf - face
        print(f'  [{i}] 천장 {c0} -> {c0 - face:.2f} (보이는 천장 중앙 {c0 - surf:.2f}, {n}칸)')
    del S

for (k, side), (surf, face) in fits.items():
    spaces[k]['_bake_' + side] = surf - face
    ax, key, sg = SIDES[side]
    spaces[k][key] = orig[k][key] + sg * face
    spaces[k]['_surf_' + side] = orig[k][key] + sg * surf
for k, c in ceil_fit.items():
    spaces[k]['ceil'] = c

# 다른 스캔끼리 같은 벽을 사이에 둔 면: 엇갈리면 두 표면 가운데로
OPP = {'N': 'S', 'S': 'N', 'W': 'E', 'E': 'W'}
for k, s in enumerate(spaces):
    for side in 'NS' 'WE':
        if (k, side) not in fits: continue
        ax, key, sg = SIDES[side]
        for k2, s2 in enumerate(spaces):
            if k2 == k or (k2, OPP[side]) not in fits: continue
            key2 = SIDES[OPP[side]][1]
            if abs(orig[k][key] - orig[k2][key2]) > 0.25: continue
            lo = max(s['z0'] if ax == 0 else s['x0'], s2['z0'] if ax == 0 else s2['x0'])
            hi = min(s['z1'] if ax == 0 else s['x1'], s2['z1'] if ax == 0 else s2['x1'])
            if hi - lo < 0.3: continue
            # s 의 면은 sg 방향(안쪽)의 반대쪽 = 바깥, s2 는 반대. 엇갈림 = s 의 면이 s2 의 면보다 s2 쪽으로 더 나감
            if (s[key] - s2[key2]) * sg < 0:
                # 엇갈려도 옮기지 않음: 면을 자기 벽 표면 앞으로 당기면 실제 벽을 가림 (12:59 확인).
                # 각 면은 한쪽에서만 보이므로 엇갈린 채로 둬도 서로 안 보임.
                print(f'  엇갈림(그대로 둠): {s["name"]} {side} / {s2["name"]} {OPP[side]}')

# 공간 사이 열린 면의 높이(복도 천장 기준)도 새 천장으로
for k, s in enumerate(spaces):
    for o in s['open']:
        if abs(o[3] - CORR_CEIL) < 1e-3:
            o[3] = min(x['ceil'] for x in spaces if x['look'] == 'Corridor' and x['owner'] == s['owner']
                       and abs(orig[spaces.index(x)]['ceil'] - CORR_CEIL) < 1e-3) if s['owner'] in (1, 2) else o[3]
    # 오목부(창문) 면도 벽이 움직인 만큼 같이
    for r in s['recess']:
        key = SIDES[r[0]][1]
        r[2] += s[key] - orig[k][key]

print('\n공간       x0     x1     z0     z1   천장  (괄호: 원래)')
for s, o in zip(spaces, orig):
    print(f"{s['name'][:22]:22s} " + ' '.join(f"{s[k]:6.2f}({s[k]-o[k]:+.2f})" for k in ['x0', 'x1', 'z0', 'z1', 'ceil']))

# ---------------------------------------------------------------- 면 목록 (StudentHallBackdrop.BuildSide 와 같은 모양)
def faces_of(s):
    """key -> 리스트[(평면 종류, 평면값, u0,u1,v0,v1, 안쪽 법선)]. 평면 종류 'x'|'z'|'y'"""
    F = {}
    add = lambda key, *q: F.setdefault(f"{s['name']}|{key}", []).append(q)
    fy = FLOOR_Y if s['look'] == 'Corridor' else FLOOR_Y - 0.005
    add('floor', 'y', fy, s['x0'], s['x1'], s['z0'], s['z1'], +1)
    rec = lambda side, face: next((r[2] for r in s['recess'] if r[0] == side), face)
    add('ceiling', 'y', s['ceil'], rec('N', s['x0']), rec('S', s['x1']), rec('W', s['z0']), rec('E', s['z1']), -1)
    for side in 'NSWE':
        ax, key, sg = SIDES[side]
        alongZ = ax == 0
        p = s[key]; a0, a1 = (s['z0'], s['z1']) if alongZ else (s['x0'], s['x1'])
        op = [o for o in s['open'] if o[0] == side]
        r = next((r for r in s['recess'] if r[0] == side), None)
        sill, q = (r[1], r[2]) if r else (s['ceil'], p)
        cuts = sorted(set([min(max(c, a0), a1) for o in op for c in (o[1], o[2])] + [a0, a1]))
        pl = 'x' if alongZ else 'z'
        for u, w in zip(cuts, cuts[1:]):
            mid = (u + w) / 2
            top = max([o[3] for o in op if o[1] <= mid <= o[2]], default=FLOOR_Y - 0.01)
            if min(sill, s['ceil']) - top > 0.005 and w - u > 0.005:
                add(side, pl, p, u, w, top, min(sill, s['ceil']), sg)
            if not r: continue
            if s['ceil'] - max(top, sill) > 0.005:
                add(side + '|upper', pl, q, u, w, max(top, sill), s['ceil'], sg)
            if top < sill:
                add(side + '|sill', 'y', sill, min(p, q), max(p, q), u, w, +1) if alongZ else \
                    add(side + '|sill', 'y', sill, u, w, min(p, q), max(p, q), +1)
        if r:
            lo, hi = min(p, q), max(p, q)
            if alongZ:
                add('W', 'z', a0, lo, hi, sill, s['ceil'], +1); add('E', 'z', a1, lo, hi, sill, s['ceil'], -1)
            else:
                add('N', 'x', a0, lo, hi, sill, s['ceil'], +1); add('S', 'x', a1, lo, hi, sill, s['ceil'], -1)
    return F

# ---------------------------------------------------------------- 텍스처 굽기
def bake_all():
    allfaces = {}
    for s in spaces:
        for key, qs in faces_of(s).items():
            allfaces[key] = (s, qs)
    out = []
    for i in range(1, 7):
        S = splats(i)
        for key, (s, qs) in allfaces.items():
            if s['owner'] != i: continue
            pl = qs[0][0]
            u0 = min(q[2] for q in qs); u1 = max(q[3] for q in qs)
            v0 = min(q[4] for q in qs); v1 = max(q[5] for q in qs)
            W_ = int(np.ceil((u1 - u0) / TEX)) + 1; H_ = int(np.ceil((v1 - v0) / TEX)) + 1
            C = np.zeros((H_, W_, 3)); A = np.zeros((H_, W_))
            for (_, pv, a0_, a1_, b0_, b1_, sg) in qs:
                # 면 앞에서 '벽 표면 층'까지만 (표면 중앙 + 15cm). 그 앞의 배관/물건까지 벽에 그려 넣으면
                # 비스듬히 볼 때 스플랫과 어긋나 얼룩처럼 보임
                part = key.split('|')[1]
                dmax = min(0.9, max(0.15, s.get('_bake_' + part, 0.15) + 0.15))
                _, c, T = view(S, pl, pv, sg, (u0, u1, v0, v1), -0.03, dmax, TEX, color=True)
                use = (1 - T) > A
                C[use] = c[use]; A[use] = (1 - T)[use]
            # 덮인 정도로 나눠 원래 밝기로, 덜 덮인 곳은 주변 색으로 채움
            img = fill(C, A)
            name = re.sub(r'[^A-Za-z0-9]+', '_', key).strip('_')
            Image.fromarray((np.clip(img[::-1], 0, 1) * 255 + 0.5).astype(np.uint8)).save(os.path.join(OUT, 'Baked', name + '.png'))
            out.append(dict(key=key, tex=name, u0=u0, u1=u1, v0=v0, v1=v1))
        print(f'  [{i}] 굽기 완료', flush=True)
        del S
    json.dump({'faces': out}, open(os.path.join(OUT, 'Baked', 'faces.json'), 'w'), indent=1)

def blur(x, sig):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(x, sigma=(sig, sig) + (0,) * (x.ndim - 2), mode='nearest')

def fill(C, A):
    """C: 미리 곱한 색, A: 덮인 정도(0..1). 잘 덮인 곳은 C/A, 나머지는 점점 넓은 평균으로 채움"""
    if A.sum() < 1e-6: return np.zeros_like(C) + 0.35
    img = np.zeros_like(C)
    have = A > 0.5
    img[have] = C[have] / A[have][:, None]
    for sig in (1.0, 2.0, 4, 8, 16, 32, 64, 128, 256):
        a = blur(C, sig); b = blur(A, sig)
        ok = (b > 0.15) & ~have
        img[ok] = a[ok] / b[ok][:, None]
        have |= ok
        if have.all(): break
    img[~have] = C.reshape(-1, 3).sum(0) / A.sum()
    return img

bake_all()

# ---------------------------------------------------------------- Unity 용 공간 목록
js = []
for s in spaces:
    js.append(dict(name=s['name'], look=s['look'], x0=s['x0'], x1=s['x1'], z0=s['z0'], z1=s['z1'], ceil=s['ceil'],
                   open=[dict(side=o[0], a=o[1], b=o[2], top=o[3]) for o in s['open']],
                   recess=[dict(side=r[0], sill=r[1], face=r[2]) for r in s['recess']]))
# 팀원 배경의 kReveals(교차로-동쪽 계단홀 사이 10cm 턱)는 면을 다시 맞추면서 두 면이 붙으므로 안 씀
json.dump({'spaces': js}, open(os.path.join(OUT, 'spaces.json'), 'w'), indent=1)

# ---------------------------------------------------------------- 자르고 합치기
def inside(P, s, grow):
    return (P[:, 0] > s['x0'] - grow) & (P[:, 0] < s['x1'] + grow) & (P[:, 2] > s['z0'] - grow) & \
           (P[:, 2] < s['z1'] + grow) & (P[:, 1] > -grow) & (P[:, 1] < s['ceil'] + grow)
total = 0
parts = []
for i in range(1, 7):
    a = load(i)
    P = np.stack([a['x'], a['y'], a['z']], 1)
    own = np.zeros(len(a), bool)
    for s in spaces:
        if s['owner'] == i: own |= inside(P, s, MARGIN)
    # 다른 공간 = 원래 상자와 다시 맞춘 상자를 합친 범위. 면을 바깥으로 민 만큼 그 사이에 남은
    # 다른 스캔 스플랫이 이 공간 안에서 보이면(벽 비침/게시판 겹침) 안 되므로 그 사이도 지움.
    # 이 스캔의 진짜 벽이 거기 걸려 지워져도, 그 자리 배경 면에 벽 모습이 구워져 있음
    for k, s in enumerate(spaces):
        if s['owner'] == i: continue
        o = orig[k]
        u = dict(x0=min(s['x0'], o['x0']), x1=max(s['x1'], o['x1']), z0=min(s['z0'], o['z0']),
                 z1=max(s['z1'], o['z1']), ceil=max(s['ceil'], o['ceil']))
        own &= ~inside(P, u, -0.01)
    parts.append(a[own]); total += own.sum()
    print(f'  [{i}] {len(a):,} -> {own.sum():,} ({own.mean()*100:.0f}%)')
    # 공간별 PLY (나중에 공간별로 켜고 끄기용)
    del a, P
m = np.concatenate(parts)
with open(PLY, 'wb') as fh:
    fh.write(('ply\nformat binary_little_endian 1.0\n' + f'element vertex {len(m)}\n' +
              ''.join(f'property float {p}\n' for p in DT.names) + 'end_header\n').encode())
    m.tofile(fh)
print(f'{PLY}: {len(m):,} splats')
