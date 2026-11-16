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
TEX = 0.02           # 텍스처 1픽셀 = 2cm

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

def surf_points(a):
    """벽 맞추기용: 불투명하고 작은 스플랫 중심"""
    o = 1 / (1 + np.exp(-a['opacity']))
    s = np.exp(np.maximum(np.maximum(a['scale_0'], a['scale_1']), a['scale_2']))
    m = (o > 0.3) & (s < 0.15)
    return np.stack([a['x'], a['y'], a['z']], 1)[m].astype(np.float64)

SIDES = {  # side: (축, 상자 키, 안쪽 방향 부호)
    'N': (0, 'x0', +1), 'S': (0, 'x1', -1), 'W': (2, 'z0', +1), 'E': (2, 'z1', -1)}

def side_band(s, side, P, depth=0.6):
    """그 면 근처 점의 (안쪽 거리 +면 안쪽, 벽 방향 위치). 문 구간/모서리 0.3m 는 뺌."""
    ax, key, sg = SIDES[side]
    p = s[key]; along = 2 if ax == 0 else 0
    a0, a1 = (s['z0'], s['z1']) if ax == 0 else (s['x0'], s['x1'])
    top = min(2.0, s['ceil'] - 0.3)
    rec = [r for r in s['recess'] if r[0] == side]
    if rec: top = min(top, rec[0][1] - 0.1)
    m = (P[:, along] > a0 + 0.3) & (P[:, along] < a1 - 0.3) & (P[:, 1] > 0.3) & (P[:, 1] < top)
    for c, oa, ob, ot in s['open']:
        if c == side and ot > 0.5:
            m &= ~((P[:, along] > oa - 0.15) & (P[:, along] < ob + 0.15))
    d = (P[:, ax] - p) * sg
    m &= (d > -depth) & (d < depth)
    return d[m], np.stack([P[m, along], P[m, 1]], 1)

def surface(d, pos, cell=0.5, lim=0.35):
    """안쪽 거리 d 의 층들 중, 면 전체에 넓게 퍼진(=진짜 벽/천장) 층 가운데 가장 바깥쪽.
    pos: 각 점의 면 위 위치(2D) - 층이 얼마나 넓게 덮는지 재는 데 씀. -> (표면, 면) 또는 None"""
    if len(d) < 150: return None
    e = np.arange(-0.6, 0.6001, 0.01)
    h, _ = np.histogram(d, bins=e)
    hs = np.convolve(h, np.ones(5) / 5, 'same')
    peaks = [k for k in range(1, len(hs) - 1) if hs[k] >= 0.08 * hs.max() and hs[k] >= hs[k - 1] and hs[k] >= hs[k + 1]]
    cells = np.unique(np.floor(pos / cell).astype(int), axis=0)
    def coverage(c):
        q = pos[np.abs(d - c) < 0.06]
        if len(q) == 0: return 0.0
        u, n = np.unique(np.floor(q / cell).astype(int), axis=0, return_counts=True)
        return (n >= 3).sum() / len(cells)
    cand = [((e[k] + e[k + 1]) / 2, coverage((e[k] + e[k + 1]) / 2)) for k in peaks]
    best = max(c for _, c in cand)
    c = min(cc for cc, cov in cand if cov >= 0.45 * best)     # 넓게 덮는 층 중 가장 바깥
    near = d[np.abs(d - c) < 0.08]
    surf = np.percentile(near, 10)
    return surf, float(np.clip(surf - BEHIND, -lim, INWARD))

# ---------------------------------------------------------------- 면 다시 맞추기
fits = {}  # (space idx, side) -> (표면 거리, 면 거리)
ceil_fit = {}
for i in range(1, 7):
    a = load(i); P = surf_points(a); del a
    # 다른 공간 상자 안의 점(문 너머로 보인 것, 떠다니는 것)은 어차피 잘려 나가니 맞추기에서도 뺌
    for s in orig:
        if s['owner'] != i:
            P = P[~((P[:, 0] > s['x0'] + 0.02) & (P[:, 0] < s['x1'] - 0.02) & (P[:, 2] > s['z0'] + 0.02) & (P[:, 2] < s['z1'] - 0.02))]
    mine = [k for k, s in enumerate(spaces) if s['owner'] == i]
    # 같은 스캔, 같은 평면의 면끼리 묶어서 함께 맞춤 (복도 조각들이 한 벽을 나눠 가짐)
    groups = {}
    for k in mine:
        s = spaces[k]
        for side in 'NSWE':
            full = any(c == side and ot >= FULL - 1 and oa <= (s['z0'] if SIDES[side][0] == 0 else s['x0']) + 0.01
                       and ob >= (s['z1'] if SIDES[side][0] == 0 else s['x1']) - 0.01 for c, oa, ob, ot in s['open'])
            if full: continue
            groups.setdefault((side, round(s[SIDES[side][1]], 2)), []).append(k)
    for (side, p), ks in groups.items():
        bands = [side_band(spaces[k], side, P) for k in ks]
        d = np.concatenate([b[0] for b in bands]); pos = np.concatenate([b[1] for b in bands])
        r = surface(d, pos)
        if r is None:
            print(f'  [{i}] {side} {p}: 점 부족({len(d)}) - 그대로'); continue
        surf, face = r
        for k in ks: fits[(k, side)] = (surf, face)
        print(f'  [{i}] {",".join(spaces[k]["name"] for k in ks)} {side}: 면 이동 {face*100:+.1f}cm (점 {len(d)})')
    # 천장: 같은 스캔, 같은 높이끼리
    cg = {}
    for k in mine: cg.setdefault(round(spaces[k]['ceil'], 2), []).append(k)
    for c0, ks in cg.items():
        ys = []; xz = []
        for k in ks:
            s = spaces[k]
            m = (P[:, 0] > s['x0'] + 0.3) & (P[:, 0] < s['x1'] - 0.3) & (P[:, 2] > s['z0'] + 0.3) & (P[:, 2] < s['z1'] - 0.3)
            ys.append(P[m, 1]); xz.append(P[m][:, [0, 2]])
        y = np.concatenate(ys); xz = np.concatenate(xz)
        d = c0 - y                                   # 천장에서 아래로 거리 (+ = 안쪽)
        m = (d > -0.6) & (d < 0.6)
        r = surface(d[m], xz[m], cell=1.0, lim=0.6)
        if r is None: print(f'  [{i}] 천장 {c0}: 점 부족'); continue
        surf, face = r
        for k in ks: ceil_fit[k] = c0 - face
        print(f'  [{i}] 천장 {c0} -> {c0 - face:.2f} ({",".join(spaces[k]["name"] for k in ks)})')
    del P

for (k, side), (surf, face) in fits.items():
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
                # 다른 스캔: 두 벽 표면의 가운데 (팀원 배경과 같은 규칙) / 같은 스캔: 두 면의 가운데
                mid = (s['_surf_' + side] + s2['_surf_' + OPP[side]]) / 2 if s2['owner'] != s['owner'] \
                    else (s[key] + s2[key2]) / 2
                print(f'  엇갈림 보정: {s["name"]} {side} / {s2["name"]} {OPP[side]} -> {mid:.3f}')
                s[key] = s2[key2] = mid

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
SH0 = 0.28209479177387814
def bake_all():
    allfaces = {}
    for s in spaces:
        for key, qs in faces_of(s).items():
            allfaces[key] = (s, qs)
    out = []
    for i in range(1, 7):
        a = load(i)
        o = 1 / (1 + np.exp(-a['opacity']))
        m = o > 0.15
        P = np.stack([a['x'], a['y'], a['z']], 1)[m]
        col = np.clip(0.5 + SH0 * np.stack([a['f_dc_0'], a['f_dc_1'], a['f_dc_2']], 1)[m], 0, 1)
        w = o[m]
        del a
        for key, (s, qs) in allfaces.items():
            if s['owner'] != i: continue
            pl = qs[0][0]
            u0 = min(q[2] for q in qs); u1 = max(q[3] for q in qs)
            v0 = min(q[4] for q in qs); v1 = max(q[5] for q in qs)
            if pl == 'y':      # 바닥/천장: q=(x,z)
                U, V, N = P[:, 0], P[:, 2], P[:, 1]
            elif pl == 'x':    # N/S 벽: q=(z,y)
                U, V, N = P[:, 2], P[:, 1], P[:, 0]
            else:              # W/E 벽: q=(x,y)
                U, V, N = P[:, 0], P[:, 1], P[:, 2]
            W_ = int(np.ceil((u1 - u0) / TEX)) + 1; H_ = int(np.ceil((v1 - v0) / TEX)) + 1
            acc = np.zeros((H_, W_, 3)); wt = np.zeros((H_, W_))
            for (_, pv, a0_, a1_, b0_, b1_, sg) in qs:
                dist = (N - pv) * sg            # 면 안쪽 거리
                depth = 0.12 if pl == 'y' else 0.30
                mm = (dist > -0.05) & (dist < depth) & (U > a0_ - 0.05) & (U < a1_ + 0.05) & (V > b0_ - 0.05) & (V < b1_ + 0.05)
                iu = np.clip(((U[mm] - u0) / TEX).astype(int), 0, W_ - 1)
                iv = np.clip(((V[mm] - v0) / TEX).astype(int), 0, H_ - 1)
                ww = w[mm] * np.exp(-np.maximum(dist[mm], 0) / 0.1)   # 면에 가까운 층을 더 믿음
                np.add.at(wt, (iv, iu), ww)
                for c in range(3): np.add.at(acc[..., c], (iv, iu), ww * col[mm, c])
            img = fill(acc, wt)
            name = re.sub(r'[^A-Za-z0-9]+', '_', key).strip('_')
            Image.fromarray((np.clip(img[::-1], 0, 1) * 255 + 0.5).astype(np.uint8)).save(os.path.join(OUT, 'Baked', name + '.png'))
            out.append(dict(key=key, tex=name, u0=u0, u1=u1, v0=v0, v1=v1))
        print(f'  [{i}] 굽기 완료')
    json.dump({'faces': out}, open(os.path.join(OUT, 'Baked', 'faces.json'), 'w'), indent=1)

def blur(x, sig):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(x, sigma=(sig, sig) + (0,) * (x.ndim - 2), mode='nearest')

def fill(acc, wt):
    """정규화 가우시안: 점이 있는 곳은 살짝 부드럽게, 빈 곳은 점점 넓은 반경의 평균으로 채움"""
    if wt.sum() == 0: return np.zeros_like(acc) + 0.35
    img = np.zeros_like(acc); have = np.zeros(wt.shape, bool)
    for sig in (1.0, 2.0, 4, 8, 16, 32, 64, 128, 256):
        a = blur(acc, sig); b = blur(wt, sig)
        ok = (b > 1e-3 * np.median(wt[wt > 0]) / (sig * sig)) & ~have
        img[ok] = a[ok] / b[ok][:, None]
        have |= ok
        if have.all(): break
    img[~have] = acc.reshape(-1, 3).sum(0) / wt.sum()
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
    for s in spaces:
        if s['owner'] != i: own &= ~inside(P, s, -0.02)
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
