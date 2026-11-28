"""v4 에 옆으로 돌려 찍는 구간 추가: t=40 초 둘레 요 +90°(가우스, 폭 1.5 s) -> <out>/ (바뀐 사진만 다시 렌더, 나머지는 v4 사진 링크).

  python make_pan.py <v4 장면 폴더> <out>
"""
import json, math, os, sys, shutil
from pathlib import Path
import numpy as np, cv2
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'prior_matching')); sys.path.insert(0, str(HERE.parents[1] / 'src'))
import multiprocessing as mp
import scene_v4 as scene
V = Path(sys.argv[1]).resolve() if len(sys.argv) > 2 else None
O = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None
TC, AMP, W = 40.0, math.radians(90), 1.5
_T = _B = None
def _init():
    global _T, _B
    _T, _B = scene.make_textures(0), scene.make_boxes(0)
def _one(a):
    i, name, R, C = a
    img = scene.render(_T, _B, R, C, rng=np.random.default_rng(1000 + i))
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 92]); (O / 'images' / name).write_bytes(buf.tobytes())
if __name__ == '__main__':
    if V is None:
        sys.exit(__doc__)
    g = json.load(open(V / 'gt_poses.json'))
    (O / 'images').mkdir(parents=True, exist_ok=True)
    (O / 'priors').mkdir(exist_ok=True)
    for f in (V / 'priors').iterdir(): shutil.copy2(f, O / 'priors' / f.name)
    shutil.copy2(V / 'gt_planes.json', O / 'gt_planes.json')
    todo = []
    for i, fr in enumerate(g['frames']):
        dy = AMP * math.exp(-((fr['t'] - TC) / W) ** 2)
        if dy > math.radians(1):
            Rz = np.array([[math.cos(dy), -math.sin(dy), 0], [math.sin(dy), math.cos(dy), 0], [0, 0, 1]])
            R = Rz @ np.array(fr['R_world_from_cam']); fr['R_world_from_cam'] = R.tolist()
            todo.append((i, fr['image'], R, np.array(fr['center'])))
        elif not (O / 'images' / fr['image']).exists():
            os.symlink((V / 'images' / fr['image']).resolve(), O / 'images' / fr['image'])
    json.dump(g, open(O / 'gt_poses.json', 'w'))
    print('rerender', len(todo), [t[1] for t in todo[:1]], [t[1] for t in todo[-1:]], flush=True)
    with mp.Pool(4, initializer=_init) as p: p.map(_one, todo, chunksize=2)
    print('done')
