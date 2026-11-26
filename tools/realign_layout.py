"""재정합한 구간의 배치 변환(layout) 다시 구하기

예전 COLMAP 모델과 새 COLMAP 모델에 공통으로 있는 프레임의 카메라 중심으로
닮음 변환 T(새 COLMAP -> 예전 COLMAP, Umeyama)를 구하고, 예전 layout 변환과 곱함:
    M_up = M_old @ T      (새 COLMAP 좌표 -> 건물 좌표)

프레임 짝짓기: 파일명 f_<순번>_<원본프레임번호>.jpg 의 원본 번호.
  같은 번호가 없으면 가장 가까운 번호 (차이 <= --tol 프레임, 한 프레임은 한 번만 씀).

사용:
  python tools/realign_layout.py --section 1 --old F:\\Sample\\4차\\2_정합데이터\\1b\\data --new F:\\Sample\\4차\\up\\1b\\data
  (모델 폴더는 cameras/images/points3D 가 있는 폴더, 또는 sparse\\0 이 들어 있는 상위 폴더. .bin/.txt 둘 다 됨)

옵션:
  --layout  예전 layout (기본 tools/layout_v4.json)
  --out     결과 (기본 tools/layout_up.json). 이미 있으면 그 파일을 바탕으로 해당 구간만 바꿈
            (구간을 하나씩 재정합해도 앞서 바꾼 구간이 유지됨). 처음엔 --layout 사본에서 시작
  --tol     원본 프레임 번호 허용 오차 (기본 3)
  --outlier-k  잔차가 중앙값의 k 배보다 큰 짝을 빼고 한 번 더 맞춤 (기본 3, 0 = 끔)
  --dry-run 파일은 쓰지 않고 결과만 출력

결과 형식은 layout_v4.json 과 같음. 바뀌는 것: matrix_4x4, scale(열 길이), pos_xz(이동의 x,z).
yaw_deg 는 행렬에서 다시 구할 수 없는 값(바닥 정렬 위에 사람이 정한 방향)이라 예전 값 유지
- 건물 안 배치 방향은 그대로이므로. unity_fix_4x4 등 나머지 키는 그대로 복사.
짝 개수·잔차 기록은 최상위 "_realign" 키에 남김 (밑줄 키는 배치 코드가 읽지 않음).
"""
import argparse
import datetime
import json
import os
import re
import struct
import sys

import numpy as np


# ---------------- COLMAP 모델 읽기 (images 만: 이름, 자세) ----------------
def qvec2rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def read_images_bin(path):
    out = {}
    data = open(path, "rb").read()
    off = 0
    n = struct.unpack_from("<Q", data, off)[0]; off += 8
    for _ in range(n):
        off += 4  # image id
        q = struct.unpack_from("<4d", data, off); off += 32
        t = struct.unpack_from("<3d", data, off); off += 24
        off += 4  # camera id
        e = data.index(b"\0", off); name = data[off:e].decode("utf-8"); off = e + 1
        n2 = struct.unpack_from("<Q", data, off)[0]; off += 8 + 24 * n2
        out[name] = (np.array(q), np.array(t))
    return out


def read_images_txt(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            e = line.split()
            q = np.array(list(map(float, e[1:5])))
            t = np.array(list(map(float, e[5:8])))
            name = " ".join(e[9:])
            out[name] = (q, t)
            f.readline()  # 2D 점 줄 (비어 있을 수 있음)
    return out


def find_model(folder):
    for sub in ("", "sparse/0", "sparse", "0"):
        d = os.path.join(folder, sub)
        for ext in ("bin", "txt"):
            if os.path.isfile(os.path.join(d, "images." + ext)):
                return d, ext
    raise SystemExit(f"COLMAP 모델(images.bin/.txt)을 찾지 못함: {folder}")


def camera_centers(folder):
    d, ext = find_model(folder)
    imgs = read_images_bin(os.path.join(d, "images.bin")) if ext == "bin" \
        else read_images_txt(os.path.join(d, "images.txt"))
    # 카메라 중심 C = -R^T t
    return d, {name: -qvec2rot(q).T @ t for name, (q, t) in imgs.items()}


# ---------------- 짝짓기 ----------------
NAME_RE = re.compile(r"f_(\d+)_(\d+)\.[A-Za-z]+$")


def frame_no(name):
    m = NAME_RE.search(os.path.basename(name.replace("\\", "/")))
    return int(m.group(2)) if m else None


def by_frame(centers):
    d = {}
    for name, c in centers.items():
        k = frame_no(name)
        if k is not None:
            d[k] = (name, c)
    return d


def pair_frames(old, new, tol):
    """old/new: {원본번호: (이름, 중심)} -> [(new번호, old번호)]"""
    pairs = [(k, k) for k in new if k in old]
    used_o = {k for k, _ in pairs}
    left_n = sorted(k for k in new if k not in old)
    left_o = np.array(sorted(k for k in old if k not in used_o))
    near = 0
    if tol > 0 and len(left_n) and len(left_o):
        cand = []
        for kn in left_n:
            i = np.searchsorted(left_o, kn)
            for j in (i - 2, i - 1, i, i + 1):
                if 0 <= j < len(left_o) and abs(int(left_o[j]) - kn) <= tol:
                    cand.append((abs(int(left_o[j]) - kn), kn, int(left_o[j])))
        cand.sort()
        used_n = set()
        for dist, kn, ko in cand:
            if kn in used_n or ko in used_o:
                continue
            pairs.append((kn, ko)); used_n.add(kn); used_o.add(ko); near += 1
    return sorted(pairs), near


# ---------------- Umeyama 닮음 변환 ----------------
def umeyama(src, dst):
    """dst ~= s R src + t 인 4x4 행렬"""
    n = len(src)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / n
    s = np.trace(np.diag(D) @ S) / var_s
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = mu_d - s * R @ mu_s
    return T


def apply(M, P):
    return P @ M[:3, :3].T + M[:3, 3]


def summary(M):
    A = np.asarray(M)[:3, :3]
    return dict(scale=round(float(np.linalg.norm(A[:, 0])), 4),
                pos_xz=[round(float(M[0][3]), 2), round(float(M[2][3]), 2)])


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--section", required=True, help='구간 키 ("1"~"6")')
    ap.add_argument("--old", required=True, help="예전 COLMAP 모델 폴더 (layout 을 만들 때 쓴 것)")
    ap.add_argument("--new", required=True, help="새 COLMAP 모델 폴더")
    ap.add_argument("--layout", default=os.path.join(here, "layout_v4.json"))
    ap.add_argument("--out", default=os.path.join(here, "layout_up.json"))
    ap.add_argument("--tol", type=int, default=3)
    ap.add_argument("--outlier-k", type=float, default=3.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    layout = json.load(open(a.layout, encoding="utf-8"))
    if a.section not in layout:
        raise SystemExit(f"layout 에 구간 {a.section} 없음 (있는 키: {[k for k in layout if not k.startswith('_')]})")
    M_old = np.array(layout[a.section]["matrix_4x4"], float)

    d_old, c_old = camera_centers(a.old)
    d_new, c_new = camera_centers(a.new)
    f_old, f_new = by_frame(c_old), by_frame(c_new)
    print(f"예전 모델 {d_old}: 사진 {len(c_old)} (이름에서 번호 읽힘 {len(f_old)})")
    print(f"새 모델   {d_new}: 사진 {len(c_new)} (이름에서 번호 읽힘 {len(f_new)})")

    pairs, near = pair_frames(f_old, f_new, a.tol)
    print(f"짝지은 프레임: {len(pairs)} (같은 번호 {len(pairs) - near}, 가까운 번호 +-{a.tol} 이내 {near})")
    if len(pairs) < 3:
        raise SystemExit("짝이 3개 미만이라 변환을 구할 수 없음 (--tol 을 늘리거나 파일명 확인)")
    if len(pairs) < 20:
        print("  [주의] 짝이 적음 - 결과를 눈으로 꼭 확인할 것")

    src = np.array([f_new[kn][1] for kn, _ in pairs])   # 새 COLMAP
    dst = np.array([f_old[ko][1] for _, ko in pairs])   # 예전 COLMAP

    def fit(mask):
        T = umeyama(src[mask], dst[mask])
        # 잔차는 건물 좌표(m)에서 잼: 예전 layout 으로 둘 다 옮긴 뒤 거리
        res = np.linalg.norm(apply(M_old, apply(T, src)) - apply(M_old, dst), axis=1) * 100
        return T, res

    mask = np.ones(len(pairs), bool)
    T, res = fit(mask)

    def report(tag, r):
        print(f"{tag}: 짝 {len(r)}, 잔차(cm) RMS {np.sqrt(np.mean(r ** 2)):.2f} / 중앙 {np.median(r):.2f} / "
              f"90% {np.percentile(r, 90):.2f} / 최대 {r.max():.2f}")

    report("1차 맞춤", res)
    if a.outlier_k > 0:
        thr = a.outlier_k * max(np.median(res), 1e-6)
        keep = res <= thr
        if (~keep).any() and keep.sum() >= 3:
            T, res_all = fit(keep)
            mask = keep
            print(f"  잔차 > {thr:.2f}cm 인 짝 {(~keep).sum()}개 빼고 다시 맞춤")
            report("2차 맞춤 (뺀 짝 제외)", res_all[keep])
            res = res_all
    s_T = np.cbrt(np.linalg.det(T[:3, :3]))
    print(f"T (새 COLMAP -> 예전 COLMAP): 축척 {s_T:.5f}")

    M_up = M_old @ T
    base = json.load(open(a.out, encoding="utf-8")) if os.path.isfile(a.out) else json.loads(json.dumps(layout))
    if os.path.isfile(a.out):
        print(f"기존 {a.out} 을 바탕으로 구간 {a.section} 만 바꿈")
    sec = dict(layout[a.section])           # 이름, yaw_deg, unity_fix_4x4 등 그대로
    sec["matrix_4x4"] = [[round(float(v), 6) for v in row] for row in M_up]
    sec.update(summary(M_up))
    # 키 순서를 예전 파일과 같게
    sec = {k: sec[k] for k in layout[a.section]}
    base[a.section] = sec
    info = base.get("_realign", {})
    info[a.section] = dict(
        date=datetime.date.today().isoformat(), old_model=d_old, new_model=d_new,
        pairs=int(len(pairs)), pairs_used=int(mask.sum()), near_pairs=int(near), tol=a.tol,
        residual_cm=dict(rms=round(float(np.sqrt(np.mean(res[mask] ** 2))), 2),
                         median=round(float(np.median(res[mask])), 2),
                         max=round(float(res[mask].max()), 2)),
        colmap_new_to_old_4x4=[[round(float(v), 8) for v in row] for row in T],
        note="matrix_4x4 = old matrix_4x4 @ colmap_new_to_old_4x4; yaw_deg kept from the old layout")
    base["_realign"] = dict(sorted(info.items()))

    print(f"구간 {a.section}: scale {layout[a.section]['scale']} -> {sec['scale']}, "
          f"pos_xz {layout[a.section]['pos_xz']} -> {sec['pos_xz']}, yaw_deg {sec['yaw_deg']} (유지)")
    if a.dry_run:
        print("--dry-run: 파일 안 씀")
        return
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(base, f, indent=1, ensure_ascii=False)
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
