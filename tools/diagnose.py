"""학습 전에 입력 데이터 품질 진단 (영상 / 프레임 / COLMAP 모델 / PLY)
3DGS 결과가 흐리거나 뭉개질 때 원인이 촬영·추출·정합·학습 중 어디인지 숫자로 확인하는 용도.

사용 (필요한 것만):
  python tools/diagnose.py --video F:\\Sample\\4차\\0_영상\\1.MP4
  python tools/diagnose.py --frames F:\\Sample\\4차\\up\\1a
  python tools/diagnose.py --model F:\\Sample\\4차\\up\\1b\\data        (sparse\\0 자동으로 찾음, .bin/.txt)
  python tools/diagnose.py --ply 기준=a.ply 새것=b.ply
  여러 개 같이 써도 됨. --ffprobe 로 ffprobe.exe 경로 지정 (기본: PATH 의 ffprobe)

보는 것
  영상  : 코덱, 비트 깊이, 색 전달 특성(HLG/PQ 면 경고), fps, 비트레이트, 밝기·대비·클리핑
  프레임: 선명도를 960px(pick_best 기준)와 원본 해상도에서 각각 재서 비교 -> 줄인 크기에서만 선명한 프레임 비율,
          밝기 변화(자동 노출 흔들림), 가장 흐린 프레임 목록
  모델  : 카메라 모델·초점·왜곡, 등록 장수, 재투영 오차, 트랙 길이, 사진당 3D 점 수(적은 사진 = 흰 벽만 보는 사진),
          촬영 궤적의 높이 변화(한 높이로만 찍었는지), 시선 방향(앞만 봤는지 / 벽을 정면으로 본 사진 비율)
          + 순간이동 검사: 촬영 순서(파일명의 원본 프레임 번호)대로 이동 속도를 재서, 중앙값보다 훨씬 빠르게
            튄 사진 = 반복되는 문·복도 때문에 엉뚱한 곳에 정합된 사진 후보 (학습 때 벽을 녹인 것처럼 뭉갬)
  카메라: --cameras-json  학습기가 남긴 3DGS cameras.json 도 같은 궤적 검사 가능
  PLY   : tools/compare_runs.py 와 같은 표
"""
import argparse
import glob
import json
import os
import shutil
import struct
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# =============== 영상 ===============
def probe(video, ffprobe):
    exe = ffprobe or shutil.which("ffprobe")
    if not exe:
        print("  (ffprobe 없음: 색 정보는 건너뜀. F:\\Sample\\tools\\ffmpeg_tool 의 ffprobe.exe 를 --ffprobe 로 지정)")
        return
    try:
        out = subprocess.run([exe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=codec_name,profile,pix_fmt,width,height,r_frame_rate,bit_rate,"
                              "color_transfer,color_primaries,color_space,color_range:format=duration,bit_rate",
                              "-of", "json", video], capture_output=True, text=True, timeout=60)
        j = json.loads(out.stdout or "{}")
    except Exception as e:  # noqa: BLE001
        print(f"  ffprobe 실패: {e}")
        return
    s = (j.get("streams") or [{}])[0]
    f = j.get("format", {})
    br = int(s.get("bit_rate") or f.get("bit_rate") or 0) / 1e6
    print(f"  코덱 {s.get('codec_name')} {s.get('profile')}, {s.get('width')}x{s.get('height')}, "
          f"{s.get('r_frame_rate')} fps, {br:.0f} Mbps, {float(f.get('duration', 0)):.0f}s")
    print(f"  픽셀 형식 {s.get('pix_fmt')}, 색: transfer={s.get('color_transfer')} "
          f"primaries={s.get('color_primaries')} range={s.get('color_range')}")
    tr = str(s.get("color_transfer"))
    if tr in ("arib-std-b67", "smpte2084"):
        print("  [경고] HDR 영상(HLG/PQ). OpenCV 는 톤매핑 없이 8bit 로 읽어 색이 물빠지고 대비가 낮아짐 "
              "-> ffmpeg 로 SDR 변환 후 추출할 것")
    if "10" in str(s.get("pix_fmt")):
        print("  [주의] 10bit 영상. D-Log M 같은 로그 프로필이면 대비가 낮아 특징점·학습 모두 불리 "
              "(아래 대비 수치가 낮으면 의심)")


def sample_video(video, n):
    import cv2
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"  열 수 없음: {video}")
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    idx = np.linspace(0, max(total - 1, 0), n).astype(int) if total else range(n)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append((f"frame {int(i)}", fr))
    return frames


# =============== 프레임 공통 지표 ===============
def lap(gray):
    import cv2
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def frame_metrics(img):
    import cv2
    h, w = img.shape[:2]
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = 960.0 / max(h, w)
    g960 = cv2.resize(gray_full, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    # 원본 해상도 선명도는 4등분 타일의 최댓값 (흰 벽 타일 때문에 낮게 나오는 것 방지)
    th, tw = h // 2, w // 2
    tiles = [gray_full[y:y + th, x:x + tw] for y in (0, th) for x in (0, tw)]
    # 실효 해상도: 가운데 정사각형의 방사 파워 스펙트럼이 고주파 잡음 바닥(나이퀴스트 근처)의 3배 아래로
    # 떨어지는 주파수 -> 그 위로는 디테일이 없음 (압축·노이즈 제거·블러로 사라짐)
    sq = min(h, w, 2048)
    c = gray_full[(h - sq) // 2:(h + sq) // 2, (w - sq) // 2:(w + sq) // 2].astype(np.float64)
    c = (c - c.mean()) * np.hanning(sq)[:, None] * np.hanning(sq)[None, :]
    P = np.abs(np.fft.fftshift(np.fft.fft2(c))) ** 2
    yy, xx = np.indices(P.shape)
    r = np.hypot(yy - sq / 2, xx - sq / 2) / (sq / 2)
    bands = [P[(r > b) & (r <= b + 0.05)].mean() for b in np.arange(0.05, 1.0, 0.05)]
    cut = next((0.05 + 0.05 * i for i, b in enumerate(bands) if b < 3 * bands[-1]), 1.0)
    return dict(sharp960=lap(g960), sharp_full=max(lap(t) for t in tiles), eff=cut,
                bright=float(np.median(gray_full)),
                contrast=float(np.percentile(gray_full, 95) - np.percentile(gray_full, 5)),
                dark=float(np.mean(gray_full < 10) * 100), clip=float(np.mean(gray_full > 250) * 100),
                w=w, h=h)


def report_frames(items, label):
    if not items:
        return
    m = [(name, frame_metrics(img)) for name, img in items]
    a = {k: np.array([x[1][k] for x in m]) for k in m[0][1]}
    print(f"  {label}: {len(m)}장 표본, 해상도 {int(a['w'][0])}x{int(a['h'][0])}")
    print(f"  밝기 중앙 {np.median(a['bright']):.0f}/255 (표본 간 범위 {a['bright'].min():.0f}~{a['bright'].max():.0f}), "
          f"대비(p95-p5) 중앙 {np.median(a['contrast']):.0f}, 검게 뭉갬 {np.mean(a['dark']):.1f}%, "
          f"하얗게 날아감 {np.mean(a['clip']):.1f}%")
    if np.median(a["contrast"]) < 80:
        print("  [주의] 대비가 낮음 (로그/HDR 프로필, 노출 부족, 물빠진 변환 의심)")
    if a["bright"].max() - a["bright"].min() > 60:
        print("  [주의] 표본 간 밝기 차이가 큼 -> 자동 노출 흔들림 가능. 학습에서 노출 보정(PPISP 등) 필요")
    # 960px 에서 선명해 보여도 원본에서는 흐린 프레임: 두 지표의 순위 차이로 봄
    r960 = a["sharp960"].argsort().argsort() / max(len(m) - 1, 1)
    rful = a["sharp_full"].argsort().argsort() / max(len(m) - 1, 1)
    hidden = np.mean((r960 > 0.5) & (rful < 0.25)) * 100
    eff = np.median(a["eff"])
    print(f"  선명도 960px 중앙 {np.median(a['sharp960']):.0f}, 원본 중앙 {np.median(a['sharp_full']):.0f}")
    print(f"  실효 해상도: 디테일이 나이퀴스트의 {eff:.2f} 까지 -> 가로 약 {int(a['w'][0] * eff)}px 상당 "
          f"(학습 해상도를 이보다 크게 해도 이득 적음, 훨씬 작게 하면 디테일 손실)")
    print(f"  960px 로는 상위 절반인데 원본에서는 하위 25%인 프레임: {hidden:.0f}% "
          f"(높으면 pick_best 가 미세한 흔들림을 못 거름)")
    worst = np.argsort(a["sharp_full"])[:5]
    print("  원본 기준 가장 흐린 표본: " + ", ".join(f"{m[i][0]}({a['sharp_full'][i]:.0f})" for i in worst))


def load_frames(folder, n):
    import cv2
    files = sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png")))
    if not files:
        print(f"  사진 없음: {folder}")
        return []
    pick = [files[i] for i in np.linspace(0, len(files) - 1, min(n, len(files))).astype(int)]
    out = []
    for p in pick:
        img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)  # 한글 경로
        if img is not None:
            out.append((os.path.basename(p), img))
    print(f"  폴더 전체 {len(files)}장")
    return out


# =============== COLMAP 모델 ===============
CAM_NAMES = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 3: "RADIAL", 4: "OPENCV",
             5: "OPENCV_FISHEYE", 6: "FULL_OPENCV", 7: "FOV", 8: "SIMPLE_RADIAL_FISHEYE",
             9: "RADIAL_FISHEYE", 10: "THIN_PRISM_FISHEYE"}
CAM_NP = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}


def qvec2rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def find_model(folder):
    for sub in ("", "sparse/0", "sparse", "0"):
        d = os.path.join(folder, sub)
        for ext in ("bin", "txt"):
            if os.path.isfile(os.path.join(d, "images." + ext)):
                return d, ext
    raise SystemExit(f"COLMAP 모델을 찾지 못함: {folder}")


def read_model(folder):
    d, ext = find_model(folder)
    cams, imgs = {}, []
    errs, tlen = [], []
    if ext == "bin":
        with open(os.path.join(d, "cameras.bin"), "rb") as f:
            for _ in range(struct.unpack("<Q", f.read(8))[0]):
                cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
                cams[cid] = (CAM_NAMES.get(model, str(model)), w, h,
                             struct.unpack("<%dd" % CAM_NP[model], f.read(8 * CAM_NP[model])))
        data = open(os.path.join(d, "images.bin"), "rb").read(); off = 8
        for _ in range(struct.unpack_from("<Q", data, 0)[0]):
            q = struct.unpack_from("<4d", data, off + 4); t = struct.unpack_from("<3d", data, off + 36)
            off += 64
            e = data.index(b"\0", off); name = data[off:e].decode("utf-8", "replace"); off = e + 1
            n2 = struct.unpack_from("<Q", data, off)[0]; off += 8
            pid = np.frombuffer(data, "<i8", count=n2 * 3, offset=off).reshape(-1, 3)[:, 2] if n2 else np.zeros(0)
            off += 24 * n2
            imgs.append((name, np.array(q), np.array(t), int(np.sum(pid != -1)), int(n2)))
        p3 = os.path.join(d, "points3D.bin")
        if os.path.isfile(p3):
            data = open(p3, "rb").read(); off = 8
            for _ in range(struct.unpack_from("<Q", data, 0)[0]):
                err, L = struct.unpack_from("<dQ", data, off + 35); off += 51 + 8 * L
                errs.append(err); tlen.append(L)
    else:
        for line in open(os.path.join(d, "cameras.txt"), encoding="utf-8"):
            if line.strip() and not line.startswith("#"):
                e = line.split()
                cams[int(e[0])] = (e[1], int(e[2]), int(e[3]), tuple(map(float, e[4:])))
        with open(os.path.join(d, "images.txt"), encoding="utf-8") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                e = line.split()
                pts = f.readline().split()
                pid = np.array(pts[2::3], float) if pts else np.zeros(0)
                imgs.append((" ".join(e[9:]), np.array(e[1:5], float), np.array(e[5:8], float),
                             int(np.sum(pid != -1)), len(pid)))
        p3 = os.path.join(d, "points3D.txt")
        if os.path.isfile(p3):
            for line in open(p3, encoding="utf-8"):
                if line.strip() and not line.startswith("#"):
                    e = line.split()
                    errs.append(float(e[7])); tlen.append((len(e) - 8) // 2)
    return d, cams, imgs, np.array(errs), np.array(tlen)


FRAME_RE = None


def frame_no(name):
    global FRAME_RE
    import re
    FRAME_RE = FRAME_RE or re.compile(r"f_(\d+)_(\d+)")
    m = FRAME_RE.search(os.path.basename(name))
    return int(m.group(2)) if m else None


def report_jumps(names, C, R_c2w, k=5.0):
    """촬영 순서대로 속도·회전 속도가 튀는 사진 찾기. C: 중심(N,3), R_c2w: (N,3,3)"""
    fr = np.array([frame_no(n) if frame_no(n) is not None else -1 for n in names])
    if np.sum(fr >= 0) < 10:
        print("  (파일명에서 원본 프레임 번호를 못 읽어 순간이동 검사는 건너뜀)")
        return
    o = np.argsort(fr); o = o[fr[o] >= 0]
    C, R, fr = C[o], R_c2w[o], fr[o]
    names = [names[i] for i in o]
    df = np.maximum(np.diff(fr), 1)
    v = np.linalg.norm(np.diff(C, axis=0), axis=1) / df                    # 단위/프레임
    rot = np.array([np.degrees(np.arccos(np.clip((np.trace(R[i].T @ R[i + 1]) - 1) / 2, -1, 1)))
                    for i in range(len(R) - 1)]) / df                       # 도/프레임
    mv, mr = np.median(v), np.median(rot)
    bad = np.where((v > k * max(mv, 1e-9)) & (v > np.percentile(v, 90)))[0]
    print(f"  촬영 순서 이동: 중앙 {mv:.4f} 단위/프레임, 회전 중앙 {mr:.2f}도/프레임, "
          f"중앙의 {k:g}배 넘게 튄 곳 {len(bad)}곳 (전체 {len(v)}구간)")
    for i in bad[np.argsort(-v[bad])][:15]:
        print(f"    {names[i]} -> {names[i + 1]}: {v[i] / mv:.0f}배 속도, 회전 {rot[i] * df[i]:.0f}도 / {df[i]} 프레임")
    if len(bad):
        print("  [주의] 튄 곳 앞뒤 사진은 엉뚱한 위치에 정합됐을 수 있음. SuperSplat/COLMAP GUI 로 카메라 위치 확인, "
              "맞으면 그 사진들을 *_skip 으로 빼고 다시 정합 (sequential 매칭은 이런 오류가 적음)")


def report_cameras_json(path):
    cams = json.load(open(path, encoding="utf-8"))
    names = [c.get("img_name", str(c.get("id"))) for c in cams]
    C = np.array([c["position"] for c in cams], float)
    R = np.array([c["rotation"] for c in cams], float)                     # 3DGS: camera-to-world 회전
    print(f"  카메라 {len(cams)}대, {cams[0].get('width')}x{cams[0].get('height')}, fx {cams[0].get('fx'):.1f}")
    report_jumps(names, C, R)


def report_model(folder):
    d, cams, imgs, errs, tlen = read_model(folder)
    print(f"  모델 {d}: 카메라 {len(cams)}개, 등록 사진 {len(imgs)}장, 3D 점 {len(errs):,}개")
    for cid, (model, w, h, p) in list(cams.items())[:3]:
        print(f"  카메라 {cid}: {model} {w}x{h}, 파라미터 " + ", ".join(f"{v:.4g}" for v in p))
        if model in ("PINHOLE", "SIMPLE_PINHOLE"):
            fx = p[0]
            print(f"    수평 화각 {np.degrees(2 * np.arctan(w / 2 / fx)):.1f}도")
    if len(errs):
        print(f"  재투영 오차 평균 {errs.mean():.2f}px (중앙 {np.median(errs):.2f}), 트랙 길이 평균 {tlen.mean():.1f} "
              f"(중앙 {np.median(tlen):.0f})")
        if errs.mean() > 1.0:
            print("  [주의] 재투영 오차 > 1px: 자세가 부정확해 학습 결과가 흐려질 수 있음 "
                  "(흔들림 보정/디워프, 롤링셔터, 잘못된 매칭 의심)")
    obs = np.array([x[3] for x in imgs])
    print(f"  사진당 3D 점: 중앙 {np.median(obs):.0f}, 하위 10% {np.percentile(obs, 10):.0f}, "
          f"100개 미만 사진 {np.sum(obs < 100)}장")

    # 궤적: 카메라 중심, 시선(카메라 +z) 방향
    R = [qvec2rot(q) for _, q, _, _, _ in imgs]
    C = np.array([-r.T @ t for r, (_, _, t, _, _) in zip(R, imgs)])
    fwd = np.array([r.T @ np.array([0, 0, 1.0]) for r in R])
    upc = np.array([r.T @ np.array([0, -1.0, 0]) for r in R])   # 카메라 위쪽(-y)
    up = upc.mean(0); up /= np.linalg.norm(up)                  # 촬영자가 대체로 수평을 유지했다고 가정
    h = C @ up
    Ch = C - np.outer(h, up)
    ev, evec = np.linalg.eigh(np.cov(Ch.T))
    length = 2 * np.sqrt(max(ev[-1], 0)) * 2
    print(f"  궤적: 수평 길이(대략) {length:.2f}, 높이 변화 표준편차 {h.std():.3f} "
          f"(= 길이의 {h.std() / max(length, 1e-9) * 100:.1f}%)")
    if h.std() / max(length, 1e-9) < 0.01:
        print("  [주의] 거의 한 높이에서만 촬영 -> VR 에서 눈높이·앉은 높이가 다르면 학습에 없던 시점")
    # 진행 방향 대비 시선 각도 (궤적 접선)
    order = np.argsort([x[0] for x in imgs])                     # 파일명 순 = 촬영 순
    Co = Ch[order]; Fo = fwd[order] - np.outer(fwd[order] @ up, up)
    k = max(1, len(Co) // 50)
    tang = np.array([Co[min(i + k, len(Co) - 1)] - Co[max(i - k, 0)] for i in range(len(Co))])
    ok = (np.linalg.norm(tang, axis=1) > 1e-9) & (np.linalg.norm(Fo, axis=1) > 1e-9)
    tn = tang[ok] / np.linalg.norm(tang[ok], axis=1, keepdims=True)
    fn = Fo[ok] / np.linalg.norm(Fo[ok], axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip(np.abs(np.sum(tn * fn, 1)), 0, 1)))  # 0 = 앞/뒤, 90 = 옆(벽 정면)
    pitch = np.degrees(np.arcsin(np.clip(fwd @ up, -1, 1)))
    print(f"  시선: 진행 방향과 30도 이내(앞만 봄) {np.mean(ang < 30) * 100:.0f}%, "
          f"60도 이상(벽을 정면) {np.mean(ang > 60) * 100:.0f}%, "
          f"위로 20도 이상(천장) {np.mean(pitch > 20) * 100:.0f}%, 아래로 20도 이상(바닥) {np.mean(pitch < -20) * 100:.0f}%")
    if np.mean(ang > 60) < 0.15:
        print("  [주의] 벽을 정면으로 본 사진이 적음 -> 벽 줄무늬·비스듬한 바늘형 스플랫의 주원인 (README 촬영 주의와 같음)")
    if np.mean(pitch > 20) < 0.05:
        print("  [주의] 천장을 본 사진이 거의 없음 -> 천장 구멍·반투명")
    report_jumps([x[0] for x in imgs], C, np.array([r.T for r in R]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", nargs="*", default=[])
    ap.add_argument("--frames", nargs="*", default=[])
    ap.add_argument("--model", nargs="*", default=[])
    ap.add_argument("--ply", nargs="*", default=[])
    ap.add_argument("--cameras-json", nargs="*", default=[])
    ap.add_argument("--ffprobe")
    ap.add_argument("--samples", type=int, default=40, help="영상/프레임 표본 수 (기본 40)")
    a = ap.parse_args()
    if not (a.video or a.frames or a.model or a.ply or a.cameras_json):
        ap.print_help(); return
    for v in a.video:
        print(f"[영상] {v}")
        probe(v, a.ffprobe)
        report_frames(sample_video(v, a.samples), "영상 표본")
    for fdir in a.frames:
        print(f"[프레임] {fdir}")
        report_frames(load_frames(fdir, a.samples), "프레임 표본")
    for m in a.model:
        print(f"[COLMAP] {m}")
        report_model(m)
    for cj in a.cameras_json:
        print(f"[cameras.json] {cj}")
        report_cameras_json(cj)
    if a.ply:
        print("[PLY]")
        sys.argv = [sys.argv[0]] + a.ply
        sys.path.insert(0, HERE)
        import compare_runs
        compare_runs.main()


if __name__ == "__main__":
    main()
