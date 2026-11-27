"""면 텍스처: 도면 면(벽·바닥·천장) 위 (u, v) 직사각형에 사진을 투영해 만든 색 이미지 + 못 본 곳 메우기.

학습 뒤 면 메우기(repair)가 새 가우시안의 색을 정할 때 쓰는 '색 출처'. 지운 아이디어: 포토샵의 내용 인식 채우기처럼
학습된 스플랫의 빈 곳·비치는 곳을 바로 메움 (Difix 식 가짜 학습 시점이 아님). 색 출처 우선순위:
  1) 실제 사진을 면에 투영 (찍혔지만 복원이 나빴던 곳 - 정확)          source = 1
  2) 예시 기반 패치 메우기 (PatchMatch 계열, 사진이 못 본 곳)          source = 2
  3) 생성형 AI (나중, 지금은 없음)
  사람이 PNG 를 고쳐 알파 > 0 으로 칠한 빈 곳                            source = 3

공개 API
  FaceTexture                                        image (H,W,3) uint8 RGB, source (H,W) uint8, support (H,W) uint16,
                                                     inside (H,W) bool, res (m/픽셀). .sample(uv) -> (rgb float, source)
  texture_grid(plane, res)                           -> (H, W, 픽셀 중심 uv (H,W,2), inside)
  face_texture_from_photos(plane, model, image_dir, *, res=0.015, min_views=2, max_views=8, max_dist=8,
                           max_incidence_deg=75, max_cameras=300, occluders=None, ...) -> FaceTexture
  inpaint_texture(tex, mask_to_fill=None, *, method="patchmatch", patch=7, max_fill_m2=4.0, ...)
                                                     -> (새 FaceTexture, 메운 곳 bool (H,W))
  build_face_textures(planes, model, image_dir, *, inpaint=True, **opts) -> dict[plane id, FaceTexture]
  save_textures(textures, out_dir) / load_textures(dir, planes=None)   PNG + JSON (walls/tex/<id>.png)
  save_texture(tex, out_dir) / load_texture(png_or_json, plane=None)

픽셀 격자
  W = ceil(size_u / res), H = ceil(size_v / res). 배열 행 i, 열 j 의 중심 = ((j + .5) res, (i + .5) res).
  배열은 행 0 = v 0 (repair 격자와 같음). PNG 로 쓸 때만 위아래를 뒤집어 위 = 큰 v (벽은 천장 쪽이 위로 보임).
  inside = plane.contains_uv(중심) (문 구멍·다각형 밖은 False). inside 가 아닌 픽셀은 늘 source 0, 검정.

사진 투영 (face_texture_from_photos, build_face_textures 는 여러 면을 사진 한 번 읽어 같이)
  보임 판정은 walls/seed.py 와 같음: 카메라 앞 (z > 0.1), 사진 안, 거리 < max_dist, 입사각 < max_incidence_deg,
  카메라가 면 안쪽, 다른 면(occluders, 문 구멍은 열림)에 가리지 않음. 카메라가 max_cameras 보다 많으면 이름 순
  k 번째마다. 사진 읽기는 cv2.imdecode(np.fromfile) (한글 경로), BGR -> RGB, 사진을 줄여 둔 경우 배율 보정.
  - 앨리어싱: 텍셀 하나가 사진에서 차지하는 크기 fx·res/z (픽셀) 로 피라미드 단(cv2.pyrDown) L = floor(log2) 을 골라
    그 단에서 쌍선형 보간 (4K 사진을 1.5 cm 텍셀로 줄일 때 점 샘플링 깜빡임 방지).
  - 시점 고르기: 점수 = cos(입사각) / max(거리, 0.3) · 선명도^sharpness_exp. 선명도 = 사진(긴 변 960 으로 줄인 회색)
    라플라시안 분산 (영상 프레임의 흔들림 블러를 덜 고름). 픽셀마다 점수 상위 max_views 개 보기를 남김.
  - 노출 보정 (Osmo 자동 노출): 사진마다 채널별 이득 g_v 하나. 픽셀별 중앙값 m_p 에 대해
    log g_v = median_p(log m_p - log c_vp) (보기 2개 이상, 어둡거나 포화된 값 제외), 이득 적용 후 중앙값 다시 - 2 번 반복.
    log 이득의 평균은 0 으로 (전체 밝기는 사진들의 평균 노출). 이득은 [0.5, 2] 로 자름. 여러 면을 같이 만들 때는
    모든 면의 샘플로 함께 추정 (노출은 사진마다라서). gain_comp=False 로 끔.
  - 합성: 보정된 상위 보기 색의 채널별 중앙값 (blend="median", 기본; 사람·이음새에 강함) 또는 중앙값에서
    3 MAD 안의 보기 중 점수 최고 하나 (blend="best", 더 선명하지만 시점 경계 이음새가 생길 수 있음).
  - 보기가 min_views 미만인 픽셀은 source 0 (support 에는 개수 기록). 도면에 없는 가만있는 물체(소화기 등)는
    벽 텍스처에 찍혀 들어감 (사진이 본 그대로).

패치 메우기 (inpaint_texture, method="patchmatch")
  Wexler-Shechtman-Irani (2007) 식 EM + PatchMatch (Barnes 2009) 최근접 패치 탐색, numpy 로 벡터화.
  - 채울 곳 = mask_to_fill (없으면 inside & source == 0) & inside & source != 1 (사진 픽셀은 안 바꿈).
    8-이웃 성분 하나의 넓이가 max_fill_m2 를 넘으면 그 성분은 안 메움 (None 이면 제한 없음).
  - 원천 패치: 패치 전체가 '아는 픽셀'(inside & source in {1,2,3} & 채울 곳 아님) 안인 중심.
  - 다중 해상도: 구멍 안 가장 먼 점까지 거리가 패치 반경 + 1 이하가 될 때까지 1/2 로 (최대 max_levels).
    가장 거친 단은 Telea 로 시작, 위 단으로는 NNF 를 2 배로 올려 투표로 시작.
  - 한 단: em_iters 번 [PatchMatch pm_iters 번 (전파: ±1, ±2, ±4, ±8 픽셀 이웃의 짝을 옮겨 시도 = 벡터화한 jump
    flooding, 무작위 탐색: 반경 max(H,W) 부터 반씩 + 전역 무작위 2 개) -> 투표 (겹치는 패치 색의 가중 평균,
    가중치 exp(-d / 2σ²) · 1.3^-(대상 중심에서 아는 픽셀까지 거리), σ² = 패치 거리 75 백분위; 거리 가중은 Wexler 의
    경계 신뢰도로 아는 무늬의 위상을 안쪽으로 끌고 들어감 - 합성 측정에서 경계 근처 오차 5.7 -> 2.7)].
    가장 고운 단은 반복을 줄임.
  - 한계: 아는 곳과 좁게만 이어진 넓은 구멍(문 뒤 띠 등)은 주기는 잇지만 위상은 틀릴 수 있음 (테스트에 측정값).
  - 패치 거리: 대상 패치의 inside 픽셀만 (문 구멍 쪽 값은 안 셈), 채널 합 SSD 평균.
  method="telea": cv2.inpaint (INPAINT_TELEA). 모르는 픽셀 전체를 마스크로 주고 채울 곳만 가져옴 (문 구멍 검정이 번지지 않게).

저장 (save_texture / load_texture)
  <out_dir>/<id>.png        RGB 텍스처 (위 = 큰 v). 포토샵에서 고쳐도 됨. 알파 채널을 넣으면 알파 0 = 지움(source 0),
                            알파 > 0 인데 원래 source 0 이었던 inside 픽셀 = 손으로 칠함(source 3).
  <out_dir>/<id>.src.png    source 0/1/2/3 (회색 0/255·1/3 단위로 보기 좋게: 0, 85, 170, 255)
  <out_dir>/<id>.support.png 16 bit 보기 수
  <out_dir>/<id>.json       plane_id, res, size, shape, 출처 통계, PNG 의 sha1 (불러올 때 다르면 meta["edited"] = True)
  파일 이름의 ':' 는 '_' 로 (윈도우). 크기가 다른 PNG 는 원래 크기로 되돌림 (INTER_AREA/LINEAR).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..colmap import model as cm
from .planes import Plane, plane_from_json
from .seed import EPS_OCC, camera_intrinsics

__all__ = ["FaceTexture", "texture_grid", "face_texture_from_photos", "inpaint_texture", "build_face_textures",
           "save_texture", "load_texture", "save_textures", "load_textures",
           "SRC_NONE", "SRC_PHOTO", "SRC_INPAINT", "SRC_MANUAL"]

SRC_NONE, SRC_PHOTO, SRC_INPAINT, SRC_MANUAL = 0, 1, 2, 3
_SRC_VIEW = 85
CONF_GAMMA = 1.3


# ---------------- 자료형 ----------------
@dataclass
class FaceTexture:
    plane_id: str
    res: float
    image: np.ndarray            # (H, W, 3) uint8 RGB, 행 0 = v 0
    source: np.ndarray           # (H, W) uint8  0 없음 / 1 사진 / 2 메움 / 3 손
    support: np.ndarray          # (H, W) uint16 보인 사진 수
    inside: np.ndarray           # (H, W) bool   면 안 (문 구멍 제외)
    size: tuple[float, float] = (0.0, 0.0)
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape[:2]

    def copy(self) -> "FaceTexture":
        return FaceTexture(self.plane_id, self.res, self.image.copy(), self.source.copy(), self.support.copy(),
                           self.inside.copy(), self.size, json.loads(json.dumps(self.meta)))

    def sample(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """면 좌표 (n,2) -> (RGB float (n,3) 쌍선형, source (n,) 가장 가까운 픽셀). 범위 밖은 source 0.
        쌍선형은 색이 있는 (source > 0) 이웃만 가중 (빈 픽셀의 검정이 섞이지 않게)."""
        uv = np.atleast_2d(np.asarray(uv, float))
        H, W = self.shape
        x = uv[:, 0] / self.res - 0.5
        y = uv[:, 1] / self.res - 0.5
        jn = np.floor(uv[:, 0] / self.res).astype(np.int64)
        in_ = np.floor(uv[:, 1] / self.res).astype(np.int64)
        ok = (jn >= 0) & (jn < W) & (in_ >= 0) & (in_ < H)
        src = np.zeros(len(uv), np.uint8)
        src[ok] = self.source[in_[ok], jn[ok]]
        x = np.clip(x, 0, W - 1)
        y = np.clip(y, 0, H - 1)
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        x1 = np.minimum(x0 + 1, W - 1)
        y1 = np.minimum(y0 + 1, H - 1)
        fx, fy = x - x0, y - y0
        img = self.image.astype(np.float64)
        has = (self.source > 0).astype(np.float64)
        acc = np.zeros((len(uv), 3))
        wsum = np.zeros(len(uv))
        for yy, xx, w in ((y0, x0, (1 - fx) * (1 - fy)), (y0, x1, fx * (1 - fy)),
                          (y1, x0, (1 - fx) * fy), (y1, x1, fx * fy)):
            ww = w * has[yy, xx]
            acc += ww[:, None] * img[yy, xx]
            wsum += ww
        near = img[np.clip(in_, 0, H - 1), np.clip(jn, 0, W - 1)]
        rgb = np.where(wsum[:, None] > 1e-9, acc / np.maximum(wsum, 1e-9)[:, None], near)
        return rgb, src

    def stats(self) -> dict:
        a = self.res * self.res
        n_in = int(self.inside.sum())
        return {"inside_m2": round(n_in * a, 4),
                "photo_m2": round(float((self.source == SRC_PHOTO).sum()) * a, 4),
                "inpaint_m2": round(float((self.source == SRC_INPAINT).sum()) * a, 4),
                "manual_m2": round(float((self.source == SRC_MANUAL).sum()) * a, 4),
                "empty_m2": round(float((self.inside & (self.source == SRC_NONE)).sum()) * a, 4)}


def texture_grid(plane: Plane, res: float):
    """(H, W, 픽셀 중심 uv (H,W,2), inside (H,W))."""
    if res <= 0:
        raise ValueError("res 는 0 보다 커야 함")
    W = max(1, int(math.ceil(plane.size[0] / res - 1e-9)))
    H = max(1, int(math.ceil(plane.size[1] / res - 1e-9)))
    uu, vv = np.meshgrid((np.arange(W) + 0.5) * res, (np.arange(H) + 0.5) * res)
    uv = np.stack([uu, vv], -1)
    inside = plane.contains_uv(uv.reshape(-1, 2)).reshape(H, W)
    return H, W, uv, inside


def _empty_texture(plane: Plane, res: float) -> FaceTexture:
    H, W, _, inside = texture_grid(plane, res)
    return FaceTexture(plane.id, float(res), np.zeros((H, W, 3), np.uint8), np.zeros((H, W), np.uint8),
                       np.zeros((H, W), np.uint16), inside, (float(plane.size[0]), float(plane.size[1])), {})


# ---------------- 사진 투영 ----------------
class _FaceAcc:
    """면 하나의 픽셀별 상위 보기 모음."""

    def __init__(self, plane: Plane, j: int, occluders: list[Plane], res: float, K: int):
        self.plane, self.j = plane, j
        self.tex = _empty_texture(plane, res)
        H, W, uv, inside = texture_grid(plane, res)
        self.pix = np.flatnonzero(inside.ravel())
        u = uv.reshape(-1, 2)[self.pix]
        X = plane.point(u[:, 0], u[:, 1])
        self.X = X - plane.signed_dist(X)[:, None] * plane.n
        self.SD = (np.stack([p.signed_dist(self.X) for p in occluders], 1) if occluders
                   else np.zeros((len(self.X), 0)))
        N = len(self.X)
        self.score = np.full((N, K), -np.inf, np.float32)
        self.col = np.full((N, K, 3), np.nan, np.float32)
        self.vid = np.full((N, K), -1, np.int32)
        self.nview = np.zeros(N, np.int32)


def _sharpness(img: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = 960.0 / max(g.shape)
    if s < 1:
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_32F).var()) + 1e-3


def _bilinear_f(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (x - x0)[:, None].astype(np.float32)
    fy = (y - y0)[:, None].astype(np.float32)
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bot = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def _visible(acc: _FaceAcc, occluders: list[Plane], C, R, t, fx, fy, cx, cy, cam_w, cam_h, sdC, *,
             max_dist, cos_min):
    """seed.seed_points 와 같은 판정. 돌려줌 (idx, cosi, dist, z, px, py)."""
    X = acc.X
    if sdC[acc.j] <= 0 or len(X) == 0:
        return None
    D = X - C
    dist = np.linalg.norm(D, axis=1)
    idx = np.flatnonzero((dist < max_dist) & (dist > 1e-6))
    if len(idx) == 0:
        return None
    cosi = -(D[idx] @ acc.plane.n) / dist[idx]
    ok = cosi > cos_min
    idx, cosi = idx[ok], cosi[ok]
    Xc = X[idx] @ R.T + t
    z = Xc[:, 2]
    front = z > 0.1
    idx, cosi, Xc, z = idx[front], cosi[front], Xc[front], z[front]
    px = fx * Xc[:, 0] / z + cx
    py = fy * Xc[:, 1] / z + cy
    ins = (px >= 0) & (px <= cam_w) & (py >= 0) & (py <= cam_h)
    idx, cosi, z, px, py = idx[ins], cosi[ins], z[ins], px[ins], py[ins]
    if len(idx) == 0:
        return None
    occl = np.zeros(len(idx), bool)
    L = dist[idx]
    for q, pq in enumerate(occluders):
        if q == acc.j or sdC[q] == 0:
            continue
        sp = acc.SD[idx, q]
        jj = np.flatnonzero((np.sign(sp) != np.sign(sdC[q])) & ~occl)
        if len(jj) == 0:
            continue
        tt = sdC[q] / (sdC[q] - sp[jj])
        before = tt * L[jj] < L[jj] - EPS_OCC
        jj, tt = jj[before], tt[before]
        if len(jj) == 0:
            continue
        Hh = C + tt[:, None] * (X[idx[jj]] - C)
        occl[jj[pq.contains_uv(pq.uv(Hh))]] = True
    v = ~occl
    return idx[v], cosi[v], L[v], z[v], px[v], py[v]


def _accumulate(accs: list[_FaceAcc], occluders: list[Plane], model: cm.Model, image_dir, *, res, max_dist,
                max_incidence_deg, max_cameras, sharpness_exp, max_level=6) -> dict:
    image_dir = Path(image_dir)
    names = sorted(model.images.values(), key=lambda im: im.name)
    step = max(1, math.ceil(len(names) / max_cameras)) if max_cameras else 1
    used = names[::step]
    info = {"n_cameras": len(names), "camera_step": step, "n_cameras_used": 0, "missing_images": [],
            "views": []}
    cos_min = math.cos(math.radians(max_incidence_deg))
    for im in used:
        path = image_dir / im.name
        img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR) if path.is_file() else None
        if img is None:
            info["missing_images"].append(im.name)
            continue
        v = len(info["views"])
        info["views"].append(im.name)
        info["n_cameras_used"] += 1
        cam = model.cameras[im.camera_id]
        fx, fy, cx, cy = camera_intrinsics(cam)
        sx, sy = img.shape[1] / cam.width, img.shape[0] / cam.height
        R, t = im.R, np.asarray(im.t, float)
        C = -R.T @ t
        sdC = np.array([p.signed_dist(C) for p in occluders])
        sharp = _sharpness(img) ** sharpness_exp if sharpness_exp else 1.0
        pyr = None
        for acc in accs:
            r = _visible(acc, occluders, C, R, t, fx, fy, cx, cy, cam.width, cam.height, sdC,
                         max_dist=max_dist, cos_min=cos_min)
            if r is None:
                continue
            idx, cosi, L, z, px, py = r
            if len(idx) == 0:
                continue
            if pyr is None:
                pyr = [img]                                   # uint8 피라미드 (4K 를 float 로 바꾸지 않음)
            foot = fx * sx * res / z                      # 텍셀 하나의 사진 픽셀 크기
            lev = np.clip(np.floor(np.log2(np.maximum(foot, 1.0))), 0, max_level).astype(int)
            while len(pyr) <= lev.max():
                pyr.append(cv2.pyrDown(pyr[-1]))
            x0 = px * sx - 0.5
            y0 = py * sy - 0.5
            rgb = np.empty((len(idx), 3), np.float32)
            for L_ in np.unique(lev):
                m = lev == L_
                s = 2.0 ** L_
                rgb[m] = _bilinear_f(pyr[L_], (x0[m] + 0.5) / s - 0.5, (y0[m] + 0.5) / s - 0.5)[:, ::-1]
            acc.nview[idx] += 1
            sc = (cosi / np.maximum(L, 0.3) * sharp).astype(np.float32)
            slot = np.argmin(acc.score[idx], axis=1)
            better = sc > acc.score[idx, slot]
            ii, ss = idx[better], slot[better]
            acc.score[ii, ss] = sc[better]
            acc.col[ii, ss] = rgb[better]
            acc.vid[ii, ss] = v
    return info


def _estimate_gains(accs: list[_FaceAcc], n_views: int, iters: int = 2) -> np.ndarray:
    """사진별 채널 이득 (n_views, 3). 모듈 설명의 노출 보정."""
    logg = np.zeros((max(n_views, 1), 3))
    if n_views == 0:
        return np.exp(logg)
    for _ in range(iters):
        lr_all, vid_all = [], []
        for acc in accs:
            if not len(acc.vid):
                continue
            ok = acc.vid >= 0
            multi = ok.sum(1) >= 2
            if not multi.any():
                continue
            vid = acc.vid[multi]
            col = acc.col[multi].astype(np.float64)
            okm = vid >= 0
            g = np.exp(logg[np.maximum(vid, 0)])
            cc = np.where(okm[..., None], col * g, np.nan)
            with np.errstate(all="ignore"):
                m = np.nanmedian(cc, axis=1)                  # (n, 3)
            good = okm[..., None] & (col > 6) & (col < 250) & (m[:, None, :] > 6)
            with np.errstate(all="ignore"):
                lr = np.log(np.maximum(m[:, None, :], 1e-6)) - np.log(np.maximum(col, 1e-6))
            for ch in range(3):
                gm = good[..., ch]
                lr_all.append((ch, lr[..., ch][gm]))
                vid_all.append((ch, vid[gm]))
        if not lr_all:
            break
        new = logg.copy()
        cnt = np.zeros(len(logg))
        for ch in range(3):
            lr = np.concatenate([a for c, a in lr_all if c == ch])
            vv = np.concatenate([a for c, a in vid_all if c == ch])
            if not len(lr):
                continue
            o = np.lexsort((lr, vv))
            lr, vv = lr[o], vv[o]
            u, start, counts = np.unique(vv, return_index=True, return_counts=True)
            med = 0.5 * (lr[start + (counts - 1) // 2] + lr[start + counts // 2])
            new[u, ch] = med
            if ch == 0:
                cnt[u] = counts
        # 평균 0 (샘플 수 가중)
        w = cnt / max(cnt.sum(), 1)
        new -= (w[:, None] * new).sum(0)
        logg = np.clip(new, math.log(0.5), math.log(2.0))
    return np.exp(logg)


def _composite(acc: _FaceAcc, gains: np.ndarray, *, min_views: int, blend: str) -> FaceTexture:
    tex = acc.tex
    ok = acc.vid >= 0
    g = gains[np.maximum(acc.vid, 0)] if len(gains) else np.ones(acc.vid.shape + (3,))
    cc = np.where(ok[..., None], acc.col * g, np.nan)
    keep = (acc.nview >= min_views) & ok.any(1)
    with np.errstate(all="ignore"):
        med = np.nanmedian(cc[keep], axis=1) if keep.any() else np.zeros((0, 3))
        if blend == "best" and keep.any():
            ck = cc[keep]
            mad = np.nanmedian(np.abs(ck - med[:, None]), axis=1) * 1.4826 + 4.0
            inl = np.all(np.abs(ck - med[:, None]) <= 3 * mad[:, None], axis=2)
            sc = np.where(inl, acc.score[keep], -np.inf)
            b = np.argmax(sc, axis=1)
            med = ck[np.arange(len(b)), b]
    H, W = tex.shape
    img = tex.image.reshape(-1, 3)
    src = tex.source.reshape(-1)
    sup = tex.support.reshape(-1)
    pk = acc.pix[keep]
    img[pk] = np.clip(np.round(np.nan_to_num(med)), 0, 255).astype(np.uint8)
    src[pk] = SRC_PHOTO
    sup[acc.pix] = np.minimum(acc.nview, 65535).astype(np.uint16)
    return tex


def _check_blend(blend):
    if blend not in ("median", "best"):
        raise ValueError("blend 는 median | best")


def _photo_textures(planes: list[Plane], occluders: list[Plane], model, image_dir, *, res, min_views, max_views,
                    max_dist, max_incidence_deg, max_cameras, gain_comp, blend, sharpness_exp):
    _check_blend(blend)
    t0 = time.time()
    idx = []
    for p in planes:
        k = next((i for i, o in enumerate(occluders) if o is p), None)
        if k is None:
            occluders = list(occluders) + [p]
            k = len(occluders) - 1
        idx.append(k)
    accs = [_FaceAcc(p, k, occluders, res, max(1, max_views)) for p, k in zip(planes, idx)]
    info = _accumulate(accs, occluders, model, image_dir, res=res, max_dist=max_dist,
                       max_incidence_deg=max_incidence_deg, max_cameras=max_cameras, sharpness_exp=sharpness_exp)
    nv = len(info["views"])
    gains = _estimate_gains(accs, nv) if gain_comp else np.ones((max(nv, 1), 3))
    out = {}
    for acc in accs:
        tex = _composite(acc, gains, min_views=min_views, blend=blend)
        tex.meta.update({"photo": {"res": res, "min_views": min_views, "max_views": max_views, "blend": blend,
                                   "gain_comp": bool(gain_comp), "n_cameras_used": info["n_cameras_used"],
                                   "camera_step": info["camera_step"]}})
        out[acc.plane.id] = tex
    gi = {"gains": {n: [round(float(x), 4) for x in gains[i]] for i, n in enumerate(info["views"])},
          "missing_images": info["missing_images"], "seconds": round(time.time() - t0, 2)}
    return out, gi


def face_texture_from_photos(plane: Plane, model: cm.Model, image_dir, *, res: float = 0.015, min_views: int = 2,
                             max_views: int = 8, max_dist: float = 8.0, max_incidence_deg: float = 75.0,
                             max_cameras: int = 300, occluders: list[Plane] | None = None, gain_comp: bool = True,
                             blend: str = "median", sharpness_exp: float = 0.25) -> FaceTexture:
    """면 하나의 사진 텍스처. occluders = 가림 판정에 쓸 다른 면들 (보통 모든 면; 없으면 가림 판정 없음)."""
    texs, gi = _photo_textures([plane], list(occluders or []), model, image_dir, res=res, min_views=min_views,
                               max_views=max_views, max_dist=max_dist, max_incidence_deg=max_incidence_deg,
                               max_cameras=max_cameras, gain_comp=gain_comp, blend=blend,
                               sharpness_exp=sharpness_exp)
    tex = texs[plane.id]
    tex.meta["photo"].update(seconds=gi["seconds"], missing_images=gi["missing_images"])
    tex.meta["gains"] = gi["gains"]
    return tex


# ---------------- 패치 메우기 ----------------
def _components_capped(mask: np.ndarray, max_px: float | None) -> np.ndarray:
    from scipy import ndimage

    if max_px is None or not mask.any():
        return mask.copy()
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask.copy()
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    ok = sizes <= max_px
    ok[0] = False
    return ok[lab]


def _telea(img: np.ndarray, unknown: np.ndarray, radius: float = 5.0) -> np.ndarray:
    u8 = np.clip(np.round(img), 0, 255).astype(np.uint8)
    out = cv2.inpaint(u8[:, :, ::-1].copy(), unknown.astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)
    return out[:, :, ::-1].astype(np.float32)


def _down(img, hole, known, valid):
    H, W = hole.shape
    H2, W2 = (H + 1) // 2, (W + 1) // 2
    ph, pw = 2 * H2 - H, 2 * W2 - W
    im = np.pad(img, ((0, ph), (0, pw), (0, 0)))
    ho = np.pad(hole, ((0, ph), (0, pw)))
    kn = np.pad(known, ((0, ph), (0, pw)))
    va = np.pad(valid, ((0, ph), (0, pw)))

    def blocks(a):
        return a.reshape(H2, 2, W2, 2, *a.shape[2:])
    kb = blocks(kn).astype(np.float32)
    ksum = kb.sum((1, 3))
    isum = (blocks(im) * kb[..., None]).sum((1, 3))
    img2 = isum / np.maximum(ksum, 1)[..., None]
    hole2 = blocks(ho).any((1, 3))
    known2 = (ksum == 4) & ~hole2
    valid2 = blocks(va).any((1, 3)) & (known2 | hole2)
    return img2.astype(np.float32), hole2, known2, valid2


class _PM:
    """한 단의 PatchMatch 상태 (패딩된 평면 배열)."""

    def __init__(self, img, hole, known, valid, r, rng):
        self.r = r
        H, W = hole.shape
        self.H, self.W = H, W
        self.Hp, self.Wp = H + 2 * r, W + 2 * r
        self.rng = rng
        pad = ((r, r), (r, r))
        self.I = np.pad(img, pad + ((0, 0),)).reshape(-1, 3).astype(np.float32)
        self.ch = [np.ascontiguousarray(self.I[:, c]) for c in range(3)]   # 채널별 (빠른 모으기)
        self.hole_p = np.pad(hole, pad).ravel()
        self.valid_p = np.pad(valid, pad).ravel().astype(np.float32)
        kp = np.pad(known, pad)
        from scipy import ndimage
        srcv = ndimage.binary_erosion(kp, np.ones((2 * r + 1, 2 * r + 1), bool), border_value=0)
        self.srcv = srcv.ravel()
        self.src_list = np.flatnonzero(self.srcv)
        dil = ndimage.binary_dilation(hole, np.ones((2 * r + 1, 2 * r + 1), bool)) if hole.any() else hole
        ty, tx = np.nonzero(dil)
        self.ty, self.tx = ty, tx
        self.t = (ty + r) * self.Wp + (tx + r)
        o = np.arange(-r, r + 1)
        oy, ox = np.meshgrid(o, o, indexing="ij")
        self.offs = (oy * self.Wp + ox).ravel()
        wsum = np.zeros(len(self.t), np.float32)
        for of in self.offs:
            wsum += self.valid_p[self.t + of]
        self.wsum = np.maximum(wsum, 1.0)
        self.nnf = np.full(len(self.t), -1, np.int64)
        self.D = np.full(len(self.t), np.inf, np.float32)
        self.tpos = np.full(self.Hp * self.Wp, -1, np.int64)   # 패딩 평면 번호 -> 대상 번호
        self.tpos[self.t] = np.arange(len(self.t))
        # 투표 신뢰도 (Wexler 2007): 구멍 안 픽셀은 아는 곳에서 멀수록 gamma^-거리 로 약하게
        dk = ndimage.distance_transform_edt(~known) if known.any() else np.zeros_like(hole, float)
        self.conf = (CONF_GAMMA ** -dk[ty, tx]).astype(np.float64)

    def dist(self, sel, s):
        t = self.t[sel]
        d = np.zeros(len(t), np.float32)
        V = self.valid_p
        R, G, B = self.ch
        for of in self.offs:
            a, b = t + of, s + of
            e = R[a] - R[b]
            q = e * e
            e = G[a] - G[b]
            q += e * e
            e = B[a] - B[b]
            q += e * e
            q *= V[a]
            d += q
        return d / self.wsum[sel]

    def try_cand(self, sel, s):
        ok = (s >= 0) & (s != self.nnf[sel])
        if not ok.any():
            return
        sel, s = sel[ok], s[ok]
        d = self.dist(sel, s)
        b = d < self.D[sel]
        self.D[sel[b]] = d[b]
        self.nnf[sel[b]] = s[b]

    def random_init(self, mask=None):
        sel = np.arange(len(self.t)) if mask is None else np.flatnonzero(mask)
        if len(sel) == 0:
            return
        s = self.src_list[self.rng.integers(0, len(self.src_list), len(sel))]
        self.nnf[sel] = s
        self.D[sel] = self.dist(sel, s)

    def _shift_valid(self, s, dy, dx):
        y, x = s // self.Wp, s % self.Wp
        y2, x2 = y + dy, x + dx
        ok = (y2 >= 0) & (y2 < self.Hp) & (x2 >= 0) & (x2 < self.Wp)
        s2 = np.where(ok, y2 * self.Wp + x2, 0)
        ok &= self.srcv[s2]
        return np.where(ok, s2, -1)

    def iterate(self, n_iter):
        n = len(self.t)
        allsel = np.arange(n)
        steps = (1, 2, 4, 8)
        for it in range(n_iter):
            sg = 1 if it % 2 == 0 else -1
            for st in steps:
                for dy, dx in ((0, sg * st), (sg * st, 0), (0, -sg * st), (-sg * st, 0)):
                    ny, nx = self.ty + self.r - dy, self.tx + self.r - dx   # 이웃 (위치 p - δ) 의 짝 + δ
                    inb = (ny >= 0) & (ny < self.Hp) & (nx >= 0) & (nx < self.Wp)
                    k = np.where(inb, self.tpos[np.where(inb, ny * self.Wp + nx, 0)], -1)
                    has = k >= 0
                    s = np.full(n, -1, np.int64)
                    s[has] = self._shift_valid(self.nnf[k[has]], dy, dx)
                    self.try_cand(allsel, s)
            Rr = max(self.H, self.W)
            while Rr >= 1:
                dy = self.rng.integers(-Rr, Rr + 1, n)
                dx = self.rng.integers(-Rr, Rr + 1, n)
                self.try_cand(allsel, self._shift_valid(self.nnf, dy, dx))
                Rr //= 2
            for _ in range(2):
                self.try_cand(allsel, self.src_list[self.rng.integers(0, len(self.src_list), n)])

    def vote(self):
        if not len(self.t):
            return
        good = np.isfinite(self.D)
        s2 = float(np.percentile(self.D[good], 75)) + 1e-3 if good.any() else 1.0
        w = np.exp(-self.D / (2 * s2)).astype(np.float64) * self.conf
        w[~good] = 0
        P, S, Wt = [], [], []
        for of in self.offs:
            p = self.t + of
            m = self.hole_p[p] & good
            P.append(p[m])
            S.append(self.nnf[m] + of)
            Wt.append(w[m])
        P = np.concatenate(P)
        S = np.concatenate(S)
        Wt = np.concatenate(Wt)
        L = len(self.I)
        ws = np.bincount(P, Wt, L)
        hp = np.flatnonzero(self.hole_p & (ws > 0))
        for ch in range(3):
            acc = np.bincount(P, Wt * self.I[S, ch], L)
            self.I[hp, ch] = (acc[hp] / ws[hp]).astype(np.float32)
            self.ch[ch][hp] = self.I[hp, ch]

    def image(self):
        r = self.r
        return self.I.reshape(self.Hp, self.Wp, 3)[r:r + self.H, r:r + self.W].copy()

    def nnf_yx(self):
        """대상 (ty, tx) -> 원천 (sy, sx), 패딩 없는 좌표, (H, W) 배열 (-1 없음)."""
        sy = np.full((self.H, self.W), -1, np.int64)
        sx = np.full((self.H, self.W), -1, np.int64)
        ok = self.nnf >= 0
        sy[self.ty[ok], self.tx[ok]] = self.nnf[ok] // self.Wp - self.r
        sx[self.ty[ok], self.tx[ok]] = self.nnf[ok] % self.Wp - self.r
        return sy, sx


def _patchmatch(img: np.ndarray, hole: np.ndarray, known: np.ndarray, valid: np.ndarray, *, patch: int,
                em_iters: int, pm_iters: int, max_levels: int, rng) -> np.ndarray:
    from scipy import ndimage

    r = max(1, patch // 2)
    pyr = [(img.astype(np.float32), hole, known, valid)]
    while len(pyr) < max_levels:
        im, ho, kn, va = pyr[-1]
        far = ndimage.distance_transform_edt(ho).max() if ho.any() else 0
        if far <= r + 1 or min(ho.shape) < 4 * patch:
            break
        nxt = _down(im, ho, kn, va)
        if not nxt[2].any():
            break
        pyr.append(nxt)
    prev = None
    out = None
    for lev in range(len(pyr) - 1, -1, -1):
        im, ho, kn, va = pyr[lev]
        im = im.copy()
        if prev is None:
            im[~kn] = _telea(im, ~kn)[~kn]
        else:
            up = cv2.resize(prev[2], (im.shape[1], im.shape[0]), interpolation=cv2.INTER_LINEAR)
            im[ho] = up[ho]
        pm = _PM(im, ho, kn, va, r, rng)
        if len(pm.src_list) == 0 or len(pm.t) == 0:
            out = im
            prev = (None, None, im)
            continue
        if prev is not None and prev[0] is not None:
            psy, psx = prev[0], prev[1]
            cy, cx = np.minimum(pm.ty // 2, psy.shape[0] - 1), np.minimum(pm.tx // 2, psx.shape[1] - 1)
            sy, sx = psy[cy, cx], psx[cy, cx]
            has = sy >= 0
            sy2 = 2 * sy + pm.ty % 2
            sx2 = 2 * sx + pm.tx % 2
            ok = has & (sy2 >= 0) & (sy2 < pm.H) & (sx2 >= 0) & (sx2 < pm.W)
            s = np.where(ok, (sy2 + r) * pm.Wp + (sx2 + r), 0)
            ok &= pm.srcv[s]
            pm.random_init(~ok)
            sel = np.flatnonzero(ok)
            pm.nnf[sel] = s[sel]
            pm.D[sel] = pm.dist(sel, s[sel])
            pm.vote()
        else:
            pm.random_init()
        n_em = em_iters if lev > 0 else max(1, em_iters // 2)
        n_pm = pm_iters if lev > 0 else max(1, pm_iters - 1)
        for _ in range(n_em):
            pm.iterate(n_pm)
            pm.vote()
        sy, sx = pm.nnf_yx()
        out = pm.image()
        prev = (sy, sx, out)
    return out


def inpaint_texture(tex: FaceTexture, mask_to_fill: np.ndarray | None = None, *, method: str = "patchmatch",
                    patch: int = 7, max_fill_m2: float | None = 4.0, em_iters: int = 4, pm_iters: int = 3,
                    max_levels: int = 6, rng=0) -> tuple[FaceTexture, np.ndarray]:
    """못 본 곳 메우기. 돌려줌 (새 FaceTexture (입력은 그대로), 메운 픽셀 bool (H, W)). 규칙은 모듈 설명."""
    if method not in ("patchmatch", "telea"):
        raise ValueError("method 는 patchmatch | telea")
    if patch < 3 or patch % 2 == 0:
        raise ValueError("patch 는 3 이상 홀수")
    t0 = time.time()
    out = tex.copy()
    H, W = tex.shape
    base = tex.inside & (tex.source == SRC_NONE) if mask_to_fill is None else np.asarray(mask_to_fill, bool)
    if base.shape != (H, W):
        raise ValueError("mask_to_fill 크기가 텍스처와 다름")
    fill = base & tex.inside & (tex.source != SRC_PHOTO)
    max_px = None if max_fill_m2 is None else max_fill_m2 / (tex.res * tex.res)
    fill = _components_capped(fill, max_px)
    known = tex.inside & (tex.source != SRC_NONE) & ~fill
    info = {"method": method, "patch": patch, "requested_m2": round(float(base.sum()) * tex.res ** 2, 4),
            "filled_m2": 0.0, "skipped_m2": 0.0}
    if not fill.any() or not known.any():
        info["skipped_m2"] = round(float((base & tex.inside & (tex.source != SRC_PHOTO)).sum()) * tex.res ** 2, 4)
        out.meta["inpaint"] = info
        return out, np.zeros((H, W), bool)
    img = tex.image.astype(np.float32)
    img[~known] = 0
    if method == "telea":
        res = _telea(img, ~known)
    else:
        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        valid = tex.inside & (known | fill)
        res = _patchmatch(img, fill, known, valid, patch=patch, em_iters=em_iters, pm_iters=pm_iters,
                          max_levels=max_levels, rng=rng)
    out.image[fill] = np.clip(np.round(res[fill]), 0, 255).astype(np.uint8)
    out.source[fill] = SRC_INPAINT
    info["filled_m2"] = round(float(fill.sum()) * tex.res ** 2, 4)
    info["skipped_m2"] = round(float((base & tex.inside & (tex.source != SRC_PHOTO) & ~fill).sum()) * tex.res ** 2, 4)
    info["seconds"] = round(time.time() - t0, 2)
    out.meta["inpaint"] = info
    return out, fill


# ---------------- 여러 면 ----------------
def build_face_textures(planes: list[Plane], model: cm.Model, image_dir, *, res: float = 0.015, min_views: int = 2,
                        max_views: int = 8, max_dist: float = 8.0, max_incidence_deg: float = 75.0,
                        max_cameras: int = 300, gain_comp: bool = True, blend: str = "median",
                        sharpness_exp: float = 0.25, inpaint: bool = True, method: str = "patchmatch",
                        patch: int = 7, max_fill_m2: float | None = 4.0, faces: list[str] | None = None,
                        rng=0) -> dict[str, FaceTexture]:
    """면마다 사진 텍스처 (+ inpaint=True 면 못 본 곳 메우기). 사진은 한 번씩만 읽음. 가림 판정은 planes 전체로.
    faces 를 주면 그 id 의 면만 만듦."""
    todo = [p for p in planes if faces is None or p.id in faces]
    texs, gi = _photo_textures(todo, list(planes), model, image_dir, res=res, min_views=min_views,
                               max_views=max_views, max_dist=max_dist, max_incidence_deg=max_incidence_deg,
                               max_cameras=max_cameras, gain_comp=gain_comp, blend=blend,
                               sharpness_exp=sharpness_exp)
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    out = {}
    for p in todo:
        t = texs[p.id]
        t.meta["photo"]["seconds_all_faces"] = gi["seconds"]
        if inpaint and t.inside.any() and (t.source == SRC_PHOTO).any():
            t, _ = inpaint_texture(t, method=method, patch=patch, max_fill_m2=max_fill_m2, rng=gen)
        out[p.id] = t
    return out


# ---------------- 저장·불러오기 ----------------
def _safe(pid: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in pid)


def _imwrite(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise IOError(f"PNG 인코딩 실패: {path}")
    buf.tofile(str(path))


def _imread(path: Path, flags=cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    if not Path(path).is_file():
        return None
    return cv2.imdecode(np.fromfile(str(path), np.uint8), flags)


def save_texture(tex: FaceTexture, out_dir, plane: Plane | None = None) -> Path:
    """<out_dir>/<id>.png (+ .src.png, .support.png, .json). 돌려줌 PNG 경로. plane 을 주면 JSON 에 면도 적음."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe(tex.plane_id)
    bgr = np.flipud(tex.image)[:, :, ::-1]
    png = out_dir / f"{stem}.png"
    _imwrite(png, np.ascontiguousarray(bgr))
    _imwrite(out_dir / f"{stem}.src.png", np.ascontiguousarray(np.flipud(tex.source.astype(np.uint8) * _SRC_VIEW)))
    _imwrite(out_dir / f"{stem}.support.png", np.ascontiguousarray(np.flipud(tex.support.astype(np.uint16))))
    meta = {"plane_id": tex.plane_id, "res": tex.res, "size": list(tex.size), "shape": list(tex.shape),
            "png": png.name, "flip": "png row 0 = top = largest v", "source_codes": {"0": "none", "1": "photo",
            "2": "inpaint", "3": "manual"}, "stats": tex.stats(),
            "sha1": hashlib.sha1(png.read_bytes()).hexdigest(), "meta": tex.meta}
    if plane is not None:
        from .planes import plane_to_json
        meta["plane"] = plane_to_json(plane)
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return png


def load_texture(path, plane: Plane | None = None) -> FaceTexture:
    """save_texture 결과 (PNG 또는 JSON 경로) 읽기. 고친 PNG 도 받음 (모듈 설명의 알파·크기 규칙).
    plane 이 없으면 JSON 에 적힌 면으로 inside 를 다시 계산하고, 그것도 없으면 src.png 의 0 아닌 곳 + 사각형 전체."""
    path = Path(path)
    stem = path.name[:-5] if path.suffix == ".json" else path.stem
    d = path.parent
    jpath = d / f"{stem}.json"
    meta = json.loads(jpath.read_text(encoding="utf-8")) if jpath.is_file() else {}
    png = d / meta.get("png", f"{stem}.png")
    raw = _imread(png)
    if raw is None:
        raise FileNotFoundError(f"텍스처 PNG 없음: {png}")
    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    if raw.dtype == np.uint16:
        raw = (raw / 257).round().astype(np.uint8)
    alpha = raw[:, :, 3] if raw.shape[2] == 4 else None
    bgr = raw[:, :, :3]
    if plane is None and meta.get("plane"):
        plane = plane_from_json(meta["plane"])
    res = float(meta.get("res", 0.015))
    if plane is not None:
        H, W, _, inside = texture_grid(plane, res)
        size = (float(plane.size[0]), float(plane.size[1]))
        pid = plane.id
    else:
        H, W = (meta["shape"] if "shape" in meta else bgr.shape[:2])
        inside = None
        size = tuple(meta.get("size", (W * res, H * res)))
        pid = meta.get("plane_id", stem)
    edited = bool(meta.get("sha1")) and hashlib.sha1(png.read_bytes()).hexdigest() != meta["sha1"]
    resized = bgr.shape[:2] != (H, W)
    if resized:
        interp = cv2.INTER_AREA if bgr.shape[0] > H else cv2.INTER_LINEAR
        bgr = cv2.resize(bgr, (W, H), interpolation=interp)
        if alpha is not None:
            alpha = cv2.resize(alpha, (W, H), interpolation=cv2.INTER_NEAREST)
    img = np.ascontiguousarray(np.flipud(bgr)[:, :, ::-1])
    s = _imread(d / f"{stem}.src.png", cv2.IMREAD_GRAYSCALE)
    if s is not None and s.shape == (H, W):
        src = np.flipud(np.round(s.astype(np.float32) / _SRC_VIEW)).astype(np.uint8)
    else:
        src = np.full((H, W), SRC_PHOTO, np.uint8)
    sp = _imread(d / f"{stem}.support.png")
    sup = (np.flipud(sp).astype(np.uint16) if sp is not None and sp.shape[:2] == (H, W)
           else np.zeros((H, W), np.uint16))
    if inside is None:
        inside = np.ones((H, W), bool)
    if alpha is not None:
        a = np.flipud(alpha)
        src = np.where(a == 0, SRC_NONE, src).astype(np.uint8)
        src = np.where((a > 0) & (src == SRC_NONE) & inside, SRC_MANUAL, src).astype(np.uint8)
    src[~inside] = SRC_NONE
    img[~inside] = 0
    m = dict(meta.get("meta", {}))
    m.update(edited=edited, resized=bool(resized), loaded_from=str(png))
    return FaceTexture(pid, res, img, np.ascontiguousarray(src), np.ascontiguousarray(sup), inside, size, m)


def save_textures(textures: dict[str, FaceTexture], out_dir, planes: list[Plane] | None = None) -> list[Path]:
    P = {p.id: p for p in planes or []}
    return [save_texture(t, out_dir, P.get(pid)) for pid, t in textures.items()]


def load_textures(tex_dir, planes: list[Plane] | None = None) -> dict[str, FaceTexture]:
    """폴더의 <id>.json 모두 (planes 를 주면 그 면들만, 면 정의는 planes 것)."""
    tex_dir = Path(tex_dir)
    out = {}
    if planes is not None:
        for p in planes:
            j = tex_dir / f"{_safe(p.id)}.json"
            if j.is_file() or (tex_dir / f"{_safe(p.id)}.png").is_file():
                out[p.id] = load_texture(j if j.is_file() else tex_dir / f"{_safe(p.id)}.png", p)
        return out
    for j in sorted(tex_dir.glob("*.json")):
        t = load_texture(j)
        out[t.plane_id] = t
    return out
