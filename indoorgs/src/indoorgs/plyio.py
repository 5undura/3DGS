"""3DGS PLY 읽기·쓰기 공용 모듈 (fill, merge, train 의 stats 가 같이 씀).

- 속성은 항상 **이름으로** 다룬다. Brush 는 f_dc, f_rest, opacity, rot, scale, x,y,z 순(xyz 맨 뒤, f_rest 는
  글자순 f_rest_0, f_rest_1, f_rest_10 ...), LichtFeld 는 x,y,z, nx,ny,nz, f_dc ... 순. 순서로 읽으면 틀린다.
- 데이터는 numpy 구조화 배열 (필드 이름 = PLY 속성 이름). binary_little_endian 만 지원 (두 학습기 모두 이 형식).
- 3DGS 규약: rot_0..3 = 쿼터니언 (w, x, y, z), scale_* = log(축 길이), opacity = logit,
  f_rest_{c*K + k} = 색 채널 c(0..2)의 k 번째 고차 계수 (K = (차수+1)^2 - 1, 채널 먼저 묶임).

공개 API
  GaussianPly                      .data(구조화 배열) .comments .names .n .xyz .sh_degree
  read(path) / write(path, ply)    파일 읽기·쓰기 (쓰기는 .tmp 에 쓴 뒤 이름 바꿈)
  read_header(path)                헤더만 (개수, 속성 이름) - 큰 파일 미리 보기
  sh_degree(names)                 f_rest 개수 -> SH 차수
  common_dtype(name_lists)         여러 PLY 를 합칠 때 쓸 공통 속성 (정책은 함수 주석)
  conform(data, dtype)             한 배열을 공통 속성으로 맞춤 (SH 차수 맞추기 포함)
  concat(plys)                     공통 속성으로 맞춰 이어 붙임
  similarity_parts(M)              4x4 -> (축척 s, 회전 R, 이동 t). 닮음 변환이 아니면 ValueError
  transform(ply, M)                위치·회전·스케일(로그)·법선·SH 고차 계수(차수 1~3) 변환
  eval_sh(coef, dirs)              SH 색 계산 (3DGS utils/sh_utils.eval_sh 와 같은 기저·부호)
  sh_rotation(R, l)                차수 l 의 (2l+1)x(2l+1) 계수 회전 행렬
  stats(ply)                       count, translucent_ratio, needle_ratio (tools/ply_stats.py 와 같은 정의)
  merge_files(out, [(path, M)])    큰 PLY 여러 개를 하나씩 읽어 변환·합쳐 바로 씀 (메모리 절약)
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814

_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "<i2", "int16": "<i2",
          "ushort": "<u2", "uint16": "<u2", "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
          "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8"}
_TYPE_NAME = {np.dtype("<f4"): "float", np.dtype("<f8"): "double", np.dtype("u1"): "uchar",
              np.dtype("i1"): "char", np.dtype("<i2"): "short", np.dtype("<u2"): "ushort",
              np.dtype("<i4"): "int", np.dtype("<u4"): "uint"}

REQUIRED = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
            "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2"]


class PlyError(ValueError):
    pass


class ShNotRotatedWarning(UserWarning):
    """(예전 호환용) transform 이 SH 고차 계수를 돌리지 않았을 때의 경고. 지금은 차수 1~3 을 돌리므로 안 나옴."""


@dataclass
class Header:
    count: int                       # vertex 개수
    dtype: np.dtype                  # vertex 한 개의 구조
    comments: list[str] = field(default_factory=list)
    data_offset: int = 0             # 파일 안 vertex 데이터 시작 바이트
    skip_before: int = 0             # vertex 앞에 있는 다른 element 의 바이트 수 (data_offset 에 이미 더함)

    @property
    def names(self) -> list[str]:
        return list(self.dtype.names)


@dataclass
class GaussianPly:
    data: np.ndarray                 # 구조화 배열
    comments: list[str] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return list(self.data.dtype.names)

    @property
    def n(self) -> int:
        return len(self.data)

    @property
    def xyz(self) -> np.ndarray:
        return np.stack([self.data["x"], self.data["y"], self.data["z"]], 1).astype(np.float64)

    @property
    def sh_degree(self) -> int:
        return sh_degree(self.names)

    def opacity(self) -> np.ndarray:
        """sigmoid 적용한 불투명도 0..1"""
        return 1.0 / (1.0 + np.exp(-self.data["opacity"].astype(np.float64)))


# ---------------- 읽기·쓰기 ----------------
def read_header(path: str | Path) -> Header:
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise PlyError(f"{path}: PLY 파일이 아님")
        elements: list[tuple[str, int, list[tuple[str, str]]]] = []
        comments, fmt = [], None
        while True:
            raw = f.readline()
            if not raw:
                raise PlyError(f"{path}: end_header 가 없음")
            line = raw.decode("ascii", "replace").strip()
            if line == "end_header":
                break
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] in ("comment", "obj_info"):
                comments.append(line)
            elif parts[0] == "element":
                elements.append((parts[1], int(parts[2]), []))
            elif parts[0] == "property":
                if not elements:
                    raise PlyError(f"{path}: element 전에 property")
                if parts[1] == "list":
                    raise PlyError(f"{path}: list 속성은 지원 안 함 ({line})")
                if parts[1] not in _TYPES:
                    raise PlyError(f"{path}: 모르는 형식 {parts[1]}")
                elements[-1][2].append((parts[2], _TYPES[parts[1]]))
        offset = f.tell()
    if fmt != "binary_little_endian":
        raise PlyError(f"{path}: format {fmt} (binary_little_endian 만 지원)")
    skip = 0
    for name, count, props in elements:
        dt = np.dtype(props)
        if name == "vertex":
            if len(set(dt.names)) != len(dt.names):
                raise PlyError(f"{path}: 속성 이름 중복")
            return Header(count, dt, comments, offset + skip, skip)
        skip += count * dt.itemsize
    raise PlyError(f"{path}: element vertex 가 없음")


def read(path: str | Path) -> GaussianPly:
    h = read_header(path)
    with open(path, "rb") as f:
        f.seek(h.data_offset)
        need = h.count * h.dtype.itemsize
        # 가상 드라이브(구글 드라이브 등)는 큰 덩어리를 한 번에 못 읽는 일이 있어 32MB 씩 (tools/fill/merge_ply.py)
        buf = bytearray(need)
        mv, got = memoryview(buf), 0
        while got < need:
            chunk = f.read(min(32 << 20, need - got))
            if not chunk:
                break
            mv[got:got + len(chunk)] = chunk
            got += len(chunk)
    if got < need:
        raise PlyError(f"{path}: 데이터가 모자람 ({got}/{need} 바이트). 파일이 다 내려받아졌는지 확인")
    return GaussianPly(np.frombuffer(bytes(buf), dtype=h.dtype).copy(), h.comments)


def _header_bytes(dtype: np.dtype, count: int, comments: list[str]) -> bytes:
    lines = ["ply", "format binary_little_endian 1.0"]
    lines += [c if c.startswith(("comment", "obj_info")) else "comment " + c for c in comments]
    lines.append(f"element vertex {count}")
    for nm in dtype.names:
        t = dtype.fields[nm][0]
        if t.newbyteorder("<") not in _TYPE_NAME and t not in _TYPE_NAME:
            raise PlyError(f"쓸 수 없는 형식 {t} ({nm})")
        lines.append(f"property {_TYPE_NAME.get(t, _TYPE_NAME.get(t.newbyteorder('<')))} {nm}")
    lines.append("end_header")
    return ("\n".join(lines) + "\n").encode("ascii")


def _le(dtype: np.dtype) -> np.dtype:
    return np.dtype([(n, dtype.fields[n][0].newbyteorder("<")) for n in dtype.names])


def write(path: str | Path, ply: GaussianPly) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dt = _le(ply.data.dtype)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(_header_bytes(dt, len(ply.data), ply.comments))
        f.write(np.ascontiguousarray(ply.data.astype(dt, copy=False)).tobytes())
    os.replace(tmp, path)
    return path


def make(n: int, names: list[str], comments: list[str] | None = None) -> GaussianPly:
    """0 으로 채운 float32 PLY."""
    return GaussianPly(np.zeros(n, np.dtype([(k, "<f4") for k in names])), list(comments or []))


def brush_names(degree: int = 2) -> list[str]:
    """Brush 내보내기 속성 순서 (fill_section.py 와 같음: f_rest 는 글자순)."""
    K = 3 * ((degree + 1) ** 2 - 1)
    return (["f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in sorted(range(K), key=str)]
            + ["opacity", "rot_0", "rot_1", "rot_2", "rot_3", "scale_0", "scale_1", "scale_2", "x", "y", "z"])


def lichtfeld_names(degree: int = 2) -> list[str]:
    """LichtFeld(원본 3DGS) 속성 순서: xyz, 법선, f_dc, f_rest(숫자순), opacity, scale, rot."""
    K = 3 * ((degree + 1) ** 2 - 1)
    return (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(K)]
            + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])


# ---------------- SH 차수 · 공통 속성 ----------------
def _rest_count(names) -> int:
    return sum(1 for k in names if k.startswith("f_rest_"))


def sh_degree(names) -> int:
    n = _rest_count(names)
    if n % 3:
        raise PlyError(f"f_rest 개수 {n} 가 3 의 배수가 아님")
    K = n // 3
    d = int(round(np.sqrt(K + 1))) - 1
    if (d + 1) ** 2 - 1 != K:
        raise PlyError(f"f_rest 개수 {n} 로 SH 차수를 정할 수 없음")
    return d


def check_required(names, where="PLY"):
    miss = [k for k in REQUIRED if k not in names]
    if miss:
        raise PlyError(f"{where}: 3DGS 필수 속성 없음 {miss}")


def common_dtype(name_lists: list[list[str]]) -> np.dtype:
    """여러 PLY 를 합칠 때 쓸 속성 목록 (모두 float32).

    정책
    - 순서: 첫 PLY 의 순서를 따름 (Brush 가 먼저면 Brush 순서).
    - 3DGS 필수 속성(위치·불투명도·스케일·회전·f_dc)은 모든 PLY 에 있어야 함. 없으면 PlyError.
    - SH 고차(f_rest): **가장 높은 차수**로 맞춤. 낮은 차수 PLY 는 없는 계수를 0 으로 채움
      (0 = 시점에 따라 색이 안 변함 -> 그 PLY 가 원래 보이던 그대로). 채널별 묶음 위치는 conform 이 맞춤.
    - 그 밖의 속성(nx,ny,nz 등): 모든 PLY 에 있을 때만 남김. 하나라도 없으면 버림 (값을 지어내지 않음).
    """
    for i, nl in enumerate(name_lists):
        check_required(nl, f"{i}번째 PLY")
    deg = max(sh_degree(nl) for nl in name_lists)
    K = 3 * ((deg + 1) ** 2 - 1)
    first = name_lists[0]
    sets = [set(nl) for nl in name_lists]
    out, rest_done = [], False
    rest_names = [f"f_rest_{i}" for i in range(K)]
    if _rest_count(first) == K:
        rest_names = [k for k in first if k.startswith("f_rest_")]  # 첫 PLY 의 f_rest 순서 유지
    for k in first:
        if k.startswith("f_rest_"):
            if not rest_done:
                out += rest_names
                rest_done = True
            continue
        if k in REQUIRED or all(k in s for s in sets):
            out.append(k)
        if k == "f_dc_2" and not rest_done and _rest_count(first) == 0 and K:
            out += rest_names                                     # 첫 PLY 가 SH0 이면 f_dc 뒤에
            rest_done = True
    return np.dtype([(k, "<f4") for k in out])


def conform(data: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """구조화 배열을 dtype 의 속성으로 맞춤. 없는 속성 0, f_rest 는 채널별 계수 위치를 다시 맞춤."""
    out = np.zeros(len(data), dtype)
    src = set(data.dtype.names)
    Ks = _rest_count(data.dtype.names) // 3
    Kd = _rest_count(dtype.names) // 3
    for k in dtype.names:
        if k.startswith("f_rest_"):
            j = int(k[7:])
            c, kk = divmod(j, Kd)
            if kk < Ks:
                out[k] = data[f"f_rest_{c * Ks + kk}"]
        elif k in src:
            out[k] = data[k]
    return out


def concat(plys: list[GaussianPly]) -> GaussianPly:
    dt = common_dtype([p.names for p in plys])
    return GaussianPly(np.concatenate([conform(p.data, dt) for p in plys]), list(plys[0].comments))


# ---------------- 변환 ----------------
def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(N,4) w,x,y,z -> (N,3,3). 정규화 후."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """(3,3) 또는 (N,3,3) -> w,x,y,z (w >= 0). 수치적으로 안정한 방식."""
    R = np.asarray(R, np.float64)
    single = R.ndim == 2
    R = R.reshape(-1, 3, 3)
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cand = np.stack([1 + tr, 1 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2],
                     1 - R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2], 1 - R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2]], 1)
    k = np.argmax(cand, 1)
    q = np.zeros((len(R), 4))
    for i in range(4):
        m = k == i
        if not m.any():
            continue
        r = R[m]
        s = np.sqrt(np.maximum(cand[m, i], 1e-300)) * 2   # = 4 * 해당 성분
        if i == 0:
            q[m] = np.stack([s / 4, (r[:, 2, 1] - r[:, 1, 2]) / s, (r[:, 0, 2] - r[:, 2, 0]) / s, (r[:, 1, 0] - r[:, 0, 1]) / s], 1)
        elif i == 1:
            q[m] = np.stack([(r[:, 2, 1] - r[:, 1, 2]) / s, s / 4, (r[:, 0, 1] + r[:, 1, 0]) / s, (r[:, 0, 2] + r[:, 2, 0]) / s], 1)
        elif i == 2:
            q[m] = np.stack([(r[:, 0, 2] - r[:, 2, 0]) / s, (r[:, 0, 1] + r[:, 1, 0]) / s, s / 4, (r[:, 1, 2] + r[:, 2, 1]) / s], 1)
        else:
            q[m] = np.stack([(r[:, 1, 0] - r[:, 0, 1]) / s, (r[:, 0, 2] + r[:, 2, 0]) / s, (r[:, 1, 2] + r[:, 2, 1]) / s, s / 4], 1)
    q *= np.where(q[:, :1] < 0, -1.0, 1.0)
    return q[0] if single else q


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """해밀턴 곱 a*b (w,x,y,z). a 는 (4,) 또는 (N,4)."""
    a = np.broadcast_to(a, b.shape)
    w1, x1, y1, z1 = a.T
    w2, x2, y2, z2 = b.T
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], 1)


def similarity_parts(M, tol: float = 2e-3) -> tuple[float, np.ndarray, np.ndarray]:
    """4x4 = [s R | t] 를 (s, R, t) 로. R 은 가장 가까운 회전(SVD).
    layout_v4.json 처럼 소수 6자리로 반올림된 행렬도 받도록 상대 오차 tol 까지 허용.
    반사(det<0)·찌그러짐(축마다 다른 축척)은 쿼터니언으로 못 나타내므로 ValueError."""
    M = np.asarray(M, np.float64)
    if M.shape != (4, 4) or not np.allclose(M[3], [0, 0, 0, 1], atol=1e-9):
        raise ValueError("4x4 동차 행렬이어야 함 (마지막 줄 0 0 0 1)")
    A = M[:3, :3]
    det = np.linalg.det(A)
    if det <= 0:
        raise ValueError("행렬식이 0 이하 (반사 또는 퇴화) - 3DGS 회전으로 옮길 수 없음")
    s = float(np.cbrt(det))
    U, _, Vt = np.linalg.svd(A / s)
    R = U @ Vt
    err = np.abs(A / s - R).max()
    if err > tol:
        raise ValueError(f"닮음 변환이 아님 (축마다 축척이 다르거나 기울어짐, 오차 {err:.2e})")
    return s, R, M[:3, 3].copy()


# ---------------- SH 회전 ----------------
# 3DGS 원본 utils/sh_utils.py 의 상수·기저 (부호 포함). Brush·LichtFeld 도 같은 기저로 PLY 를 씀.
SH_C1 = 0.4886025119029199
SH_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
SH_C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
         -0.4570457994644658, 1.445305721320277, -0.5900435899266435]


def sh_basis(dirs: np.ndarray, l: int) -> np.ndarray:
    """(N,3) 단위 방향 -> (N, 2l+1) 차수 l 기저값. eval_sh 의 각 항에서 계수를 뺀 것."""
    x, y, z = np.asarray(dirs, np.float64).T
    if l == 0:
        return np.full((len(x), 1), SH_C0)
    if l == 1:
        return np.stack([-SH_C1 * y, SH_C1 * z, -SH_C1 * x], 1)
    xx, yy, zz = x * x, y * y, z * z
    if l == 2:
        return np.stack([SH_C2[0] * x * y, SH_C2[1] * y * z, SH_C2[2] * (2 * zz - xx - yy),
                         SH_C2[3] * x * z, SH_C2[4] * (xx - yy)], 1)
    if l == 3:
        return np.stack([SH_C3[0] * y * (3 * xx - yy), SH_C3[1] * x * y * z, SH_C3[2] * y * (4 * zz - xx - yy),
                         SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy), SH_C3[4] * x * (4 * zz - xx - yy),
                         SH_C3[5] * z * (xx - yy), SH_C3[6] * x * (xx - 3 * yy)], 1)
    raise ValueError(f"SH 차수 {l} 는 지원 안 함 (0~3)")


def eval_sh(coef: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """coef (N, 3, (d+1)^2) [채널, 계수], dirs (N,3) 단위 방향 -> (N,3) 색 (0.5 더하기 전, sh_utils.eval_sh 와 같음)."""
    nk = coef.shape[-1]
    d = int(round(np.sqrt(nk))) - 1
    B = np.concatenate([sh_basis(dirs, l) for l in range(d + 1)], 1)        # (N, nk)
    return np.einsum("nck,nk->nc", coef, B)


def _fib_dirs(n: int = 64) -> np.ndarray:
    """구 위에 고르게 퍼진 n 개 방향 (최소제곱 표본)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def sh_rotation(R: np.ndarray, l: int) -> np.ndarray:
    """차수 l 계수 회전 행렬 D ((2l+1)x(2l+1), 직교). 계수 행벡터 c 에 c' = c @ D.T 로 쓰면
    eval(c', R d) == eval(c, d). 같은 차수 기저는 회전에 닫혀 있으므로 표본 방향 최소제곱으로 정확히 구해짐:
    Y(R^T u) = Y(u) @ D  ->  c'·Y(u) = c·Y(R^T u) = (c @ D.T)·Y(u)."""
    U = _fib_dirs()
    D, *_ = np.linalg.lstsq(sh_basis(U, l), sh_basis(U @ np.asarray(R, np.float64), l), rcond=None)
    return D


def _rotate_sh(d: np.ndarray, R: np.ndarray) -> None:
    """구조화 배열 d 의 f_rest (채널 우선: f_rest_{c*K+k}) 를 이름으로 읽어 R 만큼 돌림 (제자리)."""
    K = _rest_count(d.dtype.names) // 3
    deg = int(round(np.sqrt(K + 1))) - 1
    start = 0                              # 차수 l 의 계수는 f_rest 안에서 k = l^2-1 .. (l+1)^2-2
    for l in range(1, deg + 1):
        m = 2 * l + 1
        D = sh_rotation(R, l)
        for c in range(3):
            keys = [f"f_rest_{c * K + start + j}" for j in range(m)]
            C = np.stack([d[k] for k in keys], 1).astype(np.float64)
            C = C @ D.T
            for j, k in enumerate(keys):
                d[k] = C[:, j]
        start += m


def transform(ply: GaussianPly, M, warn_sh: bool = True) -> GaussianPly:
    """닮음 변환 M(4x4) 적용한 새 PLY.
    위치 p' = sR p + t, 회전 q' = q_R * q, 스케일(로그) + log s, 법선 n' = R n,
    SH 고차(f_rest, 차수 1~3) 는 sh_rotation 으로 돌림 (보는 방향 R d 에서 원래 d 의 색).
    (닮음 변환에서 f_dc·불투명도는 그대로. warn_sh 는 예전 호환용 인자로 지금은 쓰지 않음)"""
    s, R, t = similarity_parts(M)
    d = ply.data.copy()
    P = ply.xyz @ (s * R).T + t
    for i, k in enumerate("xyz"):
        d[k] = P[:, i]
    q = np.stack([d[f"rot_{i}"] for i in range(4)], 1).astype(np.float64)
    qn = quat_mul(mat_to_quat(R), q)
    for i in range(4):
        d[f"rot_{i}"] = qn[:, i]
    if s != 1.0:
        for i in range(3):
            d[f"scale_{i}"] = d[f"scale_{i}"].astype(np.float64) + np.log(s)
    if all(k in d.dtype.names for k in ("nx", "ny", "nz")):
        N = np.stack([d["nx"], d["ny"], d["nz"]], 1).astype(np.float64) @ R.T
        for i, k in enumerate(("nx", "ny", "nz")):
            d[k] = N[:, i]
    if ply.sh_degree > 0 and not np.allclose(R, np.eye(3), atol=1e-12):
        _rotate_sh(d, R)
    return GaussianPly(d, list(ply.comments))


# ---------------- 통계 ----------------
def stats(ply: GaussianPly) -> dict:
    """CONTRACT.md 4절 stats.json 값. tools/ply_stats.py 와 같은 정의:
    translucent_ratio = sigmoid(opacity) < 0.5 비율
    needle_ratio = 가장 긴 축 / **중간 축** > 10 비율 (ply_stats.py 코드 기준. CONTRACT 문장은 '가장 짧은 축'이라
    되어 있으나 실제 기록된 값(복도 1 16.3% 등)은 중간 축 기준이라 코드를 따름)"""
    d = ply.data
    if len(d) == 0:
        return {"version": 1, "count": 0, "translucent_ratio": 0.0, "needle_ratio": 0.0}
    s = np.exp(np.stack([d["scale_0"], d["scale_1"], d["scale_2"]], 1).astype(np.float64))
    s.sort(1)
    op = ply.opacity()
    ratio = s[:, 2] / np.maximum(s[:, 1], 1e-12)
    return {"version": 1, "count": int(len(d)),
            "translucent_ratio": round(float(np.mean(op < 0.5)), 4),
            "needle_ratio": round(float(np.mean(ratio > 10)), 4)}


# ---------------- 큰 파일 합치기 ----------------
def merge_files(out: str | Path, items: list[tuple[str | Path, object]], comments: list[str] | None = None,
                warn_sh: bool = True) -> dict:
    """[(PLY 경로, 4x4 또는 None)] 을 공통 속성으로 맞추고 변환해 out 하나로. 한 번에 한 파일만 메모리에.
    반환: {"count": 전체, "parts": [각 개수], "names": 속성}"""
    out = Path(out)
    heads = [read_header(p) for p, _ in items]
    dt = common_dtype([h.names for h in heads])
    total = sum(h.count for h in heads)
    if comments is None:
        comments = heads[0].comments
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    parts = []
    with open(tmp, "wb") as f:
        f.write(_header_bytes(dt, total, comments))
        for (p, M), h in zip(items, heads):
            ply = read(p)
            if M is not None:
                ply = transform(ply, M, warn_sh=warn_sh)
            f.write(conform(ply.data, dt).tobytes())
            parts.append(ply.n)
            del ply
    os.replace(tmp, out)
    return {"count": total, "parts": parts, "names": list(dt.names)}
