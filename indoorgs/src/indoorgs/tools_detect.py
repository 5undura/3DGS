"""외부 도구(COLMAP, Brush, LichtFeld) 자동 찾기.

찾는 곳 (이 순서로 모은 뒤 최신 수정 시각 순으로 정렬):
  1. PATH
  2. 흔한 위치: C:/tools (깊이 제한 안에서 전부), %LOCALAPPDATA%·%ProgramFiles%·%ProgramFiles(x86)%
     (이 셋은 너무 커서 이름에 colmap/brush/lichtfeld 가 들어간 바로 아래 폴더만)
  3. 주어진 루트 (예: F:/Sample/tools) 아래 폴더 깊이 4 까지

`indoorgs init --search <폴더>` 가 1순위 후보로 project.yaml 을 채우고,
`indoorgs check` 는 설정된 경로 말고 다른 후보를 보여 준다.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

# 도구 -> 실행 파일 이름 (대소문자 무시). 앞쪽이 Windows 이름
TOOL_FILES: dict[str, tuple[str, ...]] = {
    "colmap": ("COLMAP.bat", "colmap.exe"),
    "brush": ("brush_app.exe",),
    "lichtfeld": ("LichtFeld-Studio.exe",),
}
KEYWORDS = {"colmap": "colmap", "brush": "brush", "lichtfeld": "lichtfeld"}
MAX_DEPTH = 4
# 훑지 않는 폴더 (크고 실행 파일이 아닌 것: vcpkg 빌드 캐시 등)
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", "vcpkg", "vcpkg_installed",
             "vcpkg_cache", "cmakefiles", "_deps", "$recycle.bin", "system volume information"}


@dataclass(frozen=True)
class Candidate:
    tool: str
    path: str
    mtime: float
    source: str        # path | common | search


def _match(name: str) -> str | None:
    low = name.lower()
    for tool, files in TOOL_FILES.items():
        if low in (f.lower() for f in files):
            return tool
    return None


def _walk(root: Path, max_depth: int):
    """root 아래 파일을 폴더 깊이 max_depth 까지 (root = 0). 권한 오류·링크 고리는 건너뜀."""
    stack = [(root, 0)]
    seen: set[str] = set()
    while stack:
        d, depth = stack.pop()
        try:
            key = os.path.realpath(d)
            if key in seen:
                continue
            seen.add(key)
            entries = list(os.scandir(d))
        except OSError:
            continue
        for e in entries:
            try:
                if e.is_dir():
                    if depth < max_depth and e.name.lower() not in SKIP_DIRS and not e.name.endswith("_skip"):
                        stack.append((Path(e.path), depth + 1))
                elif e.is_file():
                    yield Path(e.path)
            except OSError:
                continue


def _cand(tool: str, p: Path, source: str) -> Candidate | None:
    try:
        return Candidate(tool, str(p), p.stat().st_mtime, source)
    except OSError:
        return None


def common_roots(env: dict | None = None) -> list[tuple[Path, bool]]:
    """(폴더, 키워드 거르기) 목록. 키워드 거르기 = 바로 아래에서 도구 이름이 든 폴더만 들어감."""
    env = os.environ if env is None else env
    out: list[tuple[Path, bool]] = []
    if os.name == "nt" or env.get("SystemDrive"):
        out.append((Path(env.get("SystemDrive", "C:") + "/tools"), False))
    for k in ("LOCALAPPDATA", "ProgramFiles", "ProgramFiles(x86)"):
        if env.get(k):
            out.append((Path(env[k]), True))
    return out


def find_tools(search_roots=(), env: dict | None = None, use_path: bool = True,
               use_common: bool = True, max_depth: int = MAX_DEPTH) -> dict[str, list[Candidate]]:
    """도구별 후보 목록 (최신 수정 시각 먼저, 같은 파일은 한 번)."""
    env = os.environ if env is None else env
    found: dict[str, list[Candidate]] = {t: [] for t in TOOL_FILES}

    def add(c: Candidate | None):
        if c and all(os.path.normcase(os.path.realpath(x.path)) != os.path.normcase(os.path.realpath(c.path))
                     for x in found[c.tool]):
            found[c.tool].append(c)

    if use_path:
        path_var = env.get("PATH", "")
        for tool, files in TOOL_FILES.items():
            for f in files:
                w = shutil.which(f, path=path_var)
                if w:
                    add(_cand(tool, Path(w), "path"))
    roots: list[tuple[Path, bool, str]] = []
    if use_common:
        roots += [(r, kw, "common") for r, kw in common_roots(env)]
    roots += [(Path(r), False, "search") for r in search_roots if r]
    for root, keyword_only, source in roots:
        if not root.is_dir():
            continue
        if keyword_only:
            try:
                subs = [Path(e.path) for e in os.scandir(root) if e.is_dir()
                        and any(k in e.name.lower() for k in KEYWORDS.values())]
            except OSError:
                continue
            walks = [(s, max_depth - 1) for s in subs]
        else:
            walks = [(root, max_depth)]
        for start, depth in walks:
            for f in _walk(start, depth):
                tool = _match(f.name)
                if tool:
                    add(_cand(tool, f, source))
    for t in found:
        found[t].sort(key=lambda c: c.mtime, reverse=True)
    return found


def best(found: dict[str, list[Candidate]]) -> dict[str, str | None]:
    return {t: (c[0].path if c else None) for t, c in found.items()}


def yaml_path(p: str) -> str:
    """project.yaml 에 쓸 경로: 슬래시로 바꾸고 작은따옴표로 감쌈 (공백·# 안전)."""
    s = p.replace("\\", "/")
    return "'" + s.replace("'", "''") + "'"


def roots_from_config(paths) -> list[str]:
    """설정된 도구 경로의 두 단계 위 폴더 (F:/Sample/tools/colmap/COLMAP.bat -> F:/Sample/tools).
    LichtFeld 는 build 아래라 더 깊어서, 'tools' 같은 상위 폴더를 쓰려면 --search 로 주는 편이 정확함."""
    out: list[str] = []
    for p in paths:
        if not p:
            continue
        r = str(Path(p).parent.parent)
        if r not in out and Path(r).is_dir():
            out.append(r)
    return out


def other_candidates(found: dict[str, list[Candidate]], configured: dict[str, str | None]) -> dict[str, list[str]]:
    """설정된 경로를 뺀 나머지 후보 경로."""
    def same(a: str, b: str | None) -> bool:
        return bool(b) and os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))
    return {t: [c.path for c in cs if not same(c.path, configured.get(t))] for t, cs in found.items()}
