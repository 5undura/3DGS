"""GUI: 학습 후 단계(walls·snap·cleanup·repair) 결과 읽기와 후처리 설정 저장.

- section_walls(project, name, run): planes.json · walls.json · clean.json · repair.json · walls/tex 목록을 한 번에.
  없는 파일은 None, 못 읽는 파일은 errors 에 이유. NaN/inf 는 None (JSON 응답이 깨지지 않게).
- texture_file(project, name, file): walls/tex 안 PNG 만 (이름 검사 + 폴더 밖 거부).
- post_settings / save_post_settings: project.yaml 의 walls·cleanup 중 GUI 가 고치는 항목만.
  저장은 그 항목 줄만 바꿔 다른 줄·주석·모르는 키는 그대로 두고, config.parse 로 검사한 뒤 원자적으로 씀.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import yaml

from .. import config as config_mod
from .. import schemas

# stages/snap.py 와 같은 파일 이름 (import 하면 무거운 단계 모듈을 다 읽어서 여기 적음)
SNAP_JSON, CLEAN_JSON, REPAIR_JSON = "walls.json", "clean.json", "repair.json"
PLYS = {"splat": "splat.ply", "walls": "splat_walls.ply", "clean": "splat_clean.ply",
        "repaired": "splat_repaired.ply", "removed": "removed_clean.ply"}
TEX_NAME = re.compile(r"^[A-Za-z0-9_\-][A-Za-z0-9_.\-]*\.png$")
RUN_NAME = re.compile(r"^[A-Za-z0-9_\-]+$")

# GUI 에서 고치는 항목: (키, 형, 최소, 최대)
WALLS_EDIT = {"enabled": bool, "seed": bool, "snap": bool, "repair": bool, "texture": bool,
              "band": (0.0, 0.5), "color_tol": (0.0, 1.0)}
CLEANUP_EDIT = {"enabled": bool, "steps": list}


class SettingsError(ValueError):
    pass


def _finite(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _finite(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_finite(x) for x in v]
    return v


def _read(path: Path, errors: dict, key: str):
    if not path.is_file():
        return None
    try:
        return _finite(schemas.read_json(path))
    except (OSError, ValueError) as e:
        errors[key] = f"{path.name} 읽기 실패: {e}"
        return None


def _run_key(d: Path):
    head, _, tail = d.name.rpartition("_")
    return (head, int(tail)) if head and tail.isdigit() else (d.name, 1)


def train_runs(sp) -> list[Path]:
    if not sp.train.is_dir():
        return []
    return sorted((d for d in sp.train.iterdir()
                   if d.is_dir() and "_failed_" not in d.name and "_prev_" not in d.name and RUN_NAME.match(d.name)),
                  key=_run_key)


def list_textures(sp) -> list[dict]:
    d = sp.textures
    if not d.is_dir():
        return []
    out = []
    for png in sorted(d.glob("*.png")):
        n = png.name
        if n.endswith(".src.png") or n.endswith(".support.png") or not TEX_NAME.match(n):
            continue
        stem = n[:-4]
        meta = {}
        j = d / f"{stem}.json"
        if j.is_file():
            try:
                meta = json.loads(j.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                meta = {}
        src = d / f"{stem}.src.png"
        out.append({"name": n, "src": src.name if src.is_file() and TEX_NAME.match(src.name) else None,
                    "plane_id": meta.get("plane_id", stem), "res": meta.get("res"),
                    "size": meta.get("size"), "shape": meta.get("shape"), "stats": _finite(meta.get("stats"))})
    return out


def section_walls(project, sp, run: str | None = None) -> dict:
    errors: dict[str, str] = {}
    runs = train_runs(sp)
    names = [d.name for d in runs]
    if run:
        if run not in names:
            raise KeyError(f"학습 폴더 없음: {run} (있는 것: {names})")
        rd = sp.train / run
    elif runs:
        t = project.cfg.train
        latest = sp.latest_train_run(t.trainer, t.preset)
        rd = latest if latest.is_dir() and latest.name in names else runs[-1]
    else:
        rd = None
    out = {"planes": _read(sp.planes_json, errors, "planes"), "runs": names,
           "run": rd.name if rd else None, "snap": None, "clean": None, "repair": None,
           "files": {}, "textures": list_textures(sp)}
    if rd is not None:
        out["snap"] = _read(rd / SNAP_JSON, errors, "snap")
        out["clean"] = _read(rd / CLEAN_JSON, errors, "clean")
        out["repair"] = _read(rd / REPAIR_JSON, errors, "repair")
        out["files"] = {k: (rd / v).is_file() for k, v in PLYS.items()}
        if out["clean"]:
            from ..stages.snap import clean_summary
            try:
                out["clean_summary"] = clean_summary(out["clean"])
            except (TypeError, ValueError, AttributeError) as e:
                errors["clean"] = f"clean.json 요약 실패: {e}"
    out["errors"] = errors
    return out


def texture_file(sp, file: str) -> Path:
    """walls/tex 안의 PNG 경로. 이름이 이상하거나 폴더 밖이면 ValueError, 없으면 FileNotFoundError."""
    if not TEX_NAME.match(file or "") or ".." in file:
        raise ValueError("텍스처 파일 이름이 잘못됨")
    root = sp.textures
    f = root / file
    try:
        f.resolve().relative_to(root.resolve())
    except ValueError:
        raise ValueError("텍스처 폴더 밖")
    if not f.is_file():
        raise FileNotFoundError(file)
    return f


# ---------------- 설정 ----------------
def post_settings(cfg) -> dict:
    w, c = cfg.walls, cfg.cleanup
    return {"walls": {k: getattr(w, k) for k in WALLS_EDIT},
            "cleanup": {"enabled": c.enabled, "steps": list(c.steps)},
            "cleanup_steps": list(config_mod.CLEANUP_STEPS)}


def _check(body: dict) -> dict:
    """GUI 가 보낸 {walls: {...}, cleanup: {...}} -> 고칠 값만 (형·범위 검사)."""
    if not isinstance(body, dict):
        raise SettingsError("JSON 객체여야 함")
    extra = set(body) - {"walls", "cleanup"}
    if extra:
        raise SettingsError(f"모르는 항목 {sorted(extra)} (walls, cleanup 만)")
    out = {}
    for sec, spec in (("walls", WALLS_EDIT), ("cleanup", CLEANUP_EDIT)):
        vals = body.get(sec)
        if vals is None:
            continue
        if not isinstance(vals, dict):
            raise SettingsError(f"{sec}: 객체여야 함")
        bad = set(vals) - set(spec)
        if bad:
            raise SettingsError(f"{sec}: GUI 에서 못 고치는 항목 {sorted(bad)}")
        got = {}
        for k, v in vals.items():
            t = spec[k]
            if t is bool:
                if not isinstance(v, bool):
                    raise SettingsError(f"{sec}.{k}: 켬/끔(true/false)이어야 함")
            elif t is list:
                if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                    raise SettingsError(f"{sec}.{k}: 글자 목록이어야 함")
            else:
                lo, hi = t
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
                    raise SettingsError(f"{sec}.{k} = {v!r}: {lo}..{hi} 사이 숫자")
                v = float(v)
            got[k] = v
        out[sec] = got
    return out


def _dump_flow(v) -> str:
    out = yaml.safe_dump(v, default_flow_style=True, allow_unicode=True, width=10 ** 6, sort_keys=False).strip()
    if out.endswith("\n..."):                  # 단일 값은 문서 끝 표시가 붙음
        out = out[:-4].strip()
    return out


def _split_comment(line: str) -> tuple[str, str]:
    """'key: value   # 설명' -> ('key: value', '   # 설명'). '#' 앞이 YAML 로 읽혀야 주석으로 봄."""
    for m in re.finditer(r"\s+#", line):
        head = line[:m.start()]
        try:
            yaml.safe_load(head)
        except yaml.YAMLError:
            continue
        return head, line[m.start():]
    return line, ""


def _edit_text(text: str, key: str, updates: dict, merged: dict) -> str:
    """최상위 key 블록에서 updates 의 항목만 바꾼 text. 블록이 없으면 끝에 merged 를 흐름 형식으로 붙임."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if re.match(rf"^{re.escape(key)}\s*:", ln)), None)
    if start is None:
        sep = "" if not text or text.endswith("\n") else "\n"
        return f"{text}{sep}{key}: {_dump_flow(merged)}\n"
    end = start + 1
    while end < len(lines) and (lines[end].strip() == "" or lines[end][:1] in (" ", "\t")):
        end += 1
    while end > start + 1 and lines[end - 1].strip() == "":
        end -= 1
    head, comment = _split_comment(lines[start])
    value = head.split(":", 1)[1].strip()
    if value:                                  # 흐름 형식 (한 줄 또는 여러 줄) -> 한 줄로 다시 씀
        if end > start + 1:
            comment = ""                       # 여러 줄 흐름 형식이면 줄 끝 주석 위치를 알기 어려움
        new_block = [f"{key}: {_dump_flow(merged)}{comment}"]
    else:                                      # 블록 형식: 바뀐 키 줄만
        body = lines[start + 1:end]
        indent = next((re.match(r"^(\s+)", ln).group(1) for ln in body if ln.strip() and not ln.strip().startswith("#")), "  ")
        for k, v in updates.items():
            idx = next((i for i, ln in enumerate(body) if re.match(rf"^\s+{re.escape(k)}\s*:", ln)), None)
            if idx is None:
                last = max((i for i, ln in enumerate(body) if ln.strip()), default=-1)
                body.insert(last + 1, f"{indent}{k}: {_dump_flow(v)}")
                continue
            ind = re.match(r"^(\s*)", body[idx]).group(1)
            h, c = _split_comment(body[idx])
            j = idx + 1                        # 아래 '- 항목' 줄(블록 목록)은 흐름 목록으로 바꾸며 지움
            while j < len(body) and body[j].strip() and len(re.match(r"^(\s*)", body[j]).group(1)) >= len(ind) \
                    and body[j].strip().startswith("-") and not re.match(r"^\s+\w+\s*:", body[j]):
                j += 1
            body[idx:j] = [f"{ind}{k}: {_dump_flow(v)}{c}"]
        new_block = [lines[start]] + body
    out = lines[:start] + new_block + lines[end:]
    return "\n".join(out) + ("\n" if text.endswith("\n") or not text else "")


def save_post_settings(path: Path, body: dict) -> dict:
    """project.yaml 의 walls·cleanup 항목 저장. 검사 실패면 SettingsError (파일 안 바뀜). 돌려줌 post_settings."""
    updates = _check(body)
    text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise SettingsError("project.yaml 최상위가 표가 아님")
    want = dict(raw)
    for sec, vals in updates.items():
        cur = want.get(sec) or {}
        if not isinstance(cur, dict):
            raise SettingsError(f"project.yaml {sec} 가 표가 아님")
        want[sec] = {**cur, **vals}
    try:
        cfg = config_mod.parse(want)
    except config_mod.ConfigError as e:
        raise SettingsError(str(e))
    new = text
    for sec, vals in updates.items():
        new = _edit_text(new, sec, vals, want[sec])
    try:
        ok = yaml.safe_load(new) == want
    except yaml.YAMLError:
        ok = False
    if not ok:                                 # 줄 편집이 예상과 다르면 그 블록을 통째로 흐름 형식으로
        new = text
        for sec in updates:
            lines = new.splitlines()
            s = next((i for i, ln in enumerate(lines) if re.match(rf"^{sec}\s*:", ln)), None)
            if s is not None:
                e = s + 1
                while e < len(lines) and (lines[e].strip() == "" or lines[e][:1] in (" ", "\t")):
                    e += 1
                lines[s:e] = [f"{sec}: {_dump_flow(want[sec])}"]
                new = "\n".join(lines) + "\n"
            else:
                new = _edit_text(new, sec, {}, want[sec])
        if yaml.safe_load(new) != want:
            raise SettingsError("project.yaml 을 안전하게 고치지 못함 (직접 고쳐 주세요)")
    config_mod.parse(yaml.safe_load(new))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(new, encoding="utf-8")
    tmp.replace(path)
    return post_settings(cfg)
