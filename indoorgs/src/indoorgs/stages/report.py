"""7. 보고서 (담당 D, 프로젝트 범위). 모든 json -> report.html (항상 다시 만듦)

- 구간별 표: 프레임 수, 등록/제외 장수, PSNR, 반투명 비율(0.8 이상이면 VR 비침 경고), 학습기
- 공간이 있으면 방 다각형 위에 priors 위치, 제외 사진은 빨간색 (인라인 SVG)
- 자체 완결 HTML (외부 리소스 없음). 없는 파일은 "아직 없음"
"""
from __future__ import annotations

import datetime as dt
import html
import json
from pathlib import Path

from .. import schemas
from .base import Stage, StageContext, StageResult

TRANSLUCENT_WARN = 0.8
NONE = '<span class="none">아직 없음</span>'


def _read(path: Path) -> dict | None:
    try:
        return schemas.read_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def _e(v) -> str:
    return html.escape(str(v))


def _num(v, fmt="{:,}") -> str:
    if v is None:
        return NONE
    try:
        return fmt.format(v)
    except (ValueError, TypeError):
        return _e(v)


def _run_key(d: Path):
    """brush_default, brush_default_2, ..._10 순서 (글자순이면 _10 이 _2 앞)"""
    head, _, tail = d.name.rpartition("_")
    return (head, int(tail)) if head and tail.isdigit() else (d.name, 1)


def _train_runs(sp, train_cfg=None) -> list[dict]:
    """학습 폴더 목록. latest = fill·merge 가 쓰는 최신 폴더 (설정된 학습기·프리셋)."""
    latest = sp.latest_train_run(train_cfg.trainer, train_cfg.preset) if train_cfg else None
    runs = []
    if sp.train.is_dir():
        for d in sorted(sp.train.iterdir(), key=_run_key):
            if not d.is_dir() or "_failed_" in d.name:
                continue
            ev, stt = _read(d / "eval.json"), _read(d / "stats.json")
            if ev is None and stt is None and not (d / "splat.ply").exists():
                continue
            runs.append({"name": d.name, "trainer": d.name.split("_")[0], "eval": ev, "stats": stt,
                         "ply": (d / "splat.ply").exists(), "latest": d == latest,
                         "walls": _read(d / "walls.json"), "clean": _read(d / "clean.json")})
    return runs


def collect(project) -> dict:
    """report 에 쓰는 값을 모음 (테스트·GUI 에서도 씀)."""
    secs = []
    for sec, sp in project.sections():
        done = _read(sp.frames_done)
        diag = _read(sp.diagnose)
        pri = _read(sp.priors)
        path = _read(sp.path_json)
        secs.append({"name": sec.name, "video": sec.video, "frames": done, "diagnose": diag,
                     "priors": pri, "path": path, "runs": _train_runs(sp, project.cfg.train),
                     "planes": _read(sp.planes_json)})
    plan = None
    if project.has_space:
        try:
            plan = schemas.load_plan(project.plan_json)
        except (schemas.SchemaError, OSError, json.JSONDecodeError, KeyError) as e:
            plan = e
    return {"sections": secs, "plan": plan, "layout": _read(project.layout_json),
            "building": project.building_ply.exists()}


# ---------------- SVG ----------------
PALETTE = ["#2563eb", "#0891b2", "#7c3aed", "#059669", "#d97706", "#db2777"]


def _svg(plan, secs) -> str:
    pts = [p for r in plan.rooms for p in r.floor]
    for s in secs:
        for f in (s["priors"] or {}).get("frames", []):
            pts.append(tuple(f["xyz"][:2]))
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs) - 1, max(xs) + 1, min(ys) - 1, max(ys) + 1
    k = min(720 / max(x1 - x0, 1e-6), 520 / max(y1 - y0, 1e-6))    # 최대 720 x 520 px
    W, H = max(160, int((x1 - x0) * k)), max(160, int((y1 - y0) * k))

    def X(x): return f"{(x - x0) * k:.1f}"
    def Y(y): return f"{(y1 - y) * k:.1f}"     # 도면 y 위쪽 -> SVG y 아래쪽

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" class="map">']
    for r in plan.rooms:
        d = " ".join(f"{X(x)},{Y(y)}" for x, y in r.floor)
        out.append(f'<polygon points="{d}" fill="#f4f4f5" stroke="#52525b" stroke-width="1.5"/>')
        cx = sum(p[0] for p in r.floor) / len(r.floor)
        cy = sum(p[1] for p in r.floor) / len(r.floor)
        out.append(f'<text x="{X(cx)}" y="{Y(cy)}" class="lbl">{_e(r.name or r.id)}</text>')
    for dr in plan.doors:
        out.append(f'<circle cx="{X(dr.center[0])}" cy="{Y(dr.center[1])}" r="{max(2, dr.width * k / 2):.1f}" '
                   f'fill="none" stroke="#a1a1aa" stroke-dasharray="3 2"/>')
    for i, s in enumerate(secs):
        color = PALETTE[i % len(PALETTE)]
        stroke = (s["path"] or {}).get("stroke")
        if stroke:
            d = " ".join(f"{X(x)},{Y(y)}" for x, y in stroke)
            out.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-opacity=".35" '
                       f'stroke-width="2" stroke-dasharray="6 4"/>')
        frames = (s["priors"] or {}).get("frames", [])
        excluded = set((s["diagnose"] or {}).get("excluded", []))
        for f in frames:
            bad = f.get("image") in excluded
            out.append(f'<circle cx="{X(f["xyz"][0])}" cy="{Y(f["xyz"][1])}" r="{3 if bad else 1.6}" '
                       f'fill="{"#dc2626" if bad else color}"><title>{_e(f.get("image"))} t={f.get("t")}</title></circle>')
    # 축척 1m
    out.append(f'<line x1="10" y1="{H - 12}" x2="{10 + k:.1f}" y2="{H - 12}" stroke="#18181b" stroke-width="2"/>'
               f'<text x="10" y="{H - 18}" class="lbl" text-anchor="start">1 m</text>')
    out.append("</svg>")
    legend = " ".join(f'<span class="dot" style="background:{PALETTE[i % len(PALETTE)]}"></span>{_e(s["name"])}'
                      for i, s in enumerate(secs) if s["priors"])
    legend += ' <span class="dot" style="background:#dc2626"></span>제외된 사진'
    return "\n".join(out) + f'<p class="legend">{legend}</p>'


# ---------------- HTML ----------------
CSS = """
body{font:14px/1.5 system-ui,-apple-system,"Segoe UI","Malgun Gothic",sans-serif;color:#18181b;background:#fff;margin:0;padding:32px;max-width:960px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:32px 0 8px}
.sub{color:#71717a;font-size:12px}
table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:6px 10px;border-bottom:1px solid #e4e4e7;font-variant-numeric:tabular-nums}
th{font-weight:600;color:#52525b;font-size:12px}
.none{color:#a1a1aa}.warn{color:#dc2626;font-weight:600}
.lbl{font-size:11px;fill:#52525b;text-anchor:middle}
.map{border:1px solid #e4e4e7;max-width:100%;height:auto}
.legend{font-size:12px;color:#52525b}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin:0 4px 0 10px}
"""


def build_html(project) -> str:
    data = collect(project)
    secs = data["sections"]
    rows = []
    for s in secs:
        fr, dg = s["frames"], s["diagnose"]
        base = [f"<td>{_e(s['name'])}</td>",
                f"<td>{_num(fr.get('count')) if fr else NONE}</td>",
                f"<td>{_num(dg.get('registered')) if dg else NONE}</td>",
                f"<td>{_num(len(dg.get('excluded', []))) if dg else NONE}</td>"]
        runs = s["runs"] or [None]
        for r in runs:
            if r is None:
                tail = [f"<td>{NONE}</td>"] * 4
            else:
                ev, st = r["eval"] or {}, r["stats"] or {}
                tr = st.get("translucent_ratio")
                if tr is None:
                    trs = NONE
                elif tr >= TRANSLUCENT_WARN:
                    trs = f'<span class="warn">{tr:.1%} · VR에서 비칠 수 있음</span>'
                else:
                    trs = f"{tr:.1%}"
                mark = " (최신·합치기에 씀)" if r.get("latest") and len(s["runs"]) > 1 else ""
                tail = [f"<td>{_e(r['name'])}{mark}</td>",
                        f"<td>{_num(ev.get('psnr'), '{:.2f}')}</td>",
                        f"<td>{_num(st.get('count'))}</td>", f"<td>{trs}</td>"]
            rows.append("<tr>" + "".join(base + tail) + "</tr>")
    table = ("<table><tr><th>구간</th><th>프레임</th><th>등록</th><th>제외</th><th>학습기_프리셋</th>"
             "<th>PSNR</th><th>스플랫 수</th><th>반투명 비율</th></tr>" + "".join(rows) + "</table>")

    plan = data["plan"]
    if plan is None:
        space = f"<p>{NONE} (floorplan/plan.json 없음)</p>"
    elif isinstance(plan, Exception):
        space = f'<p class="warn">plan.json 오류: {_e(plan)}</p>'
    else:
        space = _svg(plan, secs)
        if not any(s["priors"] for s in secs):
            space += f"<p>카메라 위치: {NONE} (colmap/priors.json)</p>"

    lay = data["layout"]
    lay_secs = (lay or {}).get("sections")
    lay_txt = (NONE if lay is None else
               f"구간 {len(lay_secs)}개" if isinstance(lay_secs, (list, dict)) else "있음")
    merged = f"building.ply: {'있음' if data['building'] else NONE} · layout.json: {lay_txt}"

    diag_notes = []
    for s in secs:
        dg = s["diagnose"]
        if dg:
            diag_notes.append(f"<li>{_e(s['name'])}: 재투영 오차 {_num(dg.get('reproj_error_px'), '{:.2f}')} px, "
                              f"순간이동 {len(dg.get('teleports', []))}, 공간 밖 {len(dg.get('outside_space', []))}, "
                              f"동선에서 멂 {len(dg.get('far_from_prior', []))}</li>")
    diag = "<ul>" + "".join(diag_notes) + "</ul>" if diag_notes else f"<p>{NONE}</p>"

    wall_notes = []
    for s in secs:
        pj = s.get("planes")
        if pj:
            fit = pj.get("fit", [])
            nf = sum(1 for i in fit if i.get("source") == "fit")
            big = [i for i in fit if i.get("source") == "fit" and abs(i.get("shift_m") or 0) >= 0.05]
            txt = (f"{_e(s['name'])}: 면 {len(fit)}개 중 {nf}개를 정합 점으로 다시 맞춤"
                   + (f", 도면과 5cm 넘게 다른 면 " + ", ".join(f"{_e(i['id'])} {i['shift_m'] * 100:+.0f}cm" for i in big)
                      if big else "")
                   + (f", 초기 점 {_num((pj.get('seed') or {}).get('n_added'))}개" if pj.get("seed") else ""))
            wall_notes.append(f"<li>{txt}</li>")
        for r in s["runs"]:
            w = r.get("walls")
            if w:
                fb, fa = w.get("flatness_before") or {}, w.get("flatness_after") or {}
                wall_notes.append(
                    f"<li>{_e(s['name'])} / {_e(r['name'])}: 벽 평탄도(두께 포함 RMS) "
                    f"{_num((fb.get('thick_m') or 0) * 100, '{:.1f}')} → {_num((fa.get('thick_m') or 0) * 100, '{:.1f}')} cm, "
                    f"붙임 {_num(w.get('n_moved'))}, 부유물 지움 {_num(w.get('n_removed'))}</li>")
    walls_html = "<ul>" + "".join(wall_notes) + "</ul>" if wall_notes else f"<p>{NONE} (도면 좌표 정합일 때만)</p>"

    from .snap import clean_summary
    clean_notes = []
    for s in secs:
        for r in s["runs"]:
            c = r.get("clean")
            if c:
                where = "도면 좌표, 면 보호" if c.get("plan_coords") else "도면 좌표 아님, 일반 정리"
                clean_notes.append(f"<li>{_e(s['name'])} / {_e(r['name'])} ({_e(c.get('input'))} → splat_clean.ply, "
                                   f"{where}): {_e(clean_summary(c))}</li>")
    clean_html = ("<ul>" + "".join(clean_notes) + "</ul>" if clean_notes else f"<p>{NONE}</p>")

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"><title>IndoorGS 보고서 · {_e(project.root.name)}</title>
<style>{CSS}</style></head><body>
<h1>IndoorGS 보고서</h1><div class="sub">{_e(project.root)} · {now}</div>
<h2>구간</h2>{table}
<p class="sub">반투명 비율 = 불투명도 0.5 미만 스플랫 비율. {TRANSLUCENT_WARN:.0%} 이상이면 VR 에서 벽이 비칠 수 있음.</p>
<h2>정합 검사</h2>{diag}
<h2>벽 평탄화</h2>{walls_html}
<h2>학습 후 정리</h2>{clean_html}
<p class="sub">지운 스플랫은 학습 폴더의 removed_clean.ply 로 열어 확인. 바늘 쪼개기로 스플랫 수가 늘어남.</p>
<h2>공간과 카메라 위치</h2>{space}
<h2>합치기</h2><p>{merged}</p>
</body></html>
"""


def write_report(project) -> Path:
    project.report.parent.mkdir(parents=True, exist_ok=True)
    tmp = project.report.with_suffix(".html.tmp")
    tmp.write_text(build_html(project), encoding="utf-8")
    tmp.replace(project.report)
    return project.report


class Report(Stage):
    name = "report"
    scope = "project"

    def outputs(self, ctx: StageContext) -> list[Path]:
        return [ctx.project.report]

    def is_done(self, ctx: StageContext) -> bool:
        return False

    def prev_targets(self, ctx: StageContext) -> list[Path]:
        return []          # 보고서는 매번 새로 만드는 화면 (원본 데이터 아님)

    def run(self, ctx: StageContext) -> StageResult:
        out = write_report(ctx.project)
        return StageResult(True, str(out), [out])
