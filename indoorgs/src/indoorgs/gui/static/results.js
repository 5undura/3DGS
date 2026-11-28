// 결과 화면: 보고서 / 벽·정리 (planes.json · walls.json · clean.json · repair.json) / 면 텍스처 보기
import { api, toast, $, esc, T } from './util.js';

const BIG = 0.05;                  // 도면과 이만큼(m) 넘게 다르면 강조
const SRC_COLOR = { 85: [34, 197, 94], 170: [245, 158, 11], 255: [168, 85, 247] };   // 사진 / 메움 / 직접 칠함
const REASON = { camera: '카메라 앞', detached: '떨어진 덩어리', through: '사진보다 앞', air: '허공', faint: '옅고 큼', big: '큰 덩어리', huge: '아주 큰 것' };

const n0 = v => v == null ? '-' : Number(v).toLocaleString();
const f = (v, d = 1) => v == null || !isFinite(v) ? '-' : Number(v).toFixed(d);
const cm = v => v == null || !isFinite(v) ? '-' : (v * 100).toFixed(1);
const pct = v => v == null || !isFinite(v) ? '-' : (v * 100).toFixed(0) + '%';
const none = '<span class="muted">아직 없음</span>';

export function initResults(state) {
  const R = { view: 'report', sec: null, run: null, data: null, tex: null, imgs: null };

  document.querySelectorAll('#result-views button').forEach(b => b.onclick = () => setView(b.dataset.view));
  // 보고서 파일은 밝은 그대로 두고, GUI 안에서 볼 때만 어두운 테마를 덧씌움
  $('#report-frame').addEventListener('load', e => {
    const d = e.target.contentDocument; if (!d?.head) return;
    const st = d.createElement('style');
    st.textContent = `body{background:${T.bg};color:${T.text};font-size:13px}a{color:#5a88cc}
      h1,h2{color:#ececec}.sub,th,.legend{color:${T.mute}}.none{color:#6e6e6e}.warn{color:${T.bad}}
      th,td{border-bottom-color:#3a3a3a}th{background:#2b2b2b}.lbl{fill:${T.label}}
      .map{border-color:#3a3a3a;background:${T.bg}}.map polygon{fill:${T.roomFill};stroke:${T.room}}.map line{stroke:${T.scale}}`;
    d.head.appendChild(st);
  });
  $('#report-rebuild').onclick = () => { $('#report-frame').src = '/api/report?rebuild=1&_=' + Date.now(); };
  const secs = state.project.sections.map(s => s.name);
  $('#walls-section').innerHTML = secs.map(s => `<option>${esc(s)}</option>`).join('');
  R.sec = secs[0] || null;
  $('#walls-section').onchange = e => { R.sec = e.target.value; R.run = null; R.tex = null; load(); };
  $('#walls-run').onchange = e => { R.run = e.target.value; load(); };
  $('#walls-reload').onclick = () => load();
  $('#tex-overlay').onchange = drawTex;
  $('#tex-alpha').oninput = drawTex;

  function setView(v) {
    R.view = v;
    document.querySelectorAll('#result-views button').forEach(b => b.classList.toggle('on', b.dataset.view === v));
    $('#report-frame').classList.toggle('hidden', v !== 'report');
    $('#walls-panel').classList.toggle('hidden', v !== 'walls');
    $('#tex-panel').classList.toggle('hidden', v !== 'tex');
    $('#result-report-tools').classList.toggle('hidden', v !== 'report');
    $('#result-sec-tools').classList.toggle('hidden', v === 'report');
    $('#walls-run').classList.toggle('hidden', v !== 'walls');
    if (v === 'report' && !$('#report-frame').src) $('#report-frame').src = '/api/report?rebuild=1';
    if (v !== 'report') load();
  }

  async function load() {
    if (!R.sec) { $('#walls-panel').innerHTML = '<p class="muted">구간 없음</p>'; return; }
    try {
      R.data = await api(`/api/walls/${encodeURIComponent(R.sec)}` + (R.run ? `?run=${encodeURIComponent(R.run)}` : ''));
    } catch (e) { $('#walls-panel').innerHTML = `<p class="warn">${esc(e.message)}</p>`; return; }
    const d = R.data;
    $('#walls-run').innerHTML = d.runs.length ? d.runs.map(r => `<option ${r === d.run ? 'selected' : ''}>${esc(r)}</option>`).join('')
      : '<option value="">학습 폴더 없음</option>';
    renderWalls(d);
    renderTexList(d);
  }

  // ---------- 벽·정리 ----------
  function renderWalls(d) {
    const errs = Object.values(d.errors || {});
    const files = d.files || {};
    const fileTags = [['walls', 'splat_walls'], ['clean', 'splat_clean'], ['repaired', 'splat_repaired'], ['removed', 'removed_clean']]
      .map(([k, n]) => `<span class="tag ${files[k] ? 'fit' : 'plan'}">${n}.ply ${files[k] ? '있음' : '없음'}</span>`).join(' ');
    $('#walls-panel').innerHTML = `
      ${errs.map(e => `<p class="warn small">${esc(e)}</p>`).join('')}
      <p class="small">${d.run ? `학습 폴더 <b>${esc(d.run)}</b> · ${fileTags}` : '<span class="muted">학습 결과 없음</span>'}</p>
      <h3>면 맞추기 <span class="muted small">walls/planes.json</span></h3>
      ${facesHtml(d.planes)}
      ${mapHtml(d.planes)}
      <div class="cards">
        ${snapHtml(d.snap, d.planes)}
        ${cleanHtml(d.clean, d.clean_summary)}
        ${repairHtml(d.repair)}
      </div>`;
  }

  function facesHtml(pj) {
    if (!pj) return `<p>${none} <span class="muted small">(도면 좌표 정합 + walls.enabled 일 때만)</span></p>`;
    const fit = pj.fit || [];
    const FITTED = new Set(['fit', 'fit_scaled']);
    const nf = fit.filter(i => FITTED.has(i.source)).length;
    const sc = pj.scale_check && pj.scale_check.warn ? `<p class="small warn">축척 경고: ${esc(pj.scale_check.message || '')}</p>` : '';
    const rows = fit.map(i => {
      const big = Math.abs(i.shift_m || 0) >= BIG;
      return `<tr class="${big ? 'big' : ''}" title="${esc(i.reason || '')}"><td>${esc(i.id)}</td><td>${esc(i.kind || '')}</td>
        <td><span class="tag ${FITTED.has(i.source) ? 'fit' : 'plan'}">${i.source === 'fit' ? '맞춤' : i.source === 'fit_scaled' ? '맞춤(축척)' : '도면'}</span></td>
        <td class="num ${big ? 'warn' : ''}">${i.shift_m == null ? '-' : (i.shift_m * 100 >= 0 ? '+' : '') + (i.shift_m * 100).toFixed(1)}</td>
        <td class="num">${f(i.tilt_deg, 2)}${i.clamped ? ' <span class="muted small">제한</span>' : ''}</td>
        <td class="num">${n0(i.n_points)}</td><td class="num">${pct(i.coverage)}</td>
        <td class="small muted">${esc(i.reason || '')}</td></tr>`;
    }).join('');
    return `${sc}<p class="small">면 ${fit.length}개 중 <b>${nf}</b>개를 정합 점으로 맞춤${pj.seed ? ` · 초기 점 ${n0(pj.seed.n_added)}개` : ''}
      · <span class="warn">빨간 줄</span> = 도면과 ${BIG * 100} cm 이상 차이</p>
      <table class="grid small"><tr><th>면</th><th>종류</th><th>출처</th><th class="num">이동 cm</th><th class="num">기울기 °</th>
      <th class="num">점</th><th class="num">덮임</th><th>이유</th></tr>${rows || '<tr><td colspan="8" class="muted">없음</td></tr>'}</table>`;
  }

  // 도면 벽(회색 점선) 대 맞춘 면(색 선) + 이동 화살표 (10배 과장)
  function mapHtml(pj) {
    if (!pj) return '';
    const seg = p => {
      if (p.kind !== 'wall' || !p.origin || !p.U || !p.size) return null;
      const a = [p.origin[0], p.origin[1]], b = [p.origin[0] + p.U[0] * p.size[0], p.origin[1] + p.U[1] * p.size[0]];
      return { id: p.id, a, b, m: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], n: p.n };
    };
    const plan = (pj.plan_planes || []).map(seg).filter(Boolean);
    const fitInfo = Object.fromEntries((pj.fit || []).map(i => [i.id, i]));
    const fitted = (pj.planes || []).map(seg).filter(s => s && ['fit', 'fit_scaled'].includes(fitInfo[s.id]?.source));
    if (!plan.length && !fitted.length) return '';
    const pts = [...plan, ...fitted].flatMap(s => [s.a, s.b]);
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
    const x0 = Math.min(...xs) - 1, x1 = Math.max(...xs) + 1, y0 = Math.min(...ys) - 1, y1 = Math.max(...ys) + 1;
    const k = Math.min(760 / Math.max(x1 - x0, 1e-6), 420 / Math.max(y1 - y0, 1e-6));
    const W = Math.max(200, (x1 - x0) * k), H = Math.max(140, (y1 - y0) * k);
    const X = x => ((x - x0) * k).toFixed(1), Y = y => ((y1 - y) * k).toFixed(1);
    const EX = 10;
    const color = s => { const v = Math.abs(s || 0); return v >= BIG ? T.bad : v >= 0.02 ? T.warn : T.ok; };
    const planById = Object.fromEntries(plan.map(s => [s.id, s]));
    let out = plan.map(s => `<line x1="${X(s.a[0])}" y1="${Y(s.a[1])}" x2="${X(s.b[0])}" y2="${Y(s.b[1])}" stroke="${T.mute}" stroke-width="1.5" stroke-dasharray="5 3"><title>${esc(s.id)} 도면</title></line>`).join('');
    for (const s of fitted) {
      const i = fitInfo[s.id], c = color(i.shift_m);
      out += `<line x1="${X(s.a[0])}" y1="${Y(s.a[1])}" x2="${X(s.b[0])}" y2="${Y(s.b[1])}" stroke="${c}" stroke-width="2.5"><title>${esc(s.id)} ${(i.shift_m * 100).toFixed(1)} cm, ${f(i.tilt_deg, 2)}°</title></line>`;
      const p = planById[s.id];
      if (p && Math.abs(i.shift_m || 0) >= 0.005) {
        const dx = (s.m[0] - p.m[0]) * EX, dy = (s.m[1] - p.m[1]) * EX;
        const ax = p.m[0] + dx, ay = p.m[1] + dy;
        out += `<line x1="${X(p.m[0])}" y1="${Y(p.m[1])}" x2="${X(ax)}" y2="${Y(ay)}" stroke="${c}" stroke-width="1.5" marker-end="url(#arr-${c.slice(1)})"/>`;
        out += `<text x="${X(ax)}" y="${Y(ay)}" dx="4" dy="-3" font-size="10" fill="${c}">${(i.shift_m * 100).toFixed(0)}cm</text>`;
      }
    }
    const marks = [T.bad, T.warn, T.ok].map(c => c.slice(1)).map(c => `<marker id="arr-${c}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" fill="#${c}"/></marker>`).join('');
    const sc = `<line x1="10" y1="${H - 10}" x2="${(10 + k).toFixed(1)}" y2="${H - 10}" stroke="${T.text}" stroke-width="2"/><text x="10" y="${H - 15}" font-size="10">1 m</text>`;
    return `<svg class="wallmap" viewBox="0 0 ${W.toFixed(0)} ${H.toFixed(0)}" width="${W.toFixed(0)}" height="${H.toFixed(0)}"><defs>${marks}</defs>${out}${sc}</svg>
      <p class="small muted">회색 점선 = 도면 벽, 색 선 = 정합 점으로 맞춘 면 (초록 &lt;2 cm · 주황 &lt;5 cm · 빨강 ≥5 cm). 화살표는 이동을 ${EX}배로 키워 그림.</p>`;
  }

  function snapHtml(w, pj) {
    const unfitted = ((pj || {}).fit || []).filter(i => i.source !== 'fit');
    const unf = unfitted.length ? `<p class="small">맞추지 못한 면 (도면 그대로, 붙이기·메우기에 안 씀):</p>
      <ul class="warnlist">${unfitted.map(i => `<li>${esc(i.id)}${i.reason ? ' · ' + esc(i.reason) : ''}</li>`).join('')}</ul>` : '';
    if (!w) return `<div class="card"><h4>벽에 붙이기 <span class="muted small">walls.json</span></h4><p>${none}</p>${unf}</div>`;
    if (w.skipped) return `<div class="card"><h4>벽에 붙이기</h4><p class="warn small">건너뜀: ${esc(w.skipped)}</p>${unf}</div>`;
    const fb = w.flatness_before || {}, fa = w.flatness_after || {};
    const skippedFaces = Object.entries((w.remove || {}).skipped_faces || {});
    const warns = w.warnings || [];
    return `<div class="card"><h4>벽에 붙이기 <span class="muted small">walls.json</span></h4><dl class="kv">
      <dt>평탄도 (두께 RMS)</dt><dd>${cm(fb.thick_m)} → <b>${cm(fa.thick_m)}</b> cm</dd>
      <dt>붙임</dt><dd>${n0(w.n_moved)}</dd>
      <dt>벽 뒤 지움</dt><dd>${n0(w.n_removed)}</dd>
      <dt>쓴 면</dt><dd>${(w.planes_used || []).length}개</dd>
      <dt>지우기 뺀 면</dt><dd>${skippedFaces.length ? skippedFaces.map(([k, v]) => `${esc(k)} <span class="muted">(${esc(v)})</span>`).join('<br>') : '없음'}</dd></dl>
      ${warns.length ? `<ul class="warnlist">${warns.map(x => `<li>${esc(x)}</li>`).join('')}</ul>` : ''}${unf}</div>`;
  }

  function cleanHtml(c, summary) {
    if (!c) return `<div class="card"><h4>학습 후 정리 <span class="muted small">clean.json</span></h4><p>${none}</p></div>`;
    const rb = { ...((c.floaters || {}).removed_by || {}) };
    if ((c.huge || {}).n_removed) rb.huge = c.huge.n_removed;
    const rows = Object.entries(rb).filter(([, v]) => v).map(([k, v]) => `<dt>${esc(REASON[k] || k)}</dt><dd>${n0(v)}</dd>`).join('');
    const q0 = (c.quality_before || {}).count, q1 = (c.quality_after || {}).count;
    const nd = c.needles || {}, sh = c.sh || {};
    return `<div class="card"><h4>학습 후 정리 <span class="muted small">clean.json</span></h4>
      <p class="small muted">${esc(c.input || '')} → splat_clean.ply · ${c.plan_coords ? '도면 좌표, 면 보호' : '도면 좌표 아님'}</p>
      <dl class="kv"><dt><b>지움</b></dt><dd>${n0(Object.values(rb).reduce((a, b) => a + (b || 0), 0))}</dd>${rows}
      <dt>바늘 쪼개기</dt><dd>${c.needles ? `${n0(nd.n_needles)} → 조각 ${n0(nd.n_pieces)}` : '안 함'}</dd>
      ${(c.huge || {}).n_split ? `<dt>거대 쪼갬</dt><dd>${n0(c.huge.n_split)}</dd>` : ''}
      <dt>SH 줄임</dt><dd>${c.sh ? n0(sh.n_damped) : '안 함'}</dd>
      <dt>스플랫 수</dt><dd>${n0(q0)} → <b>${n0(q1)}</b>${q0 ? ` (${((q1 / q0 - 1) * 100).toFixed(1)}%)` : ''}</dd></dl>
      ${summary ? `<p class="small muted">${esc(summary)}</p>` : ''}</div>`;
  }

  function repairHtml(r) {
    if (!r) return `<div class="card"><h4>면 메우기 <span class="muted small">repair.json</span></h4><p>${none}</p></div>`;
    const faces = Object.entries(r.faces || {});
    const nTexFaces = faces.filter(([, v]) => v.n_textured != null).length;
    const nTex = faces.reduce((a, [, v]) => a + (v.n_textured || 0), 0);
    const rows = faces.filter(([, v]) => v.hole_m2_before > 0 || v.n_added).map(([id, v]) => `<tr><td>${esc(id)}</td>
      <td class="num">${f(v.hole_m2_before, 2)} → ${f(v.hole_m2_after, 2)}</td><td class="num">${n0(v.n_added)}</td>
      <td class="num">${v.n_textured == null ? '-' : n0(v.n_textured)}</td></tr>`).join('');
    return `<div class="card"><h4>면 메우기 <span class="muted small">repair.json</span></h4>
      <dl class="kv"><dt>구멍 m²</dt><dd>${f(r.hole_m2_before, 2)} → <b>${f(r.hole_m2_after, 2)}</b></dd>
      <dt>더한 스플랫</dt><dd>${n0(r.n_added)}</dd>
      <dt>텍스처 색</dt><dd>${nTexFaces ? `${n0(nTex)}개 (면 ${nTexFaces}개)` : '안 씀'}</dd></dl>
      ${rows ? `<table class="grid small"><tr><th>면</th><th class="num">구멍 m²</th><th class="num">더함</th><th class="num">텍스처</th></tr>${rows}</table>` : '<p class="small muted">메울 구멍 없음</p>'}</div>`;
  }

  // ---------- 면 텍스처 ----------
  function renderTexList(d) {
    const list = d.textures || [];
    if (!list.some(t => t.name === R.tex)) R.tex = list[0]?.name || null;
    $('#tex-list').innerHTML = list.length ? list.map(t => {
      const s = t.stats || {};
      return `<button data-tex="${esc(t.name)}" class="${t.name === R.tex ? 'on' : ''}"><b>${esc(t.plane_id)}</b><br>
        <span class="muted">사진 ${f(s.photo_m2, 2)} · 메움 ${f(s.inpaint_m2, 2)} · 칠함 ${f(s.manual_m2, 2)} m²</span></button>`;
    }).join('') : '<p class="muted small">walls/tex 없음 (repair + walls.texture 일 때 만듦)</p>';
    $('#tex-list').querySelectorAll('[data-tex]').forEach(b => b.onclick = () => { R.tex = b.dataset.tex; renderTexList(d); });
    loadTex(list.find(t => t.name === R.tex));
  }

  function loadImg(url) {
    return new Promise((ok, bad) => { const im = new Image(); im.onload = () => ok(im); im.onerror = () => bad(new Error('이미지 못 읽음')); im.src = url; });
  }

  async function loadTex(t) {
    R.imgs = null;
    if (!t) { drawTex(); return; }
    const base = `/api/walls/${encodeURIComponent(R.sec)}/tex/`, bust = `?_=${Date.now()}`;
    try {
      const [img, src] = await Promise.all([loadImg(base + encodeURIComponent(t.name) + bust),
        t.src ? loadImg(base + encodeURIComponent(t.src) + bust) : Promise.resolve(null)]);
      R.imgs = { img, src, t, mask: src ? maskCanvas(src) : null };
    } catch (e) { toast(e.message, true); }
    $('#tex-overlay').disabled = !R.imgs?.mask;
    drawTex();
  }

  function maskCanvas(src) {
    const c = document.createElement('canvas'); c.width = src.naturalWidth; c.height = src.naturalHeight;
    const g = c.getContext('2d'); g.drawImage(src, 0, 0);
    const im = g.getImageData(0, 0, c.width, c.height), p = im.data;
    for (let i = 0; i < p.length; i += 4) {
      const v = p[i], key = v > 212 ? 255 : v > 127 ? 170 : v > 42 ? 85 : 0, col = SRC_COLOR[key];
      if (col) { p[i] = col[0]; p[i + 1] = col[1]; p[i + 2] = col[2]; p[i + 3] = 255; } else p[i + 3] = 0;
    }
    g.putImageData(im, 0, 0);
    return c;
  }

  function drawTex() {
    const cv = $('#tex-canvas'), g = cv.getContext('2d');
    if (!R.imgs) { cv.width = 1; cv.height = 1; $('#tex-info').textContent = ''; return; }
    const { img, mask, t } = R.imgs;
    const k = Math.max(1, Math.floor(900 / Math.max(img.naturalWidth, 1)));        // 작은 텍스처는 정수배 확대
    cv.width = img.naturalWidth * k; cv.height = img.naturalHeight * k;
    g.imageSmoothingEnabled = false;
    g.drawImage(img, 0, 0, cv.width, cv.height);
    if (mask && $('#tex-overlay').checked) {
      g.globalAlpha = +$('#tex-alpha').value; g.drawImage(mask, 0, 0, cv.width, cv.height); g.globalAlpha = 1;
    }
    const size = t.size ? ` · ${f(t.size[0], 2)} × ${f(t.size[1], 2)} m` : '';
    $('#tex-info').textContent = `${t.name} · ${img.naturalWidth}×${img.naturalHeight}px${t.res ? ` · ${(t.res * 100).toFixed(1)} cm/px` : ''}${size}`;
  }

  return { show: sub => setView(['report', 'walls', 'tex'].includes(sub) ? sub : R.view) };
}
