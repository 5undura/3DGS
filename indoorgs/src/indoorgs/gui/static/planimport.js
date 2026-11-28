// 공간 화면: 도면 불러오기 (DXF · SVG · 사진) -> 초안 미리보기 -> 바꾸기 / 추가하기 / 취소
import { toast, $, esc, T } from './util.js';
import { fromJson, roomPoly, snap, r3 } from './model.js';

const KIND = { dxf: 'dxf', svg: 'svg', png: 'image', jpg: 'image', jpeg: 'image', bmp: 'image', webp: 'image', tif: 'image', tiff: 'image' };

export function initImport(ctx) {
  const { S, view, state } = ctx;
  const panel = $('#space-import-panel'), fileIn = $('#space-file');
  let cur = null;          // {file, kind, opt, corners, url}

  function open() {
    if (S.preview) { toast('불러오기 미리보기 중', true); return; }
    fileIn.value = ''; fileIn.click();
  }
  fileIn.onchange = () => {
    const f = fileIn.files[0]; if (!f) return;
    const kind = KIND[f.name.split('.').pop().toLowerCase()];
    if (!kind) { toast('.dxf .svg 또는 사진만', true); return; }
    close();
    const old = S.model.source || {};
    cur = { file: f, kind, corners: [], opt: kind === 'svg' ? { px_per_m: old.kind === 'svg' && old.px_per_m || 100, recenter: true }
      : kind === 'dxf' ? { recenter: true } : { px_per_m: 50, size_m: [10, 10], save_png: true } };
    setPanel(true);
    if (kind === 'image') showPhoto();
    optionsPanel();
  };

  function setPanel(on) {
    panel.classList.toggle('hidden', !on);
    $('#space-props').classList.toggle('hidden', on);
  }

  // ---------- 1. 설정 ----------
  function optionsPanel() {
    const o = cur.opt, k = cur.kind;
    let body = '';
    if (k === 'svg') body = `<label>px / m</label><input type="number" step="1" id="imp-ppm" value="${o.px_per_m}">`;
    if (k === 'dxf') body = `<label>m / 도면단위</label><input type="number" step="0.001" id="imp-scale" placeholder="파일 단위로 자동" value="${o.scale_m ?? ''}">`;
    if (k === 'image') body = `<label>가로 (m)</label><input type="number" step="0.1" id="imp-w" value="${o.size_m[0]}">
      <label>세로 (m)</label><input type="number" step="0.1" id="imp-h" value="${o.size_m[1]}">
      <label>보정 px / m</label><input type="number" step="1" id="imp-ppm" value="${o.px_per_m}">
      <label></label><label class="chk"><input type="checkbox" id="imp-flat" ${cur.flat ? 'checked' : ''}> 네 점 없이 (위에서 본 그림, px/m = 그림 축척)</label>
      <label></label><label class="chk"><input type="checkbox" id="imp-png" ${o.save_png ? 'checked' : ''}> 적용할 때 plan.png 로 저장</label>`;
    if (k !== 'image') body += `<label></label><label class="chk"><input type="checkbox" id="imp-rc" ${o.recenter ? 'checked' : ''}> 왼쪽 아래를 (0, 0) 으로</label>`;
    panel.innerHTML = `<h3>도면 불러오기</h3><p class="small muted">${esc(cur.file.name)} · ${k.toUpperCase()}</p>
      ${k === 'image' ? '<p class="small">사진 위에서 도면 네 모서리를 클릭하고, 그 사각형의 실제 가로·세로를 넣으세요.</p>'
        : '<p class="small">축척을 확인하세요. 초안을 본 뒤 두 점 실측으로 고칠 수 있음.</p>'}
      <div class="form">${body}</div>
      <div class="btnrow"><button class="primary" id="imp-go">초안 만들기</button><button id="imp-cancel">취소</button></div>`;
    $('#imp-go').onclick = go;
    $('#imp-cancel').onclick = close;
    const flat = $('#imp-flat'); if (flat) flat.onchange = () => { cur.flat = flat.checked; photoMsg(); };
  }
  function readOptions() {
    const o = cur.opt, num = id => parseFloat($(id)?.value);
    if (cur.kind === 'svg') o.px_per_m = num('#imp-ppm');
    if (cur.kind === 'dxf') { const v = num('#imp-scale'); if (isNaN(v)) delete o.scale_m; else o.scale_m = v; }
    if (cur.kind === 'image') {
      o.size_m = [num('#imp-w'), num('#imp-h')]; o.px_per_m = num('#imp-ppm'); o.save_png = $('#imp-png').checked;
      if (cur.flat) { delete o.corners; } else {
        if (cur.corners.length !== 4) throw new Error('사진에서 네 모서리를 클릭하세요');
        if (!(o.size_m[0] > 0 && o.size_m[1] > 0)) throw new Error('가로·세로 실제 길이(m)를 넣으세요');
        o.corners = cur.corners.map(p => p.map(Math.round));
      }
    }
    if ($('#imp-rc')) o.recenter = $('#imp-rc').checked;
    if ('px_per_m' in o && !(o.px_per_m > 0)) throw new Error('px / m 는 양수');
  }

  async function post(opt) {
    const fd = new FormData();
    fd.append('file', cur.file, cur.file.name);
    fd.append('options', JSON.stringify(opt));
    const res = await fetch('/api/plan/import', { method: 'POST', body: fd });
    const j = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(typeof j.detail === 'string' ? j.detail : `${res.status}`);
    return j;
  }

  async function go() {
    try { readOptions(); } catch (e) { toast(e.message, true); return; }
    const btn = $('#imp-go'); btn.disabled = true; btn.textContent = '만드는 중…';
    try {
      const j = await post({ ...cur.opt, save_png: false });
      hidePhoto();
      startPreview(j);
    } catch (e) { toast(e.message, true); btn.disabled = false; btn.textContent = '초안 만들기'; }
  }

  // ---------- 사진 네 점 ----------
  function showPhoto() {
    cur.url = URL.createObjectURL(cur.file);
    const img = $('#space-photo-img');
    img.src = cur.url;
    img.onload = () => { $('#space-photo-svg').setAttribute('viewBox', `0 0 ${img.naturalWidth} ${img.naturalHeight}`); drawCorners(); };
    img.onclick = e => {
      if (cur.flat) return;
      const r = img.getBoundingClientRect();
      const p = [(e.clientX - r.left) * img.naturalWidth / r.width, (e.clientY - r.top) * img.naturalHeight / r.height];
      if (cur.corners.length >= 4) cur.corners = [];
      cur.corners.push(p); drawCorners();
    };
    $('#space-photo-reset').onclick = () => { cur.corners = []; drawCorners(); };
    $('#space-photo').classList.remove('hidden');
  }
  function hidePhoto() { $('#space-photo').classList.add('hidden'); }
  function photoMsg() {
    $('#space-photo-msg').textContent = cur.flat ? '네 점 없이: 그림 전체를 px/m 축척으로 읽음'
      : `도면 네 모서리 클릭 (순서 상관없음) · ${cur.corners.length}/4`;
  }
  function ordered(ps) {        // 왼위, 오른위, 오른아래, 왼아래 (image_import.order_corners 와 같은 방식)
    const by = f => ps.reduce((b, p) => f(p) < f(b) ? p : b);
    return [by(p => p[0] + p[1]), by(p => p[1] - p[0]), by(p => -(p[0] + p[1])), by(p => p[0] - p[1])];
  }
  function drawCorners() {
    const img = $('#space-photo-img'), k = (img.naturalWidth || 1000) / 250, c = cur.corners;
    let s = '';
    if (c.length === 4) s += `<polygon points="${ordered(c).map(p => p.join(',')).join(' ')}" fill="${T.selFill}" stroke="${T.sel}" stroke-width="${k / 2}"/>`;
    s += c.map((p, i) => `<circle cx="${p[0]}" cy="${p[1]}" r="${k * 1.6}" fill="${T.anchor}" stroke="${T.sel}" stroke-width="${k / 2}"/>
      <text x="${p[0] + k * 2.5}" y="${p[1] - k * 2}" font-size="${k * 5}" fill="${T.sel}" stroke="${T.bg}" stroke-width="${k / 3}" paint-order="stroke">${i + 1}</text>`).join('');
    $('#space-photo-svg').innerHTML = s;
    photoMsg();
  }

  // ---------- 2. 미리보기 ----------
  function startPreview(j) {
    const m = fromJson(j.plan);
    const P = {
      j, rooms: m.rooms, doors: m.doors, source: m.source, segs: j.segments, off: [0, 0], measure: null, bg: null,
      draw, down, hover: null, cancel: close,
    };
    S.preview = P; S.sel = null;
    if (j.rectified_png) {
      const img = new Image();
      img.onload = () => { P.bg = img; setBg(); };
      img.src = j.rectified_png;
    } else view.bg = null;
    previewPanel();
    view.fit([...P.rooms.flatMap(roomPoly), ...S.model.rooms.flatMap(roomPoly)]);
  }
  function setBg() {
    const P = S.preview; if (!P?.bg) return;
    const al = P.j.plan.source.image_align;
    view.bg = { img: P.bg, opacity: 0.45, pxPerM: al.px_per_m, x: P.off[0], y: P.off[1] + (al.origin_px[1] - P.bg.naturalHeight) / al.px_per_m };
    view.redraw();
  }
  function draw(g) {
    const P = S.preview;
    g.strokeStyle = T.draft; g.globalAlpha = .55; g.lineWidth = 1; g.beginPath();
    for (const [a, b] of P.segs) { g.moveTo(view.sx(a[0]), view.sy(a[1])); g.lineTo(view.sx(b[0]), view.sy(b[1])); }
    g.stroke(); g.globalAlpha = 1;
    g.setLineDash([5, 3]);
    for (const r of P.rooms) {
      const pts = roomPoly(r);
      view.poly(pts); g.fillStyle = T.draft; g.globalAlpha = .12; g.fill(); g.globalAlpha = 1;
      g.lineWidth = 1.5; g.strokeStyle = T.draft; g.stroke();
      const cx = pts.reduce((a, p) => a + p[0], 0) / pts.length, cy = pts.reduce((a, p) => a + p[1], 0) / pts.length;
      g.fillStyle = T.draft; g.font = '11px system-ui'; g.textAlign = 'center';
      g.fillText(r.id, view.sx(cx), view.sy(cy));
    }
    g.setLineDash([]);
    const ms = P.measure;
    if (ms?.length) {
      g.strokeStyle = T.cursor; g.fillStyle = T.cursor; g.lineWidth = 2; g.beginPath();
      ms.forEach((p, i) => i ? g.lineTo(view.sx(p[0]), view.sy(p[1])) : g.moveTo(view.sx(p[0]), view.sy(p[1])));
      g.stroke();
      for (const p of ms) { g.beginPath(); g.arc(view.sx(p[0]), view.sy(p[1]), 4, 0, 7); g.fill(); }
    }
  }
  function snapPt(w) {             // 선분 끝점·방 꼭짓점에 붙임 (화면 10px)
    const P = S.preview;
    let best = null, bd = 10;
    const cand = [...P.segs.flat(), ...P.rooms.flatMap(roomPoly)];
    for (const p of cand) { const d = view.px(Math.hypot(p[0] - w[0], p[1] - w[1])); if (d < bd) { bd = d; best = p; } }
    return best ? [...best] : w;
  }
  function down(w) {
    const P = S.preview;
    if (P.measure) {
      if (P.measure.length >= 2) P.measure = [];
      P.measure.push(snapPt(w)); view.redraw(); previewPanel();
      return {};
    }
    // 초안 끌어서 옮기기 (빈 곳이 아니라 초안 방 위에서만)
    const hit = P.rooms.some(r => inPoly(w, roomPoly(r)));
    if (!hit) return null;
    let last = [0, 0];
    return {
      move(p) {
        const d = [snap(p[0] - w[0]), snap(p[1] - w[1])], dx = d[0] - last[0], dy = d[1] - last[1];
        if (!dx && !dy) return;
        last = d; shift(dx, dy); view.redraw();
      },
    };
  }
  function inPoly([x, y], pts) {
    let ins = false;
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      const [xi, yi] = pts[i], [xj, yj] = pts[j];
      if ((yi > y) !== (yj > y) && x < xi + (y - yi) * (xj - xi) / (yj - yi)) ins = !ins;
    }
    return ins;
  }
  function shift(dx, dy) {
    const P = S.preview;
    P.off = [r3(P.off[0] + dx), r3(P.off[1] + dy)];
    for (const r of P.rooms) {
      if (r.kind === 'box') { r.cx = r3(r.cx + dx); r.cy = r3(r.cy + dy); }
      else r.floor = r.floor.map(([x, y]) => [r3(x + dx), r3(y + dy)]);
    }
    for (const d of P.doors) d.center = [r3(d.center[0] + dx), r3(d.center[1] + dy)];
    P.segs = P.segs.map(([a, b]) => [[a[0] + dx, a[1] + dy], [b[0] + dx, b[1] + dy]]);
    setBg();
  }

  function scaleText() {
    const s = S.preview.j.plan.source;
    if (s.kind === 'svg') return `${r3(s.px_per_m)} px/m`;
    if (s.kind === 'dxf') return `${s.scale_m} m/단위 (INSUNITS ${s.insunits})`;
    return s.size_m ? `${s.size_m[0]} × ${s.size_m[1]} m, ${s.px_per_m} px/m` : `${s.px_per_m} px/m`;
  }
  function previewPanel() {
    const P = S.preview, j = P.j, b = j.bounds;
    const ms = P.measure, mlen = ms?.length === 2 ? Math.hypot(ms[1][0] - ms[0][0], ms[1][1] - ms[0][1]) : null;
    panel.innerHTML = `<h3>도면 불러오기 · 초안</h3>
      <p class="small">방 ${P.rooms.length}개 · 선분 ${P.segs.length}개${b ? ` · 전체 ${r3(b[2] - b[0])} × ${r3(b[3] - b[1])} m` : ''}</p>
      <p class="small muted">축척 ${esc(scaleText())} · 초안 방을 끌면 위치 이동</p>
      ${j.warnings.length ? `<ul class="warnlist">${j.warnings.map(w => `<li>${esc(w)}</li>`).join('')}</ul>` : ''}
      <div class="btnrow"><button id="imp-measure" class="${P.measure ? 'on' : ''}">${P.measure ? '실측 끄기' : '두 점 실측'}</button>
        <button id="imp-back">설정 다시</button></div>
      ${P.measure ? `<p class="small">${mlen == null ? '초안에서 길이를 아는 두 점을 클릭 (끝점에 붙음)' : `잰 길이 ${r3(mlen)} m`}</p>
        ${mlen ? `<div class="form"><label>실제 길이 (m)</label><input type="number" step="0.01" id="imp-real"></div>
        <div class="btnrow"><button id="imp-rescale">이 길이로 축척 고치기</button></div>` : ''}` : ''}
      <div class="btnrow"><button class="primary" id="imp-replace" title="기존 방·문 대신 초안으로">바꾸기</button>
        <button id="imp-add" title="기존 방에 합치기 (id 겹치면 새 id)">추가하기</button><button id="imp-cancel">취소</button></div>
      <p class="small muted">적용해도 저장 전까지 plan.json 은 그대로.</p>`;
    $('#imp-measure').onclick = () => { P.measure = P.measure ? null : []; view.redraw(); previewPanel(); };
    $('#imp-back').onclick = () => { endPreview(); if (cur.kind === 'image') showPhoto(); optionsPanel(); };
    $('#imp-replace').onclick = () => apply(true);
    $('#imp-add').onclick = () => apply(false);
    $('#imp-cancel').onclick = close;
    const rs = $('#imp-rescale'); if (rs) rs.onclick = () => rescale(mlen, parseFloat($('#imp-real').value));
  }

  async function rescale(measured, real) {
    if (!(real > 0) || !(measured > 0)) { toast('실제 길이를 넣으세요', true); return; }
    const k = real / measured, s = S.preview.j.plan.source, o = cur.opt;
    if (cur.kind === 'svg') o.px_per_m = s.px_per_m / k;
    else if (cur.kind === 'dxf') o.scale_m = s.scale_m * k;
    else if (o.corners) o.size_m = o.size_m.map(v => r3(v * k));
    else o.px_per_m = s.px_per_m / k;
    try { const j = await post({ ...o, save_png: false }); endPreview(); startPreview(j); toast(`축척 고침 (×${r3(k)})`); }
    catch (e) { toast(e.message, true); }
  }

  // ---------- 3. 적용 ----------
  async function apply(replace) {
    const P = S.preview, m = S.model;
    let align = null;
    if (cur.kind === 'image' && cur.opt.save_png) {
      try {
        const j = await post({ ...cur.opt, save_png: true });
        const a = j.plan.source.image_align;
        align = { ...a, origin_px: [r3(a.origin_px[0] - P.off[0] * a.px_per_m), r3(a.origin_px[1] + P.off[1] * a.px_per_m)] };
        state.project.has_plan_image = true;
        toast(j.moved_png ? `plan.png 저장 (예전 것: ${j.moved_png})` : 'plan.png 저장');
      } catch (e) { toast(e.message, true); return; }
    }
    if (replace) {
      const src = { ...P.source }; delete src.image_align;
      if (P.off[0] || P.off[1]) src.offset = [r3((src.offset?.[0] || 0) + P.off[0]), r3((src.offset?.[1] || 0) + P.off[1])];
      if (align) src.image_align = align; else if (m.source?.image_align) src.image_align = m.source.image_align;
      m.rooms = P.rooms; m.doors = P.doors; m.source = src;
    } else {
      const ren = {}, taken = new Set(m.rooms.map(r => r.id));
      for (const r of P.rooms) {
        let id = r.id, n = 1;
        if (taken.has(id)) { const base = id.replace(/\d+$/, '') || 'room'; while (taken.has(`${base}${n}`)) n++; id = `${base}${n}`; }
        ren[r.id] = id; r.id = id; taken.add(id); m.rooms.push(r);
      }
      const dt = new Set(m.doors.map(d => d.id));
      for (const d of P.doors) {
        let id = d.id, n = 1; while (dt.has(id)) id = `door${n++}`;
        d.id = id; dt.add(id); d.rooms = d.rooms.map(x => ren[x] || x); m.doors.push(d);
      }
      if (align) m.source = { ...m.source, image_align: align };
    }
    const n = P.rooms.length;
    endPreview(); close();
    ctx.select(null); ctx.setDirty(); ctx.fit(); ctx.loadBg();
    if (!align) toast(`${replace ? '바꿈' : '추가함'}: 방 ${n}개 · 저장하면 plan.json 에 들어감`);
  }

  function endPreview() { S.preview = null; view.bg = null; ctx.applyBg(); }
  function close() {
    if (S.preview) endPreview();
    hidePhoto();
    if (cur?.url) URL.revokeObjectURL(cur.url);
    setPanel(false); panel.innerHTML = '';
    ctx.applyBg();
  }

  return { open, close };
}
