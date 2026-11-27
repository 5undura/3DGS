// 공간 화면: 박스 방 편집 (추가·이동·크기·회전), 문, 도면 이미지 배경, 3D 미리보기
import { api, toast, $, esc } from './util.js';
import { fromJson, toJson, roomPoly, boxFloor, snap, r3, pointInPoly, nearestWall } from './model.js';
import { PlanView, ACC } from './view.js';
import { initImport } from './planimport.js';

const RAD = Math.PI / 180;
const HANDLE = 6;              // 손잡이 크기 (px)

export function initSpace(state) {
  const view = new PlanView($('#space-canvas'));
  const S = { model: { rooms: [], doors: [], source: { kind: 'boxes' } }, sel: null, tool: 'select', dirty: false, exists: false };
  state.space = S;
  const snapOn = () => $('#space-snap').checked;

  const setDirty = (v = true) => { S.dirty = v; $('#space-dirty').textContent = v ? '저장 안 됨' : ''; if (v) { sched3d(); schedCheck(); } };
  const roomOf = id => S.model.rooms.find(r => r.id === id);

  // ---------- 그리기 ----------
  view.draw = g => {
    const m = S.model;
    view.rooms(m.rooms, { selected: S.sel?.type === 'room' ? S.sel.obj : null, dim: !!S.preview });
    for (const d of m.doors) { const r = roomOf(d.rooms[0]); view.door(d, r && roomPoly(r), S.sel?.obj === d); }
    if (S.sel?.type === 'room' && S.sel.obj.kind === 'box') {
      const r = S.sel.obj, pts = boxFloor(r), rh = rotHandle(r);
      g.strokeStyle = ACC; g.lineWidth = 1;
      g.beginPath(); g.moveTo(view.sx(r.cx), view.sy(r.cy)); g.lineTo(view.sx(rh[0]), view.sy(rh[1])); g.setLineDash([3, 3]); g.stroke(); g.setLineDash([]);
      g.fillStyle = '#fff';
      for (const [x, y] of pts) { g.beginPath(); g.rect(view.sx(x) - HANDLE, view.sy(y) - HANDLE, HANDLE * 2, HANDLE * 2); g.fill(); g.stroke(); }
      g.beginPath(); g.arc(view.sx(rh[0]), view.sy(rh[1]), HANDLE, 0, 7); g.fill(); g.stroke();
      // 치수
      g.fillStyle = ACC; g.font = '11px system-ui'; g.textAlign = 'center';
      g.fillText(`${r.w.toFixed(2)} × ${r.d.toFixed(2)} m${r.yaw ? ` · ${r.yaw.toFixed(1)}°` : ''}`, view.sx(r.cx), view.sy(r.cy) + 16);
    }
    if (S.preview) S.preview.draw(g);
    if (S.hoverWall) {
      const { q } = S.hoverWall;
      g.fillStyle = ACC; g.beginPath(); g.arc(view.sx(q[0]), view.sy(q[1]), 4, 0, 7); g.fill();
    }
  };

  // 방 벽 위 문: (벽 번호, 벽 위 비율, 벽에서 거리) 로 기억했다가 방이 바뀐 뒤 같은 자리에 다시 놓음.
  // 두 방 사이 문도 움직인 방 쪽 벽을 따라감. 벽에서 0.3m 안이면 벽 위로 붙임
  function doorTies(r) {
    const poly = roomPoly(r);
    return S.model.doors.filter(d => d.rooms.includes(r.id)).map(d => {
      const nw = nearestWall(d.center, poly), L = Math.hypot(nw.b[0] - nw.a[0], nw.b[1] - nw.a[1]) || 1;
      const n = [-(nw.b[1] - nw.a[1]) / L, (nw.b[0] - nw.a[0]) / L];
      let off = (d.center[0] - nw.q[0]) * n[0] + (d.center[1] - nw.q[1]) * n[1];
      if (Math.abs(off) < 0.3) off = 0;
      return { d, i: nw.i, u: nw.u, off };
    });
  }
  function applyTies(r, ties) {
    const poly = roomPoly(r);
    for (const t of ties) {
      const a = poly[t.i % poly.length], b = poly[(t.i + 1) % poly.length], L = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
      const n = [-(b[1] - a[1]) / L, (b[0] - a[0]) / L];
      t.d.center = [r3(a[0] + t.u * (b[0] - a[0]) + n[0] * t.off), r3(a[1] + t.u * (b[1] - a[1]) + n[1] * t.off)];
    }
  }

  function rotHandle(r) {
    const off = r.d / 2 + 24 / view.scale, c = Math.cos(r.yaw * RAD), s = Math.sin(r.yaw * RAD);
    return [r.cx - s * off, r.cy + c * off];
  }
  const near = (w, p, px = HANDLE + 3) => Math.hypot(view.px(w[0] - p[0]), view.px(w[1] - p[1])) <= px;

  // 벽 찾기 (문 추가용): 화면 12px 안
  function findWall(w) {
    let best = null;
    for (const r of S.model.rooms) {
      const nw = nearestWall(w, roomPoly(r));
      if (nw && view.px(nw.dist) < 12 && (!best || nw.dist < best.dist)) best = { ...nw, room: r };
    }
    return best;
  }

  // ---------- 조작 ----------
  view.handler = {
    hover(w) {
      if (S.preview) return S.preview.hover?.(w);
      if (S.tool === 'door') { S.hoverWall = findWall(w); view.redraw(); }
      else if (S.hoverWall) { S.hoverWall = null; view.redraw(); }
    },
    down(w, ev) {
      if (S.preview) return S.preview.down(w, ev);
      if (S.tool === 'door') return addDoor(w);
      const sel = S.sel;
      // 선택된 박스의 손잡이
      if (sel?.type === 'room' && sel.obj.kind === 'box') {
        const r = sel.obj, pts = boxFloor(r);
        if (near(w, rotHandle(r))) return rotateAct(r, ev);
        for (let i = 0; i < 4; i++) if (near(w, pts[i])) return resizeAct(r, pts[(i + 2) % 4]);
      }
      // 문
      for (const d of [...S.model.doors].reverse())
        if (Math.hypot(w[0] - d.center[0], w[1] - d.center[1]) < Math.max(d.width / 2, 8 / view.scale)) { select('door', d); return doorAct(d); }
      // 방 (위에 그린 것 먼저)
      for (const r of [...S.model.rooms].reverse())
        if (pointInPoly(w, roomPoly(r))) { select('room', r); return moveAct(r, w); }
      select(null);
      return null;                                     // 화면 이동
    },
  };

  function moveAct(r, w0) {
    const c0 = r.kind === 'box' ? [r.cx, r.cy] : null, f0 = r.kind === 'poly' ? r.floor.map(p => [...p]) : null;
    const ties = doorTies(r);
    let moved = false;
    return {
      move(w) {
        let dx = w[0] - w0[0], dy = w[1] - w0[1];
        if (r.kind === 'box') {
          const nx = snap(c0[0] + dx, snapOn()), ny = snap(c0[1] + dy, snapOn());
          dx = nx - c0[0]; dy = ny - c0[1]; r.cx = r3(nx); r.cy = r3(ny);
        } else {
          dx = snap(dx, snapOn()); dy = snap(dy, snapOn());
          r.floor = f0.map(([x, y]) => [r3(x + dx), r3(y + dy)]);
        }
        applyTies(r, ties);                                                     // 문도 같이
        moved = true; view.redraw(); props();
      },
      up() { if (moved) setDirty(); },
    };
  }
  function resizeAct(r, opp) {
    const ties = doorTies(r);
    return {
      move(w) {
        const m = [snap(w[0], snapOn()), snap(w[1], snapOn())];
        const c = Math.cos(r.yaw * RAD), s = Math.sin(r.yaw * RAD), vx = m[0] - opp[0], vy = m[1] - opp[1];
        const lx = c * vx + s * vy, ly = -s * vx + c * vy;           // 방 기준 좌표
        r.w = r3(Math.max(0.2, Math.abs(lx))); r.d = r3(Math.max(0.2, Math.abs(ly)));
        const sx = Math.sign(lx) || 1, sy = Math.sign(ly) || 1;
        r.cx = r3(opp[0] + (c * sx * r.w - s * sy * r.d) / 2); r.cy = r3(opp[1] + (s * sx * r.w + c * sy * r.d) / 2);
        applyTies(r, ties); view.redraw(); props();
      },
      up() { setDirty(); },
    };
  }
  function rotateAct(r) {
    const ties = doorTies(r);
    return {
      move(w, ev) {
        let a = Math.atan2(w[1] - r.cy, w[0] - r.cx) / RAD - 90;
        a = ((a + 180) % 360 + 360) % 360 - 180;
        if (snapOn() && !ev.shiftKey) a = Math.round(a / 5) * 5;     // 5도 단위 (Shift: 자유)
        r.yaw = r3(a); applyTies(r, ties); view.redraw(); props();
      },
      up() { setDirty(); },
    };
  }
  function doorAct(d) {
    return {
      move(w) {
        // 문은 벽을 따라서만
        const r = roomOf(d.rooms[0]);
        const nw = r ? nearestWall(w, roomPoly(r)) : null;
        d.center = nw ? [r3(nw.q[0]), r3(nw.q[1])] : [r3(w[0]), r3(w[1])];
        view.redraw(); props();
      },
      up() { setDirty(); },
    };
  }
  function addDoor(w) {
    const wall = findWall(w);
    if (!wall) { toast('벽 위를 클릭하세요', true); return { }; }
    const q = wall.q, rooms = [wall.room.id];
    for (const r of S.model.rooms)       // 맞붙은 다른 방 (0.3m 안)
      if (r !== wall.room && nearestWall(q, roomPoly(r)).dist < 0.3) rooms.push(r.id);
    let n = 1; while (S.model.doors.some(d => d.id === `door${n}`)) n++;
    const d = { id: `door${n}`, rooms, center: [r3(q[0]), r3(q[1])], width: 0.9, height: 2.1 };
    S.model.doors.push(d); select('door', d); setDirty();
    return doorAct(d);
  }

  function addRoom() {
    let n = S.model.rooms.length + 1; while (roomOf(`room${n}`)) n++;
    const cx = snap((view.w / 2 - view.ox) / view.scale), cy = snap((view.oy - view.h / 2) / view.scale);
    const r = { id: `room${n}`, name: '', kind: 'box', cx, cy, w: 4, d: 3, yaw: 0, floor_z: 0, ceiling_z: 2.6 };
    S.model.rooms.push(r); select('room', r); setDirty(); setTool('select');
  }
  function del() {
    const s = S.sel; if (!s) return;
    if (s.type === 'room') {
      S.model.rooms = S.model.rooms.filter(r => r !== s.obj);
      S.model.doors = S.model.doors.map(d => ({ ...d, rooms: d.rooms.filter(id => id !== s.obj.id) })).filter(d => d.rooms.length);
    } else S.model.doors = S.model.doors.filter(d => d !== s.obj);
    select(null); setDirty();
  }

  function select(type, obj) { S.sel = type ? { type, obj } : null; props(); view.redraw(); }
  function setTool(t) {
    S.tool = t; S.hoverWall = null;
    document.querySelectorAll('#space-tools button').forEach(b => b.classList.toggle('on', b.dataset.tool === t));
    $('#space-hint').textContent = t === 'door' ? '방 벽을 클릭하면 문 추가 (맞붙은 방도 자동 연결) · 끌어서 벽 따라 이동'
      : '끌기: 이동 · 모서리: 크기 · 위 동그라미: 회전(Shift: 5도 스냅 끔) · 휠: 확대 · 빈 곳 끌기: 화면 이동';
    view.c.style.cursor = t === 'door' ? 'crosshair' : '';
    view.redraw();
  }

  // ---------- 속성 패널 ----------
  function field(label, key, obj, opt = {}) {
    const v = obj[key] ?? '';
    return `<label>${label}</label><input type="${opt.text ? 'text' : 'number'}" step="${opt.step || 0.1}" data-k="${key}"
      value="${esc(typeof v === 'number' ? r3(v) : v)}" ${opt.disabled ? 'disabled' : ''}>`;
  }
  function props() {
    const el = $('#space-props'), s = S.sel;
    if (!s) {
      const n = S.model.rooms.length;
      el.innerHTML = `<h3>공간</h3><p class="muted small">방 ${n}개 · 문 ${S.model.doors.length}개${S.exists ? '' : ' · 새 공간 (아직 저장 안 함)'}</p>
        <div class="form"><label>모든 방 천장</label><input type="number" step="0.1" id="all-ceil" placeholder="m"></div>
        <p class="muted small">방이나 문을 선택하면 값을 고칠 수 있음.</p>`;
      $('#all-ceil').onchange = e => { const v = parseFloat(e.target.value); if (!isNaN(v)) { S.model.rooms.forEach(r => r.ceiling_z = r3(r.floor_z + v)); setDirty(); } };
      return;
    }
    const o = s.obj;
    if (document.activeElement && el.contains(document.activeElement) && S.editing === o) {
      // 입력 중이면 값만 갱신
      el.querySelectorAll('input[data-k]').forEach(i => { if (i !== document.activeElement && typeof o[i.dataset.k] === 'number') i.value = r3(o[i.dataset.k]); });
      return;
    }
    S.editing = o;
    if (s.type === 'room') {
      const box = o.kind === 'box';
      el.innerHTML = `<h3>방</h3><div class="form">
        ${field('id', 'id', o, { text: 1 })}${field('이름', 'name', o, { text: 1 })}
        ${box ? field('중심 x', 'cx', o) + field('중심 y', 'cy', o) + field('가로 (m)', 'w', o) + field('세로 (m)', 'd', o) + field('회전 (°)', 'yaw', o, { step: 1 })
              : '<label>모양</label><span class="small muted">다각형 (이동만)</span>'}
        ${field('바닥 z', 'floor_z', o)}${field('천장 z', 'ceiling_z', o)}
        <label>천장 높이</label><span class="small" id="ceil-h">${o.ceiling_z != null ? r3(o.ceiling_z - o.floor_z) + ' m' : '없음'}</span></div>`;
    } else {
      el.innerHTML = `<h3>문</h3><div class="form">${field('id', 'id', o, { text: 1 })}
        <label>연결 방</label><input type="text" data-k="rooms" value="${esc(o.rooms.join(', '))}">
        ${field('폭 (m)', 'width', o)}${field('높이 (m)', 'height', o)}
        <label>위치</label><span class="small">${r3(o.center[0])}, ${r3(o.center[1])}</span></div>`;
    }
    el.querySelectorAll('input[data-k]').forEach(i => i.onchange = () => setProp(o, i.dataset.k, i.value, i));
  }
  function setProp(o, k, v, input) {
    if (k === 'id') {
      v = v.trim();
      const list = S.sel.type === 'room' ? S.model.rooms : S.model.doors;
      if (!v || list.some(x => x !== o && x.id === v)) { toast('id 가 비었거나 중복', true); input.value = o.id; return; }
      if (S.sel.type === 'room') S.model.doors.forEach(d => d.rooms = d.rooms.map(id => id === o.id ? v : id));
      o.id = v;
    } else if (k === 'name') o.name = v;
    else if (k === 'rooms') o.rooms = v.split(',').map(x => x.trim()).filter(Boolean);
    else if (k === 'ceiling_z' && v === '') o.ceiling_z = null;
    else {
      const n = parseFloat(v);
      if (isNaN(n) || ((k === 'w' || k === 'd' || k === 'width' || k === 'height') && n <= 0)) { toast('양수를 넣으세요', true); setTimeout(props); return; }
      const ties = S.sel.type === 'room' && ['cx', 'cy', 'w', 'd', 'yaw'].includes(k) ? doorTies(o) : null;
      o[k] = n;
      if (ties) applyTies(o, ties);
    }
    S.editing = null; setDirty(); view.redraw();
    setTimeout(props);              // 입력칸 교체는 다음 틱에 (blur 와 겹치지 않게)
  }

  // ---------- 도면 이미지 ----------
  // 위치·축척은 plan.json source.image_align {px_per_m, origin_px:[x, y], flip_y} (origin_px = 도면 (0,0) 의 이미지 픽셀).
  // 투명도만 이 브라우저에 저장. 예전 localStorage 위치(x, y, pxPerM)는 image_align 이 없을 때만 씀
  const bgKey = 'indoorgs.bg.' + state.project.root;
  const bgCfg = Object.assign({ opacity: 0.5 }, JSON.parse(localStorage.getItem(bgKey) || '{}'));
  let bgImg = null;
  function bgAlign() {           // -> {pxPerM, x, y} (x, y = 이미지 왼쪽 아래의 도면 좌표)
    const a = S.model.source?.image_align, h = bgImg?.naturalHeight || 0;
    if (a?.px_per_m > 0 && Array.isArray(a.origin_px)) {
      const ppm = a.px_per_m;
      return { pxPerM: ppm, x: -a.origin_px[0] / ppm, y: (a.origin_px[1] - h) / ppm };
    }
    return { pxPerM: bgCfg.pxPerM || 50, x: bgCfg.x || 0, y: bgCfg.y || 0 };
  }
  function setAlign(al) {
    const h = bgImg?.naturalHeight || 0;
    S.model.source = { ...S.model.source, image_align: { px_per_m: al.pxPerM, origin_px: [r3(-al.x * al.pxPerM), r3(h + al.y * al.pxPerM)], flip_y: true } };
  }
  function applyBg() { if (bgImg && !S.preview) { view.bg = { img: bgImg, opacity: bgCfg.opacity, ...bgAlign() }; } view.redraw(); }
  function bgPanel() {
    const el = $('#space-bg');
    if (!state.project.has_plan_image) { el.innerHTML = '<p class="muted small">floorplan/plan.png 를 넣거나 사진 도면을 불러오면 배경으로 보임.</p>'; return; }
    const al = bgAlign();
    el.innerHTML = `<div class="form">
      <label>투명도</label><input type="range" min="0" max="1" step="0.05" data-b="opacity" value="${bgCfg.opacity}">
      <label>px / m</label><input type="number" step="1" data-b="pxPerM" value="${r3(al.pxPerM)}">
      <label>왼쪽 아래 x</label><input type="number" step="0.1" data-b="x" value="${r3(al.x)}">
      <label>왼쪽 아래 y</label><input type="number" step="0.1" data-b="y" value="${r3(al.y)}"></div>
      <p class="muted small">위치·축척은 plan.json 에 저장됨.</p>`;
    el.querySelectorAll('input[data-b]').forEach(i => i.oninput = () => {
      const n = parseFloat(i.value), k = i.dataset.b; if (isNaN(n) || (k === 'pxPerM' && n <= 0)) return;
      if (k === 'opacity') { bgCfg.opacity = n; localStorage.setItem(bgKey, JSON.stringify({ opacity: n })); }
      else { const al = bgAlign(); al[k] = n; setAlign(al); setDirty(); }
      applyBg();
    });
  }
  function loadBg() {
    bgPanel();
    if (!state.project.has_plan_image) { bgImg = null; view.bg = null; view.redraw(); return; }
    const img = new Image();
    img.onload = () => { bgImg = img; bgPanel(); applyBg(); };
    img.src = '/api/plan/image?t=' + Date.now();
  }

  // ---------- 검사 ----------
  let tCheck = 0;
  function schedCheck() { clearTimeout(tCheck); tCheck = setTimeout(check, 300); }
  async function check() {
    if (!S.model.rooms.length) { showIssues([]); return; }
    try { showIssues((await api('/api/plan/validate', { method: 'POST', body: toJson(S.model) })).issues); }
    catch (e) { $('#space-issues').innerHTML = `<p class="warn small">${esc(e.message)}</p>`; }
  }
  function showIssues(issues) {
    S.issues = issues;
    const el = $('#space-issues');
    if (!issues.length) { el.innerHTML = '<p class="muted small">문제 없음</p>'; return; }
    el.innerHTML = `<ul class="issues">${issues.map((x, i) => `<li class="${x.level}" data-i="${i}" title="클릭하면 선택">${x.level === 'error' ? '오류' : '경고'} · ${esc(x.message)}</li>`).join('')}</ul>`;
    el.querySelectorAll('li').forEach(li => li.onclick = () => pickIssue(issues[+li.dataset.i]));
  }
  function pickIssue(x) {
    const doorFirst = x.kind.startsWith('door');
    for (const id of x.ids) {
      const d = S.model.doors.find(o => o.id === id), r = roomOf(id);
      if (doorFirst && d) return select('door', d);
      if (r) return select('room', r);
      if (d) return select('door', d);
    }
  }

  // ---------- 3D ----------
  let three = null, t3 = null;
  function sched3d() { if (three) { clearTimeout(t3); t3 = setTimeout(() => three.update(S.model), 150); } }
  async function toggle3d() {
    const wrap = $('#space-3d-wrap');
    wrap.classList.toggle('hidden');
    $('#space-3d').classList.toggle('on', !wrap.classList.contains('hidden'));
    if (wrap.classList.contains('hidden') || three) return;
    $('#space-3d-msg').textContent = '3D 불러오는 중…';
    try {
      const mod = await import('./preview3d.js');
      three = await mod.createPreview(wrap);
      $('#space-3d-msg').textContent = '';
      three.update(S.model);
    } catch (e) {
      console.warn(e);
      $('#space-3d-msg').textContent = '3D 미리보기를 못 불러옴 (three.js 는 인터넷 연결이 필요). 위에서 본 편집은 그대로 쓸 수 있음.';
    }
  }

  // ---------- 불러오기·저장 ----------
  async function load() {
    try {
      S.model = fromJson(await api('/api/plan')); S.exists = true;
    } catch (e) {
      if (e.status !== 404) toast(e.message, true);
      S.model = { rooms: [], doors: [], source: { kind: 'boxes' } }; S.exists = false;
    }
    S.sel = null; setDirty(false); props(); fit(); sched3d(); check(); applyBg();
    state.onPlan?.(S.model);
  }
  function fit() { view.fit(S.model.rooms.flatMap(roomPoly)); }
  async function save() {
    if (S.preview) { toast('도면 불러오기를 먼저 끝내세요 (바꾸기·추가하기·취소)', true); return; }
    if (!S.model.rooms.length) { toast('방이 하나 이상 있어야 저장할 수 있음', true); return; }
    try {
      const res = await api('/api/plan', { method: 'PUT', body: toJson(S.model) });
      S.exists = true; setDirty(false); props();
      const n = (res.issues || []).length, ne = (res.issues || []).filter(x => x.level === 'error').length;
      if (n) toast(`plan.json 저장함 · 검사 ${ne ? `오류 ${ne}개, ` : ''}경고 ${n - ne}개 (오른쪽 목록)`, true); else toast('plan.json 저장함');
      showIssues(res.issues || []);
      state.onPlan?.(S.model);
    } catch (e) { toast(e.message, true); }
  }

  document.querySelectorAll('#space-tools button').forEach(b => b.onclick = () => setTool(b.dataset.tool));
  $('#space-add').onclick = addRoom;
  $('#space-del').onclick = del;
  $('#space-fit').onclick = fit;
  $('#space-save').onclick = save;
  $('#space-3d').onclick = toggle3d;
  $('#space-snap').onchange = () => view.redraw();
  document.addEventListener('keydown', e => {
    if (state.tab !== 'space' || /INPUT|SELECT|TEXTAREA/.test(document.activeElement?.tagName)) return;
    if (S.preview) { if (e.key === 'Escape') S.preview.cancel(); return; }
    if (e.key === 'Delete' || e.key === 'Backspace') { del(); e.preventDefault(); }
    if (e.key === 'Escape') { select(null); setTool('select'); }
    if ((e.ctrlKey || e.metaKey) && e.key === 's') { e.preventDefault(); save(); }
  });
  window.addEventListener('beforeunload', e => { if (S.dirty) { e.preventDefault(); e.returnValue = ''; } });

  const imp = initImport({ S, view, state, setDirty, select, fit, check, loadBg, setTool, applyBg, roomOf });
  $('#space-import').onclick = imp.open;
  loadBg(); setTool('select');
  return { load, view, fit };
}
