// 동선 화면: 브러시로 선 그리기, 영상 보며 고정점 찍기. 형식은 CONTRACT.md 4절 path.json
import { api, toast, $, esc, T } from './util.js';
import { roomPoly, r3, Polyline } from './model.js';
import { PlanView } from './view.js';

const MIN_STEP = 0.15;     // 브러시 점 간격 (m)

export function initPath(state) {
  const view = new PlanView($('#path-canvas'));
  const video = $('#path-video');
  const P = { section: null, data: empty(), tool: 'pan', dirty: false, selA: null };
  let rooms = [], doors = [];

  function empty() { return { version: 1, camera_height: 1.4, sigma_xy: 0.5, sigma_z: 0.3, stroke: null, anchors: [] }; }
  const line = () => (P.data.stroke && P.data.stroke.length >= 2 ? new Polyline(P.data.stroke) : null);
  const setDirty = (v = true) => { P.dirty = v; $('#path-dirty').textContent = v ? '저장 안 됨' : ''; };

  // 고정점 -> (선 위 위치, 거리 s)
  function anchorPos(a, pl) {
    if (a.point) { if (!pl) return { q: a.point, s: null }; const pr = pl.project(a.point); return { q: pr.q, s: pr.s }; }
    return pl ? { q: pl.at(a.s), s: a.s } : null;
  }
  // 시간 t 의 예상 위치 (pathing.PathModel 과 같은 규칙: 고정점 사이 일정 속도, 앞뒤는 멈춤)
  function posAt(t) {
    const pl = line(); if (!pl) return null;
    const ks = P.data.anchors.map(a => [a.t, anchorPos(a, pl).s]).filter(k => k[1] != null).sort((a, b) => a[0] - b[0]);
    if (!ks.length) return null;
    let s;
    if (t <= ks[0][0]) s = ks[0][1];
    else if (t >= ks[ks.length - 1][0]) s = ks[ks.length - 1][1];
    else { let i = 1; while (ks[i][0] < t) i++; const [t0, s0] = ks[i - 1], [t1, s1] = ks[i]; s = t1 > t0 ? s0 + (s1 - s0) * (t - t0) / (t1 - t0) : s1; }
    return pl.at(s);
  }

  // ---------- 그리기 ----------
  view.draw = g => {
    view.rooms(rooms, { dim: true });
    for (const d of doors) { const r = rooms.find(x => x.id === d.rooms[0]); view.door(d, r && roomPoly(r)); }
    const st = P.data.stroke, pl = line();
    if (st && st.length) {
      g.strokeStyle = T.route; g.lineWidth = 3; g.lineJoin = g.lineCap = 'round';
      g.beginPath(); st.forEach(([x, y], i) => i ? g.lineTo(view.sx(x), view.sy(y)) : g.moveTo(view.sx(x), view.sy(y))); g.stroke();
      // 시작 표시
      g.fillStyle = T.route; g.beginPath(); g.arc(view.sx(st[0][0]), view.sy(st[0][1]), 5, 0, 7); g.fill();
    }
    const order = sortedIdx();
    order.forEach((i, k) => {
      const a = P.data.anchors[i], p = anchorPos(a, pl); if (!p) return;
      const X = view.sx(p.q[0]), Y = view.sy(p.q[1]), sel = P.selA === i;
      g.fillStyle = T.anchor; g.strokeStyle = sel ? T.handle : T.route; g.lineWidth = sel ? 3 : 2;
      g.beginPath(); g.arc(X, Y, 8, 0, 7); g.fill(); g.stroke();
      g.fillStyle = T.anchorTxt; g.font = '600 10px system-ui'; g.textAlign = 'center'; g.textBaseline = 'middle';
      g.fillText(k + 1, X, Y + .5);
      g.textBaseline = 'alphabetic'; g.font = '11px system-ui'; g.fillStyle = T.label;
      view.label(`${a.t.toFixed(1)}s`, X, Y - 12);
    });
    const cur = posAt(video.currentTime || 0);
    if (cur) {
      g.fillStyle = T.cursor; g.beginPath(); g.arc(view.sx(cur[0]), view.sy(cur[1]), 5, 0, 7); g.fill();
    }
  };
  const sortedIdx = () => P.data.anchors.map((a, i) => i).sort((i, j) => P.data.anchors[i].t - P.data.anchors[j].t);

  function hitAnchor(w) {
    const pl = line();
    for (let i = P.data.anchors.length - 1; i >= 0; i--) {
      const p = anchorPos(P.data.anchors[i], pl);
      if (p && Math.hypot(view.px(w[0] - p.q[0]), view.px(w[1] - p.q[1])) < 10) return i;
    }
    return -1;
  }

  view.handler = {
    down(w) {
      if (!P.section) return null;
      if (P.tool === 'draw') {
        P.data.stroke = [[r3(w[0]), r3(w[1])]]; setDirty();
        return {
          move(w2) {
            const last = P.data.stroke[P.data.stroke.length - 1];
            if (Math.hypot(w2[0] - last[0], w2[1] - last[1]) >= MIN_STEP) { P.data.stroke.push([r3(w2[0]), r3(w2[1])]); view.redraw(); }
          },
          up() {
            if (P.data.stroke.length < 2) { P.data.stroke = null; toast('선이 너무 짧음', true); }
            // 거리(s)로 적힌 고정점은 선이 바뀌면 의미가 없어서 지금 위치 근처 점으로 바꿈
            else if (P.data.anchors.length) toast(`고정점 ${P.data.anchors.length}개가 새 선 위로 옮겨짐. 위치를 확인하세요`);
            renderAnchors(); view.redraw();
          },
        };
      }
      const hit = hitAnchor(w);
      if (hit >= 0) {                                   // 고정점 끌기: 선 위로만
        P.selA = hit; renderAnchors();
        return { move(w2) { const a = P.data.anchors[hit]; const pl = line(); const q = pl ? pl.project(w2).q : w2; delete a.s; a.point = [r3(q[0]), r3(q[1])]; setDirty(); view.redraw(); renderAnchors(); } };
      }
      if (P.tool === 'anchor') { addAnchorAt(w); return {}; }
      return null;
    },
  };

  function addAnchorAt(w) {
    const pl = line();
    if (!pl) { toast('먼저 브러시로 동선을 그리세요', true); return; }
    const pr = pl.project(w);
    if (view.px(pr.dist) > 20) { toast('선 가까이를 클릭하세요', true); return; }
    P.data.anchors.push({ t: r3(video.currentTime || 0), point: [r3(pr.q[0]), r3(pr.q[1])] });
    P.selA = P.data.anchors.length - 1; setDirty(); renderAnchors(); view.redraw();
  }
  $('#path-add-here').onclick = () => {
    const q = posAt(video.currentTime || 0) || (P.data.stroke?.[0]);
    if (!q) { toast('먼저 동선을 그리세요', true); return; }
    P.data.anchors.push({ t: r3(video.currentTime || 0), point: [r3(q[0]), r3(q[1])] });
    P.selA = P.data.anchors.length - 1; setDirty(); renderAnchors(); view.redraw();
  };

  // ---------- 고정점 목록 ----------
  function renderAnchors() {
    const pl = line(), tb = $('#path-anchors');
    const order = sortedIdx();
    if (!order.length) { tb.innerHTML = '<tr><td class="muted">고정점 없음. 최소 2개 (시작·끝). 고정점 도구로 선 위를 클릭.</td></tr>'; return; }
    tb.innerHTML = '<tr><th>#</th><th>시간 (초)</th><th>위치 (m)</th><th></th></tr>' + order.map((i, k) => {
      const a = P.data.anchors[i], p = anchorPos(a, pl);
      const pos = p ? `${p.q[0].toFixed(2)}, ${p.q[1].toFixed(2)}` : `s=${a.s}`;
      return `<tr data-i="${i}" class="${P.selA === i ? 'sel' : ''}"><td>${k + 1}</td>
        <td><input type="number" step="0.1" min="0" value="${a.t}" data-t></td><td class="small">${esc(pos)}</td>
        <td><button data-now title="영상 현재 시간으로">지금</button> <button data-seek title="이 시간으로 영상 이동">▶</button> <button data-del title="삭제">×</button></td></tr>`;
    }).join('');
    tb.querySelectorAll('tr[data-i]').forEach(tr => {
      const i = +tr.dataset.i, a = P.data.anchors[i];
      tr.onclick = e => { if (e.target.tagName !== 'INPUT') { P.selA = i; renderAnchors(); view.redraw(); } };
      // 입력칸이 사라지며 blur 가 다시 change 를 부르지 않게 다음 틱에 다시 그림
      tr.querySelector('[data-t]').onchange = e => { const v = parseFloat(e.target.value); if (!isNaN(v) && v >= 0) { a.t = r3(v); setDirty(); } setTimeout(() => { renderAnchors(); view.redraw(); }); };
      tr.querySelector('[data-now]').onclick = () => { a.t = r3(video.currentTime || 0); setDirty(); renderAnchors(); view.redraw(); };
      tr.querySelector('[data-seek]').onclick = () => { if (video.src) video.currentTime = a.t; };
      tr.querySelector('[data-del]').onclick = e => { e.stopPropagation(); P.data.anchors.splice(i, 1); P.selA = null; setDirty(); renderAnchors(); view.redraw(); };
    });
  }
  function renderSettings() {
    const d = P.data;
    $('#path-settings').innerHTML = [['카메라 높이', 'camera_height'], ['xy 오차 (m)', 'sigma_xy'], ['z 오차 (m)', 'sigma_z']]
      .map(([l, k]) => `<label>${l}</label><input type="number" step="0.1" min="0" data-k="${k}" value="${d[k]}">`).join('');
    $('#path-settings').querySelectorAll('input').forEach(i => i.onchange = () => {
      const v = parseFloat(i.value); if (isNaN(v) || v <= 0) { toast('양수를 넣으세요', true); i.value = d[i.dataset.k]; return; }
      d[i.dataset.k] = v; setDirty();
    });
  }

  // ---------- 영상 ----------
  video.addEventListener('timeupdate', () => { $('#path-time').textContent = (video.currentTime || 0).toFixed(1); view.redraw(); });
  video.addEventListener('seeked', () => view.redraw());
  video.addEventListener('error', () => { $('#path-video-msg').textContent = '영상을 열 수 없음: project.yaml 의 경로 확인. HEVC(H.265) 영상은 브라우저에 따라 재생이 안 될 수 있음.'; });
  video.addEventListener('loadedmetadata', () => { $('#path-video-msg').textContent = `길이 ${video.duration.toFixed(1)}초`; });

  // ---------- 불러오기·저장 ----------
  async function open(name) {
    if (P.dirty && !confirm('저장하지 않은 동선이 있음. 버릴까요?')) { $('#path-section').value = P.section; return; }
    P.section = name; P.selA = null;
    try { P.data = Object.assign(empty(), await api(`/api/sections/${encodeURIComponent(name)}/path`)); }
    catch (e) { if (e.status !== 404) toast(e.message, true); P.data = empty(); }
    setDirty(false);
    const sec = state.project.sections.find(s => s.name === name);
    $('#path-video-msg').textContent = sec?.video_exists ? '' : `영상 없음: ${sec?.video || ''}`;
    video.src = `/api/sections/${encodeURIComponent(name)}/video`;
    renderAnchors(); renderSettings(); fit();
  }
  function fit() {
    const pts = [...rooms.flatMap(roomPoly), ...(P.data.stroke || [])];
    view.fit(pts);
  }
  async function save() {
    if (!P.section) return;
    const d = P.data, out = { version: 1, camera_height: d.camera_height, sigma_xy: d.sigma_xy, sigma_z: d.sigma_z };
    if (d.stroke) out.stroke = d.stroke;
    out.anchors = [...d.anchors].sort((a, b) => a.t - b.t).map(a => a.point ? { t: a.t, point: a.point } : { t: a.t, s: a.s });
    try {
      await api(`/api/sections/${encodeURIComponent(P.section)}/path`, { method: 'PUT', body: out });
      setDirty(false); toast(`${P.section}/path.json 저장함`);
    } catch (e) { toast(e.message, true); }
  }
  function setTool(t) {
    P.tool = t;
    document.querySelectorAll('#path-tools button').forEach(b => b.classList.toggle('on', b.dataset.tool === t));
    $('#path-hint').textContent = { pan: '빈 곳 끌기: 화면 이동 · 고정점 끌기: 선 따라 옮기기 · 휠: 확대',
      draw: '끌어서 걸어간 길을 그림 (새로 그리면 이전 선은 지워짐)',
      anchor: '영상을 그 장면에 멈추고 선 위를 클릭 → "이 시간에 여기" 고정점' }[t];
    view.c.style.cursor = t === 'pan' ? '' : 'crosshair';
  }

  $('#path-section').innerHTML = state.project.sections.map(s => `<option value="${esc(s.name)}">${esc(s.name)}${s.has_path ? '' : ' (동선 없음)'}</option>`).join('');
  $('#path-section').onchange = e => open(e.target.value);
  document.querySelectorAll('#path-tools button').forEach(b => b.onclick = () => setTool(b.dataset.tool));
  $('#path-clear').onclick = () => { P.data.stroke = null; setDirty(); renderAnchors(); view.redraw(); };
  $('#path-fit').onclick = fit;
  $('#path-save').onclick = save;
  document.addEventListener('keydown', e => {
    if (state.tab !== 'path' || /INPUT|SELECT|TEXTAREA/.test(document.activeElement?.tagName)) return;
    if ((e.ctrlKey || e.metaKey) && e.key === 's') { e.preventDefault(); save(); }
    if (e.key === ' ') { e.preventDefault(); video.paused ? video.play() : video.pause(); }
    if (e.key === 'Delete' && P.selA != null) { P.data.anchors.splice(P.selA, 1); P.selA = null; setDirty(); renderAnchors(); view.redraw(); }
  });
  window.addEventListener('beforeunload', e => { if (P.dirty) { e.preventDefault(); e.returnValue = ''; } });

  // 공간이 바뀌면 배경 방도 갱신
  state.onPlan = m => { rooms = m.rooms; doors = m.doors; view.redraw(); };
  setTool('pan');
  // 처음엔 동선이 있는 구간부터
  const first = (state.project.sections.find(s => s.has_path) || state.project.sections[0])?.name;
  return { open: () => { $('#path-section').value = P.section || first; open(P.section || first); }, view };
}
