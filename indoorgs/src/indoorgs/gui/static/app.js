// 화면 전환, 프로젝트·실행·결과 화면
import { api, toast, $, esc } from './util.js';
import { initSpace } from './space.js';
import { initPath } from './path.js';

const state = { tab: 'project', project: null };
const TABS = ['project', 'space', 'path', 'run', 'result'];
const STATE_TXT = { done: '끝', todo: '대기', skip: '건너뜀', fail: '실패', running: '진행 중' };

async function main() {
  try { state.project = await api('/api/project'); }
  catch (e) { document.body.innerHTML = `<p class="warn" style="padding:24px">프로젝트를 못 읽음: ${esc(e.message)}</p>`; return; }
  $('#proj-name').textContent = state.project.name;
  renderProject();
  const path = initPath(state);           // onPlan 을 먼저 등록
  const space = initSpace(state);
  await space.load();
  path.open();
  initRun();
  document.querySelectorAll('#tabs button').forEach(b => b.onclick = () => show(b.dataset.tab));
  const fromHash = () => { const h = location.hash.slice(1); show(TABS.includes(h) ? h : 'project'); };
  window.addEventListener('hashchange', fromHash);
  fromHash();
}

function show(tab) {
  state.tab = tab; if (location.hash.slice(1) !== tab) location.hash = tab;
  document.querySelectorAll('#tabs button').forEach(b => b.classList.toggle('on', b.dataset.tab === tab));
  document.querySelectorAll('.tab').forEach(s => s.classList.toggle('on', s.id === `tab-${tab}`));
  if (tab === 'run') refreshRun();
  if (tab === 'result' && !$('#report-frame').src) $('#report-frame').src = '/api/report?rebuild=1';
}

// ---------- 프로젝트 ----------
function renderProject() {
  const p = state.project;
  $('#proj-warn').innerHTML = p.warnings.map(w => `<p class="warn small">${esc(w)}</p>`).join('');
  const tool = v => v ? esc(v) : '<span class="muted">없음</span>';
  const rows = [
    ['폴더', esc(p.root)],
    ['공간 (plan.json)', p.has_space ? '<span class="ok">있음</span>' : '<span class="muted">없음 · 공간 화면에서 만들 수 있음</span>'],
    ['도면 이미지', p.has_plan_image ? '있음' : '<span class="muted">없음</span>'],
    ['GPU', esc(p.gpu_profile)],
    ['학습', `${esc(p.train.trainer)} · ${esc(p.train.preset)}`],
    ['정합', `${esc(p.register.matcher)} · ${esc(p.register.mapper)} · 공간 위치 제약 ${p.register.use_floorplan_prior ? '사용' : '안 씀'}`],
    ['추출', `min_diff ${p.extract.min_diff} · sharp_ratio ${p.extract.sharp_ratio}`],
    ['COLMAP', tool(p.tools.colmap)], ['Brush', tool(p.tools.brush)], ['LichtFeld', tool(p.tools.lichtfeld)],
  ];
  $('#proj-summary').innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
  $('#proj-sections').innerHTML = '<tr><th>구간</th><th>영상</th><th>동선</th><th>프레임</th></tr>' + p.sections.map(s => `<tr>
    <td>${esc(s.name)}</td><td class="${s.video_exists ? '' : 'warn'}">${esc(s.video)}${s.video_exists ? '' : ' (없음)'}</td>
    <td>${s.has_path ? '있음' : '<span class="muted">없음</span>'}</td><td>${s.has_frames ? '있음' : '<span class="muted">아직</span>'}</td></tr>`).join('');
}

// ---------- 실행 ----------
let runTimer = 0, curLog = '';
function initRun() {
  $('#run-until').innerHTML += state.project.stages.map(s => `<option value="${s}">${s}</option>`).join('');
  $('#run-all').onclick = () => startRun(null);
  $('#run-log').onchange = e => { curLog = e.target.value; refreshLog(); };
  setInterval(() => { if (state.tab === 'run') refreshRun(); }, 2000);
}
async function startRun(section) {
  try {
    await api('/api/run', { method: 'POST', body: { section, until: $('#run-until').value || null } });
    toast(section ? `${section} 실행 시작` : '전체 실행 시작');
    refreshRun();
  } catch (e) { toast(e.message, true); }
}
async function refreshRun() {
  clearTimeout(runTimer);
  let st, rs, logs;
  try { [st, rs, logs] = await Promise.all([api('/api/status'), api('/api/run/state'), api('/api/logs')]); }
  catch (e) { $('#run-state').textContent = '서버 연결 끊김'; return; }
  const secs = [...new Set(st.rows.map(r => r.section))];
  const cell = Object.fromEntries(st.rows.map(r => [`${r.section}|${r.stage}`, r]));
  $('#run-grid').innerHTML = `<tr><th>구간</th>${st.stages.map(s => `<th style="text-align:center">${s}</th>`).join('')}<th></th></tr>` +
    secs.map(sec => `<tr><td>${sec === '*' ? '<span class="muted">프로젝트 전체</span>' : esc(sec)}</td>${st.stages.map(s => {
      const r = cell[`${sec}|${s}`];
      if (!r) return '<td class="cell"></td>';
      return `<td class="cell s-${r.state}" ${r.log ? `data-log="${esc(r.log)}" title="로그 보기"` : ''}>${STATE_TXT[r.state] || r.state}</td>`;
    }).join('')}<td>${sec === '*' ? '' : `<button data-run="${esc(sec)}" ${st.running ? 'disabled' : ''}>구간 실행</button>`}</td></tr>`).join('');
  $('#run-grid').querySelectorAll('[data-run]').forEach(b => b.onclick = () => startRun(b.dataset.run));
  $('#run-grid').querySelectorAll('[data-log]').forEach(td => td.onclick = () => { curLog = td.dataset.log; $('#run-log').value = curLog; refreshLog(); });
  $('#run-all').disabled = st.running;
  $('#run-state').textContent = rs.running ? `실행 중 (${rs.section || '전체'}${rs.until ? ` · ${rs.until}까지` : ''}, ${rs.started} 시작)`
    : rs.finished ? `마지막 실행: ${rs.ok ? '성공' : '실패'} · ${rs.finished}${rs.error ? ' · ' + rs.error : ''}` : '';
  // 로그 목록 (가장 최근 것 기본)
  const names = logs.map(l => l.name);
  if (!curLog || (!names.includes(curLog) && names.length)) curLog = names[0] || '';
  $('#run-log').innerHTML = names.length ? names.map(n => `<option ${n === curLog ? 'selected' : ''}>${esc(n)}</option>`).join('') : '<option value="">로그 없음</option>';
  if ($('#run-follow').checked) await refreshLog(rs);
}
async function refreshLog(rs) {
  const pre = $('#run-log-text');
  let lines = [];
  if (curLog) { try { lines = (await api(`/api/logs/${encodeURIComponent(curLog)}?n=300`)).lines; } catch (e) { lines = [e.message]; } }
  else if (rs?.lines?.length) lines = rs.lines;
  const atEnd = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 4;
  pre.textContent = lines.join('\n') || '로그 없음';
  if (atEnd) pre.scrollTop = pre.scrollHeight;
}

// ---------- 결과 ----------
$('#report-rebuild').onclick = () => { $('#report-frame').src = '/api/report?rebuild=1&_=' + Date.now(); };

main();
