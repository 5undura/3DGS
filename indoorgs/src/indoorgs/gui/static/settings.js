// 프로젝트 화면: 학습 후 처리 설정 (project.yaml walls·cleanup) 보기·저장
import { api, toast, $, esc } from './util.js';

const WALL_FLAGS = [
  ['enabled', '면 다시 맞추기', '정합 점으로 도면 면 위치를 다시 맞춤 (walls/planes.json)'],
  ['seed', '초기 점 심기', '맞춘 면 위에 초기 점 -> data_seed 로 학습'],
  ['snap', '벽에 붙이기', '학습 후 벽 근처 스플랫을 면에 붙이고 납작하게 (splat_walls.ply)'],
  ['repair', '면 메우기', '비치거나 빈 칸을 둘레 스플랫 복사로 메움 (splat_repaired.ply)'],
  ['texture', '면 텍스처 색', '메울 색을 사진 투영·PatchMatch 텍스처에서 (walls/tex)'],
];
const STEP_TXT = { floaters: '부유물', huge: '아주 큰 것', needles: '바늘 쪼개기', sh: 'SH 튐 줄이기' };

export function initSettings() {
  const box = $('#post-settings');
  let cur = null;

  async function load() {
    try { cur = await api('/api/settings/post'); }
    catch (e) { box.innerHTML = `<p class="warn small">설정 못 읽음: ${esc(e.message)}</p>`; return; }
    render();
  }

  function render() {
    const w = cur.walls, c = cur.cleanup;
    const order = [...c.steps, ...cur.cleanup_steps.filter(s => !c.steps.includes(s))];
    box.innerHTML = `
      <div class="card"><h4>벽 평탄화 <span class="muted small">도면 좌표 구간만</span></h4>
        ${WALL_FLAGS.map(([k, t, tip]) => `<label class="chk" title="${esc(tip)}"><input type="checkbox" data-w="${k}" ${w[k] ? 'checked' : ''}> ${t}</label>`).join('')}
        <div class="form" style="margin-top:8px">
          <label title="면에서 이 거리 안의 스플랫만 붙임 (액자 등 보호)">붙일 거리 (m)</label>
          <input type="number" id="ps-band" min="0" max="0.5" step="0.005" value="${w.band}">
          <label title="그 자리 벽 색과 채널 최대 차이가 이보다 크면 안 붙임 (0..1)">색 차이 허용</label>
          <input type="number" id="ps-color" min="0" max="1" step="0.01" value="${w.color_tol}">
        </div></div>
      <div class="card"><h4>학습 후 정리 <span class="muted small">모든 구간</span></h4>
        <label class="chk"><input type="checkbox" id="ps-clean" ${c.enabled ? 'checked' : ''}> 정리 켬 (splat_clean.ply)</label>
        <div style="margin-top:6px">${order.map(s => `<label class="chk"><input type="checkbox" data-step="${s}" ${c.steps.includes(s) ? 'checked' : ''}> ${STEP_TXT[s] || s}</label>`).join('')}</div>
        <p class="small muted">바늘 쪼개기는 스플랫 수를 늘림 (기록만).</p></div>`;
    $('#ps-dirty').textContent = '';
    box.querySelectorAll('input').forEach(i => i.oninput = () => { $('#ps-dirty').textContent = '저장 안 됨'; });
  }

  async function save() {
    const walls = {};
    box.querySelectorAll('[data-w]').forEach(i => walls[i.dataset.w] = i.checked);
    walls.band = parseFloat($('#ps-band').value);
    walls.color_tol = parseFloat($('#ps-color').value);
    if (!isFinite(walls.band) || !isFinite(walls.color_tol)) { toast('숫자를 확인하세요', true); return; }
    const steps = [...box.querySelectorAll('[data-step]')].filter(i => i.checked).map(i => i.dataset.step);
    try {
      cur = await api('/api/settings/post', { method: 'PUT', body: { walls, cleanup: { enabled: $('#ps-clean').checked, steps } } });
      render(); toast('설정 저장함 (다음 실행부터)');
    } catch (e) { toast(e.message, true); }
  }

  $('#ps-reset').onclick = () => cur && render();
  $('#ps-save').onclick = save;
  load();
}
