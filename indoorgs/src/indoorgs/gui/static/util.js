// 공통: API 호출, 알림
export const $ = s => document.querySelector(s);
export const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

export async function api(url, opt = {}) {
  const init = { method: opt.method || 'GET', headers: {} };
  if (opt.body !== undefined) { init.body = JSON.stringify(opt.body); init.headers['Content-Type'] = 'application/json'; }
  const res = await fetch(url, init);
  if (!res.ok) {
    let msg = `${res.status}`;
    try { const j = await res.json(); msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail); } catch { /* 본문 없음 */ }
    const e = new Error(msg); e.status = res.status; throw e;
  }
  return res.headers.get('content-type')?.includes('json') ? res.json() : res.text();
}

let tid = 0;
export function toast(msg, err = false) {
  const t = $('#toast');
  t.textContent = msg; t.className = `toast show ${err ? 'err' : 'ok'}`;
  clearTimeout(tid); tid = setTimeout(() => t.classList.remove('show'), err ? 6000 : 2500);
}

// 테마 색: style.css 의 --cv-* 등 CSS 변수를 한 번 읽어 캔버스·SVG·three.js 가 같이 씀
const THEME_VARS = {
  bg: 'cv-bg', grid: 'cv-grid', gridMajor: 'cv-grid-major', axisX: 'cv-axis-x', axisY: 'cv-axis-y',
  room: 'cv-room', roomDim: 'cv-room-dim', roomFill: 'cv-room-fill', roomFillDim: 'cv-room-fill-dim',
  label: 'cv-label', labelDim: 'cv-label-dim', sel: 'cv-sel', selFill: 'cv-sel-fill', handle: 'cv-handle',
  door: 'cv-door', route: 'cv-route', anchor: 'cv-anchor', anchorTxt: 'cv-anchor-txt', cursor: 'cv-cursor',
  draft: 'cv-draft', draftCorr: 'cv-draft-corr', draftLow: 'cv-draft-low', scale: 'cv-scale', bg3d: 'cv-3d-bg', wall3d: 'cv-3d-wall', floor3d: 'cv-3d-floor', edge3d: 'cv-3d-edge',
  text: 'txt', mute: 'mute', bad: 'bad', ok: 'ok', warn: 'warn', acc: 'acc',
};
const FALLBACK = { bg: '#1d1d1d', text: '#d0d0d0', sel: '#5a88cc', door: '#e8a33d', route: '#4cc9f0' };
let themeCache = null;
export const T = new Proxy({}, {
  get(_, k) {
    if (!themeCache) {
      const cs = getComputedStyle(document.documentElement);
      themeCache = Object.fromEntries(Object.entries(THEME_VARS).map(([n, v]) => [n, cs.getPropertyValue('--' + v).trim()]));
    }
    return themeCache[k] || FALLBACK[k] || '#888';
  },
});
