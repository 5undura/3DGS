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
