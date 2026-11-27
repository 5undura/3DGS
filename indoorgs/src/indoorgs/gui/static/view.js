// 위에서 본 도면 캔버스: 좌표 변환, 격자, 확대·이동, 방 그리기. 도면 좌표: x 오른쪽, y 위쪽 (미터)
import { roomPoly } from './model.js';

export const ACC = '#2563eb';

export class PlanView {
  constructor(canvas) {
    this.c = canvas; this.g = canvas.getContext('2d');
    this.scale = 30; this.ox = 200; this.oy = 300;     // 화면 px / m, 원점의 화면 위치
    this.bg = null;                                      // {img, x, y, pxPerM, opacity}
    this.draw = () => {};
    this.handler = null;                                 // {down(w, ev) -> {move, up} | null, hover(w, ev)}
    new ResizeObserver(() => this.resize()).observe(canvas);
    canvas.addEventListener('wheel', e => this.wheel(e), { passive: false });
    canvas.addEventListener('pointerdown', e => this.down(e));
    canvas.addEventListener('pointermove', e => { if (!this.drag && this.handler?.hover) this.handler.hover(this.toWorld(e), e); });
    canvas.addEventListener('contextmenu', e => e.preventDefault());
  }
  resize() {
    const r = this.c.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    if (!r.width) return;
    this.c.width = r.width * dpr; this.c.height = r.height * dpr;
    this.w = r.width; this.h = r.height; this.dpr = dpr;
    this.redraw();
  }
  sx(x) { return this.ox + x * this.scale; }
  sy(y) { return this.oy - y * this.scale; }
  toWorld(e) {
    const r = this.c.getBoundingClientRect();
    return [(e.clientX - r.left - this.ox) / this.scale, (this.oy - (e.clientY - r.top)) / this.scale];
  }
  px(m) { return m * this.scale; }
  fit(pts) {
    if (!this.w) { requestAnimationFrame(() => this.fit(pts)); return; }
    if (!pts.length) pts = [[-5, -5], [5, 5]];
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    this.scale = Math.max(2, Math.min(400, 0.85 * Math.min(this.w / Math.max(x1 - x0, 1), this.h / Math.max(y1 - y0, 1))));
    this.ox = this.w / 2 - (x0 + x1) / 2 * this.scale; this.oy = this.h / 2 + (y0 + y1) / 2 * this.scale;
    this.redraw();
  }
  wheel(e) {
    e.preventDefault();
    const r = this.c.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    const k = Math.exp(-e.deltaY * 0.0015), ns = Math.max(2, Math.min(800, this.scale * k));
    this.ox = mx - (mx - this.ox) * ns / this.scale; this.oy = my - (my - this.oy) * ns / this.scale;
    this.scale = ns; this.redraw();
  }
  down(e) {
    this.c.setPointerCapture(e.pointerId);
    let act = null;
    if (e.button === 0 && this.handler) act = this.handler.down(this.toWorld(e), e);
    if (!act) {                                          // 빈 곳 / 가운데·오른쪽 버튼: 화면 이동
      const sx = e.clientX, sy = e.clientY, ox = this.ox, oy = this.oy;
      act = { move: (w, ev) => { this.ox = ox + ev.clientX - sx; this.oy = oy + ev.clientY - sy; this.redraw(); } };
      this.c.style.cursor = 'grabbing';
    }
    this.drag = act;
    const mv = ev => act.move?.(this.toWorld(ev), ev);
    const up = ev => {
      this.c.removeEventListener('pointermove', mv); this.c.removeEventListener('pointerup', up);
      this.drag = null; this.c.style.cursor = ''; act.up?.(this.toWorld(ev), ev); this.redraw();
    };
    this.c.addEventListener('pointermove', mv);
    this.c.addEventListener('pointerup', up);
  }
  redraw() {
    if (!this.w) return;
    const g = this.g;
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    g.clearRect(0, 0, this.w, this.h);
    this.drawBg(); this.drawGrid(); this.draw(g);
  }
  drawBg() {
    const b = this.bg;
    if (!b?.img || !b.opacity) return;
    const g = this.g, wm = b.img.naturalWidth / b.pxPerM, hm = b.img.naturalHeight / b.pxPerM;
    g.globalAlpha = b.opacity;
    // 이미지 왼쪽 아래 = (x, y)
    g.drawImage(b.img, this.sx(b.x), this.sy(b.y + hm), this.px(wm), this.px(hm));
    g.globalAlpha = 1;
  }
  drawGrid() {
    const g = this.g, s = this.scale;
    const step = s > 120 ? 0.1 : s > 25 ? 1 : 5;
    const major = step === 0.1 ? 1 : step === 1 ? 5 : 25;
    const x0 = Math.floor(-this.ox / s / step) * step, x1 = (this.w - this.ox) / s;
    const y0 = Math.floor((this.oy - this.h) / s / step) * step, y1 = this.oy / s;
    g.lineWidth = 1;
    for (let pass = 0; pass < 2; pass++) {
      g.strokeStyle = pass ? '#e4e4e7' : '#f4f4f5';
      g.beginPath();
      for (let x = x0; x <= x1; x += step) {
        const isMajor = Math.abs(x / major - Math.round(x / major)) < 1e-6;
        if (isMajor !== !!pass) continue;
        const X = Math.round(this.sx(x)) + .5; g.moveTo(X, 0); g.lineTo(X, this.h);
      }
      for (let y = y0; y <= y1; y += step) {
        const isMajor = Math.abs(y / major - Math.round(y / major)) < 1e-6;
        if (isMajor !== !!pass) continue;
        const Y = Math.round(this.sy(y)) + .5; g.moveTo(0, Y); g.lineTo(this.w, Y);
      }
      g.stroke();
    }
    // 원점 축
    g.strokeStyle = '#d4d4d8'; g.beginPath();
    g.moveTo(this.sx(0) + .5, 0); g.lineTo(this.sx(0) + .5, this.h); g.moveTo(0, this.sy(0) + .5); g.lineTo(this.w, this.sy(0) + .5); g.stroke();
    // 축척
    const len = [0.5, 1, 2, 5, 10, 20].find(v => v * s >= 60) || 50;
    g.fillStyle = '#3f3f46'; g.fillRect(this.w - 20 - len * s, this.h - 22, len * s, 2);
    g.font = '11px system-ui'; g.textAlign = 'right'; g.fillText(`${len} m`, this.w - 20, this.h - 28);
  }
  poly(pts) {
    const g = this.g; g.beginPath();
    pts.forEach(([x, y], i) => i ? g.lineTo(this.sx(x), this.sy(y)) : g.moveTo(this.sx(x), this.sy(y)));
    g.closePath();
  }
  // 방들. opt: {selected, dim}
  rooms(rooms, opt = {}) {
    const g = this.g;
    for (const r of rooms) {
      const pts = roomPoly(r), sel = r === opt.selected;
      this.poly(pts);
      g.fillStyle = sel ? 'rgba(37,99,235,.08)' : opt.dim ? 'rgba(244,244,245,.6)' : 'rgba(244,244,245,.85)';
      g.fill();
      g.lineWidth = sel ? 2 : 1.5; g.strokeStyle = sel ? ACC : opt.dim ? '#a1a1aa' : '#52525b'; g.stroke();
      const cx = pts.reduce((a, p) => a + p[0], 0) / pts.length, cy = pts.reduce((a, p) => a + p[1], 0) / pts.length;
      g.fillStyle = opt.dim ? '#a1a1aa' : '#52525b'; g.font = '12px system-ui'; g.textAlign = 'center';
      g.fillText(r.name || r.id, this.sx(cx), this.sy(cy));
    }
  }
  door(d, poly, sel) {
    // 문: 벽 방향으로 폭만큼 굵은 선
    const g = this.g;
    let dir = [1, 0];
    if (poly) {
      let best = Infinity;
      for (let i = 0; i < poly.length; i++) {
        const a = poly[i], b = poly[(i + 1) % poly.length];
        const dx = b[0] - a[0], dy = b[1] - a[1], L = Math.hypot(dx, dy);
        if (!L) continue;
        const dist = Math.abs((d.center[0] - a[0]) * dy - (d.center[1] - a[1]) * dx) / L;
        if (dist < best) { best = dist; dir = [dx / L, dy / L]; }
      }
    }
    const h = d.width / 2, [cx, cy] = d.center;
    g.lineCap = 'butt'; g.lineWidth = 6; g.strokeStyle = '#fff';
    g.beginPath(); g.moveTo(this.sx(cx - dir[0] * h), this.sy(cy - dir[1] * h)); g.lineTo(this.sx(cx + dir[0] * h), this.sy(cy + dir[1] * h)); g.stroke();
    g.lineWidth = sel ? 4 : 3; g.strokeStyle = sel ? ACC : '#f59e0b'; g.stroke();
    for (const k of [-1, 1]) {
      const ex = cx + k * dir[0] * h, ey = cy + k * dir[1] * h;
      g.lineWidth = 1.5; g.beginPath();
      g.moveTo(this.sx(ex - dir[1] * .15), this.sy(ey + dir[0] * .15)); g.lineTo(this.sx(ex + dir[1] * .15), this.sy(ey - dir[0] * .15)); g.stroke();
    }
  }
}
