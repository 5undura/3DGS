// plan.json <-> 편집용 모델. 형식은 CONTRACT.md 4절
export const SNAP = 0.1;
export const snap = (v, on = true) => on ? Math.round(v / SNAP) * SNAP : v;
export const r3 = v => Math.round(v * 1000) / 1000;
const RAD = Math.PI / 180;

// 박스 -> 바닥 다각형 (schemas.box_to_floor 와 같은 식, 반시계)
export function boxFloor(r) {
  const c = Math.cos(r.yaw * RAD), s = Math.sin(r.yaw * RAD);
  return [[-r.w / 2, -r.d / 2], [r.w / 2, -r.d / 2], [r.w / 2, r.d / 2], [-r.w / 2, r.d / 2]]
    .map(([lx, ly]) => [r.cx + c * lx - s * ly, r.cy + s * lx + c * ly]);
}
export const roomPoly = r => r.kind === 'box' ? boxFloor(r) : r.floor;

function area2(p) {
  let a = 0;
  for (let i = 0; i < p.length; i++) { const [x1, y1] = p[i], [x2, y2] = p[(i + 1) % p.length]; a += x1 * y2 - x2 * y1; }
  return a;
}

// 직사각형 floor 는 박스로 바꿔서 편집할 수 있게
function rectToBox(floor) {
  if (floor.length !== 4) return null;
  let p = area2(floor) < 0 ? [...floor].reverse() : floor;
  const e0 = [p[1][0] - p[0][0], p[1][1] - p[0][1]], e1 = [p[2][0] - p[1][0], p[2][1] - p[1][1]];
  const w = Math.hypot(...e0), d = Math.hypot(...e1);
  if (w < 1e-6 || d < 1e-6) return null;
  if (Math.abs(e0[0] * e1[0] + e0[1] * e1[1]) > 1e-3 * w * d) return null;
  const p3 = [p[0][0] + e1[0], p[0][1] + e1[1]];
  if (Math.hypot(p3[0] - p[3][0], p3[1] - p[3][1]) > 1e-3) return null;
  let yaw = Math.atan2(e0[1], e0[0]) / RAD, W = w, D = d;
  // 각도를 -45..45 로 (90도 돌리면 가로·세로가 바뀜)
  while (yaw >= 45) { yaw -= 90; [W, D] = [D, W]; }
  while (yaw < -45) { yaw += 90; [W, D] = [D, W]; }
  const cx = p.reduce((a, q) => a + q[0], 0) / 4, cy = p.reduce((a, q) => a + q[1], 0) / 4;
  return { cx: r3(cx), cy: r3(cy), w: r3(W), d: r3(D), yaw: r3(yaw) };
}

export function fromJson(j) {
  const rooms = (j.rooms || []).map(r => {
    const base = { id: r.id, name: r.name || '', floor_z: r.floor_z ?? 0, ceiling_z: r.ceiling_z ?? null };
    if (r.box) return { ...base, kind: 'box', cx: r.box.center[0], cy: r.box.center[1], w: r.box.size[0], d: r.box.size[1], yaw: r.box.yaw_deg || 0 };
    const b = rectToBox(r.floor);
    return b ? { ...base, kind: 'box', ...b } : { ...base, kind: 'poly', floor: r.floor.map(p => [...p]) };
  });
  const doors = (j.doors || []).map(d => ({ id: d.id, rooms: [...(d.rooms || [])], center: [...d.center], width: d.width, height: d.height ?? 2.1 }));
  return { rooms, doors, source: j.source || { kind: 'boxes' } };
}

export function toJson(m) {
  return {
    version: 1, units: 'm',
    rooms: m.rooms.map(r => {
      const o = { id: r.id };
      if (r.name) o.name = r.name;
      if (r.kind === 'box') o.box = { center: [r3(r.cx), r3(r.cy)], size: [r3(r.w), r3(r.d)], yaw_deg: r3(r.yaw) };
      else o.floor = r.floor.map(p => [r3(p[0]), r3(p[1])]);
      o.floor_z = r3(r.floor_z);
      if (r.ceiling_z !== null && r.ceiling_z !== '' && !Number.isNaN(r.ceiling_z)) o.ceiling_z = r3(r.ceiling_z);
      return o;
    }),
    doors: m.doors.map(d => ({ id: d.id, rooms: d.rooms, center: [r3(d.center[0]), r3(d.center[1])], width: r3(d.width), height: r3(d.height) })),
    source: m.source,
  };
}

export function pointInPoly([x, y], pts) {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i], [xj, yj] = pts[j];
    if ((yi > y) !== (yj > y) && x < xi + (y - yi) * (xj - xi) / (yj - yi)) inside = !inside;
  }
  return inside;
}

// 점 -> 선분 가장 가까운 점
export function projSeg(p, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1], L2 = dx * dx + dy * dy;
  const u = L2 ? Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2)) : 0;
  const q = [a[0] + u * dx, a[1] + u * dy];
  return { q, u, dist: Math.hypot(p[0] - q[0], p[1] - q[1]) };
}

// 방 다각형 벽 중 가장 가까운 것
export function nearestWall(p, poly) {
  let best = null;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length], r = projSeg(p, a, b);
    if (!best || r.dist < best.dist) best = { ...r, a, b, i };
  }
  return best;
}

// 꺾은선: 누적 거리, 위치, 투영 (pathing.Polyline 과 같은 식)
export class Polyline {
  constructor(pts) {
    this.pts = pts; this.cum = [0];
    for (let i = 1; i < pts.length; i++) this.cum.push(this.cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
  }
  get length() { return this.cum[this.cum.length - 1]; }
  at(s) {
    s = Math.max(0, Math.min(this.length, s));
    let i = 1; while (i < this.pts.length - 1 && this.cum[i] < s) i++;
    const seg = this.cum[i] - this.cum[i - 1], u = seg ? (s - this.cum[i - 1]) / seg : 0;
    const a = this.pts[i - 1], b = this.pts[i];
    return [a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1])];
  }
  project(p) {
    let best = { s: 0, dist: Infinity, q: this.pts[0] };
    for (let i = 0; i < this.pts.length - 1; i++) {
      const r = projSeg(p, this.pts[i], this.pts[i + 1]);
      if (r.dist < best.dist) best = { s: this.cum[i] + r.u * (this.cum[i + 1] - this.cum[i]), dist: r.dist, q: r.q };
    }
    return best;
  }
}
