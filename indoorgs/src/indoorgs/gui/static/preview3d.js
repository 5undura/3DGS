// 3D 미리보기: 방 바닥 다각형을 천장 높이까지 올린 벽, 문은 구멍. three.js 는 import map(CDN)
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { roomPoly } from './model.js';
import { T } from './util.js';

export async function createPreview(wrap) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setClearColor(new THREE.Color(T.bg3d));
  wrap.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 500);
  camera.up.set(0, 0, 1);                                // 도면 좌표 그대로 (z 위)
  const controls = new OrbitControls(camera, renderer.domElement);
  scene.add(new THREE.AmbientLight(0xffffff, 0.75));
  const sun = new THREE.DirectionalLight(0xffffff, 0.8); sun.position.set(-3, -5, 10); scene.add(sun);
  let group = new THREE.Group(); scene.add(group);
  let first = true;

  const wallMat = new THREE.MeshLambertMaterial({ color: T.wall3d, transparent: true, opacity: 0.4, side: THREE.DoubleSide, depthWrite: false });
  const floorMat = new THREE.MeshLambertMaterial({ color: T.floor3d, side: THREE.DoubleSide });
  const lineMat = new THREE.LineBasicMaterial({ color: T.edge3d });
  const doorMat = new THREE.LineBasicMaterial({ color: T.door });

  function resize() {
    const w = wrap.clientWidth, h = wrap.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(wrap);

  function update(model) {
    scene.remove(group); group = new THREE.Group(); scene.add(group);
    const box = new THREE.Box3();
    for (const r of model.rooms) {
      const pts = roomPoly(r), z0 = r.floor_z || 0, z1 = r.ceiling_z ?? z0 + 2.6;
      // 바닥
      const shape = new THREE.Shape(pts.map(p => new THREE.Vector2(p[0], p[1])));
      const fl = new THREE.Mesh(new THREE.ShapeGeometry(shape), floorMat); fl.position.z = z0; group.add(fl);
      for (let i = 0; i < pts.length; i++) {
        const a = pts[i], b = pts[(i + 1) % pts.length], L = Math.hypot(b[0] - a[0], b[1] - a[1]);
        if (L < 1e-6) continue;
        const ex = new THREE.Vector3((b[0] - a[0]) / L, (b[1] - a[1]) / L, 0), up = new THREE.Vector3(0, 0, 1);
        const n = new THREE.Vector3().crossVectors(ex, up);
        // 벽 면 (u: 벽 따라, v: 위), 이 벽 위의 문은 구멍
        const ws = new THREE.Shape([new THREE.Vector2(0, 0), new THREE.Vector2(L, 0), new THREE.Vector2(L, z1 - z0), new THREE.Vector2(0, z1 - z0)]);
        for (const d of model.doors) {
          if (!d.rooms.includes(r.id)) continue;
          const u = (d.center[0] - a[0]) * ex.x + (d.center[1] - a[1]) * ex.y;
          const off = Math.abs((d.center[0] - a[0]) * ex.y - (d.center[1] - a[1]) * ex.x);
          if (off > 0.05 || u < 0 || u > L) continue;
          const u0 = Math.max(0.01, u - d.width / 2), u1 = Math.min(L - 0.01, u + d.width / 2), h = Math.min(d.height, z1 - z0 - 0.02);
          const hole = new THREE.Path([new THREE.Vector2(u0, 0.01), new THREE.Vector2(u0, h), new THREE.Vector2(u1, h), new THREE.Vector2(u1, 0.01)]);
          ws.holes.push(hole);
          const dl = new THREE.Line(new THREE.BufferGeometry().setFromPoints(
            [[u0, 0], [u0, h], [u1, h], [u1, 0]].map(([uu, vv]) => new THREE.Vector3(a[0] + ex.x * uu, a[1] + ex.y * uu, z0 + vv))), doorMat);
          group.add(dl);
        }
        const g = new THREE.ShapeGeometry(ws);
        const m = new THREE.Matrix4().makeBasis(ex, up, n).setPosition(a[0], a[1], z0);
        g.applyMatrix4(m);
        group.add(new THREE.Mesh(g, wallMat));
      }
      // 모서리 선
      const edges = [];
      for (let i = 0; i < pts.length; i++) {
        const a = pts[i], b = pts[(i + 1) % pts.length];
        for (const z of [z0, z1]) edges.push(a[0], a[1], z, b[0], b[1], z);
        edges.push(a[0], a[1], z0, a[0], a[1], z1);
      }
      const eg = new THREE.BufferGeometry(); eg.setAttribute('position', new THREE.Float32BufferAttribute(edges, 3));
      group.add(new THREE.LineSegments(eg, lineMat));
      pts.forEach(p => { box.expandByPoint(new THREE.Vector3(p[0], p[1], z0)); box.expandByPoint(new THREE.Vector3(p[0], p[1], z1)); });
    }
    if (first && !box.isEmpty()) {
      const c = box.getCenter(new THREE.Vector3()), s = box.getSize(new THREE.Vector3()).length();
      controls.target.copy(c); camera.position.set(c.x - s * 0.6, c.y - s * 0.8, c.z + s * 0.7);
      controls.update(); first = false;
    }
  }

  resize();
  renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
  return { update };
}
