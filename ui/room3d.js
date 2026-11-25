// Sala en 3D: piso, paredes, pantalla, parlantes, tu lugar y muebles. Todo se arrastra sobre el piso.
// Coordenadas de la app: x a la derecha, y hacia la pantalla (metros). En three.js: X = x − W/2, Z = D/2 − y.
import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";
import { RoundedBoxGeometry } from "three/addons/RoundedBoxGeometry.js";

const FLOOR = { madera: 0xd9c7ae, alfombra: 0xc9c6c0, baldosa: 0xe9e8e4 };
const INK = 0x1d1d1f;

function roomPolygon(r) {
  const W = r.width, D = r.depth;
  if (r.shape !== "L" || !r.cut) return [[0, 0], [W, 0], [W, D], [0, D]];
  const w = Math.min(W * 0.8, r.cut.w), d = Math.min(D * 0.8, r.cut.d), c = r.cut.corner || "bl";
  if (c === "bl") return [[w, 0], [W, 0], [W, D], [0, D], [0, d], [w, d]];
  if (c === "br") return [[0, 0], [W - w, 0], [W - w, d], [W, d], [W, D], [0, D]];
  if (c === "fl") return [[0, 0], [W, 0], [W, D], [w, D], [w, D - d], [0, D - d]];
  return [[0, 0], [W, 0], [W, D - d], [W - w, D - d], [W - w, D], [0, D]];
}

export function createRoom3D(container, hooks) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);
  const labels = document.createElement("div");
  labels.className = "r3-labels";
  container.appendChild(labels);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.47;
  controls.minDistance = 3;
  controls.maxDistance = 22;
  controls.enablePan = false;

  scene.add(new THREE.HemisphereLight(0xffffff, 0xbfbfc4, 1.6));
  const sun = new THREE.DirectionalLight(0xffffff, 1.6);
  sun.position.set(4, 9, 6);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  sun.shadow.camera.left = sun.shadow.camera.bottom = -10;
  sun.shadow.camera.right = sun.shadow.camera.top = 10;
  scene.add(sun);

  let room = new THREE.Group();
  scene.add(room);
  let W = 4, D = 5, H = 2.5;
  let picks = [];            // objetos arrastrables: {mesh, kind, idx}
  let spk = [], rings = [], youMesh = null, furn = [];
  let sig = "";
  let view = "3d";

  const toX = x => x - W / 2, toZ = y => D / 2 - y;
  const fromX = X => X + W / 2, fromY = Z => D / 2 - Z;

  function mat(color, rough = 0.8) { return new THREE.MeshStandardMaterial({ color, roughness: rough, metalness: 0 }); }

  function speakerMesh() {
    const g = new THREE.Group();
    const body = new THREE.Mesh(new RoundedBoxGeometry(0.2, 0.3, 0.2, 4, 0.04), mat(INK, 0.5));
    body.castShadow = true;
    g.add(body);
    const cone = new THREE.Mesh(new THREE.CircleGeometry(0.06, 32), mat(0x3a3a3c, 0.4));
    cone.position.set(0, -0.03, 0.101);
    g.add(cone);
    return g;
  }
  function sofa() {
    const g = new THREE.Group(), m = mat(0xa7a29a, 0.95);
    const seat = new THREE.Mesh(new RoundedBoxGeometry(1.6, 0.42, 0.8, 3, 0.08), m); seat.position.y = 0.21;
    const back = new THREE.Mesh(new RoundedBoxGeometry(1.6, 0.5, 0.2, 3, 0.08), m); back.position.set(0, 0.55, 0.32);
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.11, 24, 16), mat(INK, 0.4)); head.position.set(0, 0.92, 0.12);
    [seat, back].forEach(o => { o.castShadow = true; o.receiveShadow = true; g.add(o); });
    g.add(head);
    return g;
  }
  const FURN = {
    mesa: () => { const g = new THREE.Group(); const t = new THREE.Mesh(new RoundedBoxGeometry(1, 0.06, 0.6, 2, 0.02), mat(0xb08f6a, 0.6)); t.position.y = 0.42; t.castShadow = true; g.add(t);
      for (const [a, b] of [[-0.45, -0.25], [0.45, -0.25], [-0.45, 0.25], [0.45, 0.25]]) { const l = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.4), mat(0x8a6d4f)); l.position.set(a, 0.2, b); g.add(l); } return g; },
    planta: () => { const g = new THREE.Group(); const p = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.12, 0.3, 24), mat(0xe8e2d8)); p.position.y = 0.15;
      const f = new THREE.Mesh(new THREE.SphereGeometry(0.32, 16, 12), mat(0x6f8f6a, 0.9)); f.position.y = 0.6; f.castShadow = true; g.add(p, f); return g; },
    alfombra: () => { const r = new THREE.Mesh(new RoundedBoxGeometry(2, 0.015, 1.4, 2, 0.006), mat(0x9b9890, 1)); r.position.y = 0.008; r.receiveShadow = true; const g = new THREE.Group(); g.add(r); return g; },
    estante: () => { const g = new THREE.Group(); const b = new THREE.Mesh(new RoundedBoxGeometry(0.9, 1.6, 0.32, 2, 0.02), mat(0xece9e4, 0.7)); b.position.y = 0.8; b.castShadow = true; g.add(b); return g; },
    sillon: () => { const g = new THREE.Group(), m = mat(0x8e8a84, 0.95); const s = new THREE.Mesh(new RoundedBoxGeometry(0.8, 0.42, 0.8, 3, 0.08), m); s.position.y = 0.21;
      const b = new THREE.Mesh(new RoundedBoxGeometry(0.8, 0.45, 0.18, 3, 0.07), m); b.position.set(0, 0.55, 0.32); s.castShadow = b.castShadow = true; g.add(s, b); return g; },
  };

  function build(S) {
    scene.remove(room);
    room = new THREE.Group();
    scene.add(room);
    picks = []; spk = []; rings = []; furn = [];
    labels.innerHTML = "";
    W = S.room.width; D = S.room.depth; H = S.room.height || 2.5;
    const dec = S.decor || {};
    const poly = roomPolygon(S.room);
    const shp = new THREE.Shape(poly.map(([x, y]) => new THREE.Vector2(toX(x), -toZ(y))));
    const fg = new THREE.ExtrudeGeometry(shp, { depth: 0.06, bevelEnabled: false });
    fg.rotateX(-Math.PI / 2);
    const floor = new THREE.Mesh(fg, mat(FLOOR[dec.floor] || FLOOR.madera, dec.floor === "baldosa" ? 0.35 : 0.9));
    floor.position.y = -0.06; floor.receiveShadow = true; floor.name = "floor";
    room.add(floor);
    const wallM = new THREE.MeshStandardMaterial({ color: new THREE.Color(dec.walls || "#f2f2f0"), roughness: 0.95 });
    const open = S.room.shape === "open" ? (S.room.open_side || "back") : null;
    // paredes de «casa de muñecas»: las de la pantalla (fondo) y la izquierda, más las interiores de la L
    for (let k = 0; k < poly.length; k++) {
      const [x1, y1] = poly[k], [x2, y2] = poly[(k + 1) % poly.length];
      const len = Math.hypot(x2 - x1, y2 - y1);
      const isFront = y1 === S.room.depth && y2 === S.room.depth;
      const isLeft = x1 === 0 && x2 === 0;
      const inner = S.room.shape === "L" && !(isFront || isLeft) && !(x1 === S.room.width && x2 === S.room.width) && !(y1 === 0 && y2 === 0);
      if (!(isFront || isLeft || inner)) continue;
      if ((open === "left" && isLeft)) continue;
      const wall = new THREE.Mesh(new THREE.BoxGeometry(len, H, 0.08), wallM);
      wall.position.set((toX(x1) + toX(x2)) / 2, H / 2, (toZ(y1) + toZ(y2)) / 2);
      wall.rotation.y = -Math.atan2(toZ(y2) - toZ(y1), toX(x2) - toX(x1));
      wall.receiveShadow = true; room.add(wall);
    }
    const tv = new THREE.Mesh(new RoundedBoxGeometry(Math.min(1.6, W * 0.45), 0.92, 0.05, 2, 0.015), mat(0x0d0d0f, 0.25));
    tv.position.set(0, 1.25, -D / 2 + 0.03); room.add(tv);

    (dec.items || []).forEach((it, i) => {
      const g = (FURN[it.type] || FURN.mesa)();
      g.position.set(toX(it.x), 0, toZ(it.y)); g.rotation.y = it.rot || 0;
      room.add(g); furn.push(g); picks.push({ mesh: g, kind: "furn", idx: i });
    });

    youMesh = sofa();
    youMesh.position.set(toX(S.listener.x), 0, toZ(S.listener.y));
    room.add(youMesh); picks.push({ mesh: youMesh, kind: "you" });
    addLabel("Tú", youMesh, 1.35, "you");

    S.speakers.forEach((s, i) => {
      const g = speakerMesh();
      g.position.set(toX(s.x), (s.z ?? 1.0), toZ(s.y));
      g.lookAt(toX(S.listener.x), g.position.y, toZ(S.listener.y));
      room.add(g); spk.push(g); picks.push({ mesh: g, kind: "spk", idx: i });
      const ring = new THREE.Mesh(new THREE.RingGeometry(0.2, 0.22, 48), new THREE.MeshBasicMaterial({ color: INK, transparent: true, opacity: 0, side: THREE.DoubleSide }));
      ring.rotation.x = -Math.PI / 2; ring.position.copy(g.position); room.add(ring); rings.push({ m: ring, t: Math.random() });
      const stand = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, Math.max(0.01, (s.z ?? 1) - 0.15)), mat(0x8e8e93));
      stand.position.set(g.position.x, ((s.z ?? 1) - 0.15) / 2, g.position.z); room.add(stand); g.userData.stand = stand;
      addLabel(s.name, g, 0.32, "spk", i);
    });
    resize();
    setView(view, true);
  }

  const labelEls = [];
  function addLabel(text, obj, dy, kind, idx) {
    const el = document.createElement("button");
    el.className = "r3-label" + (kind === "you" ? " you" : "");
    el.textContent = text.length > 18 ? text.slice(0, 17) + "…" : text;
    if (kind === "spk") el.addEventListener("click", () => hooks.onSelect && hooks.onSelect(idx));
    labels.appendChild(el);
    labelEls.push({ el, obj, dy });
  }

  function setView(v, instant) {
    view = v;
    const target = new THREE.Vector3(0, 0.6, 0);
    const size = Math.max(W, D);
    let pos;
    if (v === "top") { pos = new THREE.Vector3(0, size * 1.9, 0.01); controls.enableRotate = false; }
    else { pos = new THREE.Vector3(size * 0.72, size * 0.78, size * 0.98); controls.enableRotate = true; }
    controls.target.copy(target);
    if (instant) { camera.position.copy(pos); camera.lookAt(target); return; }
    const from = camera.position.clone(); let t0 = performance.now();
    (function step(t) { const k = Math.min(1, (t - t0) / 500); const e = 1 - Math.pow(1 - k, 3);
      camera.position.lerpVectors(from, pos, e); camera.lookAt(target); if (k < 1) requestAnimationFrame(step); })(t0);
  }

  // --- arrastrar sobre el piso ---
  const ray = new THREE.Raycaster(), ndc = new THREE.Vector2(), plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  let drag = null, moved = false, downAt = null;
  function setNdc(e) { const r = renderer.domElement.getBoundingClientRect(); ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1); }
  function hit(e) {
    setNdc(e); ray.setFromCamera(ndc, camera);
    const objs = picks.map(p => p.mesh);
    const h = ray.intersectObjects(objs, true)[0];
    if (!h) return null;
    let o = h.object; while (o && !objs.includes(o)) o = o.parent;
    return picks.find(p => p.mesh === o) || null;
  }
  renderer.domElement.addEventListener("pointerdown", e => {
    const p = hit(e); downAt = [e.clientX, e.clientY]; moved = false;
    if (p) { drag = p; controls.enabled = false; renderer.domElement.setPointerCapture(e.pointerId); }
  });
  renderer.domElement.addEventListener("pointermove", e => {
    if (!drag) { renderer.domElement.style.cursor = hit(e) ? "grab" : "default"; return; }
    if (downAt && Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4) moved = true;
    if (!moved) return;
    setNdc(e); ray.setFromCamera(ndc, camera);
    const pt = new THREE.Vector3(); if (!ray.ray.intersectPlane(plane, pt)) return;
    const x = Math.max(0.1, Math.min(W - 0.1, fromX(pt.x))), y = Math.max(0.1, Math.min(D - 0.1, fromY(pt.z)));
    drag.mesh.position.x = toX(x); drag.mesh.position.z = toZ(y);
    if (drag.kind === "spk") { const r = rings[drag.idx]; r.m.position.x = drag.mesh.position.x; r.m.position.z = drag.mesh.position.z;
      const st = drag.mesh.userData.stand; if (st) { st.position.x = drag.mesh.position.x; st.position.z = drag.mesh.position.z; } }
    hooks.onMove && hooks.onMove(drag.kind, drag.idx, Math.round(x * 20) / 20, Math.round(y * 20) / 20, false);
  });
  renderer.domElement.addEventListener("pointerup", () => {
    if (drag) {
      if (moved) hooks.onMove && hooks.onMove(drag.kind, drag.idx, Math.round(fromX(drag.mesh.position.x) * 20) / 20, Math.round(fromY(drag.mesh.position.z) * 20) / 20, true);
      else if (drag.kind === "spk") hooks.onSelect && hooks.onSelect(drag.idx);
      else if (drag.kind === "furn") hooks.onFurniture && hooks.onFurniture(drag.idx);
    }
    drag = null; controls.enabled = true;
  });

  function resize() {
    const w = container.clientWidth, h = container.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(container);

  let peaks = [];
  const v = new THREE.Vector3();
  function tick(t) {
    controls.update();
    rings.forEach((r, i) => {
      const pk = Math.min(1, (peaks[i] || 0) * 2.2);
      r.t = (r.t + 0.012 + pk * 0.02) % 1;
      const s = 1 + r.t * 5;
      r.m.scale.set(s, s, s);
      r.m.material.opacity = pk > 0.02 ? (1 - r.t) * 0.45 * Math.min(1, pk * 2) : 0;
    });
    renderer.render(scene, camera);
    const w = container.clientWidth, h = container.clientHeight;
    labelEls.forEach(({ el, obj, dy }) => {
      if (!el.isConnected) return;
      v.copy(obj.position); v.y += dy; v.project(camera);
      el.style.transform = `translate(-50%,-100%) translate(${(v.x * 0.5 + 0.5) * w}px,${(-v.y * 0.5 + 0.5) * h}px)`;
      el.style.opacity = v.z < 1 ? 1 : 0;
    });
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);

  let ghosts = new THREE.Group(); scene.add(ghosts);
  function setGhosts(list, S) {
    scene.remove(ghosts); ghosts = new THREE.Group(); scene.add(ghosts);
    if (!list) return;
    const gm = new THREE.MeshStandardMaterial({ color: 0x1d1d1f, transparent: true, opacity: 0.22, roughness: 0.5 });
    const lm = new THREE.LineDashedMaterial({ color: 0x8e8e93, dashSize: 0.12, gapSize: 0.08 });
    list.speakers.forEach(g => {
      const m = new THREE.Mesh(new RoundedBoxGeometry(0.2, 0.3, 0.2, 4, 0.04), gm);
      m.position.set(toX(g.x), g.z ?? 1, toZ(g.y)); ghosts.add(m);
      const cur = S.speakers[g.idx];
      if (cur) {
        const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(toX(cur.x), 0.02, toZ(cur.y)), new THREE.Vector3(toX(g.x), 0.02, toZ(g.y))]);
        const ln = new THREE.Line(geo, lm); ln.computeLineDistances(); ghosts.add(ln);
      }
    });
    if (list.listener) {
      const r = new THREE.Mesh(new THREE.RingGeometry(0.28, 0.36, 40), new THREE.MeshBasicMaterial({ color: 0x1d1d1f, transparent: true, opacity: 0.35, side: THREE.DoubleSide }));
      r.rotation.x = -Math.PI / 2; r.position.set(toX(list.listener.x), 0.02, toZ(list.listener.y)); ghosts.add(r);
    }
  }

  return {
    setGhosts,
    update(S) {
      const s2 = JSON.stringify([S.room, S.decor, S.speakers.map(s => [s.name, s.z ?? 1]), S.listener]);
      W = S.room.width; D = S.room.depth;
      if (s2 !== sig) {
        // solo reconstruir cuando cambia algo estructural (no al arrastrar)
        const structural = JSON.stringify([S.room, S.decor, S.speakers.map(s => [s.name, s.z ?? 1])]) !==
                           (sig ? JSON.stringify(JSON.parse(sig).slice(0, 3)) : "");
        sig = s2;
        if (structural || !spk.length) { labelEls.length = 0; build(S); }
        else {
          S.speakers.forEach((s, i) => { if (spk[i] && drag?.mesh !== spk[i]) { spk[i].position.x = toX(s.x); spk[i].position.z = toZ(s.y);
            rings[i].m.position.x = spk[i].position.x; rings[i].m.position.z = spk[i].position.z;
            const st = spk[i].userData.stand; if (st) { st.position.x = spk[i].position.x; st.position.z = spk[i].position.z; } } });
          if (youMesh && drag?.mesh !== youMesh) youMesh.position.set(toX(S.listener.x), 0, toZ(S.listener.y));
        }
      }
      peaks = S.speakers.map(s => (s.live ? s.live.peak : 0));
    },
    setView,
    rebuild(S) { sig = ""; this.update(S); },
  };
}
