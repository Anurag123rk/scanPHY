import * as THREE from "three";
import { OrbitControls } from "./vendor/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "./vendor/jsm/loaders/GLTFLoader.js";

// ------------------------------------------------------------------ helpers
const $ = (s) => document.querySelector(s);
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
const fmt = (v, d = 2) => (v == null || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const api = async (path, opts = {}) => {
  const o = { ...opts };
  if (o.json !== undefined) { o.method = o.method || "POST"; o.headers = { "Content-Type": "application/json" }; o.body = JSON.stringify(o.json); delete o.json; }
  const r = await fetch(path, o);
  const data = await r.json().catch(() => ({}));
  if (!r.ok || data.ok === false) throw new Error(data.error || r.statusText);
  return data;
};
let toastT;
function toast(msg) { const t = $("#toast"); t.textContent = msg; t.classList.add("on"); clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove("on"), 2600); }

let CFG = null;
let STATE = null;

// ------------------------------------------------------------------ theme
function setTheme(t) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem("wavesense-theme", t); } catch (e) { /* storage blocked */ }
  if (scene3d) scene3d.recolor();
}
$("#themeBtn").onclick = () => setTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light");
try { const t = localStorage.getItem("wavesense-theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* ignore */ }

// ------------------------------------------------------------------ tabs
let activeTab = "live";
document.querySelectorAll("#tabs button").forEach((b) => b.onclick = () => {
  activeTab = b.dataset.tab;
  document.querySelectorAll("#tabs button").forEach((x) => x.classList.toggle("on", x === b));
  document.querySelectorAll(".tab-page").forEach((p) => p.classList.toggle("on", p.id === "page-" + activeTab));
  if (activeTab === "live" && scene3d) scene3d.resize();
  if (activeTab === "calibrate") { loadRecordings(); drawFloor(); }
  if (activeTab === "models") loadModels();
  if (activeTab === "setup") buildCfgForm();
  if (activeTab === "train") drawTrainChart();
  if (activeTab === "people") { loadHistory(); loadTeachRecs(); }
});

// ------------------------------------------------------------------ brand wave animation
(function brandWave() {
  const p = $("#brandPath");
  let t = 0;
  function frame() {
    t += 0.06;
    const amp = STATE && STATE.presence ? 9 : 3;
    let d = "M0 12";
    for (let x = 0; x <= 64; x += 2) d += ` L${x} ${12 + Math.sin(x * 0.35 + t) * amp * Math.sin((x / 64) * Math.PI)}`;
    p.setAttribute("d", d);
    requestAnimationFrame(frame);
  }
  frame();
})();

// ------------------------------------------------------------------ 3D scene
class Room3D {
  constructor(host) {
    this.host = host;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    host.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.05, 200);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.maxPolarAngle = Math.PI * 0.49;
    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.showTrail = true;
    this.showZones = true;
    this.pulses = [];
    this.target = null;
    this.clock = new THREE.Clock();
    new ResizeObserver(() => this.resize()).observe(host);
    this.loop();
  }
  col(name) { return new THREE.Color(css(name) || "#fff"); }
  recolor() { if (CFG) this.build(CFG); }
  build(cfg) {
    this.cfg = cfg;
    this.root.clear();
    this.pulses = [];
    const W = cfg.room.width, D = cfg.room.depth, H = cfg.room.height;
    const fg = this.col("--fg"), fg2 = this.col("--fg2"), fg3 = this.col("--fg3"), line = this.col("--line2");
    const lm = (c, o = 1) => new THREE.LineBasicMaterial({ color: c, transparent: o < 1, opacity: o });

    // floor grid (0.5 m)
    const pts = [];
    for (let x = 0; x <= W + 1e-6; x += 0.5) pts.push(x, 0, 0, x, 0, D);
    for (let z = 0; z <= D + 1e-6; z += 0.5) pts.push(0, 0, z, W, 0, z);
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    this.root.add(new THREE.LineSegments(g, lm(line, 0.55)));
    // walls
    const box = new THREE.BoxGeometry(W, H, D); box.translate(W / 2, H / 2, D / 2);
    this.root.add(new THREE.LineSegments(new THREE.EdgesGeometry(box), lm(fg2, 0.8)));
    // corner ticks / axis labels
    this.root.add(this.label("0,0", 0, 0.02, 0, fg3, 0.18));
    this.root.add(this.label(`${W.toFixed(1)} m`, W / 2, 0.02, -0.25, fg3, 0.2));
    this.root.add(this.label(`${D.toFixed(1)} m`, -0.35, 0.02, D / 2, fg3, 0.2));

    // zone tiles
    this.zoneTiles = [];
    const zr = cfg.zones.rows, zc = cfg.zones.cols;
    this.zoneGroup = new THREE.Group();
    for (let r = 0; r < zr; r++) for (let c = 0; c < zc; c++) {
      const w = W / zc, d = D / zr;
      const m = new THREE.Mesh(new THREE.PlaneGeometry(w * 0.96, d * 0.96), new THREE.MeshBasicMaterial({ color: fg, transparent: true, opacity: 0, depthWrite: false }));
      m.rotation.x = -Math.PI / 2; m.position.set(c * w + w / 2, 0.003, r * d + d / 2);
      this.zoneGroup.add(m); this.zoneTiles.push(m);
    }
    this.zoneGroup.visible = this.showZones;
    this.root.add(this.zoneGroup);

    // router
    const rt = cfg.router;
    const router = new THREE.Mesh(new THREE.OctahedronGeometry(0.14), new THREE.MeshBasicMaterial({ color: fg, wireframe: true }));
    router.position.set(rt.x, rt.z, rt.y);
    this.router = router;
    this.root.add(router, this.stem(rt.x, rt.y, rt.z, fg3), this.label("ROUTER", rt.x, rt.z + 0.3, rt.y, fg2, 0.17));

    // nodes + links
    this.nodeObjs = {};
    for (const n of cfg.nodes) {
      const grp = new THREE.Group();
      const body = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.05, 0.08), new THREE.MeshBasicMaterial({ color: fg }));
      const ring = new THREE.Mesh(new THREE.RingGeometry(0.16, 0.18, 48), new THREE.MeshBasicMaterial({ color: fg, transparent: true, side: THREE.DoubleSide }));
      ring.rotation.x = -Math.PI / 2;
      grp.add(body, ring);
      grp.position.set(n.x, n.z ?? 1, n.y);
      const lg = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(rt.x, rt.z, rt.y), new THREE.Vector3(n.x, n.z ?? 1, n.y)]);
      const link = new THREE.Line(lg, new THREE.LineDashedMaterial({ color: fg, dashSize: 0.08, gapSize: 0.06, transparent: true, opacity: 0.35 }));
      link.computeLineDistances();
      this.root.add(grp, link, this.stem(n.x, n.y, n.z ?? 1, fg3), this.label(n.name || "N" + n.id, n.x, (n.z ?? 1) + 0.28, n.y, fg2, 0.16));
      this.nodeObjs[n.id] = { grp, ring, link, body };
    }

    // avatar
    const av = new THREE.Group();
    const mat = new THREE.MeshBasicMaterial({ color: fg });
    const wire = new THREE.MeshBasicMaterial({ color: fg, wireframe: true, transparent: true, opacity: 0.35 });
    const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.17, 0.75, 6, 16), wire); torso.position.y = 0.95;
    const core = new THREE.Mesh(new THREE.CapsuleGeometry(0.1, 0.6, 4, 12), mat); core.position.y = 0.95;
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.12, 20, 14), mat); head.position.y = 1.62;
    av.add(torso, core, head);
    this.avatar = av; this.avatarHead = head; this.avatarCore = core;
    av.visible = false;
    this.root.add(av);
    this.unc = new THREE.Mesh(new THREE.RingGeometry(0.97, 1.0, 96), new THREE.MeshBasicMaterial({ color: fg, transparent: true, opacity: 0.8, side: THREE.DoubleSide }));
    this.unc.rotation.x = -Math.PI / 2; this.unc.position.y = 0.01; this.unc.visible = false;
    this.uncFill = new THREE.Mesh(new THREE.CircleGeometry(1, 64), new THREE.MeshBasicMaterial({ color: fg, transparent: true, opacity: 0.07, depthWrite: false }));
    this.uncFill.rotation.x = -Math.PI / 2; this.uncFill.position.y = 0.008; this.uncFill.visible = false;
    this.root.add(this.unc, this.uncFill);
    // trail
    this.trailGeom = new THREE.BufferGeometry();
    this.trailGeom.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(3 * 64), 3));
    this.trail = new THREE.Line(this.trailGeom, lm(fg2, 0.8));
    this.root.add(this.trail);
    // simulator ground truth ghost
    this.ghost = new THREE.Mesh(new THREE.RingGeometry(0.2, 0.23, 4), new THREE.MeshBasicMaterial({ color: fg3, side: THREE.DoubleSide }));
    this.ghost.rotation.x = -Math.PI / 2; this.ghost.position.y = 0.015; this.ghost.visible = false;
    this.root.add(this.ghost);

    if (cfg.room_glb) {
      new GLTFLoader().load(cfg.room_glb, (gltf) => {
        gltf.scene.traverse((o) => { if (o.isMesh) { o.material = new THREE.MeshBasicMaterial({ color: fg3, wireframe: true, transparent: true, opacity: 0.25 }); } });
        this.root.add(gltf.scene);
      }, undefined, () => toast("could not load room.glb"));
    }
    this.controls.target.set(W / 2, 0.6, D / 2);
    if (!this.viewSet) { this.iso(); this.viewSet = true; }
    this.resize();
  }
  stem(x, y, z, c) {
    const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(x, 0, y), new THREE.Vector3(x, z, y)]);
    const l = new THREE.Line(g, new THREE.LineDashedMaterial({ color: c, dashSize: 0.04, gapSize: 0.04 }));
    l.computeLineDistances();
    return l;
  }
  label(text, x, y, z, color, size) {
    const c = document.createElement("canvas"); const ctx = c.getContext("2d");
    const fs = 48; ctx.font = `600 ${fs}px JetBrains Mono, monospace`;
    c.width = Math.ceil(ctx.measureText(text).width) + 16; c.height = fs + 16;
    ctx.font = `600 ${fs}px JetBrains Mono, monospace`; ctx.fillStyle = "#" + color.getHexString(); ctx.textBaseline = "middle";
    ctx.fillText(text, 8, c.height / 2);
    const tex = new THREE.CanvasTexture(c); tex.anisotropy = 4;
    const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
    s.scale.set(size * c.width / c.height, size, 1); s.position.set(x, y, z);
    return s;
  }
  iso() {
    if (!this.cfg) return;
    const W = this.cfg.room.width, D = this.cfg.room.depth, m = Math.max(W, D);
    this.camera.position.set(W / 2 + m * 1.0, m * 1.25, D / 2 + m * 1.45);
    this.controls.target.set(W / 2, 0.5, D / 2);
  }
  top() {
    if (!this.cfg) return;
    const W = this.cfg.room.width, D = this.cfg.room.depth;
    this.camera.position.set(W / 2, Math.max(W, D) * 1.55, D / 2 + 0.001);
    this.controls.target.set(W / 2, 0, D / 2);
  }
  resize() {
    const w = this.host.clientWidth, h = this.host.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h); this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }
  update(s) {
    if (!this.cfg) return;
    this.lastState = s;
    // nodes
    for (const n of s.nodes) {
      const o = this.nodeObjs[n.id]; if (!o) continue;
      o.alive = n.alive;
      o.link.material.opacity = n.alive ? 0.25 + Math.min(0.7, s.motion * 0.25) : 0.06;
      o.body.material.opacity = n.alive ? 1 : 0.3; o.body.material.transparent = !n.alive;
    }
    // zones
    if (s.zones && this.zoneTiles.length === s.zones.probs.length) {
      s.zones.probs.forEach((p, i) => { this.zoneTiles[i].material.opacity = s.presence ? p * 0.35 : 0; });
    }
    // avatar target
    if (s.presence && s.x != null) {
      this.target = new THREE.Vector3(s.x, 0, s.y);
      if (!this.avatar.visible) this.avatar.position.copy(this.target);
      this.avatar.visible = true; this.unc.visible = true; this.uncFill.visible = true;
      this.uncR = Math.max(0.2, s.uncertainty_m);
    } else {
      this.avatar.visible = false; this.unc.visible = false; this.uncFill.visible = false; this.target = null;
    }
    // trail
    const arr = this.trailGeom.attributes.position.array;
    const tr = this.showTrail && s.presence ? s.trail.slice(-64) : [];
    for (let i = 0; i < 64; i++) {
      const p = tr[Math.min(i, tr.length - 1)];
      arr[i * 3] = p ? p[0] : 0; arr[i * 3 + 1] = 0.02; arr[i * 3 + 2] = p ? p[1] : 0;
    }
    this.trailGeom.setDrawRange(0, tr.length);
    this.trailGeom.attributes.position.needsUpdate = true;
    // truth (simulator)
    if (s.truth && s.truth.activity !== "empty") { this.ghost.visible = true; this.ghost.position.set(s.truth.x, 0.015, s.truth.y); }
    else this.ghost.visible = false;
  }
  loop() {
    requestAnimationFrame(() => this.loop());
    const dt = this.clock.getDelta(), t = this.clock.elapsedTime;
    if (this.cfg) {
      if (this.target && this.avatar.visible) {
        this.avatar.position.lerp(this.target, 1 - Math.exp(-dt * 6));
        this.unc.position.x = this.uncFill.position.x = this.avatar.position.x;
        this.unc.position.z = this.uncFill.position.z = this.avatar.position.z;
        const r = this.uncR || 0.5, cur = this.unc.scale.x || r, nr = cur + (r - cur) * Math.min(1, dt * 4);
        this.unc.scale.set(nr, nr, nr); this.uncFill.scale.set(nr, nr, nr);
        const act = this.lastState && this.lastState.activity;
        const breath = act === "still" ? 1 + 0.025 * Math.sin(t * 2 * Math.PI * ((this.lastState.vitals.breathing_bpm || 15) / 60)) : 1;
        this.avatarCore.scale.set(breath, 1, breath);
        this.avatarHead.position.y = 1.62 + (act === "moving" ? Math.abs(Math.sin(t * 6)) * 0.04 : 0);
      }
      this.router.rotation.y += dt * 0.8;
      for (const id in this.nodeObjs) {
        const o = this.nodeObjs[id];
        const ph = (t * 0.9 + Number(id) * 0.33) % 1;
        o.ring.scale.setScalar(o.alive ? 1 + ph * 2.2 : 1);
        o.ring.material.opacity = o.alive ? (1 - ph) * 0.8 : 0.15;
      }
      if (this.ghost.visible) this.ghost.rotation.z += dt;
      // expanding WiFi pulses from the router
      if (!this.nextPulse || t > this.nextPulse) {
        this.nextPulse = t + 1.4;
        const m = new THREE.Mesh(new THREE.RingGeometry(0.98, 1, 64), new THREE.MeshBasicMaterial({ color: this.col("--fg"), transparent: true, opacity: 0.5, side: THREE.DoubleSide, depthWrite: false }));
        m.rotation.x = -Math.PI / 2; m.position.set(this.cfg.router.x, 0.005, this.cfg.router.y); m.userData.t0 = t;
        this.root.add(m); this.pulses.push(m);
      }
      const maxR = Math.hypot(this.cfg.room.width, this.cfg.room.depth);
      this.pulses = this.pulses.filter((m) => {
        const age = t - m.userData.t0, r = 0.1 + age * 2.2;
        if (r > maxR) { this.root.remove(m); m.geometry.dispose(); m.material.dispose(); return false; }
        m.scale.setScalar(r); m.material.opacity = 0.35 * (1 - r / maxR);
        return true;
      });
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
const scene3d = new Room3D($("#three"));
$("#viewTop").onclick = () => scene3d.top();
$("#viewIso").onclick = () => scene3d.iso();
$("#toggleTrail").onclick = (e) => { scene3d.showTrail = !scene3d.showTrail; e.target.classList.toggle("on", scene3d.showTrail); };
$("#toggleZones").onclick = (e) => { scene3d.showZones = !scene3d.showZones; e.target.classList.toggle("on", scene3d.showZones); if (scene3d.zoneGroup) scene3d.zoneGroup.visible = scene3d.showZones; };

// ------------------------------------------------------------------ canvas sizing
// Size a canvas backing store from its CSS box. Rounded integers + a fixed CSS height are
// essential: with a fractional devicePixelRatio (Windows 125 % / 150 % scaling) comparing
// canvas.width to w * dpr never matches, and an unsized canvas then grows every frame.
function fitCanvas(c) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(c.clientWidth)), h = Math.max(1, Math.round(c.clientHeight));
  const bw = Math.round(w * dpr), bh = Math.round(h * dpr);
  if (c.width !== bw || c.height !== bh) { c.width = bw; c.height = bh; }
  const ctx = c.getContext("2d"); ctx.setTransform(bw / w, 0, 0, bh / h, 0, 0);
  return { ctx, w, h };
}

// ------------------------------------------------------------------ sparklines
function drawSpark(canvas, data, { max = null, fill = true } = {}) {
  const { ctx, w, h } = fitCanvas(canvas); ctx.clearRect(0, 0, w, h);
  if (!data.length) return;
  const m = max ?? Math.max(1e-6, ...data);
  ctx.beginPath();
  data.forEach((v, i) => { const x = (i / Math.max(1, data.length - 1)) * w, y = h - 2 - (Math.min(v, m) / m) * (h - 4); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.strokeStyle = css("--fg"); ctx.lineWidth = 1.4; ctx.stroke();
  if (fill) { ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath(); ctx.fillStyle = css("--fg") + "18"; ctx.fill(); }
}
const motionHist = [];
let breathPhase = 0;
function drawBreath(bpm) {
  const { ctx, w, h } = fitCanvas($("#breathWave")); ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = css("--fg3"); ctx.setLineDash([2, 4]); ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke(); ctx.setLineDash([]);
  if (!bpm) return;
  breathPhase += (bpm / 60) * 0.1 * 2 * Math.PI;
  ctx.beginPath();
  for (let x = 0; x <= w; x += 2) { const y = h / 2 + Math.sin(breathPhase - (x / w) * 6 * Math.PI * (bpm / 15) / 2) * (h * 0.38); x ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
  ctx.strokeStyle = css("--fg"); ctx.lineWidth = 1.6; ctx.stroke();
}

// ------------------------------------------------------------------ nodes (CSI waterfall)
const nodeCards = {};
function ensureNodeCards(cfg) {
  const row = $("#nodesRow"); row.innerHTML = "";
  for (const k in nodeCards) delete nodeCards[k];
  for (const n of cfg.nodes) {
    const c = el("div", "card node");
    c.innerHTML = `<div class="card-h"><span>${esc(n.name || "NODE-" + n.id)}</span><span class="muted">id ${n.id}</span><span class="spacer"></span><span class="state">—</span></div>
      <div class="stats"><div><small>RATE</small><b class="pps">—</b></div><div><small>RSSI</small><b class="rssi">—</b></div>
      <div><small>LOSS</small><b class="loss">—</b></div><div><small>BASE</small><b class="base">—</b></div></div>
      <canvas width="52" height="90"></canvas><div class="row muted addr" style="margin:6px 0 0;font-size:10px">CSI waterfall · 52 subcarriers · waiting for frames</div>`;
    row.appendChild(c);
    const cv = c.querySelector("canvas");
    const ctx = cv.getContext("2d");
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, 52, 90);
    nodeCards[n.id] = { card: c, cv, ctx };
  }
}
function updateNodes(s) {
  let alive = 0;
  const light = document.documentElement.dataset.theme === "light";
  for (const n of s.nodes) {
    const nc = nodeCards[n.id]; if (!nc) continue;
    if (n.alive) alive++;
    nc.card.classList.toggle("dead", !n.alive);
    nc.card.querySelector(".state").textContent = n.alive ? "● STREAMING" : "○ OFFLINE";
    nc.card.querySelector(".pps").textContent = n.alive ? n.pps.toFixed(0) + "/s" : "—";
    nc.card.querySelector(".rssi").textContent = n.rssi != null ? n.rssi.toFixed(0) : "—";
    nc.card.querySelector(".loss").textContent = n.frames ? ((100 * n.lost) / (n.frames + n.lost)).toFixed(1) + "%" : "—";
    nc.card.querySelector(".base").textContent = n.baseline ? n.baseline.toUpperCase() : "NONE";
    nc.card.querySelector(".addr").textContent = n.alive ? `CSI waterfall · ${n.addr || "?"} · ch ${n.channel || "?"} · ${n.frames} frames` : "CSI waterfall · no frames in the last 2 s";
    if (n.heat) {
      const ctx = nc.ctx;
      ctx.drawImage(nc.cv, 0, 0, 52, 89, 0, 1, 52, 89);
      const img = ctx.createImageData(52, 1);
      n.heat.forEach((v, i) => { const g = light ? 255 - v : v; img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = g; img.data[i * 4 + 3] = 255; });
      ctx.putImageData(img, 0, 0);
    }
  }
  return alive;
}

// ------------------------------------------------------------------ live panels
function updateLive(s) {
  const alive = updateNodes(s);
  $("#pillNodes").textContent = `NODES ${alive}/${s.nodes.length}`;
  $("#pillModel").textContent = "MODEL " + (s.model && s.model.name ? s.model.name : "HEURISTIC");
  const big = $("#presenceBig");
  big.textContent = s.activity === "no-signal" ? "NO SIGNAL" : s.presence ? (s.activity === "moving" ? "MOVING" : "PRESENT") : "EMPTY";
  big.classList.toggle("on", s.presence);
  const bars = $("#actBars"); bars.innerHTML = "";
  const probs = s.activity_probs || {};
  const mx = Math.max(...Object.values(probs));
  for (const k of Object.keys(probs)) {
    const b = el("div", "bar" + (probs[k] === mx ? " max" : ""));
    b.innerHTML = `<span>${k.toUpperCase()}</span><div class="track"><div class="fill" style="width:${(probs[k] * 100).toFixed(1)}%"></div></div><span>${(probs[k] * 100).toFixed(0)}%</span>`;
    bars.appendChild(b);
  }
  const ppl = s.people || { total: 0, moving: 0, still: 0 };
  $("#cTotal").textContent = ppl.total; $("#cMoving").textContent = ppl.moving; $("#cStill").textContent = ppl.still;
  $("#hudPeople").textContent = ppl.total;
  $("#motionVal").textContent = "motion " + fmt(s.motion, 2);
  motionHist.push(s.motion); if (motionHist.length > 150) motionHist.shift();
  drawSpark($("#motionSpark"), motionHist);
  $("#vBreath").textContent = s.vitals.breathing_bpm ? s.vitals.breathing_bpm.toFixed(0) : (s.vitals.settling ? "···" : "—");
  $("#vHeart").textContent = s.vitals.heart_bpm ? s.vitals.heart_bpm.toFixed(0) : "—";
  drawBreath(s.vitals.breathing_bpm);
  $("#hudX").textContent = s.x != null ? fmt(s.x) + "m" : "—";
  $("#hudY").textContent = s.y != null ? fmt(s.y) + "m" : "—";
  $("#hudZone").textContent = s.zone || "—";
  $("#hudConf").textContent = s.presence ? (s.confidence * 100).toFixed(0) + "%" : "—";
  $("#hudEngine").textContent = `ENGINE ${String(s.engine).toUpperCase()} · ${s.udp.source === "sim" ? "SIMULATOR" : "UDP " + s.udp.port}`;
  $("#hudTruth").textContent = s.truth ? `SIM TRUTH ◇ ${s.truth.people ?? 0} PEOPLE (${s.truth.moving ?? 0} MOVING · ${s.truth.still ?? 0} STILL)${s.truth.people ? ` · P1 (${fmt(s.truth.x, 1)}, ${fmt(s.truth.y, 1)})` : ""}` : "";
  // zones
  const zg = $("#zoneGrid");
  if (s.zones) {
    if (zg.childElementCount !== s.zones.probs.length) {
      zg.innerHTML = ""; zg.style.gridTemplateColumns = `repeat(${s.zones.cols}, 1fr)`;
      s.zones.names.forEach((n) => zg.appendChild(el("div", "", `<span>${n}</span><span></span>`)));
    }
    s.zones.probs.forEach((p, i) => {
      const d = zg.children[i], v = s.presence ? p : 0;
      d.style.background = `color-mix(in srgb, var(--fg) ${(v * 100).toFixed(0)}%, transparent)`;
      d.classList.toggle("hot", v > 0.45);
      d.lastChild.textContent = s.presence ? (p * 100).toFixed(0) + "%" : "";
    });
  } else zg.innerHTML = `<span class="muted">zone head needs a model</span>`;
  // events
  const ul = $("#events");
  const sig = s.events.length ? s.events[0].t + s.events.length : 0;
  if (ul.dataset.sig !== String(sig)) {
    ul.dataset.sig = String(sig);
    ul.innerHTML = s.events.map((e) => `<li class="${e.level}"><time>${new Date(e.t * 1000).toLocaleTimeString()}</time><span>${esc(e.msg)}</span></li>`).join("");
  }
  $("#udpStat").textContent = `${s.udp.frames} frames · ${s.udp.bad} rejected · uptime ${Math.floor(s.uptime)} s`;
  scene3d.update(s);
}

// ------------------------------------------------------------------ calibration
let wizard = null;
function gridPoints() {
  if (!CFG) return [];
  const step = parseFloat($("#gridStep").value), W = CFG.room.width, D = CFG.room.depth, out = [];
  const nx = Math.max(1, Math.floor((W - 0.6) / step) + 1), ny = Math.max(1, Math.floor((D - 0.6) / step) + 1);
  const ox = (W - (nx - 1) * step) / 2, oy = (D - (ny - 1) * step) / 2;
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const ii = j % 2 ? nx - 1 - i : i; // serpentine walk order
    out.push({ x: +(ox + ii * step).toFixed(2), y: +(oy + j * step).toFixed(2) });
  }
  return out;
}
let recordedSpots = [];
function drawFloor() {
  const c = $("#floorMap"); if (!CFG || !c.clientWidth) return;
  const W = CFG.room.width, D = CFG.room.depth;
  c.style.aspectRatio = `${W} / ${D}`;
  const { ctx, w, h } = fitCanvas(c);
  const pad = 18, sx = (w - 2 * pad) / W, sy = (h - 2 * pad) / D;
  const P = (x, y) => [pad + x * sx, pad + y * sy];
  const fg = css("--fg"), fg2 = css("--fg2"), fg3 = css("--fg3"), line = css("--line");
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = line; ctx.lineWidth = 1;
  for (let x = 0; x <= W + 1e-6; x += 0.5) { ctx.beginPath(); ctx.moveTo(...P(x, 0)); ctx.lineTo(...P(x, D)); ctx.stroke(); }
  for (let y = 0; y <= D + 1e-6; y += 0.5) { ctx.beginPath(); ctx.moveTo(...P(0, y)); ctx.lineTo(...P(W, y)); ctx.stroke(); }
  ctx.strokeStyle = fg; ctx.lineWidth = 2; ctx.strokeRect(pad, pad, W * sx, D * sy);
  // links
  const r = CFG.router;
  ctx.setLineDash([4, 4]); ctx.strokeStyle = fg3; ctx.lineWidth = 1;
  for (const n of CFG.nodes) { ctx.beginPath(); ctx.moveTo(...P(r.x, r.y)); ctx.lineTo(...P(n.x, n.y)); ctx.stroke(); }
  ctx.setLineDash([]);
  ctx.font = "10px JetBrains Mono, monospace";
  ctx.fillStyle = fg; const [rx, ry] = P(r.x, r.y);
  ctx.beginPath(); ctx.moveTo(rx, ry - 7); ctx.lineTo(rx + 7, ry); ctx.lineTo(rx, ry + 7); ctx.lineTo(rx - 7, ry); ctx.closePath(); ctx.fill();
  ctx.fillText("ROUTER", rx + 10, ry + 3);
  for (const n of CFG.nodes) {
    const [x, y] = P(n.x, n.y), name = n.name || "N" + n.id, tw = ctx.measureText(name).width;
    ctx.fillRect(x - 5, y - 5, 10, 10); ctx.fillText(name, x + 9 + tw > w ? x - 9 - tw : x + 9, y + 3);
  }
  // grid points
  const pts = wizard ? wizard.points : gridPoints();
  pts.forEach((p, i) => {
    const [x, y] = P(p.x, p.y);
    const done = recordedSpots.some((s) => Math.abs(s.x - p.x) < 0.05 && Math.abs(s.y - p.y) < 0.05);
    const cur = wizard && wizard.current && wizard.current.x === p.x && wizard.current.y === p.y;
    ctx.beginPath(); ctx.arc(x, y, cur ? 11 : 6, 0, Math.PI * 2);
    if (done) { ctx.fillStyle = fg2; ctx.fill(); } else { ctx.strokeStyle = fg2; ctx.lineWidth = 1; ctx.stroke(); }
    if (cur) { ctx.strokeStyle = fg; ctx.lineWidth = 2; ctx.stroke(); ctx.fillStyle = fg; ctx.fillText(String(i + 1), x + 14, y - 8); }
  });
  // live position
  if (STATE && STATE.presence && STATE.x != null) {
    const [x, y] = P(STATE.x, STATE.y);
    ctx.fillStyle = fg; ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = fg; ctx.globalAlpha = 0.4; ctx.beginPath(); ctx.arc(x, y, STATE.uncertainty_m * sx, 0, Math.PI * 2); ctx.stroke(); ctx.globalAlpha = 1;
  }
  c.onclick = async (ev) => {
    if (wizard) return;
    const rect = c.getBoundingClientRect();
    const x = +((ev.clientX - rect.left - pad) / sx).toFixed(2), y = +((ev.clientY - rect.top - pad) / sy).toFixed(2);
    if (x < 0 || y < 0 || x > W || y > D) return;
    try { await api("/api/record", { json: { activity: "still", x, y, seconds: +$("#manSecs").value } }); toast(`recording still at (${x}, ${y})`); recordedSpots.push({ x, y }); }
    catch (e) { toast(e.message); }
  };
}
$("#gridStep").onchange = drawFloor;
$("#btnViewPeople").onclick = () => document.querySelector('#tabs button[data-tab="people"]').click();
$("#btnManualFree").onclick = async () => {
  const body = { activity: $("#manAct").value, seconds: +$("#manSecs").value };
  if ($("#manMove").value !== "") body.moving_count = +$("#manMove").value;
  if ($("#manStill").value !== "") body.still_count = +$("#manStill").value;
  try { await api("/api/record", { json: body }); toast("recording " + body.activity); }
  catch (e) { toast(e.message); }
};
$("#btnBaseline").onclick = async () => {
  try { await api("/api/baseline", { json: { seconds: +$("#blSecs").value } }); toast("baseline capture started - leave the room"); }
  catch (e) { toast(e.message); }
};

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function waitRecordingDone() { while (wizard && !wizard.stop) { await sleep(300); if (STATE && !STATE.recording) return; } }
async function runWizard() {
  const secs = +$("#ptSecs").value;
  const points = gridPoints();
  const steps = points.map((p, i) => ({ kind: "still", ...p, title: `SPOT ${i + 1}/${points.length}`, text: `STAND AT (${p.x.toFixed(1)}, ${p.y.toFixed(1)})`, secs }));
  steps.push({ kind: "moving", title: "FREE WALK", text: "WALK AROUND THE WHOLE ROOM", secs: 30 });
  steps.push({ kind: "moving", title: "FREE WALK 2", text: "KEEP WALKING · VARY SPEED", secs: 30 });
  steps.push({ kind: "empty", title: "EMPTY ROOM", text: "LEAVE THE ROOM NOW", secs: 20 });
  wizard = { points, steps, stop: false, current: null };
  $("#btnWizard").disabled = true;
  try {
    for (let i = 0; i < steps.length && !wizard.stop; i++) {
      const st = steps[i];
      wizard.current = st.kind === "still" ? st : null;
      drawFloor();
      for (let c = st.kind === "empty" ? 8 : 4; c > 0 && !wizard.stop; c--) {
        $("#wizStep").textContent = `${st.title} · GET READY`; $("#wizBig").textContent = `${st.text} · ${c}`; $("#wizProg").style.width = "0%";
        await sleep(1000);
      }
      if (wizard.stop) break;
      $("#wizStep").textContent = `${st.title} · RECORDING`; $("#wizBig").textContent = st.text;
      await api("/api/record", { json: { activity: st.kind, x: st.kind === "still" ? st.x : null, y: st.kind === "still" ? st.y : null, seconds: st.secs } });
      await sleep(400);
      await waitRecordingDone();
      if (st.kind === "still") recordedSpots.push({ x: st.x, y: st.y });
      $("#wizInfo").textContent = `${i + 1}/${steps.length} steps done`;
    }
    if (!wizard.stop) {
      $("#wizStep").textContent = "COMPLETE"; $("#wizBig").textContent = "CALIBRATION RECORDED";
      if ($("#autoTrainAfter").checked) { await api("/api/train", { json: { reason: "calibration wizard" } }); toast("training started on the Jetson"); }
    } else { $("#wizStep").textContent = "STOPPED"; $("#wizBig").textContent = "—"; }
  } catch (e) { toast(e.message); $("#wizStep").textContent = "ERROR"; $("#wizBig").textContent = e.message; }
  wizard = null; $("#btnWizard").disabled = false; drawFloor(); loadRecordings();
}
$("#btnWizard").onclick = () => { if (!wizard) runWizard(); };
$("#btnWizardStop").onclick = async () => { if (wizard) wizard.stop = true; await api("/api/record/cancel", { method: "POST" }).catch(() => {}); };

function updateCalib(s) {
  const bl = s.baseline_job; $("#blProg").style.width = bl ? (100 * bl.elapsed / bl.seconds).toFixed(0) + "%" : "0%";
  if (s.recording) $("#wizProg").style.width = (100 * s.recording.elapsed / s.recording.seconds).toFixed(0) + "%";
  if (activeTab === "calibrate") drawFloor();
}
async function loadRecordings() {
  try {
    const recs = await (await fetch("/api/recordings")).json();
    $("#recCount").textContent = `${recs.length} recordings`;
    recordedSpots = recs.filter((r) => r.label.x != null).map((r) => ({ x: r.label.x, y: r.label.y }));
    $("#recTable tbody").innerHTML = recs.slice().reverse().map((r) => `<tr><td>${esc(r.id)}</td><td>${esc(r.label.activity)}</td>
      <td>${r.label.x != null ? `(${fmt(r.label.x, 1)}, ${fmt(r.label.y, 1)})` : "—"}</td><td>${r.frames.join(" / ")}</td>
      <td>${(r.size / 1024).toFixed(0)} KB</td><td><button class="chip danger" data-del="${esc(r.file)}">DELETE</button></td></tr>`).join("");
    document.querySelectorAll("[data-del]").forEach((b) => b.onclick = async () => { await api("/api/recordings/" + encodeURIComponent(b.dataset.del), { method: "DELETE" }); loadRecordings(); });
    drawFloor();
  } catch (e) { /* server offline */ }
}
$("#btnRecRefresh").onclick = loadRecordings;
$("#btnRecClear").onclick = async () => { if (confirm("Delete ALL recordings?")) { await api("/api/recordings", { method: "DELETE" }); loadRecordings(); } };

// ------------------------------------------------------------------ training
let lastTrain = null;
function updateTrain(tr) {
  lastTrain = tr;
  const st = $("#trainStatus");
  st.textContent = tr.status.toUpperCase() + (tr.status === "running" ? ` · ${tr.epoch}/${tr.epochs}` : "");
  st.classList.toggle("live", tr.status === "running");
  const p = $("#trProg");
  p.style.width = tr.epochs ? (100 * tr.epoch / tr.epochs).toFixed(1) + "%" : "0%";
  p.classList.toggle("indet", tr.status === "running" && tr.epoch === 0);
  $("#btnTrain").disabled = tr.status === "running";
  if (document.activeElement !== $("#autoTrain")) $("#autoTrain").checked = tr.autotrain;
  const h = tr.history[tr.history.length - 1] || {};
  const res = tr.result && tr.result.metrics ? (tr.result.metrics.real || tr.result.metrics.synthetic || {}) : {};
  const m = tr.status === "done" ? { ...h, ...res } : h;
  $("#trMetrics").innerHTML = [["ACTIVITY", m.activity_acc, "%"], ["ZONE", m.zone_acc, "%"], ["XY ERROR", m.xy_mae_m, "m"], ["LOSS", m.loss, ""]]
    .map(([k, v, u]) => `<div><small>${k}</small><b>${v == null ? "—" : u === "%" ? (v * 100).toFixed(1) + "%" : u === "m" ? v.toFixed(2) + " m" : v.toFixed(3)}</b></div>`).join("");
  const lines = [...tr.log];
  if (tr.reason) lines.unshift(`reason: ${tr.reason}${tr.elapsed ? ` · ${tr.elapsed}s` : ""}`);
  if (tr.result && tr.result.error) lines.push("ERROR: " + tr.result.error);
  if (tr.result && tr.result.model) lines.push(`→ activated ${tr.result.model}`);
  if (tr.pending_recordings) lines.push(`${tr.pending_recordings} new recording(s) since last training`);
  $("#trLog").textContent = lines.join("\n") || "idle";
  if (activeTab === "train") drawTrainChart();
}
function drawTrainChart() {
  const c = $("#trChart"); if (!c.clientWidth) return;
  const { ctx, w, h } = fitCanvas(c); ctx.clearRect(0, 0, w, h);
  const hist = lastTrain ? lastTrain.history : [];
  const pad = 30;
  ctx.strokeStyle = css("--line"); ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) { const y = pad + (i / 4) * (h - 2 * pad); ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w - 10, y); ctx.stroke(); }
  ctx.fillStyle = css("--fg3"); ctx.font = "10px JetBrains Mono, monospace";
  if (!hist.length) { ctx.fillText("no training run yet", pad + 10, h / 2); return; }
  const n = Math.max(hist.length, lastTrain.epochs || hist.length);
  const maxLoss = Math.max(...hist.map((x) => x.loss));
  const series = [["loss", (v) => v / maxLoss, []], ["activity_acc", (v) => v, [6, 4]], ["zone_acc", (v) => v, [2, 3]]];
  for (const [k, f, dash] of series) {
    ctx.setLineDash(dash); ctx.strokeStyle = css("--fg"); ctx.lineWidth = 1.6; ctx.beginPath();
    hist.forEach((p, i) => { const x = pad + (i / Math.max(1, n - 1)) * (w - pad - 10), y = h - pad - f(p[k]) * (h - 2 * pad); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.fillText("1.0", 4, pad + 3); ctx.fillText("0", 4, h - pad + 3); ctx.fillText(`epoch ${hist.length}/${n}`, w - 100, h - 8);
}
$("#btnTrain").onclick = async () => {
  try { await api("/api/train", { json: { epochs: +$("#trEpochs").value, synthetic: +$("#trSynth").value, use_recordings: $("#trUseRec").checked } }); toast("training started"); }
  catch (e) { toast(e.message); }
};
$("#btnTrainStop").onclick = () => api("/api/train/stop", { method: "POST" }).catch((e) => toast(e.message));
$("#autoTrain").onchange = (e) => api("/api/config", { json: { autotrain: { enabled: e.target.checked } } }).then(() => toast("auto-train " + (e.target.checked ? "on" : "off"))).catch((er) => toast(er.message));

// ------------------------------------------------------------------ models
async function loadModels() {
  try {
    const ms = await (await fetch("/api/models")).json();
    $("#modelTable tbody").innerHTML = ms.map((m) => {
      const v = m.metrics ? (m.metrics.real || m.metrics.synthetic || {}) : {};
      const pct = (x) => (x == null ? "—" : (x * 100).toFixed(1) + "%");
      return `<tr class="${m.active ? "active" : ""}"><td>${esc(m.name)}</td><td>${esc(m.kind || "?")}${m.metrics && m.metrics.real ? " · real" : ""}</td>
        <td>${esc((m.created || "").replace("T", " "))}</td><td>${pct(v.activity_acc)}</td><td>${pct(v.zone_acc)}</td><td>${v.xy_mae_m != null ? v.xy_mae_m.toFixed(2) + " m" : "—"}</td>
        <td>${esc(m.device || "")}</td><td>${m.active ? '<span class="pill live">ACTIVE</span>' : `<button class="chip" data-act="${esc(m.file)}">ACTIVATE</button>`}
        <a class="chip" href="/api/models/${encodeURIComponent(m.file)}">DOWNLOAD</a>${m.active ? "" : ` <button class="chip danger" data-mdel="${esc(m.file)}">DELETE</button>`}</td></tr>`;
    }).join("") || `<tr><td colspan="8" class="muted">no models yet</td></tr>`;
    const auto = ms.length && ms[0].auto;
    $("#modelTable tbody").insertAdjacentHTML("afterbegin", `<tr class="${auto ? "active" : ""}"><td>AUTO</td>
      <td colspan="6" class="muted">uses wavesense-1node / 2node / 3node to match the number of boards streaming right now</td>
      <td>${auto ? '<span class="pill live">ACTIVE</span>' : '<button class="chip" data-act="auto">USE AUTO</button>'}</td></tr>`);
    document.querySelectorAll("[data-act]").forEach((b) => b.onclick = async () => { try { await api("/api/models/activate", { json: { file: b.dataset.act } }); toast("model activated"); loadModels(); } catch (e) { toast(e.message); } });
    document.querySelectorAll("[data-mdel]").forEach((b) => b.onclick = async () => { if (confirm("Delete " + b.dataset.mdel + "?")) { await api("/api/models/" + encodeURIComponent(b.dataset.mdel), { method: "DELETE" }); loadModels(); } });
  } catch (e) { /* offline */ }
}
$("#btnModelsRefresh").onclick = loadModels;
async function upload(input, url, okMsg) {
  const f = input.files[0]; if (!f) return;
  const fd = new FormData(); fd.append("file", f, f.name);
  try { await api(url, { method: "POST", body: fd }); toast(okMsg); } catch (e) { toast(e.message); }
  input.value = "";
}
$("#modelUpload").onchange = async (e) => { await upload(e.target, "/api/models/upload", "model uploaded"); loadModels(); };
$("#glbUpload").onchange = (e) => upload(e.target, "/api/room.glb", "room model uploaded");

// ------------------------------------------------------------------ setup
function buildCfgForm() {
  if (!CFG) return;
  const f = $("#cfgForm");
  const num = (name, label, v, step = 0.1) => `<label>${label}<input type="number" step="${step}" name="${name}" value="${v}"></label>`;
  let html = `<fieldset><legend>ROOM</legend><label>name<input class="wide" name="room.name" value="${esc(CFG.room.name || "")}"></label>
    ${num("room.width", "width", CFG.room.width)}${num("room.depth", "depth", CFG.room.depth)}${num("room.height", "height", CFG.room.height)}</fieldset>
    <fieldset><legend>ROUTER</legend>${num("router.x", "x", CFG.router.x)}${num("router.y", "y", CFG.router.y)}${num("router.z", "z", CFG.router.z)}</fieldset>`;
  CFG.nodes.forEach((n, i) => {
    html += `<fieldset><legend>NODE ${i + 1}</legend>${num(`nodes.${i}.id`, "id", n.id, 1)}<label>name<input class="wide" name="nodes.${i}.name" value="${esc(n.name || "")}"></label>
      ${num(`nodes.${i}.x`, "x", n.x)}${num(`nodes.${i}.y`, "y", n.y)}${num(`nodes.${i}.z`, "z", n.z ?? 1)}</fieldset>`;
  });
  html += `<fieldset><legend>ZONES &amp; WINDOW</legend>${num("zones.rows", "rows", CFG.zones.rows, 1)}${num("zones.cols", "cols", CFG.zones.cols, 1)}${num("window_s", "window s", CFG.window_s, 0.25)}</fieldset>`;
  f.innerHTML = html;
  const host = location.hostname || "JETSON_IP";
  $("#flashCmds").textContent =
`# 1. build (or download from CI) the WaveSense 3D firmware, then flash every board:
python tools/flash_nodes.py --ssid "YOUR_WIFI" --password "YOUR_PASS" \\
    --target-ip ${host} --target-port 5005
#    (auto-detects each connected ESP32-S3 and assigns node ids 1, 2, 3)

# 2. or one board at a time:
python tools/provision.py --port /dev/ttyACM0 --node-id 1 \\
    --ssid "YOUR_WIFI" --password "YOUR_PASS" --target-ip ${host}

# 3. check that all nodes stream (stop the server first, or use the dashboard):
python -m wavesense probe --port 5005`;
}
$("#btnSaveCfg").onclick = async () => {
  const patch = JSON.parse(JSON.stringify(CFG));
  delete patch.model;
  for (const inp of $("#cfgForm").querySelectorAll("input")) {
    const path = inp.name.split("."); let o = patch;
    for (let i = 0; i < path.length - 1; i++) o = o[path[i]];
    o[path[path.length - 1]] = inp.type === "number" ? parseFloat(inp.value) : inp.value;
  }
  try { await api("/api/config", { json: patch }); $("#cfgMsg").textContent = "saved · retrain the model if you moved nodes"; toast("config saved"); }
  catch (e) { $("#cfgMsg").textContent = e.message; }
};

// ------------------------------------------------------------------ people
let histWin = 10, histData = null;
document.querySelectorAll("[data-win]").forEach((b) => b.onclick = () => {
  histWin = +b.dataset.win;
  document.querySelectorAll("[data-win]").forEach((x) => x.classList.toggle("on", x === b));
  loadHistory();
});
async function loadHistory() {
  try { histData = await (await fetch("/api/history?minutes=" + histWin)).json(); } catch (e) { return; }
  $("#pWin").textContent = histWin;
  $("#pPeak").textContent = histData.peak;
  $("#pOcc").textContent = histData.occupied_pct + "%";
  $("#pAvg").textContent = histData.avg_people;
  drawPeopleChart();
}
setInterval(() => { if (activeTab === "people") loadHistory(); }, 2000);
function drawDist(el_, probs) {
  if (el_.childElementCount !== 4) el_.innerHTML = [0, 1, 2, 3].map((i) => `<div><span>${i}</span></div>`).join("");
  probs.forEach((p, i) => { el_.children[i].style.height = Math.max(2, p * 46).toFixed(0) + "px"; el_.children[i].style.opacity = 0.25 + 0.75 * p; });
}
function updatePeople(s) {
  const p = s.people; if (!p) return;
  $("#pTotal").textContent = p.total; $("#pMoving").textContent = p.moving; $("#pStill").textContent = p.still;
  $("#pTotalSub").textContent = p.total ? `${p.moving} moving · ${p.still} still` : (s.activity === "no-signal" ? "no signal" : "room is empty");
  drawDist($("#pDistMove"), p.p_moving); drawDist($("#pDistStill"), p.p_still);
  const ev = s.events.filter((e) => e.msg.startsWith("people:") || e.msg.startsWith("presence") || e.msg.startsWith("room is empty"));
  $("#peopleEvents").innerHTML = ev.map((e) => `<li><time>${new Date(e.t * 1000).toLocaleTimeString()}</time><span>${esc(e.msg)}</span></li>`).join("") || `<li class="muted">no changes yet</li>`;
}
function drawPeopleChart() {
  const c = $("#peopleChart"); if (!c.clientWidth) return;
  const { ctx, w, h } = fitCanvas(c); ctx.clearRect(0, 0, w, h);
  const padL = 30, padB = 22, top = 10, ymax = 3;
  const fg = css("--fg"), fg3 = css("--fg3"), line = css("--line");
  ctx.font = "10px JetBrains Mono, monospace"; ctx.fillStyle = fg3; ctx.strokeStyle = line; ctx.lineWidth = 1;
  for (let i = 0; i <= ymax; i++) {
    const y = h - padB - (i / ymax) * (h - padB - top);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - 6, y); ctx.stroke(); ctx.fillText(String(i), 10, y + 3);
  }
  const now = Date.now() / 1000, t0 = now - histWin * 60;
  const X = (t) => padL + ((t - t0) / (now - t0)) * (w - padL - 6);
  const Y = (v) => h - padB - (Math.min(v, ymax) / ymax) * (h - padB - top);
  for (let m = 0; m <= histWin; m += histWin <= 10 ? 2 : 10) { const x = X(now - m * 60); ctx.fillText(m ? `-${m}m` : "now", x - 12, h - 6); }
  const pts = histData ? histData.points : [];
  if (!pts.length) { ctx.fillText("collecting… (1 point per second)", padL + 10, top + 20); return; }
  const bw = Math.max(1, (w - padL - 6) / (histWin * 60));
  // hatch pattern for "still"
  const pc = document.createElement("canvas"); pc.width = pc.height = 6;
  const pctx = pc.getContext("2d"); pctx.strokeStyle = fg; pctx.lineWidth = 1.2; pctx.beginPath(); pctx.moveTo(0, 6); pctx.lineTo(6, 0); pctx.stroke();
  const hatch = ctx.createPattern(pc, "repeat");
  for (const [t, , mv, st] of pts) {
    const x = X(t);
    if (mv) { ctx.fillStyle = fg; ctx.fillRect(x, Y(mv), bw + 0.5, Y(0) - Y(mv)); }
    if (st) { ctx.fillStyle = hatch; ctx.fillRect(x, Y(mv + st), bw + 0.5, Y(mv) - Y(mv + st)); }
  }
  ctx.strokeStyle = fg; ctx.lineWidth = 1.5; ctx.beginPath();
  pts.forEach(([t, tot], i) => { const x = X(t), y = Y(tot); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke();
}

// ------------------------------------------------------------------ teach counts
const SCENARIOS = [
  { act: "empty", m: 0, s: 0, title: "Nobody", sub: "everyone leaves the room" },
  { act: "still", m: 0, s: 1, title: "1 person still", sub: "sit or stand still" },
  { act: "moving", m: 1, s: 0, title: "1 person walking", sub: "walk around normally" },
  { act: "still", m: 0, s: 2, title: "2 people still", sub: "both sit / stand still", min: 2 },
  { act: "moving", m: 1, s: 1, title: "2 people: 1 walks", sub: "one walks, one sits", min: 2 },
  { act: "moving", m: 2, s: 0, title: "2 people walking", sub: "both walk around", min: 2 },
  { act: "moving", m: 1, s: 2, title: "3 people: 1 walks", sub: "one walks, two sit", min: 3 },
  { act: "still", m: 0, s: 3, title: "3 people still", sub: "all three sit still", min: 3 },
];
let teachRecs = [], teachBusy = false;
const scenKey = (act, m, s) => `${act}:${m}:${s}`;
function renderTeach() {
  const maxP = CFG ? CFG.max_people || 3 : 3;
  const count = {};
  for (const r of teachRecs) {
    const l = r.label, d = { empty: [0, 0], still: [0, 1], moving: [1, 0] }[l.activity] || [0, 0];
    const k = scenKey(l.activity, l.moving_count ?? d[0], l.still_count ?? d[1]);
    count[k] = (count[k] || 0) + 1;
  }
  $("#teachGrid").innerHTML = SCENARIOS.filter((x) => (x.min || 0) <= maxP).map((x, i) =>
    `<button data-scen="${SCENARIOS.indexOf(x)}" ${teachBusy ? "disabled" : ""}><b>${x.title}</b><small>${x.sub} · recorded ${count[scenKey(x.act, x.m, x.s)] || 0}×</small></button>`).join("");
  document.querySelectorAll("[data-scen]").forEach((b) => b.onclick = () => teach(SCENARIOS[+b.dataset.scen], b));
}
async function loadTeachRecs() {
  try { teachRecs = await (await fetch("/api/recordings")).json(); } catch (e) { teachRecs = []; }
  renderTeach();
}
async function teach(sc, btn) {
  if (teachBusy) return;
  teachBusy = true; renderTeach();
  try {
    for (let c = 5; c > 0; c--) { $("#teachInfo").textContent = `${sc.title}: get into position… ${c}`; await sleep(1000); }
    await api("/api/record", { json: { activity: sc.act, moving_count: sc.m, still_count: sc.s, seconds: +$("#teachSecs").value } });
    await sleep(400);
    while (STATE && STATE.recording) { $("#teachInfo").textContent = `${sc.title}: recording ${STATE.recording.elapsed.toFixed(0)} / ${STATE.recording.seconds} s - stay as you are`; await sleep(300); }
    $("#teachInfo").textContent = `${sc.title}: saved ✓`;
    toast("recording saved");
  } catch (e) { toast(e.message); $("#teachInfo").textContent = e.message; }
  teachBusy = false; loadTeachRecs();
}
$("#btnTeachCancel").onclick = () => api("/api/record/cancel", { method: "POST" }).catch(() => {});
$("#btnTeachTrain").onclick = async () => {
  try { await api("/api/train", { json: { reason: "teach counts" } }); toast("training started - watch the Train tab"); }
  catch (e) { toast(e.message); }
};
$("#maxPeople").onchange = async (e) => {
  try { await api("/api/config", { json: { max_people: +e.target.value } }); toast("max people = " + e.target.value); } catch (er) { toast(er.message); }
};

// ------------------------------------------------------------------ auto-calibration
$("#btnAutocal").onclick = async () => {
  try { await api("/api/autocal", { json: { seconds: +$("#acSecs").value } }); toast("auto-calibration started - leave the room for 30 s"); }
  catch (e) { toast(e.message); }
};
$("#btnAutocalStop").onclick = () => api("/api/autocal/cancel", { method: "POST" }).catch(() => {});
let lastAcRender = null;
function updateAutocal(s) {
  const job = s.autocal, res = s.autocal_result;
  $("#btnAutocal").disabled = !!job;
  if (job) {
    const left = Math.max(0, job.seconds - job.elapsed);
    $("#acStatus").textContent = `listening · ${job.frames} frames · ${left.toFixed(0)} s left`;
    $("#acProg").style.width = (100 * job.elapsed / job.seconds).toFixed(1) + "%";
    $("#acResult").innerHTML = `<div class="note">${job.elapsed < 30 ? "Phase 1 - room empty or completely still (baseline)." : "Phase 2 - keep the nodes streaming; you may move around normally now."}</div>`;
    lastAcRender = null;
    return;
  }
  $("#acProg").style.width = res ? "100%" : "0%";
  if (!res) { $("#acStatus").textContent = "idle"; return; }
  if (lastAcRender === res.finished) return;
  lastAcRender = res.finished;
  $("#acStatus").textContent = "done · " + new Date(res.finished * 1000).toLocaleTimeString();
  const sug = res.suggestion;
  const rows = res.nodes.map((n) => n.ok
    ? `<tr><td>node ${n.id}</td><td><b>${n.rate_hz}</b>/s</td><td><b>${n.rssi_dbm}</b> dBm</td><td>${n.calm_motion}</td><td>${n.link_m != null ? `<b>${n.link_m.toFixed(1)} m</b> ± ${sug ? sug.uncertainty_m.link : "?"}` : "—"}</td><td>baseline ✓ (from ${n.calm_from_s}s)</td></tr>`
    : `<tr><td>node ${n.id}</td><td colspan="5" class="muted">${esc(n.reason)}</td></tr>`).join("");
  let html = `<table class="tbl ac-table"><thead><tr><th>node</th><th>rate</th><th>rssi</th><th>noise</th><th>router distance (est.)</th><th>baseline</th></tr></thead><tbody>${rows}</tbody></table>`;
  if (sug) {
    html += `<div class="note">Rough suggested router position: <b>(${sug.router.x}, ${sug.router.y})</b> m. Distances are estimated mostly from signal strength (± ${sug.uncertainty_m.link} m on simulated rooms, usually worse on real hardware), so a tape-measured position is always better${sug.room_estimated ? ` · room ≈ <b>${sug.room.width} × ${sug.room.depth}</b> m ± ${sug.uncertainty_m.room}` : ""}.
      ${sug.note ? `<br>${esc(sug.note)}` : ""} Direction can't be measured with a single board, so check the position on the floor map.</div>
      <div class="row"><button class="btn" id="btnAcApply">APPLY ROUGH ROUTER POSITION</button><span class="muted">baselines were applied automatically</span></div>`;
  }
  $("#acResult").innerHTML = html;
  const b = $("#btnAcApply");
  if (b) b.onclick = async () => { try { await api("/api/autocal/apply", { method: "POST" }); toast("layout applied"); } catch (e) { toast(e.message); } };
}

// ------------------------------------------------------------------ websocket
function applyConfig(cfg) {
  CFG = cfg;
  $("#roomLabel").textContent = `${cfg.room.name || "room"} · ${cfg.room.width} × ${cfg.room.depth} m`;
  scene3d.build(cfg);
  ensureNodeCards(cfg);
  if (activeTab === "setup") buildCfgForm();
  $("#maxPeople").value = String(cfg.max_people || 3);
  renderTeach();
  drawFloor();
}
function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  const pill = $("#pillLink");
  ws.onopen = () => { pill.classList.add("live"); pill.lastChild.textContent = "LIVE"; };
  ws.onclose = () => { pill.classList.remove("live"); pill.lastChild.textContent = "OFFLINE"; setTimeout(connect, 1500); };
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === "hello" || m.type === "config") applyConfig(m.config);
    else if (m.type === "state") {
      STATE = m;
      if (!CFG) return;
      if (m.udp.source === "sim") pill.lastChild.textContent = "SIM";
      updateLive(m); updateCalib(m); updateTrain(m.training); updatePeople(m); updateAutocal(m);
      if (m.training.status === "done" && updateTrain.lastDone !== m.training.result.model) { updateTrain.lastDone = m.training.result.model; if (activeTab === "models") loadModels(); }
    }
  };
}
connect();
