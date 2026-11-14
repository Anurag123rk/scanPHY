"""Physics-based WiFi CSI simulator.

Models the 2.4 GHz OFDM channel between the router and each ESP32-S3 node as a sum of
propagation paths (line of sight + first-order wall/floor/ceiling reflections, image
source method), plus scattering from a human body (three scatter centres: legs, chest,
head). The body also attenuates any path that crosses its first Fresnel zone, and the
chest moves slightly with breathing.

It produces raw int8 CSI exactly like the ESP32-S3 (64 LLTF subcarriers, [imag, real]
pairs, null/guard subcarriers zeroed), so simulated data goes through the *same* code
path as real data. It is used for:
  * synthetic pre-training of the base model (domain-randomised rooms and layouts)
  * the `--source sim` demo mode and the `simulate-nodes` UDP emulator
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config as cfgmod
from .features import (ACTIVITIES, FEATURE_DIM, N_SLOTS, amplitudes, geometry_features, make_baseline,
                       shape_profile, window_features)

C = 299_792_458.0
SC_SPACING = 312.5e3
_IDX = np.arange(64)
SC_INDEX = np.where(_IDX < 32, _IDX, _IDX - 64)  # raw ESP order -> subcarrier number
NULL_MASK = (SC_INDEX == 0) | (np.abs(SC_INDEX) > 26)


def channel_freq(channel: int = 6) -> float:
    return (2407 + 5 * channel) * 1e6


class Scene:
    """Static room geometry + per-node propagation paths."""

    def __init__(self, cfg: Dict, rng: np.random.Generator, channel: int = 6):
        self.cfg = cfg
        self.rng = rng
        room = cfg["room"]
        self.W, self.D, self.H = room["width"], room["depth"], room["height"]
        self.fc = channel_freq(channel)
        self.freqs = self.fc + SC_INDEX * SC_SPACING
        self.lam = C / self.fc
        r = cfg["router"]
        self.ap = np.array([r["x"], r["y"], r["z"]], dtype=np.float64)
        self.nodes = [np.array([n["x"], n["y"], n["z"]], dtype=np.float64) for n in cfg["nodes"][:N_SLOTS]]
        # Reflection coefficient per plane: x=0, x=W, y=0, y=D, floor, ceiling.
        self.gamma = rng.uniform(0.25, 0.7, size=6)
        self.node_gain_db = rng.uniform(-4.0, 4.0, size=len(self.nodes))
        self.snr_db = rng.uniform(22.0, 34.0, size=len(self.nodes))
        self.q_target = rng.uniform(25.0, 60.0, size=len(self.nodes))
        self.clutter = [self._clutter_paths(i) for i in range(len(self.nodes))]
        self._build_paths()

    # -- geometry -----------------------------------------------------------------
    def _planes(self):
        return [(0, 0.0), (0, self.W), (1, 0.0), (1, self.D), (2, 0.0), (2, self.H)]

    def _build_paths(self) -> None:
        self.paths = []  # per node: dict(L, amp, segs, floorceil)
        for k, node in enumerate(self.nodes):
            lengths, amps, segs, fc = [], [], [], []
            d = np.linalg.norm(node - self.ap)
            lengths.append(d)
            amps.append(self.lam / (4 * math.pi * d))
            segs.append([(self.ap, node)])
            fc.append(False)
            for p, (axis, pos) in enumerate(self._planes()):
                img = self.ap.copy()
                img[axis] = 2 * pos - img[axis]
                L = np.linalg.norm(node - img)
                denom = node[axis] - img[axis]
                if abs(denom) < 1e-9:
                    continue
                t = (pos - img[axis]) / denom
                if not 0.0 < t < 1.0:
                    continue
                refl = img + t * (node - img)
                lengths.append(L)
                amps.append(self.gamma[p] * self.lam / (4 * math.pi * L))
                segs.append([(self.ap, refl), (refl, node)])
                fc.append(axis == 2)
            # Furniture / clutter: fixed random scatterers that only exist in this room.
            for (pt, sigma) in self.clutter[k]:
                d1 = np.linalg.norm(pt - self.ap)
                d2 = np.linalg.norm(node - pt)
                lengths.append(d1 + d2)
                amps.append(self.lam * math.sqrt(sigma) / ((4 * math.pi) ** 1.5 * d1 * d2))
                segs.append([(self.ap, pt), (pt, node)])
                fc.append(False)
            L = np.array(lengths)
            A = np.array(amps)
            E = np.exp(-2j * math.pi * np.outer(L, self.freqs) / C)  # (P, 64)
            self.paths.append({"L": L, "A": A, "segs": segs, "floorceil": np.array(fc), "E": E})

    def _clutter_paths(self, k: int):
        n = int(self.rng.integers(2, 6))
        pts = []
        for _ in range(n):
            pt = np.array([self.rng.uniform(0.1, self.W - 0.1), self.rng.uniform(0.1, self.D - 0.1),
                           self.rng.uniform(0.2, 1.8)])
            pts.append((pt, float(self.rng.uniform(0.2, 1.5))))
        return pts

    def perturb(self, scale: float = 0.05) -> None:
        """Small environmental drift (doors, furniture, temperature) between baseline and use."""
        for p in self.paths:
            p["A"] = p["A"] * (1.0 + self.rng.normal(0.0, scale, size=p["A"].shape))

    # -- human body -------------------------------------------------------------------
    def _blocking(self, k: int, pos: np.ndarray, depth: float) -> np.ndarray:
        """Attenuation (T, P) of every static path for person positions pos (T, 2)."""
        paths = self.paths[k]
        T = pos.shape[0]
        P = len(paths["segs"])
        att = np.ones((T, P))
        for pi, segs in enumerate(paths["segs"]):
            for (a, b) in segs:
                a2, b2 = a[:2], b[:2]
                ab = b2 - a2
                L2 = float(ab @ ab)
                if L2 < 1e-6:
                    continue
                t = np.clip(((pos - a2) @ ab) / L2, 0.0, 1.0)
                closest = a2 + t[:, None] * ab
                dist = np.linalg.norm(pos - closest, axis=1)
                seg_len = math.sqrt(L2)
                d1 = np.maximum(t * seg_len, 0.05)
                d2 = np.maximum((1 - t) * seg_len, 0.05)
                rf = np.sqrt(self.lam * d1 * d2 / (d1 + d2))
                dep = depth * (0.5 if paths["floorceil"][pi] else 1.0)
                att[:, pi] *= 1.0 - dep * np.exp(-(dist / (rf + 0.22)) ** 2)
        return att

    def body_scatter(self, k: int, pos: np.ndarray, chest: np.ndarray, body: Dict) -> np.ndarray:
        """Complex contribution (T, 64) of the body scatter centres."""
        node = self.nodes[k]
        T = pos.shape[0]
        H = np.zeros((T, 64), dtype=np.complex128)
        for (dx, dy, z, sigma, is_chest) in body["points"]:
            p3 = np.column_stack([pos[:, 0] + dx, pos[:, 1] + dy, np.full(T, z * body["height"])])
            d1 = np.linalg.norm(p3 - self.ap, axis=1)
            d2 = np.linalg.norm(p3 - node, axis=1)
            L = d1 + d2
            if is_chest:
                L = L + 1.6 * chest
            amp = self.lam * math.sqrt(sigma) / ((4 * math.pi) ** 1.5 * d1 * d2)
            H += amp[:, None] * np.exp(-2j * math.pi * L[:, None] * self.freqs[None, :] / C)
        return H

    def random_body(self) -> Dict:
        pts = [
            (self.rng.normal(0, 0.05), self.rng.normal(0, 0.05), 0.35, self.rng.uniform(0.2, 0.6), False),
            (self.rng.normal(0, 0.05), self.rng.normal(0, 0.05), 0.75, self.rng.uniform(0.5, 1.4), True),
            (self.rng.normal(0, 0.05), self.rng.normal(0, 0.05), 0.95, self.rng.uniform(0.1, 0.4), False),
        ]
        return {"points": pts, "height": self.rng.uniform(1.5, 1.9), "depth": self.rng.uniform(0.45, 0.8)}

    # -- CSI synthesis ------------------------------------------------------------------
    def channel(self, k: int, pos: Optional[np.ndarray], chest: Optional[np.ndarray], body: Optional[Dict],
                T: int) -> np.ndarray:
        p = self.paths[k]
        if pos is None:
            att = np.ones((T, len(p["A"])))
        else:
            att = self._blocking(k, pos, body["depth"])
        H = (att * p["A"][None, :]) @ p["E"]
        if pos is not None:
            H = H + self.body_scatter(k, pos, chest, body)
        return H

    def quantize(self, k: int, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Add radio impairments and quantise to ESP-style int8 IQ. Returns (iq (T,128), rssi (T,))."""
        T = H.shape[0]
        rms = np.sqrt(np.mean(np.abs(H[:, ~NULL_MASK]) ** 2, axis=1))
        rssi = 20 * np.log10(rms + 1e-12) + 5.0 + self.node_gain_db[k] + self.rng.normal(0, 0.7, size=T)
        noise_std = rms * 10 ** (-self.snr_db[k] / 20.0) / math.sqrt(2)
        H = H + noise_std[:, None] * (self.rng.normal(size=H.shape) + 1j * self.rng.normal(size=H.shape))
        # CFO / SFO / STO: random common phase + linear phase slope per frame.
        phi = self.rng.uniform(-math.pi, math.pi, size=T)
        slope = self.rng.normal(0, 0.05, size=T)
        H = H * np.exp(1j * (phi[:, None] + slope[:, None] * SC_INDEX[None, :]))
        # AGC: scale to the radio's working range with small per-frame jitter.
        peak = np.max(np.maximum(np.abs(H.real), np.abs(H.imag))[:, ~NULL_MASK], axis=1) + 1e-12
        scale = self.q_target[k] * 10 ** (self.rng.normal(0, 0.03, size=T)) / peak
        H = H * scale[:, None]
        H[:, NULL_MASK] = 0
        iq = np.empty((T, 128), dtype=np.int8)
        iq[:, 0::2] = np.clip(np.round(H.imag), -127, 127).astype(np.int8)
        iq[:, 1::2] = np.clip(np.round(H.real), -127, 127).astype(np.int8)
        return iq, np.round(np.clip(rssi, -100, -10)).astype(np.int16)


# ---------------------------------------------------------------------------------------
# Synthetic dataset for training
# ---------------------------------------------------------------------------------------

def random_layout(rng: np.random.Generator) -> Dict:
    W, D = rng.uniform(3.5, 9.0), rng.uniform(3.0, 8.0)
    cfg = {
        "room": {"name": "synthetic", "width": W, "depth": D, "height": rng.uniform(2.4, 3.2)},
        "router": _perimeter_point(rng, W, D, rng.uniform(0.8, 2.2)),
        "zones": {"rows": 3, "cols": 3},
        "window_s": 1.0,
    }
    nodes: List[Dict] = []
    tries = 0
    while len(nodes) < 3 and tries < 200:
        tries += 1
        if rng.random() < 0.75:
            p = _perimeter_point(rng, W, D, rng.uniform(0.7, 1.4))
        else:
            p = {"x": rng.uniform(0.3, W - 0.3), "y": rng.uniform(0.3, D - 0.3), "z": rng.uniform(0.7, 1.4)}
        if all(math.hypot(p["x"] - n["x"], p["y"] - n["y"]) > 1.5 for n in nodes) and \
                math.hypot(p["x"] - cfg["router"]["x"], p["y"] - cfg["router"]["y"]) > 1.0:
            nodes.append(dict(p, id=len(nodes) + 1, name="N%d" % (len(nodes) + 1)))
    cfg["nodes"] = nodes
    return cfgmod.validate(cfg)


def _perimeter_point(rng, W, D, z):
    m = rng.uniform(0.1, 0.6)
    side = rng.integers(0, 4)
    if side == 0:
        return {"x": rng.uniform(m, W - m), "y": m, "z": z}
    if side == 1:
        return {"x": rng.uniform(m, W - m), "y": D - m, "z": z}
    if side == 2:
        return {"x": m, "y": rng.uniform(m, D - m), "z": z}
    return {"x": W - m, "y": rng.uniform(m, D - m), "z": z}


def _trajectory(rng, W, D, T, rate, activity):
    t = np.arange(T) / rate
    margin = 0.3
    start = np.array([rng.uniform(margin, W - margin), rng.uniform(margin, D - margin)])
    if activity == "moving":
        speed = rng.uniform(0.3, 1.4)
        ang = rng.uniform(0, 2 * math.pi)
        turn = rng.normal(0, 0.6)
        pos = np.empty((T, 2))
        p = start.copy()
        for i in range(T):
            a = ang + turn * t[i]
            p = p + np.array([math.cos(a), math.sin(a)]) * speed / rate
            p[0] = np.clip(p[0], margin, W - margin)
            p[1] = np.clip(p[1], margin, D - margin)
            pos[i] = p
        chest = 0.01 * np.sin(2 * math.pi * rng.uniform(1.5, 2.2) * t)  # gait bounce
    else:
        pos = start[None, :] + rng.normal(0, 0.008, size=(T, 2))
        f = rng.uniform(0.15, 0.5)
        chest = rng.uniform(0.003, 0.010) * np.sin(2 * math.pi * f * t + rng.uniform(0, 6.28))
    return pos, chest


def synth_scene_samples(cfg: Optional[Dict], n: int, seed: int, rate_range=(20.0, 60.0),
                        window_s: float = 1.0):
    """Generate `n` labelled windows from one random scene. Returns arrays."""
    rng = np.random.default_rng(seed)
    layout = cfg if cfg is not None else random_layout(rng)
    scene = Scene(layout, rng, channel=int(rng.choice([1, 6, 11])))
    W, D = scene.W, scene.D
    rows, cols = layout["zones"]["rows"], layout["zones"]["cols"]
    geom = geometry_features(layout)
    n_nodes = len(scene.nodes)

    # Empty-room baseline (as the dashboard "capture baseline" button would record).
    baselines = []
    Tb = 80
    for k in range(n_nodes):
        iq, rssi = scene.quantize(k, scene.channel(k, None, None, None, Tb))
        baselines.append(make_baseline(shape_profile(amplitudes(iq)), rssi))
    scene.perturb(0.04)

    X = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    act = np.zeros(n, dtype=np.int64)
    xy = np.zeros((n, 2), dtype=np.float32)
    zone = np.zeros(n, dtype=np.int64)
    dims = np.tile(np.array([W, D], dtype=np.float32), (n, 1))
    for i in range(n):
        r = rng.random()
        a = "empty" if r < 0.22 else ("still" if r < 0.55 else "moving")
        rate = rng.uniform(*rate_range)
        T = max(8, int(rate * window_s))
        body = scene.random_body() if a != "empty" else None
        if a == "empty":
            pos = chest = None
        else:
            pos, chest = _trajectory(rng, W, D, T, rate, a)
        # Node-count dropout: the model must also work with 1 or 2 boards (or a board offline).
        r_nodes = rng.random()
        if r_nodes < 0.15 and n_nodes > 1:
            active = {int(rng.integers(0, n_nodes))}
        elif r_nodes < 0.30 and n_nodes > 2:
            active = set(rng.choice(n_nodes, size=2, replace=False).tolist())
        else:
            active = set(range(n_nodes))
        slots = []
        for k in range(N_SLOTS):
            if k not in active or rng.random() < 0.03:
                slots.append(None)
                continue
            iq, rssi = scene.quantize(k, scene.channel(k, pos, chest, body, T))
            slots.append({"profiles": shape_profile(amplitudes(iq)), "rssi": rssi})
        X[i] = window_features(slots, baselines + [None] * (N_SLOTS - n_nodes), geom)
        act[i] = ACTIVITIES.index(a)
        if pos is not None:
            m = pos.mean(axis=0)
            xy[i] = (m[0] / W, m[1] / D)
            zone[i] = cfgmod.zone_index(xy[i, 0], xy[i, 1], rows, cols)
    return X, act, xy, zone, dims


def _worker(args):
    cfg, n, seed = args
    return synth_scene_samples(cfg, n, seed)


def synth_dataset(n_samples: int, seed: int = 0, cfg: Optional[Dict] = None, per_scene: int = 24,
                  workers: int = 0):
    """Domain-randomised synthetic dataset -> (X, activity, xy_norm, zone, room_dims).

    If cfg is given its geometry is kept fixed (wall materials/clutter still vary)."""
    n_scenes = max(1, n_samples // per_scene)
    jobs = [(cfg, per_scene, seed * 100003 + s) for s in range(n_scenes)]
    if workers and workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers) as pool:
            parts = pool.map(_worker, jobs, chunksize=4)
    else:
        parts = [_worker(j) for j in jobs]
    X = np.concatenate([p[0] for p in parts])
    act = np.concatenate([p[1] for p in parts])
    xy = np.concatenate([p[2] for p in parts])
    zone = np.concatenate([p[3] for p in parts])
    dims = np.concatenate([p[4] for p in parts])
    return X, act, xy, zone, dims


# ---------------------------------------------------------------------------------------
# Live simulator (demo mode / UDP node emulator)
# ---------------------------------------------------------------------------------------

class LiveSimulator:
    """Continuously simulates the configured room with a virtual person following a script."""

    SCRIPT = [("empty", 6.0), ("moving", 14.0), ("still", 28.0), ("moving", 10.0), ("still", 24.0),
              ("moving", 8.0), ("empty", 6.0)]

    def __init__(self, cfg: Dict, rate: float = 40.0, seed: int = 7):
        self.cfg = cfg
        self.rate = rate
        self.rng = np.random.default_rng(seed)
        self.scene = Scene(cfg, self.rng)
        self.body = self.scene.random_body()
        self.t = 0.0
        self.seq = 0
        self.phase = 0
        self.phase_t = 0.0
        W, D = self.scene.W, self.scene.D
        self.pos = np.array([W * 0.5, D * 0.5])
        self.target = self._waypoint()
        self.breath_hz = 0.25  # 15 breaths / minute
        self.heart_hz = 1.2

    def _waypoint(self):
        W, D = self.scene.W, self.scene.D
        return np.array([self.rng.uniform(0.5, W - 0.5), self.rng.uniform(0.5, D - 0.5)])

    @property
    def activity(self) -> str:
        return self.SCRIPT[self.phase][0]

    def truth(self) -> Dict:
        a = self.activity
        return {"activity": a, "x": float(self.pos[0]), "y": float(self.pos[1]),
                "breathing_bpm": self.breath_hz * 60 if a == "still" else None}

    def step(self) -> List[Tuple[int, np.ndarray, int]]:
        """Advance one frame period; returns [(node_id, iq int8 (128,), rssi)]."""
        dt = 1.0 / self.rate
        self.t += dt
        self.phase_t += dt
        if self.phase_t > self.SCRIPT[self.phase][1]:
            self.phase = (self.phase + 1) % len(self.SCRIPT)
            self.phase_t = 0.0
            if self.activity == "moving":
                self.target = self._waypoint()
        a = self.activity
        if a == "moving":
            d = self.target - self.pos
            dist = float(np.linalg.norm(d))
            if dist < 0.15:
                self.target = self._waypoint()
            else:
                self.pos = self.pos + d / dist * 0.8 * dt
        chest = 0.006 * math.sin(2 * math.pi * self.breath_hz * self.t) + \
            0.0004 * math.sin(2 * math.pi * self.heart_hz * self.t)
        if a == "moving":
            chest += 0.01 * math.sin(2 * math.pi * 1.8 * self.t)
        out = []
        for k, node in enumerate(self.cfg["nodes"][:N_SLOTS]):
            if a == "empty":
                H = self.scene.channel(k, None, None, None, 1)
            else:
                H = self.scene.channel(k, self.pos[None, :], np.array([chest]), self.body, 1)
            iq, rssi = self.scene.quantize(k, H)
            out.append((int(node["id"]), iq[0], int(rssi[0])))
        self.seq += 1
        return out
