"""Automatic calibration.

While the nodes stream for a few minutes, WaveSense 3D:

  1. finds the calmest stretch of data and uses it as the empty-room baseline (no button
     press needed, as long as the room is empty or still for part of the time);
  2. measures the noise floor / packet rate of every link;
  3. *estimates* the router-to-node distance from the radio channel itself (mostly signal
     strength, plus multipath statistics), using a small regressor trained on thousands
     of simulated rooms, and turns it into a rough router position.

Step 3 is an estimate, not a measurement. A 20 MHz WiFi channel cannot resolve walls
(its range resolution is ~15 m): on simulated rooms the room-size estimate is barely
better than guessing a typical room, so it is only offered when the trained estimator
proves informative (see GeometryEstimator.informative). Room size must be entered by
the user; the router position is a suggestion the user has to confirm.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Optional

import numpy as np

from .features import make_baseline, shape_profile

GEO_FEATURES = ["rssi", "rssi_std", "sel_std", "ac1", "ac2", "ac4", "ac8", "ac13", "fades",
                "k_factor", "temporal", "tilt"]
GEO_TARGETS = ["link_m", "room_long_m", "room_short_m"]


# ----------------------------------------------------------------------------------- features

def link_stats(amp: np.ndarray, rssi: np.ndarray) -> np.ndarray:
    """Channel statistics of one calm link: amp (T, 52) amplitudes, rssi (T,)."""
    amp = np.asarray(amp, dtype=np.float64) + 1e-3
    la = np.log(amp)
    mean_prof = la.mean(axis=0)
    prof = mean_prof - mean_prof.mean()
    x = np.exp(mean_prof)
    xn = (x - x.mean()) / (x.std() + 1e-9)
    ac = [float(np.mean(xn[:-k] * xn[k:])) for k in (1, 2, 4, 8, 13)]
    p = amp ** 2
    m2 = p.mean(axis=1)
    m4 = (p ** 2).mean(axis=1)
    disc = np.clip(2 * m2 ** 2 - m4, 0, None)
    k = np.sqrt(disc) / np.maximum(m2 - np.sqrt(disc), 1e-9)
    k_db = float(np.median(10 * np.log10(np.clip(k, 1e-3, 1e3))))
    db = 20 * prof / math.log(10)
    tilt = float(np.polyfit(np.arange(len(db)), db, 1)[0])
    return np.array([
        float(np.mean(rssi)), float(np.std(rssi)), float(np.std(db)), *ac,
        float(np.mean(db < -6.0)), k_db, float(la.std(axis=0).mean()), tilt,
    ], dtype=np.float64)


def calm_segment(t: np.ndarray, amp: np.ndarray, seconds: float = 20.0):
    """Indices of the calmest `seconds` long stretch (lowest CSI fluctuation)."""
    prof = shape_profile(amp)
    if len(t) < 20:
        return None, float("inf")
    sec = np.floor(t - t[0]).astype(int)
    n_sec = int(sec.max()) + 1
    motion = np.full(n_sec, np.inf)
    for s in range(n_sec):
        m = sec == s
        if m.sum() >= 5:
            motion[s] = float(prof[m].std(axis=0).mean())
    w = int(min(seconds, n_sec))
    if w < 5:
        return None, float("inf")
    best, best_v = 0, float("inf")
    for s in range(0, n_sec - w + 1):
        v = float(np.mean(motion[s:s + w]))
        if v < best_v:
            best, best_v = s, v
    idx = np.where((sec >= best) & (sec < best + w))[0]
    return idx, best_v


# ----------------------------------------------------------------------------------- estimator

class GeometryEstimator:
    """Tiny MLP (NumPy inference) mapping link statistics -> link distance and room size."""

    def __init__(self, path: str):
        with np.load(path, allow_pickle=False) as z:
            self.meta = json.loads(str(z["meta"]))
            self.p = {k: z[k] for k in z.files if k != "meta"}

    def predict(self, feats: np.ndarray) -> Dict[str, float]:
        x = (np.atleast_2d(feats) - self.p["mu"]) / self.p["sigma"]
        h = np.tanh(x @ self.p["w0"] + self.p["b0"])
        h = np.tanh(h @ self.p["w1"] + self.p["b1"])
        y = np.exp(h @ self.p["w2"] + self.p["b2"])[0]
        return {k: float(v) for k, v in zip(GEO_TARGETS, y)}

    @property
    def errors(self) -> Dict[str, float]:
        return self.meta.get("mae", {})

    def informative(self, target: str) -> bool:
        """True if the estimate is clearly better than always guessing a typical value."""
        mae = self.meta.get("mae", {}).get(target)
        guess = self.meta.get("mae_if_guessing_median", {}).get(target)
        return mae is not None and guess is not None and mae < 0.8 * guess


def _geo_samples(seed: int, n_rooms: int):
    from .simulator import Scene, random_layout
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for _ in range(n_rooms):
        cfg = random_layout(rng)
        scene = Scene(cfg, rng, channel=int(rng.choice([1, 6, 11])))
        W, D = scene.W, scene.D
        for k in range(len(scene.nodes)):
            iq, rssi = scene.quantize(k, scene.channel(k, [], int(rng.integers(150, 400))))
            from .features import amplitudes
            X.append(link_stats(amplitudes(iq), rssi))
            Y.append([np.linalg.norm(scene.nodes[k][:2] - scene.ap[:2]), max(W, D), min(W, D)])
    return np.array(X), np.array(Y)


def train_geometry(out: str, n_rooms: int = 6000, seed: int = 0, epochs: int = 300) -> Dict:
    """Fit the geometry estimator on simulated rooms and save it (needs PyTorch)."""
    import torch
    from multiprocessing import get_context
    jobs = [(seed * 1000 + i, n_rooms // 8) for i in range(8)]
    with get_context("spawn").Pool(min(8, os.cpu_count() or 2)) as pool:
        parts = pool.starmap(_geo_samples, jobs)
    X = np.concatenate([p[0] for p in parts])
    Y = np.log(np.concatenate([p[1] for p in parts]))
    rng = np.random.default_rng(seed)
    val = rng.random(len(X)) < 0.15
    mu, sigma = X[~val].mean(0), X[~val].std(0) + 1e-6
    xt = torch.tensor((X - mu) / sigma, dtype=torch.float32)
    yt = torch.tensor(Y, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.Tanh(), torch.nn.Linear(64, 64),
                              torch.nn.Tanh(), torch.nn.Linear(64, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-5)
    tr = torch.tensor(~val)
    for ep in range(epochs):
        perm = torch.randperm(int(tr.sum()))
        xtr, ytr = xt[tr][perm], yt[tr][perm]
        for i in range(0, len(xtr), 256):
            loss = torch.nn.functional.smooth_l1_loss(net(xtr[i:i + 256]), ytr[i:i + 256], beta=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = np.exp(net(xt[torch.tensor(val)]).numpy())
    truth = np.exp(Y[val])
    mae = {k: float(np.mean(np.abs(pred[:, i] - truth[:, i]))) for i, k in enumerate(GEO_TARGETS)}
    baseline = {k: float(np.mean(np.abs(np.median(truth[:, i]) - truth[:, i]))) for i, k in enumerate(GEO_TARGETS)}
    w = [p.detach().numpy() for p in net.parameters()]
    meta = {"features": GEO_FEATURES, "targets": GEO_TARGETS, "mae": mae, "mae_if_guessing_median": baseline,
            "samples": int(len(X)), "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    np.savez_compressed(out, meta=np.array(json.dumps(meta)), mu=mu, sigma=sigma,
                        w0=w[0].T, b0=w[1], w1=w[2].T, b1=w[3], w2=w[4].T, b2=w[5])
    return meta


# ----------------------------------------------------------------------------------- run-time job

def analyse(cfg: Dict, slots: Dict[int, Dict[str, list]], estimator: Optional[GeometryEstimator]) -> Dict:
    """Turn the frames collected during auto-calibration into baselines + a suggested layout."""
    nodes_out: List[Dict] = []
    baselines: Dict[int, Dict] = {}
    for n in cfg["nodes"]:
        nid = int(n["id"])
        s = slots.get(nid)
        if not s or len(s["t"]) < 100:
            nodes_out.append({"id": nid, "ok": False, "reason": "not enough CSI frames"})
            continue
        t = np.asarray(s["t"], dtype=np.float64)
        amp = np.asarray(s["amp"], dtype=np.float32)
        rssi = np.asarray(s["rssi"], dtype=np.float32)
        idx, calm = calm_segment(t, amp)
        if idx is None:
            nodes_out.append({"id": nid, "ok": False, "reason": "recording too short"})
            continue
        baselines[nid] = make_baseline(shape_profile(amp[idx]), rssi[idx])
        total_s = float(t[-1] - t[0])
        info = {"id": nid, "ok": True, "rate_hz": round(len(t) / max(total_s, 1e-6), 1),
                "rssi_dbm": round(float(np.median(rssi)), 1), "calm_motion": round(calm * 20.0, 3),
                "calm_from_s": round(float(t[idx[0]] - t[0]), 1)}
        if estimator is not None:
            g = estimator.predict(link_stats(amp[idx], rssi[idx]))
            info.update({k: round(v, 2) for k, v in g.items()})
        nodes_out.append(info)

    suggestion = None
    good = [n for n in nodes_out if n.get("ok") and "link_m" in n]
    if good and estimator is not None:
        err = estimator.errors
        room_ok = estimator.informative("room_long_m") and estimator.informative("room_short_m")
        if room_ok:
            W = round(float(np.median([n["room_long_m"] for n in good])), 1)
            D = round(float(np.median([n["room_short_m"] for n in good])), 1)
        else:  # WiFi cannot tell us the room size: keep what the user entered
            W, D = cfg["room"]["width"], cfg["room"]["depth"]
        node_cfg = [dict(n) for n in cfg["nodes"]]
        if len(good) == 1:
            # One link: the distance is known (roughly) but not the direction. Keep the board
            # where it is and put the router `link_m` away, towards the middle of the room.
            n0 = node_cfg[0]
            d = good[0]["link_m"]
            vx, vy = W / 2 - n0["x"], D / 2 - n0["y"]
            norm = math.hypot(vx, vy) or 1.0
            rx = min(max(n0["x"] + vx / norm * d, 0.2), W - 0.2)
            ry = min(max(n0["y"] + vy / norm * d, 0.2), D - 0.2)
            router = {"x": round(rx, 2), "y": round(ry, 2), "z": cfg["router"].get("z", 1.2)}
        else:
            router = _trilaterate(cfg, good, W, D)
        suggestion = {
            "room": {"width": W, "depth": D},
            "room_estimated": room_ok,
            "router": router,
            "uncertainty_m": {"link": round(err.get("link_m", 1.0), 1),
                              "room": round(max(err.get("room_long_m", 1.0), err.get("room_short_m", 1.0)), 1)},
            "note": None if room_ok else "Room size cannot be measured from a WiFi channel - "
                                         "enter width/depth yourself (tape measure or a phone measure app).",
        }
    return {"nodes": nodes_out, "baselines": baselines, "suggestion": suggestion,
            "estimator": estimator.meta if estimator else None}


def _trilaterate(cfg: Dict, good: List[Dict], W: float, D: float) -> Dict:
    """Router position from >= 2 node distances (least squares on a grid inside the room)."""
    pos = {int(n["id"]): (n["x"], n["y"]) for n in cfg["nodes"]}
    gx, gy = np.meshgrid(np.linspace(0.2, W - 0.2, 60), np.linspace(0.2, D - 0.2, 50))
    err = np.zeros_like(gx)
    for n in good:
        x, y = pos[n["id"]]
        err += (np.hypot(gx - min(x, W), gy - min(y, D)) - n["link_m"]) ** 2
    i = np.unravel_index(int(np.argmin(err)), err.shape)
    return {"x": round(float(gx[i]), 2), "y": round(float(gy[i]), 2), "z": cfg["router"].get("z", 1.2)}
