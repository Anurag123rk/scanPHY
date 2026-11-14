"""Labelled CSI recordings (the training data collected in your own room).

Each recording is one `.npz` in data/recordings/ containing, for every node slot:
    amp<k>  float16 (T, 52)  amplitude of the usable subcarriers
    rssi<k> int16   (T,)
    t<k>    float64 (T,)     receive time (s)
plus a JSON `meta` blob: label, room/node config snapshot and the empty-room baselines
that were active while recording. Features are recomputed from the raw amplitudes at
training time, so recordings stay valid if the feature extractor evolves.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Dict, List, Optional

import numpy as np

from . import config as cfgmod
from .features import (ACTIVITIES, FEATURE_DIM, N_SLOTS, geometry_features, shape_profile, window_features)


def _baseline_to_json(b: Optional[Dict]) -> Optional[Dict]:
    if b is None:
        return None
    return {"profile": [float(v) for v in b["profile"]], "noise": [float(v) for v in b["noise"]],
            "rssi": float(b["rssi"])}


def baseline_from_json(b: Optional[Dict]) -> Optional[Dict]:
    if b is None:
        return None
    return {"profile": np.asarray(b["profile"], dtype=np.float32),
            "noise": np.asarray(b.get("noise", [0.01] * len(b["profile"])), dtype=np.float32),
            "rssi": float(b["rssi"])}


def save_recording(directory: str, slots: List[Dict[str, list]], label: Dict, cfg: Dict,
                   baselines: List[Optional[Dict]]) -> str:
    os.makedirs(directory, exist_ok=True)
    rid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    arrays = {}
    counts = []
    for k in range(N_SLOTS):
        s = slots[k] if k < len(slots) else {"t": [], "amp": [], "rssi": []}
        amp = np.asarray(s["amp"], dtype=np.float16).reshape(-1, 52)
        arrays["amp%d" % k] = amp
        arrays["rssi%d" % k] = np.asarray(s["rssi"], dtype=np.int16)
        arrays["t%d" % k] = np.asarray(s["t"], dtype=np.float64)
        counts.append(int(amp.shape[0]))
    meta = {
        "id": rid,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "label": label,
        "config": {k: cfg[k] for k in ("room", "router", "nodes", "zones", "window_s")},
        "baselines": [_baseline_to_json(b) for b in baselines],
        "frames": counts,
    }
    path = os.path.join(directory, rid + ".npz")
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)
    return path


def read_meta(path: str) -> Dict:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
    meta["file"] = os.path.basename(path)
    meta["size"] = os.path.getsize(path)
    return meta


def list_recordings(directory: str) -> List[Dict]:
    if not os.path.isdir(directory):
        return []
    out = []
    for name in sorted(os.listdir(directory)):
        if name.endswith(".npz"):
            try:
                out.append(read_meta(os.path.join(directory, name)))
            except Exception:  # corrupt file: skip it, never crash the server
                continue
    return out


def recording_windows(path: str, window_s: float = 1.0, hop_s: float = 0.25):
    """Slice a recording into labelled feature windows.

    Returns (X, activity, xy_norm, has_xy, zone, dims) or None if unusable.
    """
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        slots = [{"amp": z["amp%d" % k].astype(np.float32), "rssi": z["rssi%d" % k].astype(np.float32),
                  "t": z["t%d" % k]} for k in range(N_SLOTS)]
    cfg = cfgmod.validate(dict(meta["config"], model=None))
    label = meta["label"]
    act_name = label.get("activity", "moving")
    if act_name not in ACTIVITIES:
        return None
    W, D = cfg["room"]["width"], cfg["room"]["depth"]
    rows, cols = cfg["zones"]["rows"], cfg["zones"]["cols"]
    has_xy = label.get("x") is not None and label.get("y") is not None and act_name != "empty"
    xn = (float(label["x"]) / W, float(label["y"]) / D) if has_xy else (0.0, 0.0)
    baselines = [baseline_from_json(b) for b in meta.get("baselines", [])] + [None] * N_SLOTS
    geom = geometry_features(cfg)
    for s in slots:
        s["prof"] = shape_profile(s["amp"]) if len(s["amp"]) else np.zeros((0, 13), np.float32)
    times = [s["t"] for s in slots if len(s["t"])]
    if not times:
        return None
    t0 = min(float(t[0]) for t in times)
    t1 = max(float(t[-1]) for t in times)
    feats = []
    t = t0 + window_s
    while t <= t1 + 1e-6:
        win = []
        for s in slots:
            if not len(s["t"]):
                win.append(None)
                continue
            m = (s["t"] > t - window_s) & (s["t"] <= t)
            win.append({"profiles": s["prof"][m], "rssi": s["rssi"][m]} if m.sum() >= 5 else None)
        if any(w is not None for w in win):
            feats.append(window_features(win, baselines[:N_SLOTS], geom))
        t += hop_s
    if not feats:
        return None
    n = len(feats)
    X = np.stack(feats).astype(np.float32)
    act = np.full(n, ACTIVITIES.index(act_name), dtype=np.int64)
    xy = np.tile(np.array(xn, dtype=np.float32), (n, 1))
    hx = np.full(n, has_xy, dtype=bool)
    zone = np.full(n, cfgmod.zone_index(xn[0], xn[1], rows, cols) if has_xy else 0, dtype=np.int64)
    dims = np.tile(np.array([W, D], dtype=np.float32), (n, 1))
    return X, act, xy, hx, zone, dims


def load_all(directory: str, window_s: float = 1.0, hop_s: float = 0.25):
    """All recordings -> stacked arrays plus a recording index per window (for val split)."""
    parts, groups = [], []
    for i, meta in enumerate(list_recordings(directory)):
        r = recording_windows(os.path.join(directory, meta["file"]), window_s, hop_s)
        if r is None:
            continue
        parts.append(r)
        groups.append(np.full(len(r[0]), i, dtype=np.int64))
    if not parts:
        return None
    cat = [np.concatenate([p[j] for p in parts]) for j in range(6)]
    return cat + [np.concatenate(groups)]


EMPTY = (np.zeros((0, FEATURE_DIM), np.float32),)
