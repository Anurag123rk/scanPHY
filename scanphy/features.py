"""Feature extraction shared *exactly* by live inference, dataset building and training.

A feature vector summarises a short window (default 1 s) of CSI from up to three
nodes, relative to an empty-room baseline captured for each node:

    per node (30 values):
        13  mean shape deviation of the log-amplitude profile vs. baseline
        13  temporal std of the log-amplitude profile (motion signature)
         4  [rssi - baseline rssi, rssi std, overall motion, node valid flag]
    geometry (8 values):
         6  node (x, y) normalised by room width/depth
         2  router (x, y) normalised

Total = 3 * 30 + 8 = 98.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

FEATURE_VERSION = 3
N_GROUPS = 13
N_SLOTS = 3
PER_NODE = 2 * N_GROUPS + 4
GEOM_DIM = 8
FEATURE_DIM = N_SLOTS * PER_NODE + GEOM_DIM
MIN_FRAMES = 5

# LLTF of a 20 MHz channel in raw ESP-IDF order: index 0..31 -> subcarrier 0..31,
# index 32..63 -> subcarrier -32..-1. Usable subcarriers are -26..-1 and 1..26.
VALID64 = np.r_[38:64, 1:27]  # ordered from -26 .. +26 (DC removed)
N_VALID = VALID64.size  # 52

ACTIVITIES = ["empty", "still", "moving"]


def amplitudes(iq: np.ndarray) -> np.ndarray:
    """int8 IQ (T, 2*n_sc) -> amplitude (T, 52) over the usable LLTF subcarriers."""
    iq = np.asarray(iq, dtype=np.float32)
    if iq.ndim == 1:
        iq = iq[None, :]
    im, re = iq[:, 0::2], iq[:, 1::2]
    amp = np.sqrt(im * im + re * re)
    n_sc = amp.shape[1]
    if n_sc >= 64:
        return amp[:, :64][:, VALID64]
    # Unknown short layout: interpolate whatever we got onto 52 points.
    src = np.linspace(0.0, 1.0, n_sc)
    dst = np.linspace(0.0, 1.0, N_VALID)
    return np.stack([np.interp(dst, src, row) for row in amp]).astype(np.float32)


def shape_profile(amp: np.ndarray) -> np.ndarray:
    """Gain-invariant log-amplitude profile grouped into N_GROUPS bands -> (T, 13).

    Subtracting the per-frame mean removes the automatic gain control of the radio,
    so only the *shape* of the channel across frequency remains.
    """
    la = np.log(np.asarray(amp, dtype=np.float32) + 1.0)
    la = la - la.mean(axis=1, keepdims=True)
    return la.reshape(la.shape[0], N_GROUPS, -1).mean(axis=2)


def make_baseline(profiles: np.ndarray, rssi: np.ndarray) -> Dict[str, object]:
    """Empty-room reference for one node from (T, 13) profiles and (T,) rssi."""
    profiles = np.asarray(profiles, dtype=np.float32)
    return {
        "profile": np.median(profiles, axis=0).astype(np.float32),
        "noise": np.maximum(profiles.std(axis=0), 1e-3).astype(np.float32),
        "rssi": float(np.median(rssi)) if len(rssi) else -60.0,
    }


def node_features(profiles: np.ndarray, rssi: np.ndarray, baseline: Optional[Dict[str, object]]) -> np.ndarray:
    out = np.zeros(PER_NODE, dtype=np.float32)
    if profiles is None or len(profiles) < MIN_FRAMES:
        return out
    profiles = np.asarray(profiles, dtype=np.float32)
    rssi = np.asarray(rssi, dtype=np.float32)
    if baseline is None:
        ref = np.median(profiles, axis=0)
        ref_rssi = float(np.median(rssi))
    else:
        ref = np.asarray(baseline["profile"], dtype=np.float32)
        ref_rssi = float(baseline["rssi"])
    dev = profiles.mean(axis=0) - ref
    std = profiles.std(axis=0)
    g = N_GROUPS
    out[:g] = dev * 10.0
    out[g:2 * g] = std * 20.0
    out[2 * g + 0] = (float(rssi.mean()) - ref_rssi) / 3.0
    out[2 * g + 1] = float(rssi.std()) / 2.0
    out[2 * g + 2] = float(std.mean()) * 20.0
    out[2 * g + 3] = 1.0
    return out


def geometry_features(cfg: Dict) -> np.ndarray:
    w, d = cfg["room"]["width"], cfg["room"]["depth"]
    g = np.zeros(GEOM_DIM, dtype=np.float32)
    for i, n in enumerate(cfg["nodes"][:N_SLOTS]):
        g[2 * i] = n["x"] / w
        g[2 * i + 1] = n["y"] / d
    g[6] = cfg["router"]["x"] / w
    g[7] = cfg["router"]["y"] / d
    return g


def window_features(slots: Sequence[Optional[Dict[str, np.ndarray]]], baselines: Sequence[Optional[Dict]],
                    geom: np.ndarray) -> np.ndarray:
    """slots[i] = {"profiles": (T,13), "rssi": (T,)} or None for a silent node."""
    parts: List[np.ndarray] = []
    for i in range(N_SLOTS):
        s = slots[i] if i < len(slots) else None
        b = baselines[i] if i < len(baselines) else None
        if s is None:
            parts.append(np.zeros(PER_NODE, dtype=np.float32))
        else:
            parts.append(node_features(s["profiles"], s["rssi"], b))
    g = np.array(geom, dtype=np.float32)
    for i in range(N_SLOTS):
        # A node without data (not configured, offline or dropped) has no position either,
        # so 1-, 2- and 3-node rooms all look the same to the model.
        if parts[i][PER_NODE - 1] < 0.5:
            g[2 * i:2 * i + 2] = 0.0
    parts.append(g)
    return np.concatenate(parts)


def motion_level(vec: np.ndarray) -> float:
    """Average motion over valid nodes from a feature vector (0 = static)."""
    vals = []
    for i in range(N_SLOTS):
        base = i * PER_NODE
        if vec[base + PER_NODE - 1] > 0.5:
            vals.append(vec[base + 2 * N_GROUPS + 2])
    return float(np.mean(vals)) if vals else 0.0


def link_disturbance(vec: np.ndarray) -> np.ndarray:
    """Per-node disturbance score used by the model-free localiser."""
    out = np.zeros(N_SLOTS, dtype=np.float32)
    for i in range(N_SLOTS):
        base = i * PER_NODE
        if vec[base + PER_NODE - 1] < 0.5:
            continue
        dev = vec[base:base + N_GROUPS]
        out[i] = float(np.sqrt(np.mean(dev ** 2))) + 0.5 * vec[base + 2 * N_GROUPS + 2] \
            + 0.3 * abs(vec[base + 2 * N_GROUPS])
    return out
