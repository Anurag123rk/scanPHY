import os

import numpy as np

from scanphy import autocal
from scanphy.config import load_config, validate
from scanphy.features import amplitudes
from scanphy.simulator import Scene

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _collect(cfg, seconds=40, rate=30, seed=3):
    rng = np.random.default_rng(seed)
    scene = Scene(cfg, rng)
    slots = {}
    T = int(seconds * rate)
    t = np.arange(T) / rate
    for k, n in enumerate(cfg["nodes"]):
        # first half empty, second half someone walking near the link
        people = []
        H_empty = scene.channel(k, [], T // 2)
        pos = np.column_stack([np.linspace(1, 4, T - T // 2), np.full(T - T // 2, 2.0)])
        body = scene.random_body()
        H_walk = scene.channel(k, [(pos, np.zeros(len(pos)), body)], T - T // 2)
        iq, rssi = scene.quantize(k, np.vstack([H_empty, H_walk]))
        slots[int(n["id"])] = {"t": list(t), "amp": list(amplitudes(iq)), "rssi": list(rssi)}
    return slots


def test_calm_segment_found_in_empty_half():
    cfg = load_config(None)
    slots = _collect(cfg)
    s = slots[1]
    idx, _ = autocal.calm_segment(np.array(s["t"]), np.array(s["amp"]), seconds=10)
    assert np.array(s["t"])[idx].max() < 21.5  # baseline taken from (essentially) the empty first half


def test_analyse_single_node_suggests_router():
    cfg = load_config(os.path.join(ROOT, "config", "room.single-node.example.json"))
    geo = os.path.join(ROOT, "models", "geometry.npz")
    est = autocal.GeometryEstimator(geo) if os.path.exists(geo) else None
    res = autocal.analyse(cfg, _collect(cfg), est)
    assert res["nodes"][0]["ok"] and 1 in res["baselines"]
    if est is not None:
        sug = res["suggestion"]
        assert 0 <= sug["router"]["x"] <= cfg["room"]["width"]
        assert 0 <= sug["router"]["y"] <= cfg["room"]["depth"]
