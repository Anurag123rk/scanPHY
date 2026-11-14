import os
import time

from scanphy.config import load_config
from scanphy.engine import Engine
from scanphy.simulator import LiveSimulator

ROOT = os.path.join(os.path.dirname(__file__), "..")


def run(engine, sim, seconds, t0):
    t = t0
    state = None
    for i in range(int(seconds * sim.rate)):
        t += 1.0 / sim.rate
        for nid, iq, rssi in sim.step():
            engine.ingest_iq(nid, iq, rssi, t, sim.seq)
        if i % 4 == 0:
            state = engine.tick(t)
    return state, t


def test_engine_detects_presence_and_records(tmp_path):
    cfg = load_config(None)
    eng = Engine(cfg, ROOT, str(tmp_path), model_path="models/scanphy-base.npz")
    sim = LiveSimulator(cfg, rate=30)
    sim.SCRIPT = [("empty", 12.0), ("moving", 1e9)]
    t0 = time.time()
    st, t = run(eng, sim, 11.0, t0)
    assert all(n["alive"] for n in st["nodes"])
    assert not st["presence"]
    saved = []
    eng.on_recording_saved = saved.append
    st, t = run(eng, sim, 3.0, t)
    eng.start_recording({"activity": "moving", "x": None, "y": None}, 2.0)
    eng.recording["start"], eng.recording["end"] = t, t + 2.0
    st, t = run(eng, sim, 4.0, t)
    assert st["presence"]
    assert saved and os.path.exists(saved[0])
