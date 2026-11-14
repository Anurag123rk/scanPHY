import os

import numpy as np
import pytest

from scanphy.config import load_config, validate
from scanphy.model import ScanModel
from scanphy.simulator import synth_dataset

ROOT = os.path.join(os.path.dirname(__file__), "..")
# minimum simulated presence accuracy expected from each built-in model
MIN_PRESENCE = {1: 0.88, 2: 0.93, 3: 0.95}


@pytest.mark.parametrize("n", [1, 2, 3])
def test_builtin_model_per_board_count(n):
    path = os.path.join(ROOT, "models", "scanphy-%dnode.npz" % n)
    if not os.path.exists(path):
        pytest.skip("model not present")
    m = ScanModel(path)
    assert m.meta.get("nodes") == n
    cfg = load_config(os.path.join(ROOT, "config", "room.example.json"))
    cfg["nodes"] = cfg["nodes"][:n]
    d = synth_dataset(400, seed=4321, cfg=validate(cfg), n_active=n)
    p = m.predict(d["X"])
    assert np.allclose(p["activity"].sum(1), 1, atol=1e-5)
    assert p["n_move"].shape == (400, 4)
    presence = ((p["activity"].argmax(1) > 0) == (d["act"] > 0)).mean()
    assert presence > MIN_PRESENCE[n]
