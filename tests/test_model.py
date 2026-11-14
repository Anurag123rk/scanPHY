import os

import numpy as np
import pytest

from scanphy.config import load_config
from scanphy.model import ScanModel
from scanphy.simulator import synth_dataset

BASE = os.path.join(os.path.dirname(__file__), "..", "models", "scanphy-base.npz")


@pytest.mark.skipif(not os.path.exists(BASE), reason="base model not present")
def test_base_model_on_simulated_room():
    m = ScanModel(BASE)
    X, act, xy, zone, dims = synth_dataset(600, seed=1234, cfg=load_config(None))
    p = m.predict(X)
    assert p["activity"].shape == (600, 3)
    assert np.allclose(p["activity"].sum(1), 1, atol=1e-5)
    presence_acc = ((p["activity"].argmax(1) > 0) == (act > 0)).mean()
    assert presence_acc > 0.9
    person = act > 0
    err = np.linalg.norm((p["xy"][person] - xy[person]) * dims[person], axis=1).mean()
    assert err < 1.2
