import numpy as np

from wavesense.config import load_config
from wavesense.features import FEATURE_DIM, amplitudes, geometry_features, shape_profile, window_features
from wavesense.simulator import synth_scene_samples


def test_amplitudes_shape_and_gain_invariance():
    rng = np.random.default_rng(0)
    iq = rng.integers(-40, 40, size=(10, 128)).astype(np.int8)
    amp = amplitudes(iq)
    assert amp.shape == (10, 52)
    p1 = shape_profile(amp)
    p2 = shape_profile(amp * 3.0 + 2.0 * 0)  # pure gain change
    assert p1.shape == (10, 13)
    assert np.abs(p1 - p2).max() < 0.35


def test_window_features_dim_and_missing_nodes():
    cfg = load_config(None)
    g = geometry_features(cfg)
    v = window_features([None, None, None], [None] * 3, g)
    assert v.shape == (FEATURE_DIM,)
    assert np.all(v[:90] == 0)


def test_synthetic_samples():
    d = synth_scene_samples(None, 24, seed=1)
    assert d["X"].shape == (24, FEATURE_DIM)
    assert np.isfinite(d["X"]).all()
    assert ((d["xy"] >= 0) & (d["xy"] <= 1)).all()
    assert d["zone"].max() < 9
    total = d["n_move"] + d["n_still"]
    assert total.max() <= 3
    assert np.array_equal(d["act"] == 0, total == 0)      # empty <=> nobody
    assert not d["hx"][total != 1].any()                   # xy only labelled for one person
