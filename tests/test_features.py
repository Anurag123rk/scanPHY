import numpy as np

from scanphy.config import load_config
from scanphy.features import FEATURE_DIM, amplitudes, geometry_features, shape_profile, window_features
from scanphy.simulator import synth_scene_samples


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
    X, act, xy, zone, dims = synth_scene_samples(None, 12, seed=1)
    assert X.shape == (12, FEATURE_DIM)
    assert np.isfinite(X).all()
    assert ((xy >= 0) & (xy <= 1)).all()
    assert zone.max() < 9
