"""Breathing (and experimental heart) rate from CSI amplitude micro-variations.

Works for a single person sitting/standing still within ~3 m of a link. The grouped
log-amplitude profile is resampled to a uniform 10 Hz grid, the dominant component
(PCA) is taken and its spectrum is searched for a peak in the respiration band.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

FS = 10.0
BREATH_BAND = (0.1, 0.6)   # 6 - 36 breaths/min
HEART_BAND = (0.8, 2.0)    # 48 - 120 beats/min


def _resample(t: np.ndarray, x: np.ndarray, seconds: float) -> Optional[np.ndarray]:
    if len(t) < 20:
        return None
    t_end = float(t[-1])
    t0 = max(float(t[0]), t_end - seconds)
    if t_end - t0 < 12.0:
        return None
    grid = np.arange(t0, t_end, 1.0 / FS)
    return np.stack([np.interp(grid, t, x[:, j]) for j in range(x.shape[1])], axis=1)


def _peak(sig: np.ndarray, band) -> Optional[Dict[str, float]]:
    n = len(sig)
    nfft = 1 << int(np.ceil(np.log2(n * 8)))
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft((sig - sig.mean()) * win, nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / FS)
    m = (freqs >= band[0]) & (freqs <= band[1])
    if not m.any():
        return None
    bs, bf = spec[m], freqs[m]
    i = int(np.argmax(bs))
    snr = float(bs[i] / (np.median(bs) + 1e-12))
    return {"hz": float(bf[i]), "snr": snr}


def estimate(t: np.ndarray, profiles: np.ndarray, seconds: float = 30.0) -> Dict[str, Optional[float]]:
    """t (T,), profiles (T, G) -> {breathing_bpm, breathing_conf, heart_bpm, heart_conf}."""
    out: Dict[str, Optional[float]] = {"breathing_bpm": None, "breathing_conf": 0.0,
                                       "heart_bpm": None, "heart_conf": 0.0}
    x = _resample(np.asarray(t, dtype=np.float64), np.asarray(profiles, dtype=np.float64), seconds)
    if x is None:
        return out
    x = x - x.mean(axis=0)
    # Breathing uses the raw (mean-removed) signal; for the heart band a 2 s moving
    # average is removed first so the much stronger respiration does not leak in.
    x_b = x
    k = int(2 * FS)
    if len(x) > k:
        kern = np.ones(k) / k
        trend = np.stack([np.convolve(x[:, j], kern, mode="same") for j in range(x.shape[1])], axis=1)
        x_h = x - trend
    else:
        x_h = x
    try:
        _, _, vt = np.linalg.svd(x_b, full_matrices=False)
    except np.linalg.LinAlgError:
        return out
    comp = x_b @ vt[0]
    b = _peak(comp, BREATH_BAND)
    if b and b["snr"] > 4.0:
        out["breathing_bpm"] = round(b["hz"] * 60.0, 1)
        out["breathing_conf"] = float(min(1.0, (b["snr"] - 4.0) / 20.0 + 0.3))
    comp_h = x_h @ vt[0]
    h = _peak(comp_h, HEART_BAND)
    if h and h["snr"] > 6.0:
        # reject obvious breathing harmonics
        if not (b and any(abs(h["hz"] - n * b["hz"]) < 0.04 for n in (2, 3, 4, 5))):
            out["heart_bpm"] = round(h["hz"] * 60.0, 1)
            out["heart_conf"] = float(min(1.0, (h["snr"] - 6.0) / 30.0 + 0.1))
    return out
