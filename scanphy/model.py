"""Dependency-free (NumPy only) inference runtime for scanPHY models.

A model is a single `.npz` file holding the weights, the input normalisation and a JSON
metadata blob. It can be trained anywhere (laptop, cloud, or on the Jetson itself with
CUDA) and copied to the Jetson; inference does not need PyTorch.

Network (identical to scanphy.trainer.Net):
    x -> (x - mu) / sigma
      -> Linear(256) -> LayerNorm -> ReLU
      -> Linear(256) -> LayerNorm -> ReLU
      -> Linear(128) -> ReLU
      -> heads: activity(3) | zone(rows*cols) | xy(2, sigmoid, normalised room coords)
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np

from .features import ACTIVITIES, FEATURE_DIM, FEATURE_VERSION

LAYER_KEYS = ["l0.w", "l0.b", "n0.w", "n0.b", "l1.w", "l1.b", "n1.w", "n1.b", "l2.w", "l2.b",
              "act.w", "act.b", "zone.w", "zone.b", "xy.w", "xy.b"]


def _ln(x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    m = x.mean(axis=-1, keepdims=True)
    v = x.var(axis=-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class ScanModel:
    def __init__(self, path: str):
        self.path = path
        with np.load(path, allow_pickle=False) as z:
            self.meta: Dict = json.loads(str(z["meta"]))
            self.mu = z["mu"].astype(np.float32)
            self.sigma = z["sigma"].astype(np.float32)
            self.p = {k: z[k].astype(np.float32) for k in LAYER_KEYS}
        if self.meta.get("feature_version") != FEATURE_VERSION:
            raise ValueError("model %s was trained for feature version %s (runtime is %s)"
                             % (path, self.meta.get("feature_version"), FEATURE_VERSION))
        if self.mu.shape[0] != FEATURE_DIM:
            raise ValueError("model input size %d != %d" % (self.mu.shape[0], FEATURE_DIM))
        self.zone_rows = int(self.meta["zones"]["rows"])
        self.zone_cols = int(self.meta["zones"]["cols"])

    @property
    def name(self) -> str:
        return os.path.splitext(os.path.basename(self.path))[0]

    def _lin(self, x: np.ndarray, k: str) -> np.ndarray:
        return x @ self.p[k + ".w"].T + self.p[k + ".b"]

    def predict(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        x = np.atleast_2d(np.asarray(x, dtype=np.float32))
        h = (x - self.mu) / self.sigma
        h = np.maximum(_ln(self._lin(h, "l0"), self.p["n0.w"], self.p["n0.b"]), 0)
        h = np.maximum(_ln(self._lin(h, "l1"), self.p["n1.w"], self.p["n1.b"]), 0)
        h = np.maximum(self._lin(h, "l2"), 0)
        return {
            "activity": _softmax(self._lin(h, "act")),
            "zone": _softmax(self._lin(h, "zone")),
            "xy": 1.0 / (1.0 + np.exp(-self._lin(h, "xy"))),
        }

    def info(self) -> Dict:
        return {
            "name": self.name,
            "path": self.path,
            "created": self.meta.get("created"),
            "kind": self.meta.get("kind"),
            "metrics": self.meta.get("metrics", {}),
            "samples": self.meta.get("samples", {}),
            "device": self.meta.get("device"),
            "zones": self.meta.get("zones"),
            "activities": self.meta.get("activities", ACTIVITIES),
        }


def save_npz(path: str, params: Dict[str, np.ndarray], mu: np.ndarray, sigma: np.ndarray, meta: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    meta = dict(meta, feature_version=FEATURE_VERSION, feature_dim=FEATURE_DIM, activities=ACTIVITIES)
    arrays = {k: np.asarray(params[k], dtype=np.float32) for k in LAYER_KEYS}
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), mu=mu.astype(np.float32),
                        sigma=sigma.astype(np.float32), **arrays)
    os.replace(tmp, path)


def try_load(path: Optional[str]) -> Optional[ScanModel]:
    if not path or not os.path.exists(path):
        return None
    return ScanModel(path)
