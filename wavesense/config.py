"""Room / deployment configuration.

Coordinates are in metres with the origin at one floor corner of the room:
    x -> along the room width, y -> along the room depth, z -> height.
In the 3D viewer the floor is the Three.js X-Z plane, so (x, y) maps to (x, 0, y).
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List, Optional

DEFAULT_CONFIG: Dict[str, Any] = {
    "room": {"name": "Lab", "width": 6.0, "depth": 5.0, "height": 2.8},
    # Position of the WiFi router / access point the nodes are connected to.
    "router": {"x": 0.3, "y": 2.5, "z": 1.5},
    # Up to three ESP32-S3 nodes. `id` must match the node id provisioned on the board.
    "nodes": [
        {"id": 1, "name": "NODE-1", "x": 0.4, "y": 0.4, "z": 1.0},
        {"id": 2, "name": "NODE-2", "x": 5.6, "y": 0.4, "z": 1.0},
        {"id": 3, "name": "NODE-3", "x": 3.0, "y": 4.6, "z": 1.0},
    ],
    # Zone grid used by the zone classifier head (rows along depth, cols along width).
    "zones": {"rows": 3, "cols": 3},
    # Sliding window (seconds) used for every inference.
    "window_s": 1.0,
    # Model file (relative to the project root), or "auto" = the built-in model for the
    # number of boards currently streaming (models/wavesense-<n>node.npz).
    "model": "auto",
    # Optional GLB model of the room (served from data/ or an absolute URL).
    "room_glb": None,
    "autotrain": {"enabled": True, "epochs": 60, "mix_synthetic": True},
    # Most people that can realistically be in the room (1-3). Counts never exceed it.
    "max_people": 3,
}

MAX_NODES = 3


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def validate(cfg: Dict[str, Any]) -> Dict[str, Any]:
    room = cfg["room"]
    for key in ("width", "depth", "height"):
        room[key] = float(room[key])
        if not 1.0 <= room[key] <= 50.0:
            raise ValueError("room.%s must be between 1 and 50 metres" % key)
    nodes: List[Dict[str, Any]] = cfg["nodes"]
    if not 1 <= len(nodes) <= MAX_NODES:
        raise ValueError("between 1 and %d nodes must be configured" % MAX_NODES)
    seen = set()
    for i, n in enumerate(nodes):
        n["id"] = int(n["id"])
        if n["id"] in seen:
            raise ValueError("duplicate node id %d" % n["id"])
        seen.add(n["id"])
        n.setdefault("name", "NODE-%d" % n["id"])
        n["x"] = min(max(float(n["x"]), 0.0), room["width"])
        n["y"] = min(max(float(n["y"]), 0.0), room["depth"])
        n["z"] = float(n.get("z", 1.0))
    r = cfg["router"]
    r["x"], r["y"], r["z"] = float(r["x"]), float(r["y"]), float(r.get("z", 1.5))
    z = cfg["zones"]
    z["rows"], z["cols"] = max(1, int(z["rows"])), max(1, int(z["cols"]))
    cfg["window_s"] = min(max(float(cfg["window_s"]), 0.5), 4.0)
    cfg["max_people"] = min(max(int(cfg.get("max_people", 3)), 1), 3)
    return cfg


def load_config(path: Optional[str]) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    return validate(_merge(DEFAULT_CONFIG, data))


def save_config(cfg: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    os.replace(tmp, path)


def update_config(cfg: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    return validate(_merge(cfg, patch))


def zone_names(rows: int, cols: int) -> List[str]:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return ["%s%d" % (letters[r % 26], c + 1) for r in range(rows) for c in range(cols)]


def zone_index(xn: float, yn: float, rows: int, cols: int) -> int:
    """Zone index for normalised coordinates (0..1)."""
    c = min(cols - 1, max(0, int(xn * cols)))
    r = min(rows - 1, max(0, int(yn * rows)))
    return r * cols + c
