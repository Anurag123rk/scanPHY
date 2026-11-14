"""Real-time sensing engine: node buffers -> features -> model -> tracker -> state."""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional

import numpy as np

from . import config as cfgmod
from . import autocal, vitals
from .dataset import baseline_from_json, save_recording
from .features import (ACTIVITIES, N_GROUPS, N_SLOTS, PER_NODE, amplitudes, geometry_features, link_disturbance,
                       make_baseline, motion_level, shape_profile, window_features)
from .model import ScanModel, try_load
from .protocol import CsiFrame

HISTORY_S = 40.0


class NodeBuffer:
    def __init__(self, node_cfg: Dict, slot: int):
        self.cfg = node_cfg
        self.slot = slot
        self.id = int(node_cfg["id"])
        self.t: Deque[float] = deque()
        self.prof: Deque[np.ndarray] = deque()
        self.rssi: Deque[float] = deque()
        self.last_amp: Optional[np.ndarray] = None
        self.last_seen = 0.0
        self.addr: Optional[str] = None
        self.total = 0
        self.lost = 0
        self.last_seq: Optional[int] = None
        self.channel = 0
        self.n_sc = 0

    def push(self, t: float, amp: np.ndarray, rssi: float, seq: Optional[int] = None) -> np.ndarray:
        prof = shape_profile(amp[None, :])[0]
        self.t.append(t)
        self.prof.append(prof)
        self.rssi.append(float(rssi))
        self.last_amp = amp
        self.last_seen = t
        self.total += 1
        if seq is not None:
            if self.last_seq is not None and 0 < seq - self.last_seq < 1000:
                self.lost += seq - self.last_seq - 1
            self.last_seq = seq
        while self.t and self.t[0] < t - HISTORY_S:
            self.t.popleft()
            self.prof.popleft()
            self.rssi.popleft()
        return prof

    def window(self, now: float, seconds: float):
        n = 0
        for ts in reversed(self.t):
            if ts <= now - seconds:
                break
            n += 1
        if n < 5:
            return None
        k = len(self.t) - n
        t = np.fromiter((self.t[i] for i in range(k, len(self.t))), dtype=np.float64, count=n)
        prof = np.stack([self.prof[i] for i in range(k, len(self.prof))])
        rssi = np.fromiter((self.rssi[i] for i in range(k, len(self.rssi))), dtype=np.float32, count=n)
        return {"t": t, "profiles": prof, "rssi": rssi}

    def rate(self, now: float) -> float:
        n = 0
        for ts in reversed(self.t):
            if ts <= now - 2.0:
                break
            n += 1
        return n / 2.0

    def alive(self, now: float) -> bool:
        return now - self.last_seen < 2.0


class Engine:
    def __init__(self, cfg: Dict, root: str, data_dir: str, model_path: Optional[str] = None):
        self.root = root
        self.data_dir = data_dir
        self.cfg = cfg
        self.events: Deque[Dict] = deque(maxlen=60)
        self.on_recording_saved: Optional[Callable[[str], None]] = None
        self.model: Optional[ScanModel] = None
        self.model_error: Optional[str] = None
        self.truth: Optional[Dict] = None
        self.started = time.time()
        self._apply_config(cfg)
        self.baselines: Dict[int, Dict] = {}
        self.baseline_auto: Dict[int, bool] = {}
        self._load_baselines()
        self.load_model(model_path or cfg.get("model"))
        self._reset_tracking()
        self.baseline_job: Optional[Dict] = None
        self.recording: Optional[Dict] = None
        self._empty_streak = 0.0
        self._last_tick = time.time()
        self._last_vitals = 0.0
        self._still_since: Optional[float] = None
        self.history: Deque[List[float]] = deque(maxlen=3600)  # 1 Hz for an hour: t, total, moving, still
        self._last_hist = 0.0
        self.autocal_job: Optional[Dict] = None
        self.autocal_result: Optional[Dict] = None
        geo = os.path.join(root, "models", "geometry.npz")
        self.geometry = autocal.GeometryEstimator(geo) if os.path.exists(geo) else None
        self.vitals: Dict = {"breathing_bpm": None, "breathing_conf": 0.0, "heart_bpm": None, "heart_conf": 0.0}

    # ---- configuration ------------------------------------------------------------------
    def _apply_config(self, cfg: Dict) -> None:
        self.cfg = cfg
        old = getattr(self, "nodes", {})
        self.nodes: Dict[int, NodeBuffer] = {}
        for slot, n in enumerate(cfg["nodes"][:N_SLOTS]):
            buf = old.get(int(n["id"])) or NodeBuffer(n, slot)
            buf.cfg, buf.slot = n, slot
            self.nodes[int(n["id"])] = buf
        self.geom = geometry_features(cfg)
        self.unknown: Dict[int, Dict] = {}

    def set_config(self, cfg: Dict) -> None:
        self._apply_config(cfg)
        self.log("configuration updated")

    def _reset_tracking(self) -> None:
        self.act_probs = np.array([1.0, 0.0, 0.0])
        self.presence = False
        self.xy: Optional[np.ndarray] = None
        self.trail: Deque[List[float]] = deque(maxlen=60)
        self.zone_probs: Optional[np.ndarray] = None
        self.confidence = 0.0
        self.count_probs = (np.eye(4)[0].copy(), np.eye(4)[0].copy())  # (moving, still) over 0..3
        self.people = {"total": 0, "moving": 0, "still": 0}
        self._people_candidate = (dict(self.people), 0.0)

    def log(self, msg: str, level: str = "info") -> None:
        self.events.appendleft({"t": time.time(), "msg": msg, "level": level})

    # ---- model --------------------------------------------------------------------------
    def load_model(self, path: Optional[str]) -> bool:
        if not path:
            self.model = None
            return False
        full = path if os.path.isabs(path) else os.path.join(self.root, path)
        try:
            m = try_load(full)
        except Exception as exc:
            self.model_error = str(exc)
            self.log("model load failed: %s" % exc, "warn")
            return False
        if m is None:
            self.model_error = "model file not found: %s" % path
            self.log(self.model_error, "warn")
            return False
        z = self.cfg["zones"]
        if (m.zone_rows, m.zone_cols) != (z["rows"], z["cols"]):
            self.log("model zone grid %dx%d differs from config %dx%d - retrain recommended"
                     % (m.zone_rows, m.zone_cols, z["rows"], z["cols"]), "warn")
        self.model = m
        self.model_error = None
        self.log("model loaded: %s" % m.name)
        return True

    # ---- baselines ----------------------------------------------------------------------
    def _baseline_file(self) -> str:
        return os.path.join(self.data_dir, "baselines.json")

    def _load_baselines(self) -> None:
        try:
            with open(self._baseline_file(), "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            for k, v in raw.items():
                self.baselines[int(k)] = baseline_from_json(v)
                self.baseline_auto[int(k)] = False
            if raw:
                self.log("empty-room baselines loaded for nodes %s" % ", ".join(sorted(raw)))
        except (OSError, ValueError):
            pass

    def _save_baselines(self) -> None:
        os.makedirs(self.data_dir, exist_ok=True)
        out = {str(k): {"profile": b["profile"].tolist(), "noise": b["noise"].tolist(), "rssi": b["rssi"]}
               for k, b in self.baselines.items() if not self.baseline_auto.get(k)}
        with open(self._baseline_file(), "w", encoding="utf-8") as fh:
            json.dump(out, fh)

    def start_baseline(self, seconds: float = 15.0) -> None:
        now = time.time()
        self.baseline_job = {"start": now, "end": now + seconds, "seconds": seconds}
        self.log("capturing empty-room baseline for %.0f s - please leave the room" % seconds)

    def _finish_baseline(self, now: float) -> None:
        job = self.baseline_job
        self.baseline_job = None
        done = []
        for nid, buf in self.nodes.items():
            w = buf.window(now, now - job["start"])
            if w is None or len(w["t"]) < 20:
                continue
            self.baselines[nid] = make_baseline(w["profiles"], w["rssi"])
            self.baseline_auto[nid] = False
            done.append(str(nid))
        if done:
            self._save_baselines()
            self.log("baseline captured for nodes %s" % ", ".join(done))
        else:
            self.log("baseline failed: no CSI received", "warn")

    # ---- automatic calibration ------------------------------------------------------------
    def start_autocal(self, seconds: float = 120.0) -> None:
        if self.autocal_job is not None:
            raise RuntimeError("auto-calibration already running")
        now = time.time()
        self.autocal_job = {"start": now, "end": now + float(seconds), "seconds": float(seconds), "slots": {}}
        self.autocal_result = None
        self.log("auto-calibration started (%.0f s) - keep the room empty or still for at least 30 s" % seconds)

    def cancel_autocal(self) -> None:
        if self.autocal_job is not None:
            self.autocal_job = None
            self.log("auto-calibration cancelled", "warn")

    def _finish_autocal(self) -> None:
        job = self.autocal_job
        self.autocal_job = None
        res = autocal.analyse(self.cfg, job["slots"], self.geometry)
        for nid, b in res.pop("baselines").items():
            self.baselines[nid] = b
            self.baseline_auto[nid] = False
        self._save_baselines()
        ok = [n for n in res["nodes"] if n.get("ok")]
        res["finished"] = time.time()
        self.autocal_result = res
        if ok:
            self.log("auto-calibration done: baseline set for %d node(s)%s" % (
                len(ok), ", router ~%.1f m from node %d" % (ok[0]["link_m"], ok[0]["id"]) if "link_m" in ok[0] else ""))
        else:
            self.log("auto-calibration failed: no CSI received", "warn")

    # ---- recording ----------------------------------------------------------------------
    def start_recording(self, label: Dict, seconds: float) -> Dict:
        if self.recording is not None:
            raise RuntimeError("a recording is already running")
        act = label.get("activity")
        if act not in ACTIVITIES:
            raise ValueError("activity must be one of %s" % ACTIVITIES)
        now = time.time()
        self.recording = {"label": label, "start": now, "end": now + float(seconds), "seconds": float(seconds),
                          "slots": [{"t": [], "amp": [], "rssi": []} for _ in range(N_SLOTS)]}
        where = " at (%.1f, %.1f)" % (label["x"], label["y"]) if label.get("x") is not None else ""
        self.log("recording '%s'%s for %.0f s" % (act, where, seconds))
        return {"ok": True}

    def cancel_recording(self) -> None:
        if self.recording:
            self.recording = None
            self.log("recording cancelled", "warn")

    def _finish_recording(self) -> None:
        rec = self.recording
        self.recording = None
        frames = sum(len(s["t"]) for s in rec["slots"])
        if frames < 30:
            self.log("recording discarded: no CSI frames received", "warn")
            return
        bl = [self.baselines.get(int(n["id"])) for n in self.cfg["nodes"][:N_SLOTS]]
        path = save_recording(os.path.join(self.data_dir, "recordings"), rec["slots"], rec["label"], self.cfg, bl)
        self.log("recording saved (%d frames)" % frames)
        if self.on_recording_saved:
            self.on_recording_saved(path)

    # ---- ingest -------------------------------------------------------------------------
    def ingest(self, frame: CsiFrame, addr: Optional[str] = None) -> None:
        self.ingest_iq(frame.node_id, frame.iq, frame.rssi, frame.rx_time or time.time(), frame.seq, addr,
                       frame.channel, frame.n_subcarriers, frame.n_antennas)

    def ingest_iq(self, node_id: int, iq: np.ndarray, rssi: float, t: float, seq: Optional[int] = None,
                  addr: Optional[str] = None, channel: int = 0, n_sc: int = 64, n_ant: int = 1) -> None:
        buf = self.nodes.get(int(node_id))
        if buf is None:
            u = self.unknown.setdefault(int(node_id), {"id": int(node_id), "frames": 0})
            u.update(frames=u["frames"] + 1, addr=addr, last_seen=t)
            return
        if n_ant > 1:  # use the first antenna only
            iq = iq[: n_sc * 2]
        amp = amplitudes(iq)[0]
        buf.addr = addr or buf.addr
        buf.channel, buf.n_sc = channel, n_sc
        buf.push(t, amp, rssi, seq)
        if self.autocal_job is not None:
            a = self.autocal_job["slots"].setdefault(buf.id, {"t": [], "amp": [], "rssi": []})
            a["t"].append(t)
            a["amp"].append(amp.astype(np.float16))
            a["rssi"].append(float(rssi))
        if self.recording is not None:
            s = self.recording["slots"][buf.slot]
            s["t"].append(t)
            s["amp"].append(amp.astype(np.float16))
            s["rssi"].append(int(rssi))

    # ---- inference ----------------------------------------------------------------------
    def _heuristic(self, vec: np.ndarray, valid: List[bool]):
        """Model-free fallback: motion threshold + geometric link-disturbance localisation."""
        motion = motion_level(vec)
        dist = link_disturbance(vec)
        noise_ref = []
        for nid, buf in self.nodes.items():
            b = self.baselines.get(nid)
            if b is not None:
                noise_ref.append(float(np.mean(b["noise"])) * 20.0)
        ref = max(np.mean(noise_ref) if noise_ref else 0.25, 0.05)
        devs = [float(np.sqrt(np.mean(vec[i * PER_NODE:i * PER_NODE + N_GROUPS] ** 2))) for i in range(N_SLOTS) if valid[i]]
        moving = motion > 2.0 * ref
        present = moving or (max(devs) if devs else 0.0) > 0.6
        probs = np.array([0.0, 0.0, 0.0])
        probs[2 if moving else (1 if present else 0)] = 0.8
        probs += 0.0667
        xy = None
        if present and any(valid):
            W, D = self.cfg["room"]["width"], self.cfg["room"]["depth"]
            gx, gy = np.meshgrid(np.linspace(0.2, W - 0.2, 40), np.linspace(0.2, D - 0.2, 32))
            pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
            r = np.array([self.cfg["router"]["x"], self.cfg["router"]["y"]])
            obs = dist / (dist.max() + 1e-6)
            err = np.zeros(len(pts))
            for i, n in enumerate(self.cfg["nodes"][:N_SLOTS]):
                if not valid[i]:
                    continue
                a, b = r, np.array([n["x"], n["y"]])
                ab = b - a
                t = np.clip(((pts - a) @ ab) / (ab @ ab + 1e-9), 0, 1)
                d = np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)
                err += (np.exp(-d ** 2 / (2 * 0.9 ** 2)) - obs[i]) ** 2
            p = pts[int(np.argmin(err))]
            xy = np.array([p[0] / W, p[1] / D])
        return probs / probs.sum(), None, xy

    def tick(self, now: Optional[float] = None) -> Dict:
        now = now or time.time()
        dt = max(0.01, now - self._last_tick)
        self._last_tick = now
        if self.baseline_job and now >= self.baseline_job["end"]:
            self._finish_baseline(now)
        if self.recording and now >= self.recording["end"]:
            self._finish_recording()
        if self.autocal_job and now >= self.autocal_job["end"]:
            self._finish_autocal()

        win = self.cfg["window_s"]
        slots, bls, valid = [], [], []
        for n in self.cfg["nodes"][:N_SLOTS]:
            buf = self.nodes[int(n["id"])]
            w = buf.window(now, win) if buf.alive(now) else None
            slots.append(w)
            valid.append(w is not None)
            nid = int(n["id"])
            if nid not in self.baselines and len(buf.t) > 0 and buf.t[-1] - buf.t[0] > 10.0:
                w10 = buf.window(now, 10.0)
                if w10 is not None:
                    self.baselines[nid] = make_baseline(w10["profiles"], w10["rssi"])
                    self.baseline_auto[nid] = True
                    self.log("node %d: provisional baseline - capture an empty-room baseline for accuracy" % nid,
                             "warn")
            bls.append(self.baselines.get(nid))
        while len(slots) < N_SLOTS:
            slots.append(None)
            bls.append(None)
            valid.append(False)

        engine_kind = "idle"
        if any(valid):
            vec = window_features(slots, bls, self.geom)
            if self.model is not None:
                pred = self.model.predict(vec)
                probs, zp, xy = pred["activity"][0], pred["zone"][0], pred["xy"][0]
                counts = (pred["n_move"][0], pred["n_still"][0])
                engine_kind = "model"
            else:
                probs, zp, xy = self._heuristic(vec, valid)
                counts = (np.array([probs[0] + probs[1], probs[2], 0, 0]), np.array([probs[0] + probs[2], probs[1], 0, 0]))
                engine_kind = "heuristic"
            a = min(1.0, dt / 0.35)
            self.act_probs = (1 - a) * self.act_probs + a * probs
            self.count_probs = tuple((1 - a) * cp + a * np.asarray(c) for cp, c in zip(self.count_probs, counts))
            p_present = 1.0 - self.act_probs[0]
            if not self.presence and p_present > 0.6:
                self.presence = True
                self.log("presence detected")
            elif self.presence and p_present < 0.4:
                self.presence = False
                self.log("room is empty")
                self.trail.clear()
            if zp is not None:
                self.zone_probs = zp if self.zone_probs is None else (1 - a) * self.zone_probs + a * zp
            if self.presence and xy is not None:
                W, D = self.cfg["room"]["width"], self.cfg["room"]["depth"]
                target = np.array([xy[0] * W, xy[1] * D])
                conf = float(self.zone_probs.max()) if self.zone_probs is not None else 0.35
                alpha = min(1.0, dt * (1.0 + 3.0 * conf))
                self.xy = target if self.xy is None else self.xy + alpha * (target - self.xy)
                self.confidence = conf * p_present
                if not self.trail or math.hypot(self.trail[-1][0] - self.xy[0], self.trail[-1][1] - self.xy[1]) > 0.08:
                    self.trail.append([round(float(self.xy[0]), 3), round(float(self.xy[1]), 3)])
            elif not self.presence:
                self.xy = None
                self.confidence = 0.0
            # Slow baseline adaptation while the room is confidently empty and calm.
            if self.act_probs[0] > 0.9:
                self._empty_streak += dt
                if self._empty_streak > 20.0:
                    for i, n in enumerate(self.cfg["nodes"][:N_SLOTS]):
                        b = self.baselines.get(int(n["id"]))
                        if b is not None and slots[i] is not None:
                            b["profile"] = 0.998 * b["profile"] + 0.002 * slots[i]["profiles"].mean(axis=0)
                            b["rssi"] = 0.998 * b["rssi"] + 0.002 * float(slots[i]["rssi"].mean())
            else:
                self._empty_streak = 0.0
            motion = motion_level(vec)
        else:
            self.act_probs = np.array([1.0, 0.0, 0.0])
            self.presence = False
            self.xy = None
            motion = 0.0

        activity = ACTIVITIES[int(np.argmax(self.act_probs))] if any(valid) else "no-signal"
        self._update_people(now, activity)
        if activity != "still":
            self._still_since = None
        elif self._still_since is None:
            self._still_since = now
        if now - self._last_vitals > 1.0:
            self._last_vitals = now
            self._update_vitals(now, activity)
        return self._state(now, activity, motion, engine_kind)

    def _update_vitals(self, now: float, activity: str) -> None:
        if activity != "still":
            self.vitals = {"breathing_bpm": None, "breathing_conf": 0.0, "heart_bpm": None, "heart_conf": 0.0}
            return
        # Only analyse the stretch since the person became still (movement swamps breathing).
        span = min(32.0, now - (self._still_since or now) + 1.0)
        best = None
        if span < 13.0:
            self.vitals = {"breathing_bpm": None, "breathing_conf": 0.0, "heart_bpm": None, "heart_conf": 0.0,
                           "settling": round(13.0 - span, 1)}
            return
        for buf in self.nodes.values():
            w = buf.window(now, span)
            if w is None:
                continue
            v = vitals.estimate(w["t"], w["profiles"], seconds=span)
            if v["breathing_bpm"] is not None and (best is None or v["breathing_conf"] > best["breathing_conf"]):
                best = v
        if best is not None:
            self.vitals = best

    def _update_people(self, now: float, activity: str) -> None:
        """Debounced people counts (moving / still / total) + 1 Hz history."""
        if not self.presence:
            cand = {"total": 0, "moving": 0, "still": 0}
        else:
            mv = int(np.argmax(self.count_probs[0]))
            st = int(np.argmax(self.count_probs[1]))
            if mv + st == 0:  # model unsure about the count but someone is there
                mv, st = (1, 0) if activity == "moving" else (0, 1)
            cand = {"total": mv + st, "moving": mv, "still": st}
        if cand != self._people_candidate[0]:
            self._people_candidate = (cand, now)
        elif cand != self.people and now - self._people_candidate[1] > 2.0:
            self.people = cand
            self.log("people: %d (%d moving, %d still)" % (cand["total"], cand["moving"], cand["still"]))
        if now - self._last_hist >= 1.0:
            self._last_hist = now
            p = self.people
            self.history.append([round(now, 1), p["total"], p["moving"], p["still"]])

    def history_summary(self, minutes: float = 10.0) -> Dict:
        now = time.time()
        pts = [h for h in self.history if h[0] >= now - minutes * 60]
        if not pts:
            return {"points": [], "peak": 0, "occupied_pct": 0.0, "avg_people": 0.0, "minutes": minutes}
        tot = np.array([h[1] for h in pts], dtype=np.float32)
        return {"points": pts, "peak": int(tot.max()), "occupied_pct": round(float((tot > 0).mean() * 100), 1),
                "avg_people": round(float(tot.mean()), 2), "minutes": minutes,
                "peak_moving": int(max(h[2] for h in pts)), "peak_still": int(max(h[3] for h in pts))}

    def _state(self, now: float, activity: str, motion: float, engine_kind: str) -> Dict:
        nodes = []
        for n in self.cfg["nodes"][:N_SLOTS]:
            buf = self.nodes[int(n["id"])]
            amp = buf.last_amp
            heat = None
            if amp is not None and buf.alive(now):
                la = np.log1p(amp)
                lo, hi = float(la.min()), float(la.max())
                heat = np.round((la - lo) / (hi - lo + 1e-6) * 255).astype(int).tolist()
            nodes.append({
                "id": buf.id, "name": n.get("name"), "x": n["x"], "y": n["y"], "z": n.get("z", 1.0),
                "alive": buf.alive(now), "addr": buf.addr, "pps": round(buf.rate(now), 1),
                "rssi": round(float(np.mean(list(buf.rssi)[-20:])), 1) if buf.rssi else None,
                "frames": buf.total, "lost": buf.lost, "channel": buf.channel,
                "baseline": ("auto" if self.baseline_auto.get(buf.id) else "set") if buf.id in self.baselines else None,
                "heat": heat,
            })
        zones = None
        if self.zone_probs is not None:
            z = self.cfg["zones"]
            names = cfgmod.zone_names(z["rows"], z["cols"])
            if len(self.zone_probs) == len(names):
                zones = {"rows": z["rows"], "cols": z["cols"], "names": names,
                         "probs": [round(float(p), 3) for p in self.zone_probs]}
        zone_name = zones["names"][int(np.argmax(zones["probs"]))] if (zones and self.presence) else None
        x = y = None
        if self.xy is not None:
            x, y = round(float(self.xy[0]), 3), round(float(self.xy[1]), 3)
        rec = None
        if self.recording:
            rec = {"label": self.recording["label"], "elapsed": round(now - self.recording["start"], 1),
                   "seconds": self.recording["seconds"]}
        bjob = None
        if self.baseline_job:
            bjob = {"elapsed": round(now - self.baseline_job["start"], 1), "seconds": self.baseline_job["seconds"]}
        return {
            "type": "state",
            "t": now,
            "uptime": round(now - self.started, 1),
            "engine": engine_kind,
            "presence": self.presence,
            "activity": activity,
            "activity_probs": {k: round(float(v), 3) for k, v in zip(ACTIVITIES, self.act_probs)},
            "motion": round(float(motion), 3),
            "x": x, "y": y,
            "confidence": round(float(self.confidence), 3),
            "uncertainty_m": round(0.25 + 1.5 * (1.0 - float(self.confidence)), 2),
            "zone": zone_name,
            "zones": zones,
            "trail": list(self.trail),
            "vitals": self.vitals,
            "nodes": nodes,
            "unknown_nodes": list(self.unknown.values()),
            "recording": rec,
            "baseline_job": bjob,
            "model": self.model.info() if self.model else {"name": None, "error": self.model_error},
            "truth": self.truth,
            "events": list(self.events)[:25],
            "people": dict(self.people,
                           p_moving=[round(float(v), 3) for v in self.count_probs[0]],
                           p_still=[round(float(v), 3) for v in self.count_probs[1]]),
            "autocal": ({"elapsed": round(now - self.autocal_job["start"], 1), "seconds": self.autocal_job["seconds"],
                         "frames": sum(len(v["t"]) for v in self.autocal_job["slots"].values())}
                        if self.autocal_job else None),
            "autocal_result": self.autocal_result,
        }
