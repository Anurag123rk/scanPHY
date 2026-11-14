"""Background training manager (auto-train on the Jetson).

Runs `python -m scanphy train ... --json-progress` as a subprocess so the sensing loop
never stalls, streams its progress to the dashboard and hot-loads the new model when
training finishes.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional

from .dataset import list_recordings


class AutoTrainer:
    def __init__(self, root: str, data_dir: str, config_path: str, get_cfg: Callable[[], Dict],
                 on_model: Callable[[str], None], log: Callable[..., None]):
        self.root = root
        self.data_dir = data_dir
        self.config_path = config_path
        self.get_cfg = get_cfg
        self.on_model = on_model
        self.log = log
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.status = "idle"
        self.epoch = 0
        self.epochs = 0
        self.history: List[Dict] = []
        self.lines: Deque[str] = deque(maxlen=40)
        self.result: Optional[Dict] = None
        self.started: Optional[float] = None
        self.out_path: Optional[str] = None
        self.reason = ""
        self.pending_recordings = 0

    @property
    def models_dir(self) -> str:
        return os.path.join(self.root, "models")

    @property
    def recordings_dir(self) -> str:
        return os.path.join(self.data_dir, "recordings")

    def state(self) -> Dict:
        return {
            "status": self.status, "epoch": self.epoch, "epochs": self.epochs, "reason": self.reason,
            "history": self.history[-200:], "log": list(self.lines)[-12:], "result": self.result,
            "elapsed": round(time.time() - self.started, 1) if self.started and self.status == "running" else None,
            "pending_recordings": self.pending_recordings,
            "autotrain": bool(self.get_cfg().get("autotrain", {}).get("enabled", True)),
        }

    def recording_saved(self) -> None:
        self.pending_recordings += 1

    async def start(self, epochs: Optional[int] = None, reason: str = "manual", synthetic: int = 24000,
                    use_recordings: bool = True) -> Dict:
        if self.status == "running":
            raise RuntimeError("training already running")
        cfg = self.get_cfg()
        epochs = int(epochs or cfg.get("autotrain", {}).get("epochs", 60))
        has_rec = use_recordings and len(list_recordings(self.recordings_dir)) > 0
        name = time.strftime("room-%Y%m%d-%H%M%S.npz")
        self.out_path = os.path.join(self.models_dir, name)
        init = cfg.get("model")
        init_path = init if (init and os.path.isabs(init)) else os.path.join(self.root, init or "")
        args = [sys.executable, "-m", "scanphy", "train", "--json-progress", "--onnx",
                "--config", self.config_path, "--out", self.out_path, "--epochs", str(epochs),
                "--synthetic", str(int(synthetic) if cfg.get("autotrain", {}).get("mix_synthetic", True) or not has_rec
                                   else 0)]
        if has_rec:
            args += ["--recordings", self.recordings_dir]
        if init and os.path.exists(init_path):
            args += ["--init", init_path]
        self.status, self.epoch, self.epochs, self.history = "running", 0, epochs, []
        self.result, self.reason, self.started = None, reason, time.time()
        self.lines.clear()
        self.pending_recordings = 0
        self.log("training started (%s, %d epochs%s)" % (reason, epochs, ", with your recordings" if has_rec else
                                                          ", synthetic data for your layout"))
        self.proc = await asyncio.create_subprocess_exec(
            *args, cwd=self.root, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        asyncio.ensure_future(self._pump())
        return {"ok": True, "out": os.path.basename(self.out_path)}

    async def _pump(self) -> None:
        assert self.proc and self.proc.stdout
        done_msg = None
        while True:
            raw = await self.proc.stdout.readline()
            if not raw:
                break
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                self.lines.append(line[-300:])
                continue
            ev = msg.get("event")
            if ev == "epoch":
                self.epoch = msg["epoch"]
                v = msg.get("val", {})
                self.history.append({"epoch": msg["epoch"], "loss": round(msg["loss"], 4),
                                     "activity_acc": round(v.get("activity_acc", 0), 4),
                                     "zone_acc": round(v.get("zone_acc", 0), 4),
                                     "xy_mae_m": round(v.get("xy_mae_m", 0), 3)})
            elif ev in ("log", "error"):
                self.lines.append(msg.get("message", ""))
            elif ev == "done":
                done_msg = msg
        code = await self.proc.wait()
        self.proc = None
        if code == 0 and done_msg and self.out_path and os.path.exists(self.out_path):
            self.status = "done"
            self.result = {"model": os.path.basename(self.out_path), "metrics": done_msg.get("metrics"),
                           "seconds": done_msg.get("seconds")}
            self.log("training finished in %.0f s - activating %s" % (done_msg.get("seconds", 0),
                                                                       os.path.basename(self.out_path)))
            self.on_model(self.out_path)
        else:
            self.status = "failed"
            self.result = {"error": self.lines[-1] if self.lines else "exit code %s" % code}
            self.log("training failed: %s" % self.result["error"], "warn")

    async def stop(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            self.status = "failed"
            self.result = {"error": "stopped by user"}
