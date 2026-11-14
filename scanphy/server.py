"""scanPHY edge server: UDP CSI receiver + sensing engine + HTTP/WebSocket dashboard.

    UDP  :5005   CSI frames from the ESP32-S3 nodes
    HTTP :3000   dashboard + REST API, WebSocket at /ws
    WS   :3001   plain WebSocket stream (same JSON) for your own clients / Three.js viewers
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import socket
import time
from typing import Dict, Optional, Set

from aiohttp import WSMsgType, web

from . import APP_NAME, __version__
from . import config as cfgmod
from . import protocol
from .autotrain import AutoTrainer
from .dataset import list_recordings
from .engine import Engine
from .model import ScanModel

log = logging.getLogger("scanphy")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEB = os.path.join(os.path.dirname(__file__), "web")
AUTO_TRAIN_MIN_RECORDINGS = 8
AUTO_TRAIN_QUIET_S = 30.0


class UdpReceiver(asyncio.DatagramProtocol):
    def __init__(self, hub: "Hub"):
        self.hub = hub

    def datagram_received(self, data: bytes, addr) -> None:
        frame = protocol.parse(data, time.time())
        if frame is not None:
            self.hub.engine.ingest(frame, addr[0])
            self.hub.udp_frames += 1
        else:
            self.hub.udp_bad += 1


class Hub:
    def __init__(self, args):
        self.args = args
        self.config_path = os.path.abspath(args.config)
        self.data_dir = os.path.abspath(args.data)
        os.makedirs(os.path.join(self.data_dir, "recordings"), exist_ok=True)
        os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
        self.cfg = cfgmod.load_config(self.config_path)
        if not os.path.exists(self.config_path):
            cfgmod.save_config(self.cfg, self.config_path)
        self.engine = Engine(self.cfg, ROOT, self.data_dir, args.model)
        self.engine.on_recording_saved = self._recording_saved
        self.trainer = AutoTrainer(ROOT, self.data_dir, self.config_path, lambda: self.cfg,
                                   self._activate_model, self.engine.log)
        self.clients: Set[web.WebSocketResponse] = set()
        self.udp_frames = 0
        self.udp_bad = 0
        self.last_recording_end = 0.0
        self.sim = None
        self.state: Dict = {}

    # ---- callbacks ----------------------------------------------------------------------
    def _recording_saved(self, path: str) -> None:
        self.trainer.recording_saved()
        self.last_recording_end = time.time()

    def _activate_model(self, path: str) -> None:
        if self.engine.load_model(path):
            rel = os.path.relpath(path, ROOT)
            self.cfg["model"] = rel
            cfgmod.save_config(self.cfg, self.config_path)

    # ---- loops --------------------------------------------------------------------------
    async def tick_loop(self) -> None:
        while True:
            try:
                st = self.engine.tick()
                st["training"] = self.trainer.state()
                st["udp"] = {"frames": self.udp_frames, "bad": self.udp_bad, "port": self.args.udp_port,
                             "source": self.args.source}
                self.state = st
                await self.broadcast(st)
                await self._maybe_autotrain()
            except Exception:  # keep the loop alive no matter what
                log.exception("tick failed")
            await asyncio.sleep(0.1)

    async def _maybe_autotrain(self) -> None:
        at = self.cfg.get("autotrain", {})
        if not at.get("enabled", True) or self.trainer.status == "running":
            return
        if self.engine.recording is not None:
            return
        if self.trainer.pending_recordings >= AUTO_TRAIN_MIN_RECORDINGS and \
                time.time() - self.last_recording_end > AUTO_TRAIN_QUIET_S:
            await self.trainer.start(reason="auto: %d new recordings" % self.trainer.pending_recordings)

    async def sim_loop(self) -> None:
        from .simulator import LiveSimulator
        rate = 40.0
        self.sim = LiveSimulator(self.cfg, rate=rate)
        period = 1.0 / rate
        nxt = time.monotonic()
        while True:
            now = time.time()
            for node_id, iq, rssi in self.sim.step():
                self.engine.ingest_iq(node_id, iq, rssi, now, self.sim.seq, "sim", 6, 64)
                self.udp_frames += 1
            self.engine.truth = self.sim.truth()
            nxt += period
            await asyncio.sleep(max(0.0, nxt - time.monotonic()))

    async def broadcast(self, msg: Dict) -> None:
        if not self.clients:
            return
        data = json.dumps(msg, separators=(",", ":"))
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_str(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


# ---- HTTP handlers ------------------------------------------------------------------------

def _json(data, status=200):
    return web.json_response(data, status=status, dumps=lambda o: json.dumps(o, default=str))


def _err(msg: str, status: int = 400):
    return _json({"ok": False, "error": msg}, status)


def make_app(hub: Hub) -> web.Application:
    app = web.Application(client_max_size=64 * 1024 * 1024)
    r = app.router

    async def index(_req):
        return web.FileResponse(os.path.join(WEB, "index.html"))

    async def ws_handler(req):
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(req)
        hub.clients.add(ws)
        try:
            await ws.send_str(json.dumps({"type": "hello", "app": APP_NAME, "version": __version__,
                                          "config": hub.cfg}))
            async for msg in ws:
                if msg.type == WSMsgType.ERROR:
                    break
        finally:
            hub.clients.discard(ws)
        return ws

    async def status(_req):
        return _json(hub.state)

    async def get_config(_req):
        return _json(hub.cfg)

    async def post_config(req):
        try:
            patch = await req.json()
            cfg = cfgmod.update_config(hub.cfg, patch)
        except (ValueError, KeyError, TypeError) as exc:
            return _err(str(exc))
        hub.cfg = cfg
        hub.engine.set_config(cfg)
        if hub.sim is not None:
            from .simulator import LiveSimulator
            hub.sim = LiveSimulator(cfg)
        cfgmod.save_config(cfg, hub.config_path)
        await hub.broadcast({"type": "config", "config": cfg})
        return _json({"ok": True, "config": cfg})

    async def baseline(req):
        body = await _body(req)
        hub.engine.start_baseline(float(body.get("seconds", 15)))
        return _json({"ok": True})

    async def record_start(req):
        body = await _body(req)
        label = {"activity": body.get("activity", "still"), "x": body.get("x"), "y": body.get("y"),
                 "note": body.get("note", "")}
        for k in ("moving_count", "still_count"):
            if body.get(k) is not None:
                label[k] = max(0, min(3, int(body[k])))
        if label["x"] is not None:
            label["x"], label["y"] = float(label["x"]), float(label["y"])
        try:
            return _json(hub.engine.start_recording(label, float(body.get("seconds", 10))))
        except (RuntimeError, ValueError) as exc:
            return _err(str(exc))

    async def autocal_start(req):
        body = await _body(req)
        try:
            hub.engine.start_autocal(float(body.get("seconds", 120)))
        except RuntimeError as exc:
            return _err(str(exc))
        return _json({"ok": True})

    async def autocal_cancel(_req):
        hub.engine.cancel_autocal()
        return _json({"ok": True})

    async def autocal_apply(_req):
        res = hub.engine.autocal_result
        if not res or not res.get("suggestion"):
            return _err("no auto-calibration result to apply")
        sug = res["suggestion"]
        patch = {"router": sug["router"]}
        if sug.get("room_estimated"):
            patch["room"] = sug["room"]
        try:
            cfg = cfgmod.update_config(hub.cfg, patch)
        except (ValueError, KeyError) as exc:
            return _err(str(exc))
        hub.cfg = cfg
        hub.engine.set_config(cfg)
        cfgmod.save_config(cfg, hub.config_path)
        await hub.broadcast({"type": "config", "config": cfg})
        hub.engine.log("auto-calibration layout applied")
        return _json({"ok": True, "config": cfg})

    async def history(req):
        try:
            minutes = float(req.query.get("minutes", 10))
        except ValueError:
            minutes = 10.0
        return _json(hub.engine.history_summary(max(1.0, min(60.0, minutes))))

    async def record_cancel(_req):
        hub.engine.cancel_recording()
        return _json({"ok": True})

    async def recordings(_req):
        recs = list_recordings(os.path.join(hub.data_dir, "recordings"))
        return _json([{"file": m["file"], "id": m["id"], "created": m["created"], "label": m["label"],
                       "frames": m["frames"], "size": m["size"]} for m in recs])

    async def recording_delete(req):
        name = os.path.basename(req.match_info["name"])
        path = os.path.join(hub.data_dir, "recordings", name)
        if not name.endswith(".npz") or not os.path.exists(path):
            return _err("not found", 404)
        os.remove(path)
        return _json({"ok": True})

    async def recordings_clear(_req):
        d = os.path.join(hub.data_dir, "recordings")
        n = 0
        for f in os.listdir(d):
            if f.endswith(".npz"):
                os.remove(os.path.join(d, f))
                n += 1
        hub.engine.log("deleted %d recordings" % n, "warn")
        return _json({"ok": True, "deleted": n})

    async def train(req):
        body = await _body(req)
        try:
            res = await hub.trainer.start(epochs=body.get("epochs"), reason=body.get("reason", "manual"),
                                          synthetic=int(body.get("synthetic", 24000)),
                                          use_recordings=bool(body.get("use_recordings", True)))
        except RuntimeError as exc:
            return _err(str(exc))
        return _json(res)

    async def train_stop(_req):
        await hub.trainer.stop()
        return _json({"ok": True})

    async def models(_req):
        out = []
        d = os.path.join(ROOT, "models")
        am = getattr(hub.engine, "active_model", None)
        active = os.path.abspath(am.path) if am is not None else ""
        for f in sorted(os.listdir(d)):
            if not f.endswith(".npz"):
                continue
            p = os.path.join(d, f)
            try:
                info = ScanModel(p).info()
            except Exception as exc:
                info = {"name": f[:-4], "error": str(exc)}
            info.update(file=f, size=os.path.getsize(p), active=os.path.abspath(p) == active,
                        auto=hub.cfg.get("model") in (None, "", "auto"),
                        onnx=os.path.exists(p[:-4] + ".onnx"))
            info.pop("path", None)
            out.append(info)
        return _json(out)

    async def model_activate(req):
        body = await _body(req)
        name = os.path.basename(body.get("file", ""))
        if name == "auto":
            hub.engine.load_model("auto")
            hub.cfg["model"] = "auto"
            cfgmod.save_config(hub.cfg, hub.config_path)
            return _json({"ok": True})
        path = os.path.join(ROOT, "models", name)
        if not os.path.exists(path):
            return _err("model not found", 404)
        hub._activate_model(path)
        return _json({"ok": hub.engine.model is not None and hub.engine.model.name == name[:-4]})

    async def model_upload(req):
        reader = await req.multipart()
        field = await reader.next()
        if field is None or not field.filename or not field.filename.endswith(".npz"):
            return _err("upload a .npz model file")
        name = os.path.basename(field.filename)
        dest = os.path.join(ROOT, "models", name)
        tmp = dest + ".upload"
        with open(tmp, "wb") as fh:
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                fh.write(chunk)
        try:
            ScanModel(tmp)
        except Exception as exc:
            os.remove(tmp)
            return _err("invalid model: %s" % exc)
        shutil.move(tmp, dest)
        hub.engine.log("model uploaded: %s" % name)
        return _json({"ok": True, "file": name})

    async def model_download(req):
        name = os.path.basename(req.match_info["name"])
        path = os.path.join(ROOT, "models", name)
        if not os.path.exists(path):
            return _err("not found", 404)
        return web.FileResponse(path, headers={"Content-Disposition": "attachment; filename=%s" % name})

    async def model_delete(req):
        name = os.path.basename(req.match_info["name"])
        if name == os.path.basename(hub.cfg.get("model") or "") or \
                name in ("scanphy-1node.npz", "scanphy-2node.npz", "scanphy-3node.npz"):
            return _err("cannot delete the active or a built-in model")
        path = os.path.join(ROOT, "models", name)
        if not os.path.exists(path):
            return _err("not found", 404)
        for ext in (".npz", ".onnx", ".json"):
            p = path[:-4] + ext
            if os.path.exists(p):
                os.remove(p)
        return _json({"ok": True})

    async def room_glb(_req):
        p = os.path.join(hub.data_dir, "room.glb")
        if not os.path.exists(p):
            return _err("no room.glb uploaded", 404)
        return web.FileResponse(p)

    async def room_glb_upload(req):
        reader = await req.multipart()
        field = await reader.next()
        if field is None:
            return _err("no file")
        dest = os.path.join(hub.data_dir, "room.glb")
        with open(dest, "wb") as fh:
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                fh.write(chunk)
        hub.cfg["room_glb"] = "/api/room.glb"
        cfgmod.save_config(hub.cfg, hub.config_path)
        await hub.broadcast({"type": "config", "config": hub.cfg})
        return _json({"ok": True})

    r.add_get("/", index)
    r.add_get("/ws", ws_handler)
    r.add_get("/api/status", status)
    r.add_get("/api/config", get_config)
    r.add_post("/api/config", post_config)
    r.add_post("/api/baseline", baseline)
    r.add_post("/api/record", record_start)
    r.add_post("/api/record/cancel", record_cancel)
    r.add_post("/api/autocal", autocal_start)
    r.add_post("/api/autocal/cancel", autocal_cancel)
    r.add_post("/api/autocal/apply", autocal_apply)
    r.add_get("/api/history", history)
    r.add_get("/api/recordings", recordings)
    r.add_delete("/api/recordings", recordings_clear)
    r.add_delete("/api/recordings/{name}", recording_delete)
    r.add_post("/api/train", train)
    r.add_post("/api/train/stop", train_stop)
    r.add_get("/api/models", models)
    r.add_post("/api/models/activate", model_activate)
    r.add_post("/api/models/upload", model_upload)
    r.add_get("/api/models/{name}", model_download)
    r.add_delete("/api/models/{name}", model_delete)
    r.add_get("/api/room.glb", room_glb)
    r.add_post("/api/room.glb", room_glb_upload)
    r.add_static("/", WEB, show_index=False)
    return app


def make_ws_only_app(hub: Hub, ws_handler_app: web.Application) -> web.Application:
    app = web.Application()

    async def handler(req):
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(req)
        hub.clients.add(ws)
        try:
            async for _ in ws:
                pass
        finally:
            hub.clients.discard(ws)
        return ws

    app.router.add_get("/", handler)
    app.router.add_get("/ws", handler)
    return app


async def _body(req) -> Dict:
    try:
        return await req.json()
    except Exception:
        return {}


def _local_ips():
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return sorted(ips) or ["127.0.0.1"]


async def _main(args) -> None:
    hub = Hub(args)
    loop = asyncio.get_running_loop()
    if args.source == "esp32":
        await loop.create_datagram_endpoint(lambda: UdpReceiver(hub), local_addr=(args.host, args.udp_port))
    app = make_app(hub)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, args.host, args.http_port).start()
    if args.ws_port:
        ws_runner = web.AppRunner(make_ws_only_app(hub, app), access_log=None)
        await ws_runner.setup()
        await web.TCPSite(ws_runner, args.host, args.ws_port).start()
    ip = _local_ips()[0]
    print("")
    print("  scanPHY %s  -  WiFi CSI human sensing" % __version__)
    print("  ----------------------------------------------")
    print("  source     : %s" % ("ESP32-S3 nodes on UDP %d" % args.udp_port if args.source == "esp32"
                                 else "built-in simulator"))
    print("  dashboard  : http://%s:%d" % (ip, args.http_port))
    if args.ws_port:
        print("  websocket  : ws://%s:%d" % (ip, args.ws_port))
    print("  model      : %s" % (hub.engine.model.name if hub.engine.model else
                                 "auto (%d per-board-count models%s)" % (
                                     len(hub.engine.specialists), " + general" if hub.engine.general else "")))
    print("  node target: provision boards with --target-ip %s --target-port %d" % (ip, args.udp_port))
    print("")
    tasks = [asyncio.ensure_future(hub.tick_loop())]
    if args.source == "sim":
        tasks.append(asyncio.ensure_future(hub.sim_loop()))
    await asyncio.gather(*tasks)


def run(args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        pass
    return 0
