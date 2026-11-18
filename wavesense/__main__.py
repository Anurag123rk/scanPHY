"""WaveSense 3D command line.

    python -m wavesense serve            # run the edge server + dashboard (Jetson)
    python -m wavesense serve --source sim   # demo without hardware
    python -m wavesense train ...        # train / fine-tune a model (see trainer.py)
    python -m wavesense emulate          # emulate 3 ESP32-S3 nodes over UDP
    python -m wavesense probe            # check which boards are streaming CSI
    python -m wavesense info MODEL.npz   # print a model card
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time


def cmd_serve(a) -> int:
    from .server import run
    return run(a)


def cmd_train(a) -> int:
    from .trainer import train
    return train(a)


def cmd_emulate(a) -> int:
    from . import protocol
    from .config import load_config
    from .simulator import LiveSimulator
    cfg = load_config(a.config)
    sim = LiveSimulator(cfg, rate=a.rate, seed=a.seed)
    host, port = a.target.rsplit(":", 1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    period = 1.0 / a.rate
    nxt = time.monotonic()
    print("emulating %d nodes -> %s:%s at %.0f Hz (Ctrl+C to stop)" % (len(cfg["nodes"]), host, port, a.rate))
    try:
        while True:
            for node_id, iq, rssi in sim.step():
                sock.sendto(protocol.build(node_id, sim.seq, iq, rssi, timestamp_ms=int(sim.t * 1000)),
                            (host, int(port)))
            nxt += period
            time.sleep(max(0.0, nxt - time.monotonic()))
    except KeyboardInterrupt:
        return 0


def cmd_probe(a) -> int:
    from . import protocol
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", a.port))
    sock.settimeout(0.2)
    seen = {}
    end = time.time() + a.seconds
    print("listening on UDP %d for %.0f s ..." % (a.port, a.seconds))
    while time.time() < end:
        try:
            data, addr = sock.recvfrom(4096)
        except socket.timeout:
            continue
        f = protocol.parse(data)
        if f is None:
            continue
        s = seen.setdefault(f.node_id, {"ip": addr[0], "frames": 0, "rssi": [], "sc": f.n_subcarriers})
        s["frames"] += 1
        s["rssi"].append(f.rssi)
    if not seen:
        print("no CSI frames received. Check the node target IP / port and the firewall (UDP %d)." % a.port)
        return 1
    for nid in sorted(seen):
        s = seen[nid]
        print("node %3d  %-15s  %6.1f frames/s  rssi %4.0f dBm  %d subcarriers" % (
            nid, s["ip"], s["frames"] / a.seconds, sum(s["rssi"]) / len(s["rssi"]), s["sc"]))
    return 0


def cmd_info(a) -> int:
    from .model import ScanModel
    m = ScanModel(a.model)
    print(json.dumps({k: v for k, v in m.meta.items() if k != "history"}, indent=2))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="wavesense", description="WaveSense 3D WiFi CSI human sensing")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the sensing server and dashboard")
    s.add_argument("--source", choices=["esp32", "sim"], default="esp32")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--udp-port", type=int, default=5005)
    s.add_argument("--http-port", type=int, default=3000)
    s.add_argument("--ws-port", type=int, default=3001, help="extra plain WebSocket port (0 = off)")
    s.add_argument("--config", default="config/room.json")
    s.add_argument("--data", default="data")
    s.add_argument("--model", default=None, help="override the model from the config")
    s.set_defaults(fn=cmd_serve)

    from .trainer import add_args
    t = sub.add_parser("train", help="train / fine-tune a model")
    add_args(t)
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("emulate", help="emulate ESP32-S3 nodes sending CSI over UDP")
    e.add_argument("--target", default="127.0.0.1:5005")
    e.add_argument("--rate", type=float, default=40.0)
    e.add_argument("--config", default="config/room.json")
    e.add_argument("--seed", type=int, default=7)
    e.set_defaults(fn=cmd_emulate)

    pr = sub.add_parser("probe", help="listen for CSI frames and report each node")
    pr.add_argument("--port", type=int, default=5005)
    pr.add_argument("--seconds", type=float, default=5.0)
    pr.set_defaults(fn=cmd_probe)

    i = sub.add_parser("info", help="show a model card")
    i.add_argument("model")
    i.set_defaults(fn=cmd_info)

    a = p.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
