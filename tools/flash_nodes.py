#!/usr/bin/env python3
"""Flash + provision every connected ESP32-S3 in one go, then verify they stream.

Boards are detected on USB, flashed with the WaveSense 3D firmware and given incrementing
node ids (1, 2, 3 ...). Finally the script listens on the UDP port for a few seconds
and reports which nodes are sending CSI (run it on the Jetson, with the server stopped,
or pass --no-verify).

    python tools/flash_nodes.py --ssid "MyWiFi" --password "secret" --target-ip 192.168.1.50
"""

from __future__ import annotations

import argparse
import getpass
import os
import socket
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provision import provision  # noqa: E402

# USB vendor ids: Espressif native USB, Silicon Labs CP210x, WCH CH34x, FTDI.
KNOWN_VIDS = {0x303A: "Espressif USB", 0x10C4: "CP210x", 0x1A86: "CH34x", 0x0403: "FTDI"}


def detect_ports():
    try:
        from serial.tools import list_ports
    except ImportError:
        print("pyserial missing: pip install esptool", file=sys.stderr)
        return []
    out = []
    for p in sorted(list_ports.comports(), key=lambda x: x.device):
        if p.vid in KNOWN_VIDS:
            out.append((p.device, KNOWN_VIDS[p.vid], p.serial_number or ""))
    return out


def local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def verify(port: int, seconds: float) -> dict:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("0.0.0.0", port))
    except OSError:
        print("UDP %d is busy (is `wavesense serve` running?) - check the dashboard instead." % port)
        return {}
    sock.settimeout(0.3)
    seen = {}
    end = time.time() + seconds
    while time.time() < end:
        try:
            data, addr = sock.recvfrom(4096)
        except socket.timeout:
            continue
        if len(data) >= 24 and struct.unpack_from("<I", data)[0] in (0x59485053, 0xC5110001):
            node = data[5] if struct.unpack_from("<I", data)[0] == 0x59485053 else data[4]
            seen.setdefault(node, [addr[0], 0])[1] += 1
    return seen


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ssid")
    p.add_argument("--password")
    p.add_argument("--target-ip", default=None, help="Jetson IP (default: this machine's LAN IP)")
    p.add_argument("--target-port", type=int, default=5005)
    p.add_argument("--rate", type=int, default=50)
    p.add_argument("--first-id", type=int, default=1)
    p.add_argument("--config-only", action="store_true", help="only rewrite the config, keep the firmware")
    p.add_argument("--firmware-dir", default=None)
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--verify-seconds", type=float, default=15.0)
    a = p.parse_args()

    ports = detect_ports()
    if not ports:
        print("no ESP32-S3 boards found on USB. Use a data-capable USB-C cable (the port marked USB or UART).")
        return 1
    print("found %d board(s):" % len(ports))
    for i, (dev, kind, sn) in enumerate(ports):
        print("  node %d  <-  %s  (%s %s)" % (a.first_id + i, dev, kind, sn))
    ssid = a.ssid or input("WiFi SSID: ").strip()
    password = a.password if a.password is not None else getpass.getpass("WiFi password: ")
    target = a.target_ip or local_ip()
    print("server target: %s:%d" % (target, a.target_port))

    failed = []
    for i, (dev, _, _) in enumerate(ports):
        try:
            provision(dev, a.first_id + i, ssid, password, target, a.target_port, a.rate,
                      flash_firmware=not a.config_only, firmware_dir=a.firmware_dir)
        except Exception as exc:  # keep going with the other boards
            print("  !! %s failed: %s" % (dev, exc))
            failed.append(dev)
    if failed:
        print("failed: %s" % ", ".join(failed))
    if a.no_verify:
        return 1 if failed else 0
    print("waiting %.0f s for the nodes to join WiFi and stream ..." % a.verify_seconds)
    seen = verify(a.target_port, a.verify_seconds)
    expected = set(range(a.first_id, a.first_id + len(ports)))
    for nid in sorted(expected | set(seen)):
        if nid in seen:
            ip, n = seen[nid]
            print("  node %d  OK   %s  %.0f frames/s" % (nid, ip, n / a.verify_seconds))
        else:
            print("  node %d  --   no frames (check SSID/password, 2.4 GHz band, firewall on UDP %d)"
                  % (nid, a.target_port))
    return 0 if expected <= set(seen) and not failed else 1


if __name__ == "__main__":
    sys.exit(main())
