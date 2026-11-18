#!/usr/bin/env python3
"""Provision (and optionally flash) one WaveSense 3D ESP32-S3 node over USB.

Writes WiFi credentials, server IP/port, node id and CSI rate into the board's NVS
partition. With --flash-firmware the WaveSense 3D firmware is written first.

    pip install esptool esp-idf-nvs-partition-gen
    python tools/provision.py --port /dev/ttyACM0 --node-id 1 \
        --ssid "MyWiFi" --password "secret" --target-ip 192.168.1.50 --flash-firmware
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NVS_OFFSET = "0x9000"
NVS_SIZE = "0x6000"

# (offset, candidate paths relative to the repo) - first existing path wins.
FIRMWARE_PARTS = [
    ("0x0", ["firmware/release/bootloader.bin", "firmware/wavesense-node/build/bootloader/bootloader.bin"]),
    ("0x8000", ["firmware/release/partition-table.bin",
                "firmware/wavesense-node/build/partition_table/partition-table.bin"]),
    ("0x10000", ["firmware/release/wavesense-node.bin", "firmware/wavesense-node/build/wavesense-node.bin"]),
]


def find_firmware(firmware_dir: str = None):
    parts = []
    for offset, cands in FIRMWARE_PARTS:
        if firmware_dir:
            cands = [os.path.join(firmware_dir, os.path.basename(c)) for c in cands[:1]]
        else:
            cands = [os.path.join(ROOT, c) for c in cands]
        path = next((c for c in cands if os.path.exists(c)), None)
        if path is None:
            raise FileNotFoundError(
                "firmware binary not found (looked for %s). Build it with `idf.py build` in "
                "firmware/wavesense-node or download the CI artifact into firmware/release/." % cands[0])
        parts.append((offset, path))
    return parts


def make_nvs(ssid: str, password: str, ip: str, port: int, node: int, rate: int, out: str) -> None:
    if len(ssid) > 32 or len(password) > 64:
        raise ValueError("SSID must be <= 32 and password <= 64 characters")
    csv_path = out + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "type", "encoding", "value"])
        w.writerow(["wavesense", "namespace", "", ""])
        w.writerow(["ssid", "data", "string", ssid])
        w.writerow(["pass", "data", "string", password])
        w.writerow(["ip", "data", "string", ip])
        w.writerow(["port", "data", "u16", str(port)])
        w.writerow(["node", "data", "u8", str(node)])
        w.writerow(["rate", "data", "u16", str(rate)])
    subprocess.check_call([sys.executable, "-m", "esp_idf_nvs_partition_gen", "generate", csv_path, out, NVS_SIZE],
                          stdout=subprocess.DEVNULL)
    os.remove(csv_path)


def provision(port: str, node_id: int, ssid: str, password: str, target_ip: str, target_port: int = 5005,
              rate: int = 50, flash_firmware: bool = False, firmware_dir: str = None, baud: int = 460800) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        nvs_bin = os.path.join(tmp, "nvs.bin")
        make_nvs(ssid, password, target_ip, target_port, node_id, rate, nvs_bin)
        args = [sys.executable, "-m", "esptool", "--chip", "esp32s3", "-p", port, "-b", str(baud),
                "write_flash"]
        if flash_firmware:
            for offset, path in find_firmware(firmware_dir):
                args += [offset, path]
        args += [NVS_OFFSET, nvs_bin]
        print("[%s] node %d -> %s:%d %s" % (port, node_id, target_ip, target_port,
                                           "(firmware + config)" if flash_firmware else "(config only)"))
        subprocess.check_call(args)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True, help="serial port, e.g. /dev/ttyACM0, /dev/ttyUSB0, COM3")
    p.add_argument("--node-id", type=int, required=True)
    p.add_argument("--ssid", required=True)
    p.add_argument("--password", default="")
    p.add_argument("--target-ip", required=True, help="IP address of the Jetson running `wavesense serve`")
    p.add_argument("--target-port", type=int, default=5005)
    p.add_argument("--rate", type=int, default=50, help="CSI frames per second (5-200)")
    p.add_argument("--flash-firmware", action="store_true", help="also flash the WaveSense 3D firmware")
    p.add_argument("--firmware-dir", default=None, help="directory with bootloader.bin, partition-table.bin, "
                                                         "wavesense-node.bin")
    p.add_argument("--baud", type=int, default=460800)
    a = p.parse_args()
    try:
        provision(a.port, a.node_id, a.ssid, a.password, a.target_ip, a.target_port, a.rate,
                  a.flash_firmware, a.firmware_dir, a.baud)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    print("done - reset the board; it will stream CSI to %s:%d" % (a.target_ip, a.target_port))
    return 0


if __name__ == "__main__":
    sys.exit(main())
