"""UDP wire format for CSI frames sent by the ESP32-S3 nodes.

WaveSense 3D v1 frame (little endian), 24-byte header followed by CSI samples:

    off  size  field
    0    4     magic            0x59485053 ("SPHY")
    4    1     version          1
    5    1     node_id
    6    1     n_antennas       (1 on ESP32-S3)
    7    1     flags
    8    2     n_subcarriers
    10   2     channel
    12   4     seq
    16   4     timestamp_ms     node uptime in ms
    20   1     rssi (int8, dBm)
    21   1     noise_floor (int8, dBm)
    22   2     reserved
    24   ...   int8 pairs [imag, real] per subcarrier (raw ESP-IDF order)

For convenience the receiver also understands the 20-byte "0xC5110001" frame
emitted by boards that are still flashed with the older RuView firmware, so an
existing deployment keeps working while you migrate the boards to WaveSense 3D firmware.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional

import numpy as np

MAGIC = 0x59485053
VERSION = 1
HEADER = struct.Struct("<IBBBBHHIIbbH")
HEADER_SIZE = HEADER.size  # 24

LEGACY_MAGIC = 0xC5110001
LEGACY_HEADER = struct.Struct("<IBBHIIbbBB")
LEGACY_HEADER_SIZE = LEGACY_HEADER.size  # 20


@dataclass
class CsiFrame:
    node_id: int
    seq: int
    rssi: int
    noise_floor: int
    channel: int
    n_subcarriers: int
    n_antennas: int
    timestamp_ms: int
    iq: np.ndarray  # int8, shape (n_antennas * n_subcarriers * 2,)
    rx_time: float = 0.0


def _freq_to_channel(freq_mhz: int) -> int:
    if 2412 <= freq_mhz <= 2472:
        return (freq_mhz - 2407) // 5
    if freq_mhz == 2484:
        return 14
    if freq_mhz >= 5000:
        return (freq_mhz - 5000) // 5
    return 0


def parse(data: bytes, rx_time: float = 0.0) -> Optional[CsiFrame]:
    """Parse one UDP datagram. Returns None for anything that is not a CSI frame."""
    if len(data) < 4:
        return None
    magic = struct.unpack_from("<I", data, 0)[0]
    if magic == MAGIC and len(data) >= HEADER_SIZE:
        (_, ver, node, n_ant, _flags, n_sc, chan, seq, ts, rssi, noise, _r) = HEADER.unpack_from(data, 0)
        if ver != VERSION:
            return None
        payload = data[HEADER_SIZE:]
    elif magic == LEGACY_MAGIC and len(data) >= LEGACY_HEADER_SIZE:
        (_, node, n_ant, n_sc, freq, seq, rssi, noise, _a, _b) = LEGACY_HEADER.unpack_from(data, 0)
        chan, ts = _freq_to_channel(freq), 0
        payload = data[LEGACY_HEADER_SIZE:]
    else:
        return None
    n_ant = max(1, n_ant)
    need = n_ant * n_sc * 2
    if n_sc == 0 or need > len(payload):
        # Tolerate firmware that reports a sample count but sends fewer bytes.
        n_sc = len(payload) // (2 * n_ant)
        need = n_ant * n_sc * 2
        if n_sc < 8:
            return None
    iq = np.frombuffer(payload[:need], dtype=np.int8).copy()
    return CsiFrame(node, seq, rssi, noise, chan, n_sc, n_ant, ts, iq, rx_time)


def build(node_id: int, seq: int, iq: np.ndarray, rssi: int, noise_floor: int = -95,
          channel: int = 6, timestamp_ms: int = 0) -> bytes:
    """Encode a WaveSense 3D v1 frame (used by the simulator and the tests)."""
    iq = np.asarray(iq, dtype=np.int8).ravel()
    n_sc = iq.size // 2
    head = HEADER.pack(MAGIC, VERSION, node_id & 0xFF, 1, 0, n_sc, channel,
                       seq & 0xFFFFFFFF, timestamp_ms & 0xFFFFFFFF,
                       int(max(-128, min(127, rssi))), int(max(-128, min(127, noise_floor))), 0)
    return head + iq.tobytes()
