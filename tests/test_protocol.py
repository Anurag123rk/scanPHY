import struct

import numpy as np

from wavesense import protocol


def test_roundtrip():
    iq = np.arange(-64, 64, dtype=np.int8)
    data = protocol.build(3, 42, iq, -51, channel=11, timestamp_ms=1234)
    f = protocol.parse(data)
    assert f.node_id == 3 and f.seq == 42 and f.rssi == -51 and f.channel == 11
    assert f.n_subcarriers == 64 and f.timestamp_ms == 1234
    assert np.array_equal(f.iq, iq)


def test_legacy_frame():
    iq = np.ones(128, dtype=np.int8)
    head = struct.pack("<IBBHIIbbBB", 0xC5110001, 2, 1, 64, 2437, 7, -60, -95, 0, 0)
    f = protocol.parse(head + iq.tobytes())
    assert f.node_id == 2 and f.channel == 6 and f.n_subcarriers == 64 and f.rssi == -60


def test_garbage_rejected():
    assert protocol.parse(b"") is None
    assert protocol.parse(b"hello world, not csi at all") is None
    assert protocol.parse(struct.pack("<I", protocol.MAGIC) + b"\x00" * 10) is None
