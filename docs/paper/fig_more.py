import json, sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/user/scanPHY")
from wavesense.config import load_config
from wavesense.simulator import Scene
from wavesense.features import amplitudes, shape_profile
plt.rcParams["font.family"] = "DejaVu Serif"
out = sys.argv[1]

# ---------- Fig: CSI patterns (simulated, one link) ----------
cfg = load_config("/home/user/scanPHY/config/room.single-node.example.json")
rng = np.random.default_rng(5)
sc = Scene(cfg, rng)
rate = 40
T1, T2, T3 = 10 * rate, 10 * rate, 20 * rate
body = sc.random_body()
H_empty = sc.channel(0, [], T1)
t2 = np.arange(T2) / rate
walk = np.column_stack([1.0 + 2.0 * t2 / t2[-1], 2.0 + 0.3 * np.sin(t2)])
H_walk = sc.channel(0, [(walk, 0.01 * np.sin(2 * math.pi * 1.8 * t2), body)], T2)
t3 = np.arange(T3) / rate
sit = np.tile([[2.2, 2.05]], (T3, 1))
H_sit = sc.channel(0, [(sit, 0.006 * np.sin(2 * math.pi * 0.25 * t3), body)], T3)
iq, rssi = sc.quantize(0, np.vstack([H_empty, H_walk, H_sit]))
amp = amplitudes(iq)
prof = shape_profile(amp)
t = np.arange(len(amp)) / rate
base = prof[:T1].mean(axis=0)
dev = prof - base
k = int(0.25 * rate)
smooth = np.stack([np.convolve(dev[:, j], np.ones(k) / k, mode="same") for j in range(dev.shape[1])], axis=1)
w = rate
motion = np.array([dev[max(0, i - w):i + 1].std(axis=0).mean() * 20 for i in range(len(dev))])
# breathing signal: first principal component of the sitting segment, band-pass 0.1-0.6 Hz
seg = dev[T1 + T2:] - dev[T1 + T2:].mean(axis=0)
_, _, vt = np.linalg.svd(seg, full_matrices=False)
comp = seg @ vt[0]
F = np.fft.rfft(comp); f = np.fft.rfftfreq(len(comp), 1 / rate)
F[(f < 0.1) | (f > 0.6)] = 0
breath = np.fft.irfft(F, len(comp))
fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(9, 5.6), dpi=220, sharex=True,
                                 gridspec_kw={"height_ratios": [1.3, 0.8, 0.8]})
lim = np.percentile(np.abs(smooth), 99)
a1.imshow(np.abs(smooth.T), aspect="auto", cmap="Greys", origin="lower", extent=[0, t[-1], 0.5, 13.5], vmin=0, vmax=lim)
a1.set_ylabel("Band (1-13)", fontsize=8)
a2.plot(t, motion, color="black", lw=1.0)
a2.set_ylabel("Motion score", fontsize=8)
a3.plot(t[T1 + T2:], breath / np.abs(breath).max(), color="black", lw=1.2)
a3.set_ylabel("Breathing\nsignal", fontsize=8); a3.set_yticks([]); a3.set_xlabel("Time (s)", fontsize=8)
for a in (a1, a2, a3):
    a.tick_params(labelsize=7)
    for x in (10, 20):
        a.axvline(x, color="black", ls="--", lw=0.8)
for x, lab in ((5, "empty room"), (15, "person walking"), (30, "person sitting still (breathing 15 per minute)")):
    a1.text(x, 14.2, lab, ha="center", fontsize=7.5, weight="bold")
a3.set_xlim(0, t[-1])
plt.tight_layout(); plt.savefig(out + "/fig_csi.png", bbox_inches="tight", facecolor="white"); plt.close()

# ---------- Fig: training curves (real training logs) ----------
fig, (b1, b2) = plt.subplots(1, 2, figsize=(9, 3.0), dpi=220)
styles = {1: "-", 2: "--", 3: ":"}
for n in (1, 2, 3):
    h = json.load(open("/home/user/scanPHY/models/wavesense-%dnode.json" % n))["history"]
    ep = [r["epoch"] for r in h]
    b1.plot(ep, [r["loss"] for r in h], styles[n], color="black", lw=1.3, label="%d board%s" % (n, "" if n == 1 else "s"))
    b2.plot(ep, [100 * r["activity_acc"] for r in h], styles[n], color="black", lw=1.3, label="%d board%s" % (n, "" if n == 1 else "s"))
b1.set_title("Training loss", fontsize=9, weight="bold"); b2.set_title("Validation activity accuracy (%)", fontsize=9, weight="bold")
for b in (b1, b2):
    b.set_xlabel("Epoch", fontsize=8); b.tick_params(labelsize=7); b.grid(lw=0.4, ls=":")
    for s in ("top", "right"): b.spines[s].set_visible(False)
b2.legend(fontsize=7, frameon=False, loc="lower right")
plt.tight_layout(); plt.savefig(out + "/fig_training.png", bbox_inches="tight", facecolor="white"); plt.close()

# ---------- Fig: real serial log from the hardware test (network details masked) ----------
lines = [  # verbatim from the real hardware test (firmware was still called scanPHY then)
    "I (394) scanphy: scanPHY node firmware starting",
    "I (397) scanphy: configuration loaded from NVS",
    "I (397) scanphy: node 1 -> x.x.x.x:5005, ssid '<hotspot>', 50 Hz",
    "I (13396) scanphy: csi cb 256, sent 183, dropped 10, heap 269948",
    "I (18412) scanphy: csi cb 507, sent 350, dropped 10, heap 269948",
    "I (23413) scanphy: csi cb 757, sent 532, dropped 10, heap 270044",
    "I (28429) scanphy: csi cb 1006, sent 716, dropped 10, heap 269948",
]
fig = plt.figure(figsize=(9, 1.9), dpi=220)
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
ax.add_patch(plt.Rectangle((0.005, 0.02), 0.99, 0.96, transform=ax.transAxes, fc="#f2f2f2", ec="black", lw=0.8))
for i, l in enumerate(lines):
    ax.text(0.02, 0.88 - i * 0.125, l, family="DejaVu Sans Mono", fontsize=8.6, transform=ax.transAxes, va="center")
plt.savefig(out + "/fig_serial.png", facecolor="white"); plt.close()
print("ok")
