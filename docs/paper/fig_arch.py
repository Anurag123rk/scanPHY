import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams["font.family"] = "DejaVu Serif"
fig, ax = plt.subplots(figsize=(10, 4.6), dpi=220)
ax.set_xlim(0, 100); ax.set_ylim(-3, 46); ax.axis("off")
def box(x, y, w, h, title, lines, fill="#ffffff", lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fill, ec="black", lw=lw))
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=8.6, weight="bold")
    for i, l in enumerate(lines):
        ax.text(x + w / 2, y + h - 5.4 - i * 2.75, l, ha="center", va="top", fontsize=6.9)
def arrow(x0, y0, x1, y1, label=None, dy=1.4):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=1.3, color="black"))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, label, ha="center", fontsize=7.2, style="italic")
# sensing layer
box(1, 30, 17, 14, "WiFi router / AP", ["2.4 GHz, 20 MHz", "OFDM, 64 subcarriers", "replies to pings"])
box(1, 3, 17, 22, "ESP32-S3 nodes", ["1-3 boards", "ping AP at 50 Hz", "LLTF CSI capture", "int8 I/Q per subcarrier", "UDP frames (24 B hdr)"], fill="#f0f0f0")
arrow(9.5, 30, 9.5, 25.4)
ax.text(10.4, 27.7, "CSI", fontsize=7.4, style="italic", va="center")
ax.text(8.6, 27.7, "body changes\nmultipath", fontsize=6.2, ha="right", va="center")
# edge server
ax.add_patch(FancyBboxPatch((24, 2), 52, 42, boxstyle="round,pad=0.3,rounding_size=1.5", fc="none", ec="black", lw=1.0, ls="--"))
ax.text(50, 42.4, "NVIDIA Jetson Orin  -  scanPHY edge server", ha="center", fontsize=9.5, weight="bold")
box(26, 25, 14, 14, "Receiver", ["UDP :5005", "per-node", "ring buffers"])
box(43, 25, 14, 14, "DSP features", ["|H| on 52 sc", "gain-invariant", "13 bands, 1 s + 4 s"])
box(60, 25, 14, 14, "Edge-AI model", ["one per board count", "MLP 149-384-384-128", "5 output heads"])
box(26, 5, 14, 15, "Calibration", ["auto baseline", "labelled", "recordings"])
box(43, 5, 14, 15, "Trainer", ["PyTorch / CUDA", "sim + real data", "auto fine-tune"])
box(60, 5, 14, 15, "Tracker", ["smoothing", "counts, zones", "breathing rate"])
arrow(18.3, 14, 25.7, 30)
ax.text(19.0, 22.5, "UDP", fontsize=7.0, style="italic")
arrow(40.3, 32, 42.7, 32); arrow(57.3, 32, 59.7, 32); arrow(67, 24.7, 67, 20.3)
arrow(40.3, 12.5, 42.7, 12.5); arrow(57.3, 15, 61, 24.7)
ax.text(59.8, 21.8, "hot-swap", fontsize=6.4, style="italic", ha="right")
arrow(33, 24.7, 33, 20.3)
# outputs
box(81, 22, 18, 22, "3D dashboard", ["HTTP :3000 / WS :3001", "presence, activity", "people count", "position, zones", "breathing rate"], fill="#f0f0f0")
box(81, 3, 18, 15, "Simulator", ["multipath physics", "domain-randomised", "synthetic pre-training"])
arrow(74.3, 12.5, 80.7, 30)
ax.text(78.6, 23.5, "10 Hz", fontsize=7.0, style="italic", ha="center")
ax.plot([90, 90, 50], [2.7, -1.6, -1.6], color="black", lw=1.3)
arrow(50, -1.6, 50, 4.6)
ax.text(70, -1.0, "synthetic training data", ha="center", fontsize=7.0, style="italic")
plt.savefig(__import__("sys").argv[1], bbox_inches="tight", facecolor="white")
