import sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
plt.rcParams["font.family"] = "DejaVu Serif"
metrics = ["Presence", "Activity", "Exact count", "Count ±1"]
general = {1: [88.8, 74.8, 45.2, 84.0], 2: [97.6, 85.3, 57.7, 94.6], 3: [99.4, 89.2, 62.6, 96.5]}
special = {1: [95.4, 87.4, 56.5, 93.7], 2: [98.8, 92.1, 64.2, 95.5], 3: [99.8, 95.4, 68.7, 97.8]}
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), dpi=220, sharey=True)
x = np.arange(len(metrics)); w = 0.38
for ax, n in zip(axes, (1, 2, 3)):
    b1 = ax.bar(x - w / 2, general[n], w, color="white", edgecolor="black", hatch="///", label="single general model")
    b2 = ax.bar(x + w / 2, special[n], w, color="black", edgecolor="black", label="per-board-count model")
    for b, v in list(zip(b2, special[n])):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, "%.0f" % v, ha="center", fontsize=6.5)
    ax.set_title("%d board%s" % (n, "" if n == 1 else "s"), fontsize=9.5, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=7, rotation=20)
    ax.set_ylim(40, 105); ax.grid(axis="y", lw=0.4, ls=":")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel("Accuracy on held-out simulated rooms (%)", fontsize=7.5)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.07))
plt.tight_layout(); plt.savefig(sys.argv[1], bbox_inches="tight", facecolor="white")
