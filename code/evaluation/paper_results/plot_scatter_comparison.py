"""Response-level scatter: (helpfulness, harmlessness) for all 1000 prompts per method."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

base = Path(__file__).resolve().parent.parent

HELP_KEY = "help_score (high better)"
HARM_KEY = "harm_score (low better)"

methods = {
    "PARM": {
        "path": base / "results/PARM_0.5help_0.5harm/reward_result.json",
        "color": "purple", "marker": "^", "ms": 4, "zorder": 2,
    },
    "GenARM": {
        "path": base / "results_genarm_1000/GenARM_0.5help_0.5harm/reward_result.json",
        "color": "steelblue", "marker": "s", "ms": 4, "zorder": 3,
    },
    "MOD": {
        "path": base / "results_mod/MOD_0.5help_0.5harm/reward_result.json",
        "color": "darkorange", "marker": "P", "ms": 4, "zorder": 3,
    },
    "EPEC+GenARM": {
        "path": base / "results_epec_1000/EPEC_GenARM_0.5help_0.5harm_tau0.1_k50/reward_result.json",
        "color": "crimson", "marker": "o", "ms": 4, "zorder": 4,
    },
    "EPEC+PARM": {
        "path": base / "results_epec_parm/EPEC_PARM_0.5help_0.5harm_tau0.1_k50/reward_result.json",
        "color": "forestgreen", "marker": "D", "ms": 4, "zorder": 5,
    },
}

scores = {}
for name, cfg in methods.items():
    raw = json.load(open(cfg["path"]))[:1000]
    h = np.array([r[HELP_KEY] for r in raw])
    s = np.array([-r[HARM_KEY] for r in raw])  # negate so higher = safer
    scores[name] = (h, s)

fig = plt.figure(figsize=(10, 8))
gs = GridSpec(4, 4, hspace=0.05, wspace=0.05)

ax_main = fig.add_subplot(gs[1:, :3])
ax_top = fig.add_subplot(gs[0, :3], sharex=ax_main)
ax_right = fig.add_subplot(gs[1:, 3], sharey=ax_main)

for name, cfg in methods.items():
    h, s = scores[name]
    ax_main.scatter(h, s, c=cfg["color"], marker=cfg["marker"], s=cfg["ms"]**2,
                    alpha=0.2, zorder=cfg["zorder"], rasterized=True)
    ax_main.scatter(h.mean(), s.mean(), c=cfg["color"], marker=cfg["marker"],
                    s=150, edgecolors="black", linewidths=1.2, zorder=cfg["zorder"]+10,
                    label=f'{name} (mean: h={h.mean():.1f}, s={s.mean():.1f})')

    # KDE contours
    try:
        xy = np.vstack([h, s])
        kde = gaussian_kde(xy)
        xmin, xmax = h.min()-1, h.max()+1
        ymin, ymax = s.min()-1, s.max()+1
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        levels = np.percentile(zz[zz > 0], [5, 32])
        ax_main.contour(xx, yy, zz, levels=sorted(levels), colors=[cfg["color"]],
                        linewidths=1.2, alpha=0.7, zorder=cfg["zorder"])
    except Exception:
        pass

    # Marginal histograms
    ax_top.hist(h, bins=50, color=cfg["color"], alpha=0.3, density=True)
    ax_right.hist(s, bins=50, color=cfg["color"], alpha=0.3, density=True,
                  orientation="horizontal")

ax_top.tick_params(labelbottom=False)
ax_right.tick_params(labelleft=False)
ax_top.set_ylabel("Density", fontsize=10)
ax_right.set_xlabel("Density", fontsize=10)

ax_main.set_xlabel("Helpfulness Score (higher = better)", fontsize=12)
ax_main.set_ylabel("Harmlessness Score (higher = safer)", fontsize=12)
ax_main.legend(fontsize=9, loc="upper left", framealpha=0.9)
ax_main.grid(True, linestyle="--", alpha=0.3)
ax_main.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax_main.axvline(0, color="gray", linewidth=0.5, linestyle=":")

fig.suptitle("Response-Level Reward Distribution at α=(0.5, 0.5) — 1000 prompts",
             fontsize=13, y=0.98)
plt.tight_layout()

out = Path(__file__).parent / "scatter_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")

# Print stats
print("\n=== Statistics ===")
print(f"{'Method':<15} {'help_mean':>10} {'help_std':>10} {'harm_mean':>10} {'harm_std':>10} {'% safe':>8}")
for name in methods:
    h, s = scores[name]
    pct_safe = (s > 0).mean() * 100
    print(f"{name:<15} {h.mean():>+10.2f} {h.std():>10.2f} {s.mean():>+10.2f} {s.std():>10.2f} {pct_safe:>7.1f}%")
