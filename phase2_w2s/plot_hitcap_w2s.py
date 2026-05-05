"""
3-curve cap-hit rate plot: PARM, GenARM (logit-sum baselines, n=300) +
CAGE = EPEC+GenARM (n=300), all on W2S 65B + 7B-ARM stack.

Reads phase2_w2s/hitcap_w2s.json (produced by compute_hitcap_w2s.py).

Style matches plot_3curve_w2s_epec.py (paper Figure 4 colors / markers, white
background, lower-left legend, large fontsize). x = preference α_help, y =
cap-hit rate (%).

Naming convention (paper-aligned):
  EPEC + GenARM   →  CAGE
  GenARM logit-sum →  GenARM
  PARM logit-sum   →  PARM

Usage:
  python phase2_w2s/plot_hitcap_w2s.py            # default → hitcap_w2s.png
  python phase2_w2s/plot_hitcap_w2s.py --out /tmp/foo.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE = Path(__file__).parent
DEFAULT_DATA = BASE / "hitcap_w2s.json"


def plot_method(ax, points, label, marker, color):
    if not points:
        return
    xs = [p["a"] for p in points]
    ys = [p["pct"] for p in points]
    ax.plot(
        xs, ys,
        marker=marker, linestyle="-",
        color=color, linewidth=2.0, markersize=7,
        markerfacecolor=color, markeredgecolor=color,
        zorder=3, label=label,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DEFAULT_DATA),
                   help="path to hitcap_w2s.json")
    p.add_argument("--out",  default=str(BASE / "hitcap_w2s.png"))
    args = p.parse_args()

    data = json.loads(Path(args.data).read_text())
    parm   = data.get("PARM",   [])
    genarm = data.get("GenARM", [])
    cage   = data.get("CAGE",   [])

    print(f"PARM:   {len(parm)} α  | max pct = {max((p['pct'] for p in parm), default=0):.1f}%")
    print(f"GenARM: {len(genarm)} α  | max pct = {max((p['pct'] for p in genarm), default=0):.1f}%")
    print(f"CAGE:   {len(cage)} α  | max pct = {max((p['pct'] for p in cage), default=0):.1f}%")

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Paper Figure 4 convention:
    #   CAGE (EPEC+GenARM) → tab:blue,  marker o
    #   GenARM (logit-sum) → tab:green, marker ^
    #   PARM   (logit-sum) → tab:red,   marker D
    plot_method(ax, cage,   label="CAGE",   marker="o", color="tab:blue")
    plot_method(ax, genarm, label="GenARM", marker="^", color="tab:green")
    plot_method(ax, parm,   label="PARM",   marker="D", color="tab:red")

    ax.set_xlabel(r"Preference $\alpha_{\mathrm{help}}$", fontsize=32)
    ax.set_ylabel("Cap-hit rate (%)", fontsize=32)

    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35, color="gray")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.tick_params(axis="both", which="major", labelsize=18)

    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="lightgray",
        framealpha=1.0,
        fontsize=24,
    )

    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
