"""
Per-preference Δ-EPEC heatmap: visualize where EPEC outperforms its non-EPEC counterpart.

For each preference vector α and each reward-model family (GenARM, PARM) we compute
the scalarized utility u(α) = α · q̂(α) under per-axis [0,1] normalization, then plot

     Δ_family(α) = u_{EPEC+family}(α) - u_{family}(α)

as a diverging heatmap (green = EPEC wins, red = EPEC loses), with one row per
family and one column per preference. Columns are sorted by mean Δ across the
two families so the EPEC-favorable regime is on the left.

Reads:    code/evaluation/results/HH-RLHF/<method>_..../mean_result.json
Writes:   code/evaluation/plots/HH-RLHF/epec_delta_heatmap.png / .pdf
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE, "results", "HH-RLHF")
PLOTS_DIR = os.path.join(BASE, "plots", "HH-RLHF")
PNG = os.path.join(PLOTS_DIR, "epec_delta_heatmap.png")
PDF = os.path.join(PLOTS_DIR, "epec_delta_heatmap.pdf")

METHODS = ["GenARM", "EPEC_GenARM", "PARM", "EPEC_PARM"]


def load_data():
    """Returns: pref_tuple -> {method: (help, harm, humor) raw scores}."""
    by_pref = {}
    for name in os.listdir(RESULTS_DIR):
        for m in METHODS:
            if not name.startswith(m + "_"):
                continue
            tail = name[len(m) + 1:]
            try:
                h = float(tail.split("help")[0])
                s = float(tail.split("help_")[1].split("harm")[0])
                u = float(tail.split("harm_")[1].split("humor")[0])
            except (IndexError, ValueError):
                break
            mp = os.path.join(RESULTS_DIR, name, "mean_result.json")
            if not os.path.isfile(mp):
                break
            r = json.load(open(mp))
            by_pref.setdefault((h, s, u), {})[m] = (r["help"], r["harm"], r["humor"])
            break
    return by_pref


def main():
    by_pref = load_data()
    if not by_pref:
        raise SystemExit("No mean_result.json files found.")

    all_pts = np.array([s for d in by_pref.values() for s in d.values()])
    gmin = all_pts.min(axis=0)
    gmax = all_pts.max(axis=0)
    span = (gmax - gmin) + 1e-12

    prefs = sorted(by_pref.keys())
    families = [("GenARM", "EPEC_GenARM"), ("PARM", "EPEC_PARM")]
    family_labels = ["EPEC+GenARM − GenARM", "EPEC+PARM − PARM"]

    deltas = np.full((len(families), len(prefs)), np.nan)
    for j, p in enumerate(prefs):
        a = np.array(p)
        for i, (base_m, epec_m) in enumerate(families):
            scores = by_pref[p]
            if base_m not in scores or epec_m not in scores:
                continue
            q_base = (np.array(scores[base_m]) - gmin) / span
            q_epec = (np.array(scores[epec_m]) - gmin) / span
            u_base = float(np.dot(a, q_base))
            u_epec = float(np.dot(a, q_epec))
            deltas[i, j] = u_epec - u_base

    # Sort prefs by mean Δ descending (so EPEC-favorable regime is on the left)
    mean_d = np.nanmean(deltas, axis=0)
    order = np.argsort(-mean_d)
    deltas = deltas[:, order]
    prefs_sorted = [prefs[k] for k in order]

    # Plot — taller rows for readability, wide enough for 16 columns
    fig_w = max(15, 0.78 * len(prefs_sorted) + 5)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))
    vmax = float(np.nanmax(np.abs(deltas)))
    im = ax.imshow(deltas, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    # Cell value annotations
    for i in range(deltas.shape[0]):
        for j in range(deltas.shape[1]):
            v = deltas[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=12, color="#555")
                continue
            color = "white" if abs(v) > 0.55 * vmax else "#111"
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    # Column labels (preference vectors)
    col_labels = [f"({h:g}, {s:g}, {u:g})" for h, s, u in prefs_sorted]
    ax.set_xticks(range(len(prefs_sorted)))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=11)

    # Row labels
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels(family_labels, fontsize=13, fontweight="bold")

    # Win-rate annotation per row (placed in axis fraction coords so it can't get clipped)
    for i in range(deltas.shape[0]):
        wins = int(np.nansum(deltas[i] > 0))
        total = int(np.sum(~np.isnan(deltas[i])))
        ax.annotate(
            f"{wins}/{total} wins\nmean Δ = {np.nanmean(deltas[i]):+.4f}",
            xy=(1.012, i), xycoords=("axes fraction", "data"),
            ha="left", va="center", fontsize=11, color="#222",
            annotation_clip=False,
        )

    ax.set_xlabel("Preference vector α = (α_help, α_harm, α_humor)  —  sorted by mean Δ-EPEC, descending",
                  fontsize=12)
    ax.set_title("Δ-EPEC: scalarized utility of EPEC vs non-EPEC, per preference\n"
                 "(green = EPEC wins, red = EPEC loses)",
                 fontsize=14, fontweight="bold", pad=14)

    # Colorbar (small, well-separated from the wins-annotation)
    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.16)
    cbar.set_label("u(EPEC) − u(non-EPEC)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    plt.subplots_adjust(left=0.13, right=0.84, top=0.80, bottom=0.30)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(PNG, dpi=150, bbox_inches="tight")
    plt.savefig(PDF, bbox_inches="tight")
    print(f"Saved: {PNG}")
    print(f"Saved: {PDF}")

    # Also print the sorted Δ table
    print("\nSorted Δ table:")
    print(f"{'pref':<22}{'Δ EPEC+GenARM':>16}{'Δ EPEC+PARM':>16}{'mean Δ':>12}")
    for j, p in enumerate(prefs_sorted):
        d_g = deltas[0, j]; d_p = deltas[1, j]
        avg = float(np.nanmean(deltas[:, j]))
        s_g = "  --" if np.isnan(d_g) else f"{d_g:+.4f}"
        s_p = "  --" if np.isnan(d_p) else f"{d_p:+.4f}"
        print(f"({p[0]:g},{p[1]:g},{p[2]:g})".ljust(22) + f"{s_g:>16}{s_p:>16}{avg:>+12.4f}")


if __name__ == "__main__":
    main()
