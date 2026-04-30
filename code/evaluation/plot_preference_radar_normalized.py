"""
Normalized radar plot of per-method preference responsiveness.

The original `plot_preference_sensitivity.py` builds a radar from RAW per-axis
ranges, which is unfair: helpfulness has a natural width of ~4, harmlessness
~2.7, humor ~0.83. The polygon area is dominated by whichever axis happens
to have the largest raw scale.

This script normalizes each method's per-axis range by the GLOBAL per-axis
range (max-min across all methods), so each axis is in [0, 1] and represents
"fraction of the achievable spread captured by this method on that axis".

Reads:    code/evaluation/results/HH-RLHF/<method>_..../mean_result.json
Writes:   code/evaluation/plots/HH-RLHF/preference_radar_normalized.png / .pdf
          code/evaluation/metrics/HH-RLHF/radar_normalized_values.json

Does NOT modify plot_preference_sensitivity.py.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "HH-RLHF")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots", "HH-RLHF")
METRICS_DIR = os.path.join(os.path.dirname(__file__), "metrics", "HH-RLHF")
PNG_PATH = os.path.join(PLOTS_DIR, "preference_radar_normalized.png")
PDF_PATH = os.path.join(PLOTS_DIR, "preference_radar_normalized.pdf")
JSON_PATH = os.path.join(METRICS_DIR, "radar_normalized_values.json")

METHODS = ["EPEC_PARM", "EPEC_GenARM", "GenARM", "PARM"]
OBJECTIVES = ["help", "harm", "humor"]
COLORS = {"EPEC_PARM": "#e74c3c", "EPEC_GenARM": "#95a5a6",
          "GenARM": "#3498db", "PARM": "#2ecc71"}
LABELS = {"EPEC_PARM": "EPEC+PARM", "EPEC_GenARM": "EPEC+GenARM",
          "GenARM": "GenARM", "PARM": "PARM"}


def load_scores():
    """method -> list of (help, harm, humor) score tuples."""
    by_method = {m: [] for m in METHODS}
    for name in sorted(os.listdir(RESULTS_DIR)):
        mp = os.path.join(RESULTS_DIR, name, "mean_result.json")
        if not os.path.isfile(mp):
            continue
        for m in METHODS:
            if name.startswith(m + "_"):
                d = json.load(open(mp))
                by_method[m].append((d["help"], d["harm"], d["humor"]))
                break
    return by_method


def main():
    by_method = load_scores()
    n_per = {m: len(by_method[m]) for m in METHODS}
    print(f"Points per method: {n_per}")

    # Per-method per-axis range (raw)
    raw_range = {}
    for m in METHODS:
        if not by_method[m]:
            raw_range[m] = {o: 0.0 for o in OBJECTIVES}
            continue
        arr = np.array(by_method[m])  # (n, 3)
        raw_range[m] = {OBJECTIVES[i]: float(arr[:, i].max() - arr[:, i].min())
                        for i in range(3)}

    # Global per-axis range (max range across methods)
    global_range = {o: max(raw_range[m][o] for m in METHODS) for o in OBJECTIVES}
    print(f"\nPer-axis raw ranges per method:")
    print(f"{'method':<14}{'help':>10}{'harm':>10}{'humor':>10}")
    for m in METHODS:
        print(f"{m:<14}{raw_range[m]['help']:>10.4f}{raw_range[m]['harm']:>10.4f}{raw_range[m]['humor']:>10.4f}")
    print(f"{'GLOBAL_MAX':<14}{global_range['help']:>10.4f}{global_range['harm']:>10.4f}{global_range['humor']:>10.4f}")

    # Normalize: each method's range / global max range
    norm_range = {m: {o: (raw_range[m][o] / global_range[o] if global_range[o] > 0 else 0.0)
                      for o in OBJECTIVES} for m in METHODS}

    print(f"\nNormalized (range / global max range) ∈ [0, 1]:")
    print(f"{'method':<14}{'help':>10}{'harm':>10}{'humor':>10}")
    for m in METHODS:
        print(f"{m:<14}{norm_range[m]['help']:>10.4f}{norm_range[m]['harm']:>10.4f}{norm_range[m]['humor']:>10.4f}")

    # Radar
    angles = np.linspace(0, 2 * np.pi, len(OBJECTIVES), endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for m in METHODS:
        vals = [norm_range[m][o] for o in OBJECTIVES]
        vals_closed = vals + vals[:1]
        ax.plot(angles_closed, vals_closed, "o-", linewidth=2.2,
                label=LABELS[m], color=COLORS[m])
        ax.fill(angles_closed, vals_closed, alpha=0.12, color=COLORS[m])

    ax.set_xticks(angles)
    ax.set_xticklabels([
        "Helpfulness\nResponsiveness",
        "Harmlessness\nResponsiveness",
        "Humor\nResponsiveness",
    ], fontsize=12)

    # Axis ticks at 0, 0.25, 0.5, 0.75, 1.0
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_rlabel_position(150)

    ax.set_title(
        "Normalized Preference Responsiveness\n"
        "(per-axis range ÷ best per-axis range across methods)",
        fontsize=13, fontweight="bold", pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.10), fontsize=11)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=150, bbox_inches="tight")
    plt.savefig(PDF_PATH, bbox_inches="tight")
    print(f"\nSaved: {PNG_PATH}")
    print(f"Saved: {PDF_PATH}")

    # Save raw + normalized values for the paper table
    os.makedirs(METRICS_DIR, exist_ok=True)
    out = {
        "n_points_per_method": n_per,
        "global_max_range": global_range,
        "raw_range": raw_range,
        "normalized_range": norm_range,
    }
    with open(JSON_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {JSON_PATH}")


if __name__ == "__main__":
    main()
