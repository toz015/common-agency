"""
Plot preference sensitivity comparison: EPEC vs GenARM vs PARM.

For each objective (help, harm, humor), shows how each method's reward score
changes as the preference weight for that objective increases — demonstrating
EPEC's superior preference responsiveness.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir="./results/HH-RLHF"):
    """Load all mean_result.json files."""
    data = {}
    for name in sorted(os.listdir(results_dir)):
        p = os.path.join(results_dir, name, "mean_result.json")
        if os.path.isfile(p):
            d = json.load(open(p))
            # Parse method and alphas from dir name
            parts = name.split("_")
            alpha_start = 0
            for i, part in enumerate(parts):
                if part and part[0].isdigit():
                    alpha_start = i
                    break
            method = "_".join(parts[:alpha_start])
            alphas = {}
            for part in parts[alpha_start:]:
                for obj in ["help", "harm", "humor"]:
                    if part.endswith(obj):
                        alphas[obj] = float(part.replace(obj, ""))
            data[name] = {"method": method, "alphas": alphas, "scores": d}
    return data


def plot_sensitivity(data, output_path="plots/HH-RLHF/preference_sensitivity.png"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    objectives = [("help", "Helpfulness Score", "help"),
                  ("harm", "Harmlessness Score (cost)", "harm"),
                  ("humor", "Humor Score", "humor")]
    methods = {"EPEC": {"color": "#e74c3c", "marker": "o", "ls": "-"},
               "GenARM": {"color": "#3498db", "marker": "s", "ls": "--"},
               "PARM": {"color": "#2ecc71", "marker": "^", "ls": ":"}}

    for ax, (obj, ylabel, score_key) in zip(axes, objectives):
        for method, style in methods.items():
            # Collect (alpha_for_this_obj, score) pairs
            points = []
            for name, d in data.items():
                if d["method"] == method and obj in d["alphas"]:
                    alpha = d["alphas"][obj]
                    score = d["scores"][score_key]
                    points.append((alpha, score))
            points.sort()
            if not points:
                continue
            xs, ys = zip(*points)
            ax.plot(xs, ys, color=style["color"], marker=style["marker"],
                    linestyle=style["ls"], linewidth=2, markersize=7,
                    label=method, alpha=0.85)

        ax.set_xlabel(f"α_{obj}", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(f"Preference Sensitivity: {obj.capitalize()}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")

    # Also save PDF
    pdf_path = output_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved to {pdf_path}")


def plot_radar(data, output_path="plots/HH-RLHF/preference_radar.png"):
    """Radar plot showing range of each score per method."""
    methods = ["EPEC", "GenARM", "PARM"]
    objectives = ["help", "harm", "humor"]
    colors = {"EPEC": "#e74c3c", "GenARM": "#3498db", "PARM": "#2ecc71"}

    # Compute score ranges per method
    ranges = {}
    for method in methods:
        scores = {obj: [] for obj in objectives}
        for name, d in data.items():
            if d["method"] == method:
                for obj in objectives:
                    scores[obj].append(d["scores"][obj])
        ranges[method] = {obj: max(scores[obj]) - min(scores[obj]) for obj in objectives}

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(objectives), endpoint=False).tolist()
    angles += angles[:1]

    for method in methods:
        values = [ranges[method][obj] for obj in objectives]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=method, color=colors[method])
        ax.fill(angles, values, alpha=0.1, color=colors[method])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["Helpfulness\nRange", "Harmlessness\nRange", "Humor\nRange"], fontsize=12)
    ax.set_title("Score Range Across Preferences\n(larger = more responsive)", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    data = load_results()
    plot_sensitivity(data)
    plot_radar(data)
