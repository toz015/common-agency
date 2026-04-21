"""
High-quality Pareto front plot for meeting presentation.
Reproduces PARM paper Figure 3 style.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

RESULTS_DIR = "./results_hh"


def load_all_results():
    data = []
    for name in sorted(os.listdir(RESULTS_DIR)):
        p = os.path.join(RESULTS_DIR, name, "mean_result.json")
        if not os.path.isfile(p):
            continue
        d = json.load(open(p))
        parts = name.split("_")
        method = parts[0]
        alphas = {}
        for part in parts[1:]:
            for obj in ["help", "harm", "humor"]:
                if part.endswith(obj):
                    alphas[obj] = float(part.replace(obj, ""))
        data.append({
            "name": name, "method": method,
            "alpha_help": alphas.get("help", 0),
            "alpha_harm": alphas.get("harm", 0),
            "alpha_humor": alphas.get("humor", 0),
            "help": d["help"], "harm": d["harm"], "humor": d["humor"],
        })
    return data


def main():
    data = load_all_results()
    methods = ["GenARM", "EPEC", "PARM"]
    colors = {"EPEC": "#e74c3c", "GenARM": "#3498db", "PARM": "#2ecc71"}
    markers = {"EPEC": "o", "GenARM": "s", "PARM": "^"}
    labels = {"EPEC": "EPEC (ours)", "GenARM": "GenARM (Xu et al., 2025)", "PARM": "PARM (Lin et al., 2025)"}
    sizes = {"EPEC": 100, "GenARM": 80, "PARM": 90}

    fig = plt.figure(figsize=(22, 6))

    # (a) 3D
    ax3d = fig.add_subplot(141, projection="3d")
    for method in methods:
        pts = [(d["help"], d["harm"], d["humor"]) for d in data if d["method"] == method]
        if not pts:
            continue
        xs, ys, zs = zip(*pts)
        ax3d.scatter(xs, ys, zs, c=colors[method], marker=markers[method],
                     s=sizes[method], label=labels[method], alpha=0.9,
                     edgecolors="white", linewidths=0.8, zorder=3)
    ax3d.set_xlabel("Helpfulness", fontsize=11, labelpad=8)
    ax3d.set_ylabel("Harmlessness", fontsize=11, labelpad=8)
    ax3d.set_zlabel("Humor", fontsize=11, labelpad=8)
    ax3d.set_title("(a) 3D Visualization", fontsize=14, fontweight="bold", pad=10)
    ax3d.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax3d.view_init(elev=20, azim=135)
    ax3d.tick_params(labelsize=9)

    # 2D projections
    proj_configs = [
        ("help", "humor", "(b) Helpfulness vs. Humor", 142),
        ("help", "harm", "(c) Helpfulness vs. Harmlessness", 143),
        ("harm", "humor", "(d) Harmlessness vs. Humor", 144),
    ]
    obj_labels = {"help": "Helpfulness", "harm": "Harmlessness", "humor": "Humor"}

    for x_obj, y_obj, title, subplot_idx in proj_configs:
        ax = fig.add_subplot(subplot_idx)
        for method in methods:
            method_data = [d for d in data if d["method"] == method]
            xs = [d[x_obj] for d in method_data]
            ys = [d[y_obj] for d in method_data]

            # Plot points
            ax.scatter(xs, ys, c=colors[method], marker=markers[method],
                       s=sizes[method], label=labels[method], alpha=0.85,
                       edgecolors="white", linewidths=0.8, zorder=3)

            # Connect with line (sorted by x)
            if len(xs) > 1:
                pts = sorted(zip(xs, ys))
                px, py = zip(*pts)
                ax.plot(px, py, color=colors[method], linewidth=1.8, alpha=0.4, zorder=2)

        ax.set_xlabel(obj_labels[x_obj], fontsize=12)
        ax.set_ylabel(obj_labels[y_obj], fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=8.5, framealpha=0.9)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=10)

    plt.tight_layout(w_pad=3)
    plt.savefig("pareto_hh_rlhf_meeting.png", dpi=200, bbox_inches="tight")
    plt.savefig("pareto_hh_rlhf_meeting.pdf", bbox_inches="tight")
    print("Saved pareto_hh_rlhf_meeting.png / .pdf")


if __name__ == "__main__":
    main()
