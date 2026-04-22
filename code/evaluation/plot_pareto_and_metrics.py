"""
Reproduce PARM paper Figure 3 style plots and compute HV/MIP metrics.

Figure 3 layout:
  (a) 3D Pareto front visualization
  (b) Helpfulness vs Humor (fixing harmlessness weight = 0)
  (c) Helpfulness vs Harmlessness (fixing humor weight = 0)
  (d) Harmlessness vs Humor (fixing helpfulness weight = 0)

Metrics (Appendix D of PARM paper):
  HV  = Hypervolume indicator (larger = better coverage)
  MIP = Mean Inner Product between α and normalized q (larger = better alignment)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


RESULTS_DIR = "./results/HH-RLHF"


def load_all_results():
    """Load all mean_result.json files, parse method/alphas/scores."""
    data = []
    for name in sorted(os.listdir(RESULTS_DIR)):
        p = os.path.join(RESULTS_DIR, name, "mean_result.json")
        if not os.path.isfile(p):
            continue
        d = json.load(open(p))
        # Parse method name (handles EPEC_PARM, EPEC_GenARM, GenARM, PARM)
        parts = name.split("_")
        # Find where alpha values start (first part starting with a digit)
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
        data.append({
            "name": name, "method": method,
            "alpha_help": alphas.get("help", 0),
            "alpha_harm": alphas.get("harm", 0),
            "alpha_humor": alphas.get("humor", 0),
            "help": d["help"],
            "harm": d["harm"],    # higher = less harmful = better
            "humor": d["humor"],  # higher = funnier = better
        })
    return data


def compute_hv(points, ref_point):
    """Compute hypervolume (all objectives maximized)."""
    try:
        from pymoo.indicators.hv import HV
        indicator = HV(ref_point=-np.array(ref_point))
        return indicator(-np.array(points))
    except ImportError:
        pass

    # Monte Carlo fallback
    points = np.array(points)
    ref = np.array(ref_point)
    non_dom = []
    n = len(points)
    for i in range(n):
        dominated = False
        for j in range(n):
            if i != j and all(points[j] >= points[i]) and any(points[j] > points[i]):
                dominated = True
                break
        if not dominated:
            non_dom.append(points[i])
    non_dom = np.array(non_dom) if non_dom else points

    np.random.seed(42)
    n_samples = 500000
    bounds_high = np.max(non_dom, axis=0)
    if any(bounds_high <= ref):
        return 0.0
    samples = np.random.uniform(ref, bounds_high, size=(n_samples, len(ref)))
    total_vol = np.prod(bounds_high - ref)
    is_dominated = np.zeros(n_samples, dtype=bool)
    for pt in non_dom:
        is_dominated |= np.all(samples <= pt, axis=1)
    return total_vol * np.mean(is_dominated)


def compute_mip_normalized(data_for_method, global_min, global_max):
    """
    MIP with per-objective min-max normalization to [0,1].
    MIP = (1/N) Σ α · q_normalized
    """
    n = len(data_for_method)
    if n == 0:
        return 0.0
    mip = 0.0
    for d in data_for_method:
        alpha = np.array([d["alpha_help"], d["alpha_harm"], d["alpha_humor"]])
        q_raw = np.array([d["help"], d["harm"], d["humor"]])
        # Normalize each objective to [0, 1]
        q_norm = (q_raw - global_min) / (global_max - global_min + 1e-12)
        mip += np.dot(alpha, q_norm)
    return mip / n


def plot_pareto_figure3(data, output_path="plots/HH-RLHF/pareto_hh_rlhf.png"):
    """Reproduce Figure 3: (a) 3D, (b-d) 2D projections."""
    methods = ["GenARM", "EPEC_GenARM", "PARM", "EPEC_PARM"]
    colors = {"EPEC_GenARM": "#95a5a6", "GenARM": "#3498db", "PARM": "#2ecc71", "EPEC_PARM": "#e74c3c"}
    markers = {"EPEC_GenARM": "x", "GenARM": "s", "PARM": "^", "EPEC_PARM": "o"}
    labels = {"EPEC_GenARM": "EPEC+GenARM", "GenARM": "GenARM", "PARM": "PARM", "EPEC_PARM": "EPEC+PARM"}

    fig = plt.figure(figsize=(20, 5.5))

    # (a) 3D scatter
    ax3d = fig.add_subplot(141, projection="3d")
    for method in methods:
        pts = [(d["help"], d["harm"], d["humor"]) for d in data if d["method"] == method]
        if not pts:
            continue
        xs, ys, zs = zip(*pts)
        ax3d.scatter(xs, ys, zs, c=colors[method], marker=markers[method],
                     s=70, label=labels[method], alpha=0.85, edgecolors="white", linewidths=0.5)
    ax3d.set_xlabel("Helpfulness", fontsize=10, labelpad=5)
    ax3d.set_ylabel("Harmlessness", fontsize=10, labelpad=5)
    ax3d.set_zlabel("Humor", fontsize=10, labelpad=5)
    ax3d.set_title("(a) 3D Visualization", fontsize=13, fontweight="bold")
    ax3d.legend(fontsize=9, loc="upper left")
    ax3d.view_init(elev=25, azim=135)

    # 2D projections
    proj_configs = [
        # (x_obj, y_obj, fixed_alpha_key, title, subplot)
        ("help", "humor", "alpha_harm", "(b) Helpfulness vs. Humor", 142),
        ("help", "harm", "alpha_humor", "(c) Helpfulness vs. Harmlessness", 143),
        ("harm", "humor", "alpha_help", "(d) Harmlessness vs. Humor", 144),
    ]

    obj_labels = {"help": "Helpfulness", "harm": "Harmlessness", "humor": "Humor"}

    for x_obj, y_obj, fixed_key, title, subplot_idx in proj_configs:
        ax = fig.add_subplot(subplot_idx)
        for method in methods:
            method_data = [d for d in data if d["method"] == method]
            # All points for this method
            xs = [d[x_obj] for d in method_data]
            ys = [d[y_obj] for d in method_data]

            ax.scatter(xs, ys, c=colors[method], marker=markers[method],
                       s=60, label=labels[method], alpha=0.8, edgecolors="white",
                       linewidths=0.5, zorder=3)

            # Connect Pareto front (convex hull boundary)
            if len(xs) > 2:
                pts = sorted(zip(xs, ys))
                px, py = zip(*pts)
                ax.plot(px, py, color=colors[method], linewidth=1.2, alpha=0.4, zorder=2)

        ax.set_xlabel(obj_labels[x_obj], fontsize=12)
        ax.set_ylabel(obj_labels[y_obj], fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved {output_path}")


def main():
    data = load_all_results()
    methods = ["EPEC_PARM", "EPEC_GenARM", "GenARM", "PARM"]

    # --- Plot ---
    plot_pareto_figure3(data)

    # --- Compute metrics ---
    all_help = [d["help"] for d in data]
    all_harm = [d["harm"] for d in data]
    all_humor = [d["humor"] for d in data]

    # HV reference point: worst per objective - margin
    ref_point = [min(all_help) - 1.0, min(all_harm) - 1.0, min(all_humor) - 0.1]

    # MIP normalization bounds (global across all methods)
    global_min = np.array([min(all_help), min(all_harm), min(all_humor)])
    global_max = np.array([max(all_help), max(all_harm), max(all_humor)])

    print("\n" + "=" * 60)
    print("  Metrics (PARM paper Table 4 style)")
    print("=" * 60)
    print(f"\nHV reference: ({ref_point[0]:.2f}, {ref_point[1]:.2f}, {ref_point[2]:.2f})")
    print(f"MIP normalization: min=({global_min[0]:.2f}, {global_min[1]:.2f}, {global_min[2]:.4f})")
    print(f"                   max=({global_max[0]:.2f}, {global_max[1]:.2f}, {global_max[2]:.4f})")

    print(f"\n{'Method':<15} {'HV':>10} {'MIP':>10}")
    print("-" * 38)

    results_table = {}
    for method in methods:
        method_data = [d for d in data if d["method"] == method]
        points = [[d["help"], d["harm"], d["humor"]] for d in method_data]
        hv = compute_hv(points, ref_point)
        mip = compute_mip_normalized(method_data, global_min, global_max)
        results_table[method] = {"HV": round(hv, 2), "MIP": round(mip, 4)}
        print(f"{method:<15} {hv:>10.2f} {mip:>10.4f}")

    # Save
    with open("metrics/HH-RLHF/hv_mip_results.json", "w") as f:
        json.dump(results_table, f, indent=2)

    # Score range per method (for discussion)
    print(f"\n{'Method':<15} {'Help range':>12} {'Harm range':>12} {'Humor range':>12}")
    print("-" * 55)
    for method in methods:
        md = [d for d in data if d["method"] == method]
        hr = max(d["help"] for d in md) - min(d["help"] for d in md)
        sr = max(d["harm"] for d in md) - min(d["harm"] for d in md)
        ur = max(d["humor"] for d in md) - min(d["humor"] for d in md)
        print(f"{method:<15} {hr:>12.2f} {sr:>12.2f} {ur:>12.4f}")

    print(f"\nSaved metrics to metrics/HH-RLHF/hv_mip_results.json")


if __name__ == "__main__":
    main()
