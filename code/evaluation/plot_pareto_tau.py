"""
Plot EPEC_GenARM, EPEC_PARM, GenARM baseline, and PARM baseline
Pareto frontiers together, and save a CSV with HV / MIP for all methods.

Data layout:

1. EPEC_GenARM:
   results/EPEC_GenARM_*_tau0.2_k50/mean_result.json

2. EPEC_PARM:
   results/EPEC_PARM_*_tau0.2_k50/mean_result.json

3. GenARM baseline:
   results_genarm/GenARM_*help_*harm/mean_result.json

4. PARM baseline:
   results_parm/PARM_*help_*harm/mean_result.json

Outputs:
- sensitivity analysis/tau=0.2_N=50/pareto_frontier_tau0.2_k50.png
- sensitivity analysis/tau=0.2_N=50/hv_mip_comparison_tau0.2_k50.csv
"""

import json
import re
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ============================================================
# Paths and parameters
# ============================================================

BASE_DIR = Path(__file__).parent

RESULTS_DIR = BASE_DIR / "results"
GENARM_BASELINE_DIR = BASE_DIR / "results_genarm"
PARM_BASELINE_DIR = BASE_DIR / "results_parm"

TAU = 0.2
K = 50

SENS_DIR = BASE_DIR / "sensitivity analysis"
RUN_DIR = SENS_DIR / f"tau={TAU}_N={K}"
RUN_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Pareto / metric helpers
# ============================================================

def get_nondominated_points(points_2d):
    """
    Exact Pareto frontier for 2D maximization.
    points_2d columns = [safety, helpfulness].
    """
    pts = np.array(points_2d, dtype=float)
    if len(pts) == 0:
        return pts

    # Sort by safety ascending; if same safety, keep higher help first
    pts = pts[np.lexsort((-pts[:, 1], pts[:, 0]))]

    frontier = []
    best_help_to_right = -float("inf")

    # Scan from high safety to low safety.
    # A point is nondominated iff its help is better than all points with higher safety.
    for safety, help_score in pts[::-1]:
        if help_score > best_help_to_right:
            frontier.append([safety, help_score])
            best_help_to_right = help_score

    frontier = np.array(frontier[::-1], dtype=float)
    return frontier


def compute_hv_2d(points_2d, ref_point):
    """
    Correct 2D hypervolume for maximization.

    Each point dominates rectangle:
        [ref_safety, safety] x [ref_help, help]

    points_2d columns = [safety, helpfulness].
    """
    pts = get_nondominated_points(points_2d)
    if len(pts) == 0:
        return 0.0

    ref_safety, ref_help = ref_point

    # Safety ascending, help descending
    pts = pts[np.argsort(pts[:, 0])]

    hv = 0.0
    prev_safety = ref_safety

    for safety, help_score in pts:
        width = safety - prev_safety
        height = help_score - ref_help

        if width > 0 and height > 0:
            hv += width * height

        prev_safety = safety

    return float(hv)


def compute_mip_2d(
    alpha_h_vals,
    alpha_s_vals,
    help_scores,
    safety_scores,
    global_q_min,
    global_q_max,
):
    """
    MIP = average alpha dot normalized q.
    q = [safety, helpfulness]
    alpha = [alpha_harm, alpha_help]
    """
    q = np.column_stack([safety_scores, help_scores]).astype(float)
    alphas = np.column_stack([alpha_s_vals, alpha_h_vals]).astype(float)

    q_min = np.array(global_q_min, dtype=float)
    q_max = np.array(global_q_max, dtype=float)

    q_norm = (q - q_min) / (q_max - q_min + 1e-12)
    mip = np.mean(np.sum(alphas * q_norm, axis=1))

    return float(mip)


def compute_method_metrics(points, global_q_min, global_q_max, global_ref_point):
    alpha_h_vals = [p["alpha_help"] for p in points]
    alpha_s_vals = [p["alpha_harm"] for p in points]
    help_scores = np.array([p["help"] for p in points], dtype=float)
    safe_scores = np.array([p["safe"] for p in points], dtype=float)

    # Unnormalized HV: directly use original safety/help scores
    points_2d = np.column_stack([safe_scores, help_scores])
    hv = compute_hv_2d(points_2d, global_ref_point)

    # MIP still uses global normalization
    mip = compute_mip_2d(
        alpha_h_vals=alpha_h_vals,
        alpha_s_vals=alpha_s_vals,
        help_scores=help_scores,
        safety_scores=safe_scores,
        global_q_min=global_q_min,
        global_q_max=global_q_max,
    )

    return {
        "HV": hv,
        "MIP": mip,
        "ref_point_safety": global_ref_point[0],
        "ref_point_help": global_ref_point[1],
    }

# def compute_method_metrics(points, global_q_min, global_q_max, global_ref_point=None):
#     alpha_h_vals = [p["alpha_help"] for p in points]
#     alpha_s_vals = [p["alpha_harm"] for p in points]
#     help_scores = np.array([p["help"] for p in points], dtype=float)
#     safe_scores = np.array([p["safe"] for p in points], dtype=float)

#     q = np.column_stack([safe_scores, help_scores])
#     q_norm = (q - global_q_min) / (global_q_max - global_q_min + 1e-12)

#     # normalized reference point
#     ref_point = (-0.05, -0.05)

#     hv = compute_hv_2d(q_norm, ref_point)

#     mip = compute_mip_2d(
#         alpha_h_vals,
#         alpha_s_vals,
#         help_scores,
#         safe_scores,
#         global_q_min=global_q_min,
#         global_q_max=global_q_max,
#     )

#     return {
#         "HV": hv,
#         "MIP": mip,
#         "ref_point_safety": ref_point[0],
#         "ref_point_help": ref_point[1],
#     }
# ============================================================
# Loaders
# ============================================================

def load_epec_points_from_mean_results(results_dir, method_name, tau=0.2, k=50):
    """
    Load EPEC results stored as:
        results/{method_name}_{alpha_help}help_{alpha_harm}harm_tau{tau}_k{k}/mean_result.json
    """
    pattern = f"{method_name}_*_tau{tau}_k{k}/mean_result.json"
    points = []

    for mean_file in sorted(results_dir.glob(pattern)):
        folder = mean_file.parent.name

        m = re.search(
            rf"{re.escape(method_name)}_([\d.]+)help_([\d.]+)harm_tau{tau}_k{k}",
            folder,
        )
        if m is None:
            continue

        alpha_help = float(m.group(1))
        alpha_harm = float(m.group(2))

        d = json.loads(mean_file.read_text())

        help_score = float(d["help"])
        harm_score = float(d["harm"])
        safety_score = -harm_score

        points.append({
            "method": method_name,
            "alpha_help": alpha_help,
            "alpha_harm": alpha_harm,
            "help": help_score,
            "harm": harm_score,
            "safe": safety_score,
        })

    points.sort(key=lambda x: x["alpha_help"])
    return points


def load_baseline_points_from_mean_results(results_dir, method_name):
    """
    Load baseline results stored as:
        results_dir/{method_name}_{alpha_help}help_{alpha_harm}harm/mean_result.json

    Examples:
        results_genarm/GenARM_0.1help_0.9harm/mean_result.json
        results_parm/PARM_0.1help_0.9harm/mean_result.json
    """
    pattern = f"{method_name}_*help_*harm/mean_result.json"
    points = []

    for mean_file in sorted(results_dir.glob(pattern)):
        folder = mean_file.parent.name

        m = re.search(
            rf"{re.escape(method_name)}_([\d.]+)help_([\d.]+)harm",
            folder,
        )
        if m is None:
            continue

        alpha_help = float(m.group(1))
        alpha_harm = float(m.group(2))

        d = json.loads(mean_file.read_text())

        help_score = float(d["help"])
        harm_score = float(d["harm"])
        safety_score = -harm_score

        points.append({
            "method": method_name,
            "alpha_help": alpha_help,
            "alpha_harm": alpha_harm,
            "help": help_score,
            "harm": harm_score,
            "safe": safety_score,
        })

    points.sort(key=lambda x: x["alpha_help"])
    return points


# ============================================================
# Plot helper
# ============================================================

def plot_method(
    ax,
    points,
    label,
    marker_style,
    color,
    linestyle="-",
    annotate_offset=(6, 4),
    annotate=True,
):
    if len(points) == 0:
        return

    x = [p["safe"] for p in points]
    y = [p["help"] for p in points]
    labels = [f'({p["alpha_help"]:.1f}, {p["alpha_harm"]:.1f})' for p in points]

    ax.plot(
        x,
        y,
        marker=marker_style,
        linestyle=linestyle,
        color=color,
        linewidth=2,
        markersize=7,
        zorder=3,
        label=label,
    )

    if annotate:
        for xx, yy, lbl in zip(x, y, labels):
            ax.annotate(
                lbl,
                (xx, yy),
                textcoords="offset points",
                xytext=annotate_offset,
                fontsize=7.5,
                color=color,
            )


# ============================================================
# Main
# ============================================================

def main():
    epec_genarm_points = load_epec_points_from_mean_results(
        RESULTS_DIR,
        "EPEC_GenARM",
        tau=TAU,
        k=K,
    )

    epec_parm_points = load_epec_points_from_mean_results(
        RESULTS_DIR,
        "EPEC_PARM",
        tau=TAU,
        k=K,
    )

    genarm_baseline_points = load_baseline_points_from_mean_results(
        GENARM_BASELINE_DIR,
        "GenARM",
    )

    parm_baseline_points = load_baseline_points_from_mean_results(
        PARM_BASELINE_DIR,
        "PARM",
    )

    all_methods = {}

    if len(epec_genarm_points) > 0:
        all_methods["EPEC_GenARM"] = epec_genarm_points
    else:
        print(f"Warning: No EPEC_GenARM points found in {RESULTS_DIR}")

    if len(epec_parm_points) > 0:
        all_methods["EPEC_PARM"] = epec_parm_points
    else:
        print(f"Warning: No EPEC_PARM points found in {RESULTS_DIR}")

    if len(genarm_baseline_points) > 0:
        all_methods["GenARM"] = genarm_baseline_points
    else:
        print(f"Warning: No GenARM baseline points found in {GENARM_BASELINE_DIR}")

    if len(parm_baseline_points) > 0:
        all_methods["PARM"] = parm_baseline_points
    else:
        print(f"Warning: No PARM baseline points found in {PARM_BASELINE_DIR}")

    if len(all_methods) == 0:
        raise RuntimeError("No method points found. Please check your folders.")

    print("Loaded points:")
    for method, pts in all_methods.items():
        print(f"  {method}: {len(pts)} points")

    # --------------------------------------------------------
    # Global normalization and reference point
    # --------------------------------------------------------
    all_safe = []
    all_help = []

    for pts in all_methods.values():
        all_safe.extend([p["safe"] for p in pts])
        all_help.extend([p["help"] for p in pts])

    global_q_min = np.array([min(all_safe), min(all_help)], dtype=float)
    global_q_max = np.array([max(all_safe), max(all_help)], dtype=float)

    global_ref_point = (
        min(all_safe) - 1.0,
        min(all_help) - 1.0,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    metrics_by_method = {}

    for method, pts in all_methods.items():
        metrics_by_method[method] = compute_method_metrics(
            pts,
            global_q_min=global_q_min,
            global_q_max=global_q_max,
            global_ref_point=global_ref_point,
        )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------
    csv_out = RUN_DIR / f"hv_mip_comparison_tau{TAU}_k{K}.csv"

    with open(csv_out, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "method",
            "num_points",
            "HV",
            "MIP",
            "ref_point_safety",
            "ref_point_help",
            "norm_min_safety",
            "norm_min_help",
            "norm_max_safety",
            "norm_max_help",
        ])

        for method, pts in all_methods.items():
            m = metrics_by_method[method]

            writer.writerow([
                method,
                len(pts),
                m["HV"],
                m["MIP"],
                m["ref_point_safety"],
                m["ref_point_help"],
                global_q_min[0],
                global_q_min[1],
                global_q_max[0],
                global_q_max[1],
            ])

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    plot_method(
        ax,
        epec_genarm_points,
        label="EPEC_GenARM",
        marker_style="o",
        color="steelblue",
        linestyle="-",
        annotate_offset=(6, 4),
        annotate=True,
    )

    plot_method(
        ax,
        epec_parm_points,
        label="EPEC_PARM",
        marker_style="^",
        color="darkorange",
        linestyle="-.",
        annotate_offset=(6, 8),
        annotate=True,
    )

    plot_method(
        ax,
        genarm_baseline_points,
        label="GenARM",
        marker_style="D",
        color="purple",
        linestyle=":",
        annotate_offset=(6, 4),
        annotate=True,
    )

    plot_method(
        ax,
        parm_baseline_points,
        label="PARM",
        marker_style="s",
        color="seagreen",
        linestyle="--",
        annotate_offset=(6, -10),
        annotate=True,
    )

    ax.set_xlabel("Harmlessness", fontsize=12)
    ax.set_ylabel("Helpfulness", fontsize=12)

    # ax.set_title(
    #     f"EPEC_GenARM vs EPEC_PARM vs GenARM vs PARM\n"
    #     f"alpha sweep, tau = {TAU}, k = {K}",
    #     fontsize=13,
    # )

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.legend()

    plt.tight_layout()

    fig_out = RUN_DIR / f"pareto_frontier_tau{TAU}_k{K}.png"
    plt.savefig(fig_out, dpi=150)

    print(f"Saved figure: {fig_out}")
    print(f"Saved CSV: {csv_out}")

    plt.show()


if __name__ == "__main__":
    main()