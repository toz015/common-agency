"""
Plot EPEC_GenARM and PARM Pareto frontiers together,
and save a CSV with HV / MIP for both methods.

PARM selection rule in this version:
- For each uid, select the candidate with the largest log_prob
- Then average q_help / q_harm over selected candidates

Assumptions:
1. EPEC_GenARM results live under:
   results/EPEC_GenARM_*_tau0.2_k50/mean_result.json

2. PARM scored candidate files live under:
   scored_parm/scored_candidates_PARM_*help_*harm_*.json

3. In both methods here:
   - help: larger is better
   - harm: larger is more harmful
   - so safety = -harm

Outputs:
- pareto_frontier_epec_genarm_vs_parm_tau0.2_k50_logprob.png
- hv_mip_comparison_tau0.2_k50_logprob.csv
"""

import json
import re
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ============================================================
# Paths / config
# ============================================================

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
PARM_DIR = BASE_DIR / "scored_parm"

TAU = 0.2
K = 50


# ============================================================
# Metric helpers
# ============================================================

def get_nondominated_points(points_2d):
    """
    Keep only Pareto non-dominated points for 2D maximization.
    points_2d: array of shape (N, 2), columns = [safety, help]
    """
    pts = np.array(points_2d, dtype=float)
    if len(pts) == 0:
        return pts

    nondom = []
    for i in range(len(pts)):
        dominated = False
        for j in range(len(pts)):
            if i == j:
                continue
            if np.all(pts[j] >= pts[i]) and np.any(pts[j] > pts[i]):
                dominated = True
                break
        if not dominated:
            nondom.append(pts[i])

    return np.array(nondom, dtype=float)


def compute_hv_2d(points_2d, ref_point):
    """
    Hypervolume for 2D maximization.
    points_2d: array-like of shape (N, 2), columns = [safety, help]
    ref_point: (ref_safety, ref_help), should be worse than all points
    """
    pts = get_nondominated_points(points_2d)
    if len(pts) == 0:
        return 0.0

    pts = pts[np.argsort(pts[:, 0])]  # sort by safety ascending

    hv = 0.0
    cur_best_help = ref_point[1]

    for s, h in pts:
        if h > cur_best_help:
            hv += (s - ref_point[0]) * (h - cur_best_help)
            cur_best_help = h

    return float(hv)


def compute_mip_2d(alpha_h_vals, alpha_s_vals, help_scores, safety_scores,
                   global_q_min=None, global_q_max=None):
    """
    MIP = average of alpha dot normalized q
    q = [safety, help]
    alpha = [alpha_s, alpha_h]
    """
    q = np.column_stack([safety_scores, help_scores]).astype(float)
    alphas = np.column_stack([alpha_s_vals, alpha_h_vals]).astype(float)

    if global_q_min is None:
        q_min = q.min(axis=0)
    else:
        q_min = np.array(global_q_min, dtype=float)

    if global_q_max is None:
        q_max = q.max(axis=0)
    else:
        q_max = np.array(global_q_max, dtype=float)

    q_norm = (q - q_min) / (q_max - q_min + 1e-12)
    mip = np.mean(np.sum(alphas * q_norm, axis=1))
    return float(mip), q_min, q_max


# ============================================================
# Load EPEC_GenARM points
# ============================================================

def load_epec_genarm_points(results_dir, tau=0.2, k=50):
    """
    Load EPEC_GenARM mean_result.json files.
    """
    pattern = f"EPEC_GenARM_*_tau{tau}_k{k}/mean_result.json"
    points = []

    for mean_file in sorted(results_dir.glob(pattern)):
        folder = mean_file.parent.name
        m = re.search(
            rf"EPEC_GenARM_([\d.]+)help_([\d.]+)harm_tau{tau}_k{k}",
            folder
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
            "method": "EPEC_GenARM",
            "alpha_help": alpha_help,
            "alpha_harm": alpha_harm,
            "help": help_score,
            "harm": harm_score,
            "safe": safety_score,
        })

    points.sort(key=lambda x: x["alpha_help"])
    return points


# ============================================================
# Load PARM points (select candidate with largest log_prob)
# ============================================================

def summarize_parm_file_by_logprob(json_file):
    """
    For one scored_candidates_PARM_*.json file:
    - parse alpha_help, alpha_harm from filename
    - group by uid
    - for each uid, select the candidate with the largest log_prob
    - average selected candidates to obtain one final point
    """
    m = re.search(r"scored_candidates_PARM_([\d.]+)help_([\d.]+)harm", json_file.name)
    if m is None:
        return None

    alpha_help = float(m.group(1))
    alpha_harm = float(m.group(2))

    data = json.loads(json_file.read_text())

    by_uid = defaultdict(list)
    for row in data:
        by_uid[row["uid"]].append(row)

    chosen = []
    for uid, rows in by_uid.items():
        # largest log_prob = most probable candidate
        best_row = max(rows, key=lambda r: float(r["log_prob"]))
        chosen.append(best_row)

    mean_help = float(np.mean([float(r["q_help"]) for r in chosen]))
    mean_harm = float(np.mean([float(r["q_harm"]) for r in chosen]))
    mean_safe = -mean_harm

    return {
        "method": "PARM",
        "alpha_help": alpha_help,
        "alpha_harm": alpha_harm,
        "help": mean_help,
        "harm": mean_harm,
        "safe": mean_safe,
        "num_prompts": len(chosen),
    }


def load_parm_points_by_logprob(parm_dir):
    """
    Load all PARM scored candidate files and summarize each into one point
    using largest-log_prob selection.
    """
    points = []
    for f in sorted(parm_dir.glob("scored_candidates_PARM_*help_*harm_*.json")):
        summary = summarize_parm_file_by_logprob(f)
        if summary is not None:
            points.append(summary)

    points.sort(key=lambda x: x["alpha_help"])
    return points


# ============================================================
# Compute method metrics
# ============================================================

def compute_method_metrics(points, global_q_min=None, global_q_max=None, global_ref_point=None):
    alpha_h_vals = [p["alpha_help"] for p in points]
    alpha_s_vals = [p["alpha_harm"] for p in points]
    help_scores = [p["help"] for p in points]
    safe_scores = [p["safe"] for p in points]

    points_2d = np.column_stack([safe_scores, help_scores])

    if global_ref_point is None:
        ref_point = (
            min(safe_scores) - 1.0,
            min(help_scores) - 1.0,
        )
    else:
        ref_point = tuple(global_ref_point)

    hv = compute_hv_2d(points_2d, ref_point)
    mip, _, _ = compute_mip_2d(
        alpha_h_vals,
        alpha_s_vals,
        help_scores,
        safe_scores,
        global_q_min=global_q_min,
        global_q_max=global_q_max,
    )

    return {
        "HV": hv,
        "MIP": mip,
        "ref_point_safety": ref_point[0],
        "ref_point_help": ref_point[1],
    }


# ============================================================
# Main
# ============================================================

def main():
    epec_points = load_epec_genarm_points(RESULTS_DIR, tau=TAU, k=K)
    parm_points = load_parm_points_by_logprob(PARM_DIR)

    if len(epec_points) == 0:
        raise RuntimeError(
            f"No EPEC_GenARM points found under {RESULTS_DIR} with tau={TAU}, k={K}."
        )

    if len(parm_points) == 0:
        raise RuntimeError(
            f"No PARM scored candidate files found under {PARM_DIR}."
        )

    # Global normalization / reference point for fair comparison
    all_safe = [p["safe"] for p in epec_points] + [p["safe"] for p in parm_points]
    all_help = [p["help"] for p in epec_points] + [p["help"] for p in parm_points]

    global_q_min = np.array([min(all_safe), min(all_help)], dtype=float)
    global_q_max = np.array([max(all_safe), max(all_help)], dtype=float)

    global_ref_point = (
        min(all_safe) - 1.0,
        min(all_help) - 1.0,
    )

    # Metrics
    epec_metrics = compute_method_metrics(
        epec_points,
        global_q_min=global_q_min,
        global_q_max=global_q_max,
        global_ref_point=global_ref_point,
    )

    parm_metrics = compute_method_metrics(
        parm_points,
        global_q_min=global_q_min,
        global_q_max=global_q_max,
        global_ref_point=global_ref_point,
    )

    # Save CSV
    csv_out = BASE_DIR / f"hv_mip_comparison_tau{TAU}_k{K}_logprob.csv"
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
        writer.writerow([
            "EPEC_GenARM",
            len(epec_points),
            epec_metrics["HV"],
            epec_metrics["MIP"],
            epec_metrics["ref_point_safety"],
            epec_metrics["ref_point_help"],
            global_q_min[0],
            global_q_min[1],
            global_q_max[0],
            global_q_max[1],
        ])
        writer.writerow([
            "PARM",
            len(parm_points),
            parm_metrics["HV"],
            parm_metrics["MIP"],
            parm_metrics["ref_point_safety"],
            parm_metrics["ref_point_help"],
            global_q_min[0],
            global_q_min[1],
            global_q_max[0],
            global_q_max[1],
        ])

    # Arrays for plotting
    epec_safe = [p["safe"] for p in epec_points]
    epec_help = [p["help"] for p in epec_points]
    epec_labels = [f'({p["alpha_help"]:.1f}, {p["alpha_harm"]:.1f})' for p in epec_points]

    parm_safe = [p["safe"] for p in parm_points]
    parm_help = [p["help"] for p in parm_points]
    parm_labels = [f'({p["alpha_help"]:.1f}, {p["alpha_harm"]:.1f})' for p in parm_points]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        epec_safe,
        epec_help,
        'o-',
        color='steelblue',
        linewidth=2,
        markersize=7,
        zorder=3,
        label='EPEC_GenARM'
    )

    ax.plot(
        parm_safe,
        parm_help,
        's--',
        color='seagreen',
        linewidth=2,
        markersize=6,
        zorder=3,
        label='PARM (max log_prob)'
    )

    for x, y, lbl in zip(epec_safe, epec_help, epec_labels):
        ax.annotate(
            lbl, (x, y),
            textcoords='offset points',
            xytext=(6, 4),
            fontsize=7.5,
            color='steelblue'
        )

    for x, y, lbl in zip(parm_safe, parm_help, parm_labels):
        ax.annotate(
            lbl, (x, y),
            textcoords='offset points',
            xytext=(6, -10),
            fontsize=7.5,
            color='seagreen'
        )

    ax.set_xlabel('Safety Score (higher = safer)', fontsize=12)
    ax.set_ylabel('Helpfulness Score (higher = better)', fontsize=12)
    ax.set_title(
        f"EPEC_GenARM vs PARM Pareto Frontier\n"
        f"(α_help, α_harm) sweep, τ = {TAU}, k = {K}",
        fontsize=13
    )
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.legend()

    # textbox = (
    #     f"EPEC_GenARM: HV={epec_metrics['HV']:.4f}, MIP={epec_metrics['MIP']:.4f}\n"
    #     f"PARM(max log_prob): HV={parm_metrics['HV']:.4f}, MIP={parm_metrics['MIP']:.4f}"
    # )
    # ax.text(
    #     0.03, 0.97,
    #     textbox,
    #     transform=ax.transAxes,
    #     fontsize=10,
    #     verticalalignment='top',
    #     bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    # )

    plt.tight_layout()
    fig_out = BASE_DIR / f"pareto_frontier_epec_genarm_vs_parm_tau{TAU}_k{K}_logprob.png"
    plt.savefig(fig_out, dpi=150)
    print(f"Saved figure: {fig_out}")
    print(f"Saved CSV: {csv_out}")
    plt.show()


if __name__ == "__main__":
    main()