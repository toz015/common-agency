"""
Create two summary figures:

1) Tau sensitivity:
   - fixed N = 50
   - tau in {0.2, 0.3, 0.4}

2) N sensitivity:
   - fixed tau = 0.1
   - N in {10, 20, 100}

Each figure contains 8 lines:
- 3 x EPEC_GenARM
- 3 x EPEC_PARM
- GenARM baseline
- PARM baseline

Only the six target preference points are kept:
    alpha_help = 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    alpha_harm = 1 - alpha_help

Expected data layout:

1. EPEC_GenARM:
   results/EPEC_GenARM_*help_*harm_tau*_k*/mean_result.json

2. EPEC_PARM:
   results/EPEC_PARM_*help_*harm_tau*_k*/mean_result.json

3. GenARM baseline:
   results_genarm_new/GenARM_*help_*harm/mean_result.json

4. PARM baseline:
   results_parm/PARM_*help_*harm/mean_result.json

Outputs:
- sensitivity analysis_summary/tau_sensitivity/pareto_tau_sensitivity.png
- sensitivity analysis_summary/tau_sensitivity/hv_mip_tau_sensitivity.csv
- sensitivity analysis_summary/tau_sensitivity/selected_points_tau_sensitivity.csv

- sensitivity analysis_summary/N_sensitivity/pareto_N_sensitivity.png
- sensitivity analysis_summary/N_sensitivity/hv_mip_N_sensitivity.csv
- sensitivity analysis_summary/N_sensitivity/selected_points_N_sensitivity.csv
"""

import json
import re
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "results"
GENARM_BASELINE_DIR = BASE_DIR / "results_genarm_new"   # 如果你实际叫 results_genarm，就改这里
PARM_BASELINE_DIR = BASE_DIR / "results_parm"

OUT_DIR = BASE_DIR / "sensitivity analysis_summary"
TAU_OUT_DIR = OUT_DIR / "tau_sensitivity"
N_OUT_DIR = OUT_DIR / "N_sensitivity"

TAU_OUT_DIR.mkdir(parents=True, exist_ok=True)
N_OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Settings
# ============================================================

TARGET_ALPHA_HELPS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Tau sensitivity: fixed N = 50
FIXED_N_FOR_TAU = 50
TAU_VALUES = [0.2, 0.3, 0.4]

# N sensitivity: fixed tau = 0.1
FIXED_TAU_FOR_N = 0.1
N_VALUES = [10, 20, 100]

EPS = 1e-8


# ============================================================
# Helper functions
# ============================================================

def is_close(a, b, eps=EPS):
    return abs(a - b) < eps


def keep_target_alpha_pair(alpha_help, alpha_harm, target_alpha_helps=TARGET_ALPHA_HELPS):
    """
    Keep only:
    (0.0,1.0), (0.2,0.8), ..., (1.0,0.0)
    """
    for ah in target_alpha_helps:
        if is_close(alpha_help, ah) and is_close(alpha_harm, 1.0 - ah):
            return True
    return False


def sort_points_by_target_order(points, target_alpha_helps=TARGET_ALPHA_HELPS):
    order_map = {round(a, 6): i for i, a in enumerate(target_alpha_helps)}
    return sorted(points, key=lambda x: order_map.get(round(x["alpha_help"], 6), 999))


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

    # If same safety appears multiple times, keep the one with largest help
    best_by_safety = {}
    for safety, help_score in pts:
        if safety not in best_by_safety:
            best_by_safety[safety] = help_score
        else:
            best_by_safety[safety] = max(best_by_safety[safety], help_score)

    pts = np.array([[s, h] for s, h in best_by_safety.items()], dtype=float)
    pts = pts[np.argsort(pts[:, 0])]  # safety ascending

    frontier = []
    best_help_to_right = -float("inf")

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
    pts = pts[np.argsort(pts[:, 0])]  # safety ascending

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

    points_2d = np.column_stack([safe_scores, help_scores])
    hv = compute_hv_2d(points_2d, global_ref_point)

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


# ============================================================
# Loaders
# ============================================================

def load_epec_points_from_mean_results(results_dir, method_name, tau, k):
    """
    Load EPEC results stored roughly as:
        results/{method_name}_{alpha_help}help_{alpha_harm}harm_tau{tau}_k{k}/mean_result.json

    This loader is slightly robust: it reads all candidates matching the broad pattern
    and then filters by extracted tau/k.
    """
    pattern = f"{method_name}_*help_*harm_tau*_k*/mean_result.json"
    regex = re.compile(
        rf"{re.escape(method_name)}_([\d.]+)help_([\d.]+)harm_tau([\d.]+)_k(\d+)"
    )

    points = []

    for mean_file in sorted(results_dir.glob(pattern)):
        folder = mean_file.parent.name
        m = regex.search(folder)
        if m is None:
            continue

        alpha_help = float(m.group(1))
        alpha_harm = float(m.group(2))
        tau_val = float(m.group(3))
        k_val = int(m.group(4))

        if not is_close(tau_val, tau):
            continue
        if k_val != k:
            continue
        if not keep_target_alpha_pair(alpha_help, alpha_harm):
            continue

        d = json.loads(mean_file.read_text())

        help_score = float(d["help"])
        harm_score = float(d["harm"])
        safety_score = -harm_score

        points.append({
            "method": method_name,
            "tau": tau_val,
            "k": k_val,
            "alpha_help": alpha_help,
            "alpha_harm": alpha_harm,
            "help": help_score,
            "harm": harm_score,
            "safe": safety_score,
        })

    return sort_points_by_target_order(points)


def load_baseline_points_from_mean_results(results_dir, method_name):
    """
    Load baseline results stored as:
        results_dir/{method_name}_{alpha_help}help_{alpha_harm}harm/mean_result.json
    """
    pattern = f"{method_name}_*help_*harm/mean_result.json"
    regex = re.compile(
        rf"{re.escape(method_name)}_([\d.]+)help_([\d.]+)harm"
    )

    points = []

    for mean_file in sorted(results_dir.glob(pattern)):
        folder = mean_file.parent.name
        m = regex.search(folder)
        if m is None:
            continue

        alpha_help = float(m.group(1))
        alpha_harm = float(m.group(2))

        if not keep_target_alpha_pair(alpha_help, alpha_harm):
            continue

        d = json.loads(mean_file.read_text())

        help_score = float(d["help"])
        harm_score = float(d["harm"])
        safety_score = -harm_score

        points.append({
            "method": method_name,
            "tau": None,
            "k": None,
            "alpha_help": alpha_help,
            "alpha_harm": alpha_harm,
            "help": help_score,
            "harm": harm_score,
            "safe": safety_score,
        })

    return sort_points_by_target_order(points)


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
    annotate=False,
    annotate_offset=(6, 4),
):
    if len(points) == 0:
        return

    # x-axis = helpfulness
    # y-axis = harmlessness / safety
    x = [p["help"] for p in points]
    y = [p["safe"] for p in points]

    ax.plot(
        x,
        y,
        marker=marker_style,
        linestyle=linestyle,
        color=color,
        linewidth=2.0,
        markersize=7,
        markerfacecolor=color,
        markeredgecolor=color,
        zorder=3,
        label=label,
    )

    if annotate:
        labels = [f'({p["alpha_help"]:.1f}, {p["alpha_harm"]:.1f})' for p in points]
        for xx, yy, lbl in zip(x, y, labels):
            ax.annotate(
                lbl,
                (xx, yy),
                textcoords="offset points",
                xytext=annotate_offset,
                fontsize=7.5,
                color=color,
            )


def set_axis_limits(ax, configs):
    all_x = []
    all_y = []

    for cfg in configs:
        all_x.extend([p["help"] for p in cfg["points"]])
        all_y.extend([p["safe"] for p in cfg["points"]])

    if len(all_x) == 0 or len(all_y) == 0:
        return

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    x_pad = max(0.02, 0.05 * (x_max - x_min + 1e-12))
    y_pad = max(0.02, 0.05 * (y_max - y_min + 1e-12))

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)


# ============================================================
# Build collections for the two summary figures
# ============================================================

def build_tau_sensitivity_configs():
    """
    fixed N = 50
    tau in {0.2, 0.3, 0.4}
    total 8 lines:
      3 EPEC_GenARM + 3 EPEC_PARM + GenARM + PARM
    """
    configs = []

    tau_style = {
        0.2: {"linestyle": "-",  "marker": "o"},
        0.3: {"linestyle": "--", "marker": "s"},
        0.4: {"linestyle": "-.", "marker": "^"},
    }

    method_color = {
        "EPEC_GenARM": "tab:blue",
        "EPEC_PARM": "tab:orange",
        "GenARM": "tab:green",
        "PARM": "tab:red",
    }

    # EPEC_GenARM
    for tau in TAU_VALUES:
        pts = load_epec_points_from_mean_results(
            RESULTS_DIR,
            "EPEC_GenARM",
            tau=tau,
            k=FIXED_N_FOR_TAU,
        )
        if len(pts) > 0:
            configs.append({
                "label": f"EPEC_GenARM, τ={tau}",
                "method": "EPEC_GenARM",
                "setting_name": f"tau={tau}",
                "tau": tau,
                "k": FIXED_N_FOR_TAU,
                "points": pts,
                "color": method_color["EPEC_GenARM"],
                "linestyle": tau_style[tau]["linestyle"],
                "marker": tau_style[tau]["marker"],
            })
        else:
            print(f"Warning: missing EPEC_GenARM, tau={tau}, N={FIXED_N_FOR_TAU}")

    # EPEC_PARM
    for tau in TAU_VALUES:
        pts = load_epec_points_from_mean_results(
            RESULTS_DIR,
            "EPEC_PARM",
            tau=tau,
            k=FIXED_N_FOR_TAU,
        )
        if len(pts) > 0:
            configs.append({
                "label": f"EPEC_PARM, τ={tau}",
                "method": "EPEC_PARM",
                "setting_name": f"tau={tau}",
                "tau": tau,
                "k": FIXED_N_FOR_TAU,
                "points": pts,
                "color": method_color["EPEC_PARM"],
                "linestyle": tau_style[tau]["linestyle"],
                "marker": tau_style[tau]["marker"],
            })
        else:
            print(f"Warning: missing EPEC_PARM, tau={tau}, N={FIXED_N_FOR_TAU}")

    # Baselines
    genarm_pts = load_baseline_points_from_mean_results(
        GENARM_BASELINE_DIR,
        "GenARM",
    )
    if len(genarm_pts) > 0:
        configs.append({
            "label": "GenARM baseline",
            "method": "GenARM",
            "setting_name": "baseline",
            "tau": None,
            "k": None,
            "points": genarm_pts,
            "color": method_color["GenARM"],
            "linestyle": ":",
            "marker": "D",
        })
    else:
        print(f"Warning: missing GenARM baseline in {GENARM_BASELINE_DIR}")

    parm_pts = load_baseline_points_from_mean_results(
        PARM_BASELINE_DIR,
        "PARM",
    )
    if len(parm_pts) > 0:
        configs.append({
            "label": "PARM baseline",
            "method": "PARM",
            "setting_name": "baseline",
            "tau": None,
            "k": None,
            "points": parm_pts,
            "color": method_color["PARM"],
            "linestyle": ":",
            "marker": "P",
        })
    else:
        print(f"Warning: missing PARM baseline in {PARM_BASELINE_DIR}")

    return configs


def build_n_sensitivity_configs():
    """
    fixed tau = 0.1
    N in {10, 20, 100}
    total 8 lines:
      3 EPEC_GenARM + 3 EPEC_PARM + GenARM + PARM
    """
    configs = []

    n_style = {
        10:  {"linestyle": "-",  "marker": "o"},
        20:  {"linestyle": "--", "marker": "s"},
        100: {"linestyle": "-.", "marker": "^"},
    }

    method_color = {
        "EPEC_GenARM": "tab:blue",
        "EPEC_PARM": "tab:orange",
        "GenARM": "tab:green",
        "PARM": "tab:red",
    }

    # EPEC_GenARM
    for n in N_VALUES:
        pts = load_epec_points_from_mean_results(
            RESULTS_DIR,
            "EPEC_GenARM",
            tau=FIXED_TAU_FOR_N,
            k=n,
        )
        if len(pts) > 0:
            configs.append({
                "label": f"EPEC_GenARM, N={n}",
                "method": "EPEC_GenARM",
                "setting_name": f"N={n}",
                "tau": FIXED_TAU_FOR_N,
                "k": n,
                "points": pts,
                "color": method_color["EPEC_GenARM"],
                "linestyle": n_style[n]["linestyle"],
                "marker": n_style[n]["marker"],
            })
        else:
            print(f"Warning: missing EPEC_GenARM, tau={FIXED_TAU_FOR_N}, N={n}")

    # EPEC_PARM
    for n in N_VALUES:
        pts = load_epec_points_from_mean_results(
            RESULTS_DIR,
            "EPEC_PARM",
            tau=FIXED_TAU_FOR_N,
            k=n,
        )
        if len(pts) > 0:
            configs.append({
                "label": f"EPEC_PARM, N={n}",
                "method": "EPEC_PARM",
                "setting_name": f"N={n}",
                "tau": FIXED_TAU_FOR_N,
                "k": n,
                "points": pts,
                "color": method_color["EPEC_PARM"],
                "linestyle": n_style[n]["linestyle"],
                "marker": n_style[n]["marker"],
            })
        else:
            print(f"Warning: missing EPEC_PARM, tau={FIXED_TAU_FOR_N}, N={n}")

    # Baselines
    genarm_pts = load_baseline_points_from_mean_results(
        GENARM_BASELINE_DIR,
        "GenARM",
    )
    if len(genarm_pts) > 0:
        configs.append({
            "label": "GenARM baseline",
            "method": "GenARM",
            "setting_name": "baseline",
            "tau": None,
            "k": None,
            "points": genarm_pts,
            "color": method_color["GenARM"],
            "linestyle": ":",
            "marker": "D",
        })
    else:
        print(f"Warning: missing GenARM baseline in {GENARM_BASELINE_DIR}")

    parm_pts = load_baseline_points_from_mean_results(
        PARM_BASELINE_DIR,
        "PARM",
    )
    if len(parm_pts) > 0:
        configs.append({
            "label": "PARM baseline",
            "method": "PARM",
            "setting_name": "baseline",
            "tau": None,
            "k": None,
            "points": parm_pts,
            "color": method_color["PARM"],
            "linestyle": ":",
            "marker": "P",
        })
    else:
        print(f"Warning: missing PARM baseline in {PARM_BASELINE_DIR}")

    return configs


# ============================================================
# Save CSV / compute metrics
# ============================================================

def compute_metrics_for_configs(configs):
    if len(configs) == 0:
        return {}, None, None, None

    all_safe = []
    all_help = []

    for cfg in configs:
        all_safe.extend([p["safe"] for p in cfg["points"]])
        all_help.extend([p["help"] for p in cfg["points"]])

    global_q_min = np.array([min(all_safe), min(all_help)], dtype=float)
    global_q_max = np.array([max(all_safe), max(all_help)], dtype=float)

    global_ref_point = (
        min(all_safe) - 1.0,
        min(all_help) - 1.0,
    )

    metrics = {}
    for cfg in configs:
        metrics[cfg["label"]] = compute_method_metrics(
            cfg["points"],
            global_q_min=global_q_min,
            global_q_max=global_q_max,
            global_ref_point=global_ref_point,
        )

    return metrics, global_q_min, global_q_max, global_ref_point


def save_selected_points_csv(configs, out_csv):
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "line_label",
            "method",
            "setting_name",
            "tau",
            "k",
            "alpha_help",
            "alpha_harm",
            "help",
            "harm",
            "safe",
        ])

        for cfg in configs:
            for p in cfg["points"]:
                writer.writerow([
                    cfg["label"],
                    cfg["method"],
                    cfg["setting_name"],
                    cfg["tau"],
                    cfg["k"],
                    p["alpha_help"],
                    p["alpha_harm"],
                    p["help"],
                    p["harm"],
                    p["safe"],
                ])


def save_metrics_csv(configs, metrics, global_q_min, global_q_max, out_csv):
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "line_label",
            "method",
            "setting_name",
            "tau",
            "k",
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

        for cfg in configs:
            m = metrics[cfg["label"]]
            writer.writerow([
                cfg["label"],
                cfg["method"],
                cfg["setting_name"],
                cfg["tau"],
                cfg["k"],
                len(cfg["points"]),
                m["HV"],
                m["MIP"],
                m["ref_point_safety"],
                m["ref_point_help"],
                global_q_min[0],
                global_q_min[1],
                global_q_max[0],
                global_q_max[1],
            ])


# ============================================================
# Plot figure
# ============================================================

def make_summary_figure(configs, title, out_png):
    if len(configs) == 0:
        print(f"Warning: no configs to plot for {title}")
        return

    plt.style.use("default")

    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for cfg in configs:
        plot_method(
            ax,
            cfg["points"],
            label=cfg["label"],
            marker_style=cfg["marker"],
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            annotate=False,
        )

    #ax.set_title(title, fontsize=15)
    ax.set_xlabel("Helpfulness", fontsize=22)
    ax.set_ylabel("Harmlessness", fontsize=22)

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.8,
        alpha=0.35,
        color="gray",
    )

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    ax.tick_params(axis="both", which="major", labelsize=12)

    set_axis_limits(ax, configs)

    # legend 放右侧，避免挡住图
    ax.legend(
        loc="lower left",
        #bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        facecolor="white",
        edgecolor="lightgray",
        framealpha=1.0,
        fontsize=16
        #borderaxespad=0.0,
    )

    plt.tight_layout()

    plt.savefig(
        out_png,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    print(f"Saved figure: {out_png}")
    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    # --------------------------------------------------------
    # Tau sensitivity
    # --------------------------------------------------------
    tau_configs = build_tau_sensitivity_configs()

    print("\n" + "=" * 80)
    print("Tau sensitivity configs")
    print("=" * 80)
    for cfg in tau_configs:
        print(f"{cfg['label']}: {len(cfg['points'])} points")

    if len(tau_configs) > 0:
        tau_metrics, tau_q_min, tau_q_max, _ = compute_metrics_for_configs(tau_configs)

        tau_selected_csv = TAU_OUT_DIR / "selected_points_tau_sensitivity.csv"
        tau_metrics_csv = TAU_OUT_DIR / "hv_mip_tau_sensitivity.csv"
        tau_fig = TAU_OUT_DIR / "pareto_tau_sensitivity.png"

        save_selected_points_csv(tau_configs, tau_selected_csv)
        save_metrics_csv(tau_configs, tau_metrics, tau_q_min, tau_q_max, tau_metrics_csv)
        make_summary_figure(
            tau_configs,
            title=f"Tau Sensitivity (fixed N={FIXED_N_FOR_TAU})",
            out_png=tau_fig,
        )

        print(f"Saved CSV: {tau_metrics_csv}")
        print(f"Saved selected points CSV: {tau_selected_csv}")

    # --------------------------------------------------------
    # N sensitivity
    # --------------------------------------------------------
    n_configs = build_n_sensitivity_configs()

    print("\n" + "=" * 80)
    print("N sensitivity configs")
    print("=" * 80)
    for cfg in n_configs:
        print(f"{cfg['label']}: {len(cfg['points'])} points")

    if len(n_configs) > 0:
        n_metrics, n_q_min, n_q_max, _ = compute_metrics_for_configs(n_configs)

        n_selected_csv = N_OUT_DIR / "selected_points_N_sensitivity.csv"
        n_metrics_csv = N_OUT_DIR / "hv_mip_N_sensitivity.csv"
        n_fig = N_OUT_DIR / "pareto_N_sensitivity.png"

        save_selected_points_csv(n_configs, n_selected_csv)
        save_metrics_csv(n_configs, n_metrics, n_q_min, n_q_max, n_metrics_csv)
        make_summary_figure(
            n_configs,
            title=f"N Sensitivity (fixed τ={FIXED_TAU_FOR_N})",
            out_png=n_fig,
        )

        print(f"Saved CSV: {n_metrics_csv}")
        print(f"Saved selected points CSV: {n_selected_csv}")


if __name__ == "__main__":
    main()