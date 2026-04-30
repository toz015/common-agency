"""
Recompute Hypervolume with per-axis min-max normalization to [0,1]^3.

Why: the original HV in `plot_pareto_and_metrics.py` operates in raw reward units,
so axes with a larger natural range (e.g. helpfulness ~4.14) dominate the volume,
and axes with a smaller range (e.g. humor in [0,1]) contribute disproportionately
little. Normalizing each axis to [0,1] gives every objective equal weight in the HV.

Reads:    code/evaluation/results/HH-RLHF/<method>_<h>help_<s>harm_<u>humor/mean_result.json
Writes:   code/evaluation/metrics/HH-RLHF/hv_mip_normalized_results.json
Prints:   side-by-side comparison of raw HV vs normalized HV with rank changes.

Does NOT modify plot_pareto_and_metrics.py or hv_mip_results.json.
"""
import json
import os
import sys
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "HH-RLHF")
METRICS_DIR = os.path.join(os.path.dirname(__file__), "metrics", "HH-RLHF")
OUT_PATH = os.path.join(METRICS_DIR, "hv_mip_normalized_results.json")
METHODS = ["GenARM", "EPEC_GenARM", "PARM", "EPEC_PARM"]


def load_data():
    data = {m: [] for m in METHODS}
    for name in sorted(os.listdir(RESULTS_DIR)):
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
            data[m].append({
                "alpha": (h, s, u),
                "score": (r["help"], r["harm"], r["humor"]),
            })
            break
    return data


def hv_pymoo_or_mc(points: np.ndarray, ref: np.ndarray) -> float:
    """Hypervolume for maximization. Uses pymoo if available, else Monte Carlo."""
    try:
        from pymoo.indicators.hv import HV
        return float(HV(ref_point=-ref)(-points))
    except ImportError:
        pass

    # MC fallback (matches plot_pareto_and_metrics.py)
    non_dom = []
    n = len(points)
    for i in range(n):
        dominated = any(
            j != i and np.all(points[j] >= points[i]) and np.any(points[j] > points[i])
            for j in range(n)
        )
        if not dominated:
            non_dom.append(points[i])
    non_dom = np.array(non_dom) if non_dom else points

    np.random.seed(42)
    n_samples = 500_000
    bounds_high = np.max(non_dom, axis=0)
    if np.any(bounds_high <= ref):
        return 0.0
    samples = np.random.uniform(ref, bounds_high, size=(n_samples, len(ref)))
    total_vol = float(np.prod(bounds_high - ref))
    is_dominated = np.zeros(n_samples, dtype=bool)
    for pt in non_dom:
        is_dominated |= np.all(samples <= pt, axis=1)
    return total_vol * float(np.mean(is_dominated))


def mip_normalized(method_data, gmin, gmax):
    if not method_data:
        return 0.0
    span = (gmax - gmin) + 1e-12
    s = 0.0
    for d in method_data:
        alpha = np.array(d["alpha"])
        q = np.array(d["score"])
        s += float(np.dot(alpha, (q - gmin) / span))
    return s / len(method_data)


def main():
    data = load_data()
    n_per = {m: len(data[m]) for m in METHODS}
    print(f"Loaded points per method: {n_per}\n")

    all_pts = np.array([d["score"] for m in METHODS for d in data[m]])
    if len(all_pts) == 0:
        print("No mean_result.json files found.")
        sys.exit(1)
    gmin = all_pts.min(axis=0)
    gmax = all_pts.max(axis=0)

    axis_names = ["Help", "Harm", "Humor"]
    print("Per-axis ranges (across all methods):")
    for i, n in enumerate(axis_names):
        print(f"  {n:<6}: [{gmin[i]:8.4f}, {gmax[i]:8.4f}]   width = {gmax[i]-gmin[i]:.4f}")
    print()

    # Reference points
    ref_raw = np.array([gmin[0] - 1.0, gmin[1] - 1.0, gmin[2] - 0.1])  # matches original
    ref_norm = np.zeros(3)
    span = (gmax - gmin) + 1e-12

    rows = []
    for m in METHODS:
        if not data[m]:
            rows.append((m, 0, 0.0, 0.0, 0.0))
            continue
        pts_raw = np.array([d["score"] for d in data[m]])
        pts_norm = (pts_raw - gmin) / span
        hv_raw = hv_pymoo_or_mc(pts_raw, ref_raw)
        hv_norm = hv_pymoo_or_mc(pts_norm, ref_norm)
        mip = mip_normalized(data[m], gmin, gmax)
        rows.append((m, len(data[m]), hv_raw, hv_norm, mip))

    # Ranks (higher = better)
    raw_rank = {m: r + 1 for r, m in enumerate(sorted(rows, key=lambda x: -x[2]))}
    norm_rank = {m: r + 1 for r, m in enumerate(sorted(rows, key=lambda x: -x[3]))}
    raw_rank = {row[0]: i + 1 for i, row in enumerate(sorted(rows, key=lambda x: -x[2]))}
    norm_rank = {row[0]: i + 1 for i, row in enumerate(sorted(rows, key=lambda x: -x[3]))}

    print(f"{'Method':<14}{'N':>4}{'HV_raw':>11}{'HV_norm':>10}{'MIP':>10}   rank raw -> norm")
    print("-" * 70)
    for m, n, hv_r, hv_n, mip in rows:
        rr, nr = raw_rank[m], norm_rank[m]
        arrow = "==" if rr == nr else f"{rr} -> {nr}"
        print(f"{m:<14}{n:>4}{hv_r:>11.3f}{hv_n:>10.4f}{mip:>10.4f}   {arrow}")

    # Save
    os.makedirs(METRICS_DIR, exist_ok=True)
    out = {
        "axis_ranges": {
            "help":  {"min": float(gmin[0]), "max": float(gmax[0]), "width": float(gmax[0] - gmin[0])},
            "harm":  {"min": float(gmin[1]), "max": float(gmax[1]), "width": float(gmax[1] - gmin[1])},
            "humor": {"min": float(gmin[2]), "max": float(gmax[2]), "width": float(gmax[2] - gmin[2])},
        },
        "ref_raw": ref_raw.tolist(),
        "ref_norm": ref_norm.tolist(),
        "methods": {
            m: {"n_points": n, "HV_raw": round(hv_r, 4), "HV_norm": round(hv_n, 4), "MIP": round(mip, 4)}
            for m, n, hv_r, hv_n, mip in rows
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
