"""
Per-preference comparison of EPEC vs non-EPEC, plus subset-restricted HV/MIP.

For each preference vector α, compute the scalarized utility u_method(α) = α · q̂(α)
where q̂ is the per-axis [0,1]-normalized achieved reward. Then:

  1. Show a row per preference, with each method's u and Δ_EPEC (EPEC minus non-EPEC).
  2. Identify the "EPEC-favorable" subsets:
       - S_GenARM = { α : u_{EPEC+GenARM}(α) > u_{GenARM}(α) }
       - S_PARM   = { α : u_{EPEC+PARM}(α)   > u_{PARM}(α) }
       - S_both   = S_GenARM ∩ S_PARM
  3. Recompute HV (normalized) and MIP on the full set and on each subset,
     so you can quote things like
        "On the 7 interior balanced preferences where EPEC outperforms its
         non-EPEC counterpart, EPEC+PARM achieves HV_norm = 0.27 vs PARM 0.24."

Methodological note: restricting to a subset where one method wins is biased
by definition. Use this as a *regime characterization* tool, not as the
headline metric.

Reads:    code/evaluation/results/HH-RLHF/<method>_..../mean_result.json
Writes:   code/evaluation/metrics/HH-RLHF/epec_subset_analysis.json
"""
import json
import os
import sys
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "HH-RLHF")
METRICS_DIR = os.path.join(os.path.dirname(__file__), "metrics", "HH-RLHF")
OUT_PATH = os.path.join(METRICS_DIR, "epec_subset_analysis.json")
METHODS = ["GenARM", "EPEC_GenARM", "PARM", "EPEC_PARM"]


def load_data():
    """Returns dict: pref_tuple -> {method: score_tuple}."""
    by_pref = {}
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
            by_pref.setdefault((h, s, u), {})[m] = (r["help"], r["harm"], r["humor"])
            break
    return by_pref


def hv(points: np.ndarray, ref: np.ndarray) -> float:
    """Hypervolume for maximization. Uses pymoo if available."""
    if len(points) == 0:
        return 0.0
    try:
        from pymoo.indicators.hv import HV
        return float(HV(ref_point=-ref)(-points))
    except ImportError:
        # MC fallback
        np.random.seed(42)
        n_samples = 200_000
        bounds_high = np.max(points, axis=0)
        if np.any(bounds_high <= ref):
            return 0.0
        samples = np.random.uniform(ref, bounds_high, size=(n_samples, len(ref)))
        total_vol = float(np.prod(bounds_high - ref))
        is_dom = np.zeros(n_samples, dtype=bool)
        for pt in points:
            is_dom |= np.all(samples <= pt, axis=1)
        return total_vol * float(np.mean(is_dom))


def main():
    by_pref = load_data()
    if not by_pref:
        print("No mean_result.json files found.")
        sys.exit(1)

    # Global axis bounds across every (method, pref) score we have
    all_pts = np.array([s for d in by_pref.values() for s in d.values()])
    gmin = all_pts.min(axis=0)
    gmax = all_pts.max(axis=0)
    span = (gmax - gmin) + 1e-12
    print(f"Per-axis ranges: help [{gmin[0]:.3f},{gmax[0]:.3f}]  "
          f"harm [{gmin[1]:.3f},{gmax[1]:.3f}]  humor [{gmin[2]:.3f},{gmax[2]:.3f}]\n")

    # Compute utility per (pref, method)
    util = {}     # util[pref][method] = scalarized utility in [0,1]
    qnorm = {}    # qnorm[pref][method] = normalized score vector
    for pref, m_to_score in by_pref.items():
        a = np.array(pref)
        util[pref] = {}
        qnorm[pref] = {}
        for m, sc in m_to_score.items():
            qn = (np.array(sc) - gmin) / span
            qnorm[pref][m] = qn
            util[pref][m] = float(np.dot(a, qn))

    # Per-preference table
    sorted_prefs = sorted(by_pref.keys())
    print("Per-preference scalarized utility u(α) = α · q̂(α)  (higher = better)")
    print("=" * 90)
    print(f"{'pref (h/s/u)':<18}{'GenARM':>10}{'EPEC+G':>10}{'ΔEPEC_G':>10}"
          f"{'PARM':>10}{'EPEC+P':>10}{'ΔEPEC_P':>10}")
    print("-" * 90)

    rows = []
    for p in sorted_prefs:
        u = util[p]
        ug = u.get("GenARM"); ueg = u.get("EPEC_GenARM")
        up = u.get("PARM");   uep = u.get("EPEC_PARM")
        d_g = (ueg - ug) if (ug is not None and ueg is not None) else None
        d_p = (uep - up) if (up is not None and uep is not None) else None

        def _f(x): return f"{x:>10.4f}" if x is not None else f"{'--':>10}"
        def _d(x):
            if x is None: return f"{'--':>10}"
            sign = "+" if x >= 0 else ""
            return f"{sign+f'{x:.4f}':>10}"
        print(f"{p[0]:g}/{p[1]:g}/{p[2]:g}  ".ljust(18)
              + _f(ug) + _f(ueg) + _d(d_g) + _f(up) + _f(uep) + _d(d_p))
        rows.append({"pref": p, "u_GenARM": ug, "u_EPEC_GenARM": ueg,
                     "u_PARM": up, "u_EPEC_PARM": uep,
                     "delta_GenARM": d_g, "delta_PARM": d_p})

    # Identify EPEC-favorable subsets
    s_genarm = [p for p in sorted_prefs
                if util[p].get("EPEC_GenARM") is not None
                and util[p].get("GenARM") is not None
                and util[p]["EPEC_GenARM"] > util[p]["GenARM"]]
    s_parm = [p for p in sorted_prefs
              if util[p].get("EPEC_PARM") is not None
              and util[p].get("PARM") is not None
              and util[p]["EPEC_PARM"] > util[p]["PARM"]]
    s_both = [p for p in s_genarm if p in s_parm]
    s_full = sorted_prefs

    print("\nEPEC-favorable subsets:")
    print(f"  S_GenARM (EPEC+G > G):           {len(s_genarm)}/{len(sorted_prefs)} prefs")
    for p in s_genarm: print(f"     {p}")
    print(f"  S_PARM   (EPEC+P > P):           {len(s_parm)}/{len(sorted_prefs)} prefs")
    for p in s_parm: print(f"     {p}")
    print(f"  S_both   (EPEC wins both fams): {len(s_both)}/{len(sorted_prefs)} prefs")
    for p in s_both: print(f"     {p}")

    # Compute HV_norm and MIP on each subset, per method
    def metrics_on(subset_prefs, method):
        pts = []
        mip_vals = []
        for p in subset_prefs:
            if method not in qnorm[p]:
                continue
            qn = qnorm[p][method]
            pts.append(qn)
            mip_vals.append(float(np.dot(np.array(p), qn)))
        if not pts:
            return {"n": 0, "HV_norm": 0.0, "MIP": 0.0}
        return {
            "n": len(pts),
            "HV_norm": round(hv(np.array(pts), np.zeros(3)), 4),
            "MIP": round(float(np.mean(mip_vals)), 4),
        }

    subsets = {"FULL": s_full, "S_GenARM": s_genarm, "S_PARM": s_parm, "S_both": s_both}

    print("\n\nMetrics on full set vs EPEC-favorable subsets:")
    print("=" * 92)
    print(f"{'Subset':<12}{'|S|':>5}  ", end="")
    for m in METHODS:
        print(f"{m+'_HV':>14}{m+'_MIP':>13}", end="")
    print()
    print("-" * 92)
    for name, subset in subsets.items():
        print(f"{name:<12}{len(subset):>5}  ", end="")
        for m in METHODS:
            mv = metrics_on(subset, m)
            print(f"{mv['HV_norm']:>14.4f}{mv['MIP']:>13.4f}", end="")
        print()

    # Save
    os.makedirs(METRICS_DIR, exist_ok=True)
    out = {
        "axis_min": gmin.tolist(),
        "axis_max": gmax.tolist(),
        "n_prefs_total": len(sorted_prefs),
        "subsets": {
            "S_GenARM": [list(p) for p in s_genarm],
            "S_PARM":   [list(p) for p in s_parm],
            "S_both":   [list(p) for p in s_both],
        },
        "per_preference": [
            {"pref": list(r["pref"]),
             "u_GenARM": r["u_GenARM"], "u_EPEC_GenARM": r["u_EPEC_GenARM"],
             "u_PARM":   r["u_PARM"],   "u_EPEC_PARM":   r["u_EPEC_PARM"],
             "delta_GenARM": r["delta_GenARM"], "delta_PARM": r["delta_PARM"]}
            for r in rows
        ],
        "metrics_by_subset": {
            name: {m: metrics_on(subset, m) for m in METHODS}
            for name, subset in subsets.items()
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
