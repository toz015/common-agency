"""
Local re-scoring patch for the Phase-2 sanity check.

Per-response reward/cost scores are stored in reward_result.json, so we can
recompute the Pareto means after dropping format-drift contaminated responses
without needing the GPU.

Strategy: for each (method, alpha) config, mark a response as drifted if it
contains any of the leak markers below. Recompute mean help/harm over the
clean subset and emit a new pareto_sanity_filtered.csv plus a side-by-side
diff against the original means.
"""
import argparse
import json
import re
from pathlib import Path
import csv

_DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "phase2_results" / "results_phase2_sanity"

_ap = argparse.ArgumentParser()
_ap.add_argument("--root", type=Path, default=_DEFAULT_ROOT,
                 help="Results root containing parm/ and genarm/ subdirs.")
_ap.add_argument("--tag", type=str, default="sanity",
                 help="Suffix for output files: pareto_{tag}_filtered.(csv|png)")
_args = _ap.parse_args()
ROOT = _args.root
TAG = _args.tag
DRIFT_RE = re.compile(r"###\s*(Instruction|Response|Human|Assistant|Input)", re.IGNORECASE)

CONFIGS = [
    ("parm",   "PARM",   "PARM_0.2help_0.8harm", 0.2, 0.8),
    ("parm",   "PARM",   "PARM_0.4help_0.6harm", 0.4, 0.6),
    ("parm",   "PARM",   "PARM_0.8help_0.2harm", 0.8, 0.2),
    ("genarm", "GenARM", "GenARM_0.2help_0.8harm", 0.2, 0.8),
    ("genarm", "GenARM", "GenARM_0.4help_0.6harm", 0.4, 0.6),
    ("genarm", "GenARM", "GenARM_0.8help_0.2harm", 0.8, 0.2),
]


def drift_mark(text: str):
    m = DRIFT_RE.search(text or "")
    return m.start() if m else -1


rows = []
for sub, method, name, ah, ahm in CONFIGS:
    p = ROOT / sub / name / "reward_result.json"
    data = json.load(open(p))
    n = len(data)
    drifted = []
    clean_help, clean_harm = [], []
    all_help, all_harm = [], []
    for rec in data:
        h = rec["help_score (high better)"]
        c = rec["harm_score (low better)"]
        all_help.append(h)
        all_harm.append(c)
        idx = drift_mark(rec["response"])
        if idx >= 0:
            drifted.append((rec["uid"], idx, h, c))
        else:
            clean_help.append(h)
            clean_harm.append(c)

    def avg(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    rows.append({
        "method": method,
        "alpha_help": ah,
        "alpha_harm": ahm,
        "n_total": n,
        "n_drift": len(drifted),
        "n_clean": len(clean_help),
        "help_all": avg(all_help),
        "harm_all": avg(all_harm),
        "help_clean": avg(clean_help),
        "harm_clean": avg(clean_harm),
        "safety_all": -avg(all_harm),
        "safety_clean": -avg(clean_harm),
        "drift_uids": [u for u, *_ in drifted],
    })

# --- write CSV ---
out_csv = ROOT / f"pareto_{TAG}_filtered.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "method", "alpha_help", "alpha_harm",
        "n_total", "n_drift", "n_clean",
        "help_all", "harm_all", "safety_all",
        "help_clean", "harm_clean", "safety_clean",
    ])
    for r in rows:
        w.writerow([
            r["method"], r["alpha_help"], r["alpha_harm"],
            r["n_total"], r["n_drift"], r["n_clean"],
            f'{r["help_all"]:.4f}', f'{r["harm_all"]:.4f}', f'{r["safety_all"]:.4f}',
            f'{r["help_clean"]:.4f}', f'{r["harm_clean"]:.4f}', f'{r["safety_clean"]:.4f}',
        ])
print(f"wrote {out_csv}")

# --- console diff ---
print("\n=== drift-filtered Pareto ===")
print(f"{'method':<7} {'a_h':>4} {'a_c':>4} {'drift':>6} {'help_all':>10} {'help_clr':>10} {'safe_all':>10} {'safe_clr':>10}")
for r in rows:
    print(
        f"{r['method']:<7} {r['alpha_help']:>4.1f} {r['alpha_harm']:>4.1f} "
        f"{r['n_drift']:>3}/{r['n_total']:<2} "
        f"{r['help_all']:>10.3f} {r['help_clean']:>10.3f} "
        f"{r['safety_all']:>10.3f} {r['safety_clean']:>10.3f}"
    )

print("\n=== drift uids per config ===")
for r in rows:
    if r["drift_uids"]:
        print(f"{r['method']} a_h={r['alpha_help']}: {r['drift_uids']}")

# --- plot ---
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for method, marker in [("PARM", "o"), ("GenARM", "s")]:
        rs = [r for r in rows if r["method"] == method]
        rs.sort(key=lambda r: r["alpha_help"])
        ax.plot([r["help_all"] for r in rs], [r["safety_all"] for r in rs],
                marker=marker, linestyle="--", alpha=0.5, label=f"{method} (all)")
        ax.plot([r["help_clean"] for r in rs], [r["safety_clean"] for r in rs],
                marker=marker, linestyle="-", label=f"{method} (clean)")
        for r in rs:
            ax.annotate(f"αh={r['alpha_help']}",
                        (r["help_clean"], r["safety_clean"]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("help score (higher = better)")
    ax.set_ylabel("safety score = -harm (higher = better)")
    ax.set_title(f"Phase-2 {TAG} Pareto: all vs drift-filtered")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_png = ROOT / f"pareto_{TAG}_filtered.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"\nwrote {out_png}")
except ImportError:
    print("\nmatplotlib not available, skipping plot")
