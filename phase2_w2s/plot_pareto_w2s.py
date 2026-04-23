"""
Read Beaver mean_result.json files produced by compute_reward.py for both
PARM and GenARM sweeps in the Phase-2 weak-to-strong sanity check, then plot
the Pareto front (helpfulness vs safety).

Directory layout expected:
    parm_dir/PARM_{ah}help_{as}harm/mean_result.json
    genarm_dir/GenARM_{ah}help_{as}harm/mean_result.json

compute_reward.py writes "help" and "harm" keys. Fallback keys are probed in
case of schema drift.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HELP_KEYS = [
    "help",
    "help_score (high better)",
    "mean_help_score",
    "help_score",
    "helpfulness",
]
HARM_KEYS = [
    "harm",
    "harm_score (low better)",
    "mean_harm_score",
    "harm_score",
    "harmlessness",
    "cost",
]

DIR_RE = re.compile(r"^(?P<method>PARM|GenARM)_(?P<ah>[0-9.]+)help_(?P<as>[0-9.]+)harm")


def extract_scores(mean_dict):
    def first_match(keys):
        for k in keys:
            if k in mean_dict:
                return mean_dict[k]
        return None

    return first_match(HELP_KEYS), first_match(HARM_KEYS)


def load_sweep(root_dir, method_prefix):
    """Return list of (alpha_help, alpha_harm, help_score, harm_score) sorted by alpha_help."""
    root = Path(root_dir)
    out = []
    if not root.exists():
        print(f"WARN: {root_dir} does not exist, skipping.")
        return out
    for sub in sorted(root.iterdir()):
        m = DIR_RE.match(sub.name)
        if not m or m.group("method") != method_prefix:
            continue
        mf = sub / "mean_result.json"
        if not mf.exists():
            print(f"WARN: no mean_result.json in {sub}")
            continue
        with open(mf, "r", encoding="utf-8") as f:
            d = json.load(f)
        h, m_harm = extract_scores(d)
        if h is None or m_harm is None:
            print(f"WARN: could not parse scores from {mf}; keys={list(d.keys())}")
            continue
        out.append((float(m.group("ah")), float(m.group("as")), float(h), float(m_harm)))
    out.sort(key=lambda r: r[0])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parm_dir",   required=True)
    p.add_argument("--genarm_dir", required=True)
    p.add_argument("--output_png", required=True)
    p.add_argument("--output_csv", default=None)
    p.add_argument("--n_per_cell", type=int, default=100, help="prompts per (method, alpha) cell — used in the plot title only")
    args = p.parse_args()

    parm = load_sweep(args.parm_dir,   "PARM")
    gen  = load_sweep(args.genarm_dir, "GenARM")

    # Write combined CSV
    if args.output_csv:
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["method", "alpha_help", "alpha_harm", "help_score", "harm_score", "safety_score"])
            for ah, as_, h, m in parm:
                w.writerow(["PARM",   ah, as_, h, m, -m])
            for ah, as_, h, m in gen:
                w.writerow(["GenARM", ah, as_, h, m, -m])
        print(f"Wrote CSV -> {args.output_csv}")

    # Plot Pareto: x=helpfulness (higher better), y=safety = -harm (higher better)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for pts, label, marker, color in [
        (parm, "PARM (PBLoRA, 7B→65B)", "o", "tab:blue"),
        (gen,  "GenARM (7B→65B)",       "s", "tab:orange"),
    ]:
        if not pts:
            continue
        xs = [p[2] for p in pts]           # help_score
        ys = [-p[3] for p in pts]          # -harm = safety
        ahs = [p[0] for p in pts]
        ax.plot(xs, ys, marker=marker, linestyle="-", color=color, label=label, markersize=7)
        for x, y, a in zip(xs, ys, ahs):
            ax.annotate(f"α={a}", (x, y), fontsize=7,
                        xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Helpfulness (Beaver-7B reward, higher is better)")
    ax.set_ylabel("Safety (= -Harm cost, higher is better)")
    n_alphas = max(len(parm), len(gen))
    ax.set_title(
        f"Phase-2 — Weak-to-Strong PARM vs GenARM Pareto Front\n"
        f"({args.n_per_cell} prompts × {n_alphas} α ∈ [0, 1], "
        f"4-bit GPTQ Alpaca-65B base + 7B ARM)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(args.output_png, dpi=150)
    print(f"Wrote plot -> {args.output_png}")


if __name__ == "__main__":
    main()
