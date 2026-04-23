"""
Overlay post-hoc EPEC sweeps (per method + combined) on the logit-sum
Pareto to show whether EPEC's menu-selection dominates the single-α
baselines for each method.
"""
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_logitsum(csv_path):
    parm, gen = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rec = (float(r["alpha_help"]), float(r["help_score"]), float(r["harm_score"]))
            (parm if r["method"] == "PARM" else gen).append(rec)
    parm.sort(); gen.sort()
    return parm, gen


def load_epec(json_path):
    sweep = json.load(open(json_path))
    return [(float(s["w_help"]), float(s["mean_help"]), float(s["mean_harm"])) for s in sweep]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pareto_csv",  default="pareto_n100.csv")
    p.add_argument("--epec_parm",   default="epec_sweep_w2s_PARM.json")
    p.add_argument("--epec_genarm", default="epec_sweep_w2s_GenARM.json")
    p.add_argument("--epec_combined", default="epec_sweep_w2s.json",
                   help="Optional combined N=22 sweep; pass '' to omit")
    p.add_argument("--out_png",     default="pareto_posthoc_epec.png")
    args = p.parse_args()

    parm, gen   = load_logitsum(args.pareto_csv)
    epec_parm   = load_epec(args.epec_parm)
    epec_gen    = load_epec(args.epec_genarm)
    epec_comb   = load_epec(args.epec_combined) if args.epec_combined else None

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    series = [
        (parm,      "Logit-sum PARM",              "o", "tab:blue",   "-"),
        (gen,       "Logit-sum GenARM",            "s", "tab:orange", "-"),
        (epec_parm, "EPEC over PARM menu (N=11)",   "o", "tab:blue",   "--"),
        (epec_gen,  "EPEC over GenARM menu (N=11)", "s", "tab:orange", "--"),
    ]
    if epec_comb is not None:
        series.append((epec_comb, "EPEC over combined menu (N=22)", "D", "tab:green", ":"))

    for pts, label, marker, color, ls in series:
        xs = [p[1] for p in pts]
        ys = [-p[2] for p in pts]
        ax.plot(xs, ys, marker=marker, linestyle=ls, color=color, label=label, markersize=6,
                linewidth=1.8, alpha=0.9)
        for (a, _, _), x, y in zip(pts, xs, ys):
            ax.annotate(f"{a}", (x, y), fontsize=6,
                        xytext=(3, 3), textcoords="offset points", alpha=0.7)

    ax.set_xlabel("Helpfulness (Beaver-7B reward, higher is better)")
    ax.set_ylabel("Safety (= −Harm cost, higher is better)")
    ax.set_title(
        "Phase-2 W2S — Post-hoc EPEC vs Logit-sum Pareto\n"
        "(100 prompts × 11 α ∈ [0, 1]; solid = logit-sum, dashed = EPEC-rerank same method's menu)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    print(f"Wrote {args.out_png}")


if __name__ == "__main__":
    main()
