"""
4-curve Pareto: PARM/GenARM × logit-sum/EPEC, all on W2S 65B base.

Naming convention (paper-aligned):
  EPEC + GenARM   →  CAGE  (Ours)
  EPEC + PARM     →  CAGE+ (Ours)
  GenARM logit-sum baseline →  GenARM
  PARM logit-sum baseline   →  PARM

Style: matches plot_pareto_tau.py from epec_parm_0.1_0.9 branch
(white background, large axis fontsize, no title, paper-style colors/markers).

Usage:
  python phase2_w2s/plot_4curve_w2s_epec.py            # use existing mean_result.json
  python phase2_w2s/plot_4curve_w2s_epec.py --n 50     # re-aggregate logit-sum over first 50 prompts
  python phase2_w2s/plot_4curve_w2s_epec.py --n 300    # re-aggregate logit-sum over first 300 prompts
"""
import argparse
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE = Path(__file__).parent
LS_ROOT   = BASE / 'results_n100_t512'           # has reward_result.json (n=100 baseline)
LS_ROOT_300 = BASE / 'results_n300_t512'         # has reward_result.json (n=300 baseline)
EPEC_ROOT = BASE / 'results_n50_t512_epec'       # EPEC mean_result.json (n=50 PARM, n=300 GenARM)


def load_curve(root: Path, kind: str, prefix: str, n=None):
    """
    Load configs under root/kind/{prefix}_*.

    If n is None: read mean_result.json directly.
    If n is set:  re-aggregate from reward_result.json over first n prompts.
    """
    out = []
    pat = re.compile(rf'{re.escape(prefix)}_([\d.]+)help_([\d.]+)harm')
    for d in sorted((root / kind).glob(f'{prefix}_*')):
        if not d.is_dir(): continue
        m = pat.search(d.name)
        if not m: continue
        ah = float(m.group(1))

        if n is None:
            mr = d / 'mean_result.json'
            if not mr.exists(): continue
            mean = json.loads(mr.read_text())
            help_avg, harm_avg = mean['help'], mean['harm']
        else:
            rr = d / 'reward_result.json'
            if not rr.exists():
                mr = d / 'mean_result.json'
                if not mr.exists(): continue
                mean = json.loads(mr.read_text())
                help_avg, harm_avg = mean['help'], mean['harm']
            else:
                records = json.loads(rr.read_text())[:n]
                if not records: continue
                help_avg = sum(r['help_score (high better)'] for r in records) / len(records)
                harm_avg = sum(r['harm_score (low better)'] for r in records) / len(records)

        out.append({'a': ah, 'help': help_avg, 'safe': -harm_avg})
    out.sort(key=lambda r: r['a'])
    return out


def hv2d(pts, ref):
    """2D hypervolume above ref point (max-objectives)."""
    pts = sorted([(x, y) for x, y in pts if x > ref[0] and y > ref[1]],
                 key=lambda p: -p[0])
    if not pts: return 0.0
    hv, last_y = 0.0, ref[1]
    for x, y in pts:
        if y > last_y:
            hv += (x - ref[0]) * (y - last_y)
            last_y = y
    return hv


def plot_method(ax, points, label, marker, color, linestyle='-'):
    """Plot one curve in Tong's paper style."""
    if not points: return
    x = [p['help'] for p in points]
    y = [p['safe'] for p in points]
    ax.plot(
        x, y,
        marker=marker,
        linestyle=linestyle,
        color=color,
        linewidth=2.0,
        markersize=7,
        markerfacecolor=color,
        markeredgecolor=color,
        zorder=3,
        label=label,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=None,
                   help='If set, re-aggregate logit-sum baseline over first N prompts.')
    p.add_argument('--out', default='/tmp/pareto_4curve_w2s_epec.png')
    args = p.parse_args()

    # Pick LS dir based on n
    ls_root = LS_ROOT_300 if args.n and args.n > 100 else LS_ROOT

    parm_ls    = load_curve(ls_root,   'parm',   'PARM',         n=args.n)
    genarm_ls  = load_curve(ls_root,   'genarm', 'GenARM',       n=args.n)
    parm_epec  = load_curve(EPEC_ROOT, 'parm',   'EPEC_PARM',    n=None)
    genarm_epec= load_curve(EPEC_ROOT, 'genarm', 'EPEC_GenARM',  n=None)

    n_label = f"n={args.n}" if args.n else "n=mean_result"
    print(f"Loaded: {len(parm_ls)} PARM ({n_label}), {len(genarm_ls)} GenARM ({n_label}), "
          f"{len(parm_epec)} CAGE+ EPEC_PARM (n=50), {len(genarm_epec)} CAGE EPEC_GenARM\n")

    # ============================================================
    # Plot — style matches plot_pareto_tau.py
    # ============================================================
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # CAGE  = EPEC+GenARM   → tab:blue, marker o
    # CAGE+ = EPEC+PARM     → tab:orange, marker s
    # GenARM (logit-sum)    → tab:green, marker ^
    # PARM (logit-sum)      → tab:red, marker D

    plot_method(ax, genarm_epec, label="CAGE (Ours)",  marker="o", color="tab:blue")
    plot_method(ax, parm_epec,   label="CAGE+ (Ours)", marker="s", color="tab:orange")
    plot_method(ax, genarm_ls,   label="GenARM",       marker="^", color="tab:green")
    plot_method(ax, parm_ls,     label="PARM",         marker="D", color="tab:red")

    ax.set_xlabel("Helpfulness", fontsize=32)
    ax.set_ylabel("Harmlessness", fontsize=32)

    # 白底 + 浅灰网格
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35, color="gray")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # 黑色边框
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    ax.tick_params(axis="both", which="major", labelsize=12)

    ax.legend(
        loc="best",
        frameon=True,
        facecolor="white",
        edgecolor="lightgray",
        framealpha=1.0,
        fontsize=17,
    )

    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {args.out}\n")

    # ============================================================
    # HV table
    # ============================================================
    all_pts = parm_ls + genarm_ls + parm_epec + genarm_epec
    if all_pts:
        ref = (min(r['help'] for r in all_pts) - 1.0,
               min(r['safe'] for r in all_pts) - 1.0)
        print(f"HV ref point: help={ref[0]:.2f}, safety={ref[1]:.2f}\n")
        rows = [
            ('PARM',        parm_ls),
            ('GenARM',      genarm_ls),
            ('CAGE+ (EPEC+PARM)',   parm_epec),
            ('CAGE (EPEC+GenARM)',  genarm_epec),
        ]
        print(f"{'Method':<28} {'n_alpha':>8} {'HV':>10}")
        print('-' * 50)
        for name, data in rows:
            hv = hv2d([(r['help'], r['safe']) for r in data], ref)
            print(f"{name:<28} {len(data):>8} {hv:>10.2f}")
        print()
        for method, ls, ep in [('GenARM/CAGE', genarm_ls, genarm_epec),
                               ('PARM/CAGE+',  parm_ls, parm_epec)]:
            if ls and ep:
                hv_ls = hv2d([(r['help'], r['safe']) for r in ls], ref)
                hv_ep = hv2d([(r['help'], r['safe']) for r in ep], ref)
                d = hv_ep - hv_ls
                print(f"  {method}: baseline HV {hv_ls:.2f} -> EPEC HV {hv_ep:.2f}  "
                      f"(Δ {d:+.2f}, {100*d/hv_ls:+.1f}%)")


if __name__ == '__main__':
    main()
