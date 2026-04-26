"""
4-curve Pareto: PARM/GenARM × logit-sum/EPEC, all on W2S 65B base.

Loads either mean_result.json directly or re-aggregates from reward_result.json
(useful for restricting logit-sum baseline to first N prompts to be apples-to-apples
with EPEC's n=50 sweep).

Tong-style: glob per-config dirs and dump (sorted-by-α) curves.
Adapted from epec-parm-sweep-20260424:code/evaluation/plot_epec_vs_baselines_1000.py.

Usage:
  python phase2_w2s/plot_4curve_w2s_epec.py            # use existing mean_result.json
  python phase2_w2s/plot_4curve_w2s_epec.py --n 50     # re-aggregate over first 50 prompts
"""
import argparse
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
LS_ROOT   = BASE / 'results_n100_t512'           # logit-sum baseline (n=100)
EPEC_ROOT = BASE / 'results_n50_t512_epec'       # EPEC (n=50)


def load_curve(root: Path, kind: str, prefix: str, n=None):
    """
    Load all configs under root/kind/{prefix}_*; return sorted-by-α list of dicts.

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
                # fallback: use mean_result.json (some EPEC dirs may not have reward_result.json)
                mr = d / 'mean_result.json'
                if not mr.exists(): continue
                mean = json.loads(mr.read_text())
                help_avg, harm_avg = mean['help'], mean['harm']
            else:
                records = json.loads(rr.read_text())[:n]
                if not records: continue
                help_avg = sum(r['help_score (high better)'] for r in records) / len(records)
                harm_avg = sum(r['harm_score (low better)'] for r in records) / len(records)

        out.append({'a': ah, 'help': help_avg, 'harm': harm_avg})
    out.sort(key=lambda r: r['a'])
    return out


def hv2d(pts, ref):
    pts = sorted([(x, y) for x, y in pts if x > ref[0] and y > ref[1]],
                 key=lambda p: -p[0])
    if not pts: return 0.0
    hv, last_y = 0.0, ref[1]
    for x, y in pts:
        if y > last_y:
            hv += (x - ref[0]) * (y - last_y)
            last_y = y
    return hv


def plot_curve(ax, data, marker, color, label, label_offset=(5, 8), label_size=8):
    if not data: return
    x = [r['help'] for r in data]
    y = [-r['harm'] for r in data]
    ax.plot(x, y, marker, color=color, label=label, lw=2, ms=8, alpha=0.9)
    for r in data:
        ax.annotate(f"({r['a']:.1f},{round(1-r['a'],1)})", (r['help'], -r['harm']),
                    textcoords='offset points', xytext=label_offset,
                    fontsize=label_size, color=color, alpha=0.85)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=None,
                   help='If set, re-aggregate logit-sum baseline over first N prompts '
                        '(EPEC always uses its own mean_result.json).')
    p.add_argument('--out', default='/tmp/pareto_4curve_w2s_epec.png')
    args = p.parse_args()

    # Logit-sum: optionally re-aggregate to n
    parm_ls    = load_curve(LS_ROOT,   'parm',   'PARM',         n=args.n)
    genarm_ls  = load_curve(LS_ROOT,   'genarm', 'GenARM',       n=args.n)
    # EPEC: always use mean_result.json (already at n=50)
    parm_epec  = load_curve(EPEC_ROOT, 'parm',   'EPEC_PARM',    n=None)
    genarm_epec= load_curve(EPEC_ROOT, 'genarm', 'EPEC_GenARM',  n=None)

    n_label = f"n={args.n}" if args.n else "n=100"
    print(f"Loaded: {len(parm_ls)} PARM-LS ({n_label}), {len(genarm_ls)} GenARM-LS ({n_label}), "
          f"{len(parm_epec)} PARM-EPEC (n=50), {len(genarm_epec)} GenARM-EPEC (n=50)\n")

    fig, ax = plt.subplots(figsize=(11, 8))
    plot_curve(ax, parm_ls,    '^--', '#7570b3', f'PARM logit-sum ({n_label})',   (5, -10))
    plot_curve(ax, genarm_ls,  'o--', '#4682b4', f'GenARM logit-sum ({n_label})', (5, -10))
    plot_curve(ax, parm_epec,  's-',  '#dc143c', 'PARM EPEC (n=50)',              (5, 8), 9)
    plot_curve(ax, genarm_epec,'D-',  '#8a2be2', 'GenARM EPEC (n=50)',            (5, 8), 9)

    ax.set_xlabel('Helpfulness  (Beaver-7B reward; higher = better)', fontsize=12)
    ax.set_ylabel('Harmlessness  (-Beaver-7B cost; higher = safer)', fontsize=12)
    title_n = f"all curves at n={args.n}" if args.n else "logit-sum n=100, EPEC n=50"
    ax.set_title('W2S 65B: token-level EPEC vs logit-sum (PARM and GenARM)\n'
                 f'base: alpaca-lora-65B-GPTQ  |  ARM: alpaca-7b-reproduced  |  '
                 f'{title_n}, max_tok=512', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend(loc='lower left', fontsize=11, framealpha=0.95)
    ax.axhline(0, color='gray', lw=0.6, alpha=0.4)
    ax.axvline(0, color='gray', lw=0.6, alpha=0.4)
    ax.text(0.02, 0.98, 'Upper-right = better\n(more helpful AND safer)',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.85))

    plt.tight_layout()
    plt.savefig(args.out, dpi=140, bbox_inches='tight')
    print(f"Saved: {args.out}\n")

    all_pts = parm_ls + genarm_ls + parm_epec + genarm_epec
    if all_pts:
        ref = (min(r['help'] for r in all_pts) - 1,
               min(-r['harm'] for r in all_pts) - 1)
        print(f"HV ref point: help={ref[0]:.2f}, safety={ref[1]:.2f}")
        for name, data in [('PARM logit-sum', parm_ls), ('GenARM logit-sum', genarm_ls),
                           ('PARM EPEC', parm_epec),    ('GenARM EPEC', genarm_epec)]:
            hv = hv2d([(r['help'], -r['harm']) for r in data], ref)
            print(f"  {name:24}  HV = {hv:.2f}")
        print()
        for method, ls, ep in [('GenARM', genarm_ls, genarm_epec), ('PARM', parm_ls, parm_epec)]:
            if ls and ep:
                hv_ls = hv2d([(r['help'], -r['harm']) for r in ls], ref)
                hv_ep = hv2d([(r['help'], -r['harm']) for r in ep], ref)
                d = hv_ep - hv_ls
                print(f"  {method}: ls HV {hv_ls:.2f} -> ep HV {hv_ep:.2f}  "
                      f"(Δ {d:+.2f}, {100*d/hv_ls:+.1f}%)")


if __name__ == '__main__':
    main()
