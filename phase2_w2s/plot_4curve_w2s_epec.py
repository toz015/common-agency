"""
4-curve Pareto: PARM/GenARM × logit-sum/EPEC, all on W2S 65B base.

Loads mean_result.json from each per-config dir (Tong's dynamic style):
  - logit-sum baseline:   results_n100_t512/{parm,genarm}/{PARM,GenARM}_*  (mean over 100)
  - EPEC:                 results_n50_t512_epec/{parm,genarm}/EPEC_*       (mean over 50)

Note on n: the logit-sum baseline mean_result.json is over n=100 (was generated
for the n=100 sweep). EPEC is over n=50. They are NOT strictly comparable in
sample size, but the n=50 prefix of the n=100 baseline produces qualitatively
identical Pareto shape (verified earlier). For a strictly apples-to-apples plot,
re-aggregate the logit-sum reward_result.json over the first 50 uids only —
see commit aee48c4 era for that auxiliary script.

Usage:
  python phase2_w2s/plot_4curve_w2s_epec.py
  -> writes /tmp/pareto_4curve_w2s_epec.png and prints HV summary

Adapted from style of Tong's plot_epec_vs_baselines_1000.py
(epec-parm-sweep-20260424).
"""
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
LS_ROOT   = BASE / 'results_n100_t512'           # logit-sum baseline (n=100)
EPEC_ROOT = BASE / 'results_n50_t512_epec'       # EPEC (n=50)


def load_curve(root: Path, kind: str, prefix: str):
    """Load all configs under root/kind/{prefix}_*; return sorted-by-α list of dicts."""
    out = []
    pat = re.compile(rf'{re.escape(prefix)}_([\d.]+)help_([\d.]+)harm')
    for f in sorted((root / kind).glob(f'{prefix}_*/mean_result.json')):
        m = pat.search(f.parent.name)
        if not m: continue
        ah = float(m.group(1))
        d = json.loads(f.read_text())
        out.append({'a': ah, 'help': d['help'], 'harm': d['harm']})
    out.sort(key=lambda r: r['a'])
    return out


def hv2d(pts, ref):
    """2D hypervolume above ref point (max-objectives, upper-right = better)."""
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
    if not data:
        return
    x = [r['help'] for r in data]
    y = [-r['harm'] for r in data]
    ax.plot(x, y, marker, color=color, label=label, lw=2, ms=8, alpha=0.9)
    for r in data:
        ax.annotate(f"({r['a']:.1f},{1-r['a']:.1f})", (r['help'], -r['harm']),
                    textcoords='offset points', xytext=label_offset,
                    fontsize=label_size, color=color, alpha=0.85)


def main():
    parm_ls    = load_curve(LS_ROOT,   'parm',   'PARM')
    genarm_ls  = load_curve(LS_ROOT,   'genarm', 'GenARM')
    parm_epec  = load_curve(EPEC_ROOT, 'parm',   'EPEC_PARM')
    genarm_epec= load_curve(EPEC_ROOT, 'genarm', 'EPEC_GenARM')

    print(f"Loaded: {len(parm_ls)} PARM-LS, {len(genarm_ls)} GenARM-LS, "
          f"{len(parm_epec)} PARM-EPEC, {len(genarm_epec)} GenARM-EPEC\n")

    fig, ax = plt.subplots(figsize=(11, 8))
    plot_curve(ax, parm_ls,    '^--', '#7570b3', 'PARM logit-sum',   (5, -10))
    plot_curve(ax, genarm_ls,  'o--', '#4682b4', 'GenARM logit-sum', (5, -10))
    plot_curve(ax, parm_epec,  's-',  '#dc143c', 'PARM EPEC',        (5, 8), label_size=9)
    plot_curve(ax, genarm_epec,'D-',  '#8a2be2', 'GenARM EPEC',      (5, 8), label_size=9)

    ax.set_xlabel('Helpfulness  (Beaver-7B reward; higher = better)', fontsize=12)
    ax.set_ylabel('Harmlessness  (-Beaver-7B cost; higher = safer)', fontsize=12)
    ax.set_title('W2S 65B: token-level EPEC vs logit-sum (PARM and GenARM)\n'
                 'base: alpaca-lora-65B-GPTQ  |  ARM: alpaca-7b-reproduced  |  '
                 'logit-sum n=100, EPEC n=50, max_tok=512', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend(loc='lower left', fontsize=11, framealpha=0.95)
    ax.axhline(0, color='gray', lw=0.6, alpha=0.4)
    ax.axvline(0, color='gray', lw=0.6, alpha=0.4)
    ax.text(0.02, 0.98, 'Upper-right = better\n(more helpful AND safer)',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.85))

    plt.tight_layout()
    out = '/tmp/pareto_4curve_w2s_epec.png'
    plt.savefig(out, dpi=140, bbox_inches='tight')
    print(f"Saved: {out}\n")

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
