"""
plot_pareto_pref_sweep.py
─────────────────────────
Reads the preference-vector sweep logs from solved_pref_sweep/ and
the tau-sweep logs from solved_tau_sweep/, then plots the TRUE Pareto
frontier: for each fixed tau, PARM traces a tradeoff curve as w varies,
while EPEC produces a single endogenous point.
"""
import os, glob, re, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PREF_DIR   = os.path.join(SCRIPT_DIR, 'solved_pref_sweep')
TAU_DIR    = os.path.join(SCRIPT_DIR, 'solved_tau_sweep')

# ── helpers ──────────────────────────────────────────────────────────
def parse_summary(logpath):
    """Return dict of {method: (E_help, E_harm)} from SUMMARY block."""
    with open(logpath) as f:
        content = f.read()
    result = {}
    for method in ['PARM', 'Naive', 'EPEC']:
        hm = re.search(rf'E\[q_help\]\s*.\s*{method}\s+([-0-9.]+)', content)
        sm = re.search(rf'E\[q_harm\]\s*.\s*{method}\s+([-0-9.]+)', content)
        if hm and sm:
            result[method] = (float(hm.group(1)), float(sm.group(1)))
    return result

def parse_per_prompt(logpath):
    """Fallback: parse per-prompt E[q_help] lines for partial runs."""
    with open(logpath) as f:
        content = f.read()
    help_lines = re.findall(
        r'E\[q_help\]: PARM=([-\d.]+)\s+Naive=([-\d.]+)\s+EPEC=([-\d.]+)', content)
    harm_lines = re.findall(
        r'E\[q_harm\]: PARM=([-\d.]+)\s+Naive=([-\d.]+)\s+EPEC=([-\d.]+)', content)
    if not help_lines:
        return {}
    result = {}
    for method, idx in [('PARM', 0), ('Naive', 1), ('EPEC', 2)]:
        eh = np.mean([float(row[idx]) for row in help_lines])
        if harm_lines:
            es = np.mean([float(row[idx]) for row in harm_lines])
        else:
            es = np.nan
        result[method] = (eh, es)
    return result

# ── collect preference sweep data ───────────────────────────────────
def collect_pref_sweep():
    """Returns list of dicts: {tau, w_help, method, E_help, E_harm}"""
    records = []
    if not os.path.isdir(PREF_DIR):
        return records
    for logpath in sorted(glob.glob(os.path.join(PREF_DIR, 'run_tau*.log'))):
        bn = os.path.basename(logpath)
        m = re.search(r'tau([\d.]+)_w([\d.]+)\.log', bn)
        if not m:
            continue
        tau = float(m.group(1))
        w_help = float(m.group(2))

        parsed = parse_summary(logpath)
        if not parsed:
            parsed = parse_per_prompt(logpath)
        if not parsed:
            continue

        for method, (eh, es) in parsed.items():
            records.append({
                'tau': tau, 'w_help': w_help, 'method': method,
                'E_help': eh, 'E_harm': es  # negate harm → safety
            })
    return records

# ── collect tau sweep data (for EPEC reference points) ──────────────
def collect_tau_sweep():
    records = []
    if not os.path.isdir(TAU_DIR):
        return records
    for logpath in sorted(glob.glob(os.path.join(TAU_DIR, 'run_*.log'))):
        bn = os.path.basename(logpath)
        m = re.search(r'run_([\d.]+)\.log', bn)
        if not m:
            continue
        tau = float(m.group(1))
        parsed = parse_summary(logpath)
        if not parsed:
            continue
        for method, (eh, es) in parsed.items():
            records.append({
                'tau': tau, 'w_help': 0.5, 'method': method,
                'E_help': eh, 'E_harm': es
            })
    return records

# ── main ─────────────────────────────────────────────────────────────
def main():
    pref_data = collect_pref_sweep()
    tau_data  = collect_tau_sweep()

    # Filter out records with NaN safety
    pref_data = [r for r in pref_data if not np.isnan(r['E_harm'])]
    
    print(f"Preference sweep (usable): {len(pref_data)} records")
    print(f"Tau sweep:                 {len(tau_data)} records")

    if not pref_data and not tau_data:
        print("No data found! Make sure logs are downloaded.")
        return

    # ── plotting ─────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'axes.labelsize': 16, 'axes.titlesize': 18,
        'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'legend.fontsize': 11, 'mathtext.fontset': 'cm',
    })

    # Check which taus have COMPLETE pref sweep data (at least 3 w values for PARM)
    pref_taus_with_data = []
    for tau in sorted(set(r['tau'] for r in pref_data)):
        n_parm = len([r for r in pref_data if r['method'] == 'PARM' and r['tau'] == tau])
        if n_parm >= 2:
            pref_taus_with_data.append(tau)

    if pref_taus_with_data:
        # ── Full preference sweep mode ──
        n_taus = len(pref_taus_with_data)
        import math
        n_cols = min(n_taus, 4)
        n_rows = math.ceil(n_taus / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows), squeeze=False)
        
        # Hide any unused subplots
        for j in range(n_taus, n_rows * n_cols):
            fig.delaxes(axes.flatten()[j])

        for i, tau in enumerate(pref_taus_with_data):
            row_idx = i // n_cols
            col_idx = i % n_cols
            ax = axes[row_idx, col_idx]
            
            # PARM tradeoff curve
            parm_pts = [(r['w_help'], r['E_help'], r['E_harm'])
                         for r in pref_data
                         if r['method'] == 'PARM' and r['tau'] == tau]
            if parm_pts:
                parm_pts.sort(key=lambda x: x[0])
                ax.plot([p[1] for p in parm_pts], [p[2] for p in parm_pts],
                        '-o', color='#d7191c', lw=2.5, ms=8,
                        label='PARM (sweep $w$)', zorder=3)
                for w, x, y in parm_pts:
                    ax.annotate(f'$w$={w:.1f}', (x, y), xytext=(5, 5),
                                textcoords='offset points', fontsize=8,
                                color='#d7191c', zorder=10)

            # Naive point
            naive_pts = [(r['E_help'], r['E_harm']) for r in pref_data
                          if r['method'] == 'Naive' and r['tau'] == tau]
            if not naive_pts:
                naive_pts = [(r['E_help'], r['E_harm']) for r in tau_data
                              if r['method'] == 'Naive' and abs(r['tau'] - tau) < 0.01]
            if naive_pts:
                ax.scatter([naive_pts[0][0]], [naive_pts[0][1]], marker='s', s=150,
                           color='#1a9641', edgecolors='black', linewidths=1.5,
                           label='Naive', zorder=5)

            # EPEC tradeoff curve point
            epec_pts = [(r['w_help'], r['E_help'], r['E_harm'])
                         for r in pref_data
                         if r['method'] == 'EPEC' and r['tau'] == tau]
            if not epec_pts:
                # Fallback to tau_data for legacy single-point
                epec_pts = [(0.5, r['E_help'], r['E_harm']) for r in tau_data
                             if r['method'] == 'EPEC' and abs(r['tau'] - tau) < 0.01]
            
            if epec_pts:
                epec_pts.sort(key=lambda x: x[0])
                if len(epec_pts) > 1:
                    ax.plot([p[1] for p in epec_pts], [p[2] for p in epec_pts],
                            '-^', color='#2c7bb6', lw=2.5, ms=10,
                            label='EPEC (sweep $w$)', zorder=4)
                    for w, x, y in epec_pts:
                        ax.annotate(f'$w$={w:.1f}', (x, y), xytext=(5, -15),
                                    textcoords='offset points', fontsize=8,
                                    color='#2c7bb6', zorder=10)
                else:
                    # Single point representation
                    ax.scatter([epec_pts[0][1]], [epec_pts[0][2]], marker='^', s=200,
                               color='#2c7bb6', edgecolors='black', linewidths=1.5,
                               label='EPEC (ours)', zorder=6)

            ax.set_xlabel(r'Expected Helpfulness $\longrightarrow$')
            ax.set_ylabel(r'Harmfulness Cost (lower is better) $\longrightarrow$')
            ax.invert_yaxis()
            ax.set_title(rf'$\tau = {tau}$', fontweight='bold')
            ax.legend(loc='best', frameon=True, shadow=True)

        fig.suptitle('Pareto Frontier: Preference Vector Sweep',
                     fontsize=20, fontweight='bold', y=1.02)

    else:
        # ── Preview mode: use tau sweep data ──
        print("\nPreference sweep not ready yet. Using tau sweep data for preview...")
        fig, ax = plt.subplots(figsize=(10, 7))

        styles = {
            'EPEC':  {'color': '#2c7bb6', 'marker': '^', 'ms': 11, 'lw': 3.0, 'ls': '-',  'zorder': 4},
            'PARM':  {'color': '#d7191c', 'marker': 'o', 'ms': 8,  'lw': 2.0, 'ls': '--', 'zorder': 3},
            'Naive': {'color': '#1a9641', 'marker': 's', 'ms': 9,  'lw': 2.5, 'ls': '-',  'zorder': 2},
        }

        for method, sty in styles.items():
            pts = [(r['tau'], r['E_help'], r['E_harm']) for r in tau_data
                   if r['method'] == method]
            if not pts:
                continue
            pts.sort(key=lambda p: p[1])  # sort by helpfulness
            # Deduplicate
            seen = set()
            unique = []
            for t, h, s in pts:
                key = (round(h, 6), round(s, 6))
                if key not in seen:
                    seen.add(key)
                    unique.append((t, h, s))
            
            taus = [p[0] for p in unique]
            xs = [p[1] for p in unique]
            ys = [p[2] for p in unique]
            
            label = f'EPEC (ours)' if method == 'EPEC' else f'{method} ($w$=[0.5,0.5])'
            ax.plot(xs, ys, color=sty['color'], marker=sty['marker'],
                    markersize=sty['ms'], linewidth=sty['lw'], linestyle=sty['ls'],
                    label=label, zorder=sty['zorder'])
            
            # Annotate tau values on each point
            for t, x, y in unique:
                if t < 1.0:
                    lbl = rf'$\tau$={t}'
                elif t == min(tt for tt, _, _ in unique if tt >= 1.0):
                    lbl = r'$\tau \geq 1$'
                else:
                    continue
                offset = (0, 8) if method == 'EPEC' else (0, -14) if method == 'PARM' else (8, 0)
                ax.annotate(lbl, (x, y), xytext=offset, textcoords='offset points',
                            fontsize=8, color=sty['color'], zorder=10)

        ax.set_xlabel(r'Expected Helpfulness $\longrightarrow$')
        ax.set_ylabel(r'Harmfulness Cost (lower is better) $\longrightarrow$')
        ax.set_title('Preview: Helpfulness vs. Harmfulness (Tau Sweep, $w$=[0.5,0.5])',
                     fontweight='bold')
        ax.legend(loc='upper left', frameon=True, shadow=True)
        
        # Add note about ongoing sweep
        ax.text(0.98, 0.02, 'Preference sweep in progress on GCP...\n'
                'Re-run after downloading solved_pref_sweep/',
                transform=ax.transAxes, fontsize=9, color='grey',
                ha='right', va='bottom', style='italic')

    plt.tight_layout()

    out_png = os.path.join(SCRIPT_DIR, 'pareto_pref_sweep.png')
    out_pdf = os.path.join(SCRIPT_DIR, 'pareto_pref_sweep.pdf')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"\nSaved:\n  {out_png}\n  {out_pdf}")

if __name__ == '__main__':
    main()

