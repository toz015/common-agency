import os
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ── Configuration ───────────────────────────────────────────────────
LOG_DIR = "/Users/Rui/Library/Mobile Documents/com~apple~CloudDocs/Mechanism Design + LLM/code/evaluation/solved_pref_sweep"

# Use specific taus to prevent plot overcrowding
TARGET_TAUS = [0.05, 0.1, 0.2, 0.5]

def parse_summary(filepath):
    """
    Parses the SUMMARY block for KL and Expected Rewards.
    Returns: { 'PARM': (kl, eh, es), 'Naive': ..., 'EPEC': ... } 
    where es is E[q_harm]
    """
    result = {}
    with open(filepath, 'r') as f:
        content = f.read()
    
    match = re.search(r'SUMMARY[\s\S]+?={10,}([\s\S]+?)={10,}', content)
    if not match:
        return None
    
    block = match.group(1)
    
    def extract_val(pattern):
        m = re.search(pattern, block)
        return float(m.group(1)) if m else np.nan

    # KL Divergence
    kl_parm  = extract_val(r'KL\(\S*_parm.*?\)\s+([-\d.]+)')
    kl_naive = extract_val(r'KL\(\S*_naive.*?\)\s+([-\d.]+)')
    kl_star  = extract_val(r'KL\(\S*_star.*?\)\s+([-\d.]+)')

    # PARM
    eh_parm = extract_val(r'E\[q_help\] — PARM\s+([-\d.]+)')
    es_parm = extract_val(r'E\[q_harm\] — PARM\s+([-\d.]+)')
    
    # Naive
    eh_naive = extract_val(r'E\[q_help\] — Naive\s+([-\d.]+)')
    es_naive = extract_val(r'E\[q_harm\] — Naive\s+([-\d.]+)')
    
    # EPEC
    eh_star = extract_val(r'E\[q_help\] — EPEC\s+([-\d.]+)')
    es_star = extract_val(r'E\[q_harm\] — EPEC\s+([-\d.]+)')
    
    if not np.isnan(kl_parm):
        result['PARM'] = (kl_parm, eh_parm, es_parm)
        result['Naive'] = (kl_naive, eh_naive, es_naive)
        result['EPEC'] = (kl_star, eh_star, es_star)
        
    return result

def main():
    records = []
    
    for logpath in sorted(glob.glob(os.path.join(LOG_DIR, 'run_tau*.log'))):
        bn = os.path.basename(logpath)
        m = re.search(r'tau([\d.]+)_w([\d.]+)\.log', bn)
        if not m: continue
        
        tau = float(m.group(1))
        w_help = float(m.group(2))
        
        if tau not in TARGET_TAUS:
            continue
            
        parsed = parse_summary(logpath)
        if not parsed: continue
        
        for method, (kl, eh, es) in parsed.items():
            records.append({
                'tau': tau,
                'w_help': w_help,
                'method': method,
                'KL': kl,
                'Total_Reward': eh - es  # Help - Harm (Cost)
            })

    if not records:
        print("No valid logs found with KL data.")
        return

    # Plotting
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'axes.labelsize': 14, 'axes.titlesize': 16,
        'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'legend.fontsize': 11,
    })

    fig, axes = plt.subplots(1, len(TARGET_TAUS), figsize=(6 * len(TARGET_TAUS), 5), squeeze=False)

    for col, tau in enumerate(TARGET_TAUS):
        ax = axes[0, col]
        
        # Plot PARM Curve
        parm_pts = [(r['w_help'], r['Total_Reward'], r['KL']) for r in records if r['method']=='PARM' and r['tau']==tau]
        if parm_pts:
            parm_pts.sort(key=lambda x: x[0])
            ax.plot([p[1] for p in parm_pts], [p[2] for p in parm_pts], '-o', color='#d7191c', lw=2.5, ms=8, label='PARM')

        # Plot EPEC Curve
        epec_pts = [(r['w_help'], r['Total_Reward'], r['KL']) for r in records if r['method']=='EPEC' and r['tau']==tau]
        if epec_pts:
            epec_pts.sort(key=lambda x: x[0])
            if len(epec_pts) > 1:
                ax.plot([p[1] for p in epec_pts], [p[2] for p in epec_pts], '-^', color='#2c7bb6', lw=2.5, ms=10, label='EPEC (Ours)')
            else:
                ax.scatter([epec_pts[0][1]], [epec_pts[0][2]], marker='^', s=200, color='#2c7bb6', edgecolors='black', linewidths=1.5, label='EPEC (Ours)', zorder=6)

        # Plot Naive Point
        naive_pts = [(r['w_help'], r['Total_Reward'], r['KL']) for r in records if r['method']=='Naive' and r['tau']==tau]
        if naive_pts:
            # Naive doesn't depend on w_help, just pick the first representation
            ax.scatter([naive_pts[0][1]], [naive_pts[0][2]], marker='s', s=150, color='#1a9641', edgecolors='black', linewidths=1.5, label='Naive', zorder=5)

        ax.set_title(rf'$\tau = {tau}$', fontweight='bold')
        ax.set_xlabel(r'Total Expected Reward $(Help - Harm) \longrightarrow$')
        ax.set_ylabel(r'Alignment Tax (KL Divergence $\longrightarrow$)')
        
        if col == 0:
            ax.legend()
            
    plt.tight_layout()
    out_png = os.path.join(os.path.dirname(LOG_DIR), 'kl_divergence_tax.png')
    plt.savefig(out_png, dpi=300)
    print(f"Plot saved to: {out_png}")

if __name__ == '__main__':
    main()
