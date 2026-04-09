"""Overlay PARM and EPEC Pareto curves on PKU-SafeRLHF-10K test."""
import json, re
from pathlib import Path
import matplotlib.pyplot as plt

base = Path(__file__).parent

# PARM
parm = []
for f in sorted((base / 'results').glob('PARM_*/mean_result.json')):
    m = re.search(r'PARM_([\d.]+)help_([\d.]+)harm', f.parent.name)
    d = json.loads(f.read_text())
    parm.append((float(m.group(1)), float(m.group(2)), d['help'], d['harm']))
parm.sort(key=lambda x: x[0])

# EPEC
epec = json.loads((base / 'epec_results' / 'epec_sweep.json').read_text())
epec.sort(key=lambda r: r['w_help'])

# Best-of-N baseline: argmax (w_help * q_help - w_harm * q_harm) per prompt
import numpy as np
scored = json.loads((base / 'scored' / 'scored_candidates_test_prompt_only_alpaca-7b-reproduced_N20.json').read_text())
by_uid = {}
for r in scored:
    by_uid.setdefault(r['uid'], []).append(r)
# Pick a* = argmax log_prob (most-likely base candidate), independent of w
maxprob_helps, maxprob_harms = [], []
for uid, g in by_uid.items():
    g.sort(key=lambda r: r['candidate_id'])
    a = int(np.argmax([r['log_prob'] for r in g]))
    maxprob_helps.append(g[a]['q_help'])
    maxprob_harms.append(g[a]['q_harm'])
mp_help = float(np.mean(maxprob_helps))
mp_harm = float(np.mean(maxprob_harms))
# Per-w scoring of the SAME a*: (w_h * q_help[a*], w_s * q_harm[a*])
bon = [{'w_help': r['w_help'], 'w_harm': r['w_harm'],
        'mean_help': r['w_help'] * mp_help,
        'mean_harm': r['w_harm'] * mp_harm} for r in epec]

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([p[2] for p in parm], [-p[3] for p in parm],
        'o-', color='steelblue', lw=2, ms=7, label='PARM', zorder=3)
ax.plot([r['mean_help'] for r in epec], [-r['mean_harm'] for r in epec],
        's-', color='crimson', lw=2, ms=7, label='EPEC (ours)', zorder=3)
ax.plot([mp_help], [-mp_harm], '*', color='darkorange', ms=18,
        label='Base model (argmax π_base)', zorder=4)

for p in parm:
    ax.annotate(f'({p[0]:.1f},{p[1]:.1f})', (p[2], -p[3]),
                textcoords='offset points', xytext=(6, 4), fontsize=7, color='steelblue')
for r in epec:
    ax.annotate(f'({r["w_help"]:.1f},{r["w_harm"]:.1f})', (r['mean_help'], -r['mean_harm']),
                textcoords='offset points', xytext=(6, -10), fontsize=7, color='crimson')

ax.set_xlabel('Helpfulness Score  (higher = better)', fontsize=12)
ax.set_ylabel('Harmlessness Score  (higher = safer)', fontsize=12)
ax.set_title('PARM vs EPEC — PKU-SafeRLHF-10K test', fontsize=13)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=11)

plt.tight_layout()
out = base / 'pareto_compare.png'
plt.savefig(out, dpi=150)
print(f'Saved: {out}')
