"""Pareto plot: EPEC+GenARM vs GenARM vs PARM, 1000-prompt full results."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

base = Path(__file__).parent

alphas = [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]


def load_mean(path):
    d = json.load(open(path))
    return d['help'], d['harm']


epec, genarm = [], []
for ah, aharm in alphas:
    eh, em = load_mean(base / f'results_epec_1000/EPEC_GenARM_{ah}help_{aharm}harm_tau0.1_k50/mean_result.json')
    gh, gm = load_mean(base / f'results_genarm_1000/GenARM_{ah}help_{aharm}harm/mean_result.json')
    epec.append({'a_help': ah, 'a_harm': aharm, 'help': eh, 'harm': em})
    genarm.append({'a_help': ah, 'a_harm': aharm, 'help': gh, 'harm': gm})

parm = json.load(open('/tmp/parm_1000.json'))

fig, ax = plt.subplots(figsize=(9, 7))

ax.plot([r['help'] for r in parm], [-r['harm'] for r in parm],
        '^-', color='purple', lw=2, ms=10, label='PARM (10 α, ModelArithmetic)', zorder=3)
ax.plot([r['help'] for r in genarm], [-r['harm'] for r in genarm],
        's-', color='steelblue', lw=2, ms=11, label='GenARM linear blend (8 α)', zorder=4)
ax.plot([r['help'] for r in epec], [-r['harm'] for r in epec],
        'o-', color='crimson', lw=2.5, ms=13, label='EPEC + GenARM (8 α, τ=0.1, k=50)', zorder=5)

for r in parm:
    ax.annotate(f'({r["a_help"]:.1f},{r["a_harm"]:.1f})', (r['help'], -r['harm']),
                textcoords='offset points', xytext=(5, -12), fontsize=7, color='purple')
for r in genarm:
    ax.annotate(f'({r["a_help"]:.1f},{r["a_harm"]:.1f})', (r['help'], -r['harm']),
                textcoords='offset points', xytext=(5, 6), fontsize=8, color='steelblue')
for r in epec:
    ax.annotate(f'({r["a_help"]:.1f},{r["a_harm"]:.1f})', (r['help'], -r['harm']),
                textcoords='offset points', xytext=(5, -14), fontsize=8, color='crimson')

ax.set_xlabel('Helpfulness Score  (higher = better)', fontsize=13)
ax.set_ylabel('Harmlessness Score  (higher = safer)', fontsize=13)
ax.set_title('Token-level EPEC vs Baselines — 1000 PKU-SafeRLHF test prompts', fontsize=13)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=11, loc='best')

plt.tight_layout()
out = base / 'pareto_epec_vs_baselines_1000.png'
plt.savefig(out, dpi=150)
print(f'Saved: {out}')

print('\n=== Summary ===')
print(f'{"Method":<25} {"α":>12} {"help":>8} {"harm":>8}')
print('-' * 55)
for r in epec:
    print(f'{"EPEC+GenARM":<25} ({r["a_help"]},{r["a_harm"]}){r["help"]:>+8.2f}{r["harm"]:>+8.2f}')
for r in genarm:
    print(f'{"GenARM":<25} ({r["a_help"]},{r["a_harm"]}){r["help"]:>+8.2f}{r["harm"]:>+8.2f}')
for r in parm:
    print(f'{"PARM":<25} ({r["a_help"]},{r["a_harm"]}){r["help"]:>+8.2f}{r["harm"]:>+8.2f}')
