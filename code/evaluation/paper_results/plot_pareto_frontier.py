"""Pareto plot: EPEC+PARM vs EPEC+GenARM vs GenARM vs PARM vs MOD."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

base = Path(__file__).resolve().parent.parent

alphas = [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6), (0.5, 0.5),
          (0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]


def load_mean(path):
    d = json.load(open(path))
    return d['help'], d['harm']


epec_genarm, genarm, parm = [], [], []
for ah, am in alphas:
    eh, em = load_mean(base / f'results_epec_1000/EPEC_GenARM_{ah}help_{am}harm_tau0.1_k50/mean_result.json')
    gh, gm = load_mean(base / f'results_genarm_1000/GenARM_{ah}help_{am}harm/mean_result.json')
    ph, pm_ = load_mean(base / f'results/PARM_{ah}help_{am}harm/mean_result.json')
    epec_genarm.append({'a_help': ah, 'a_harm': am, 'help': eh, 'harm': em})
    genarm.append({'a_help': ah, 'a_harm': am, 'help': gh, 'harm': gm})
    parm.append({'a_help': ah, 'a_harm': am, 'help': ph, 'harm': pm_})

# Add PARM 0.9 point
ph9, pm9 = load_mean(base / 'results/PARM_0.9help_0.1harm/mean_result.json')
parm.append({'a_help': 0.9, 'a_harm': 0.1, 'help': ph9, 'harm': pm9})
parm.sort(key=lambda x: x['a_help'])

epec_parm = []
for d in sorted(base.glob('results_epec_parm/*/mean_result.json')):
    r = json.load(open(d))
    name = d.parent.name
    ah = float(name.split('_')[2].replace('help', ''))
    am = float(name.split('_')[3].replace('harm', ''))
    epec_parm.append({'a_help': ah, 'a_harm': am, 'help': r['help'], 'harm': r['harm']})
epec_parm.sort(key=lambda x: x['a_help'])

mod = []
for d in sorted(base.glob('results_mod/*/mean_result.json')):
    r = json.load(open(d))
    name = d.parent.name
    ah = float(name.split('_')[1].replace('help', ''))
    am = float(name.split('_')[2].replace('harm', ''))
    mod.append({'a_help': ah, 'a_harm': am, 'help': r['help'], 'harm': r['harm']})
mod.sort(key=lambda x: x['a_help'])

fig, ax = plt.subplots(figsize=(9, 7))

ax.plot([r['help'] for r in parm], [-r['harm'] for r in parm],
        '^-', color='purple', lw=2, ms=10, label='PARM', zorder=3)
ax.plot([r['help'] for r in genarm], [-r['harm'] for r in genarm],
        's-', color='steelblue', lw=2, ms=11, label='GenARM', zorder=4)
ax.plot([r['help'] for r in mod], [-r['harm'] for r in mod],
        'P-', color='darkorange', lw=2, ms=11, label='MOD', zorder=4)
ax.plot([r['help'] for r in epec_genarm], [-r['harm'] for r in epec_genarm],
        'o-', color='crimson', lw=2, ms=11, label='EPEC+GenARM', zorder=5)
ax.plot([r['help'] for r in epec_parm], [-r['harm'] for r in epec_parm],
        'D-', color='forestgreen', lw=2.5, ms=13, label='EPEC+PARM', zorder=6)

ax.set_xlabel('Helpfulness Score  (higher = better)', fontsize=13)
ax.set_ylabel('Harmlessness Score  (higher = safer)', fontsize=13)
ax.set_title('Token-level EPEC vs Baselines — 1000 PKU-SafeRLHF test prompts', fontsize=13)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=11, loc='best')

plt.tight_layout()
out = Path(__file__).parent / 'pareto_frontier.png'
plt.savefig(out, dpi=150)
print(f'Saved: {out}')
