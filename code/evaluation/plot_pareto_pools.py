"""EPEC solver on two candidate pools (base-alpaca vs PARM), 1000-prompt subset.
Same solver, same prompts, different candidate source — isolates pool quality.
Also plots argmax-log_prob baselines on both pools (most-likely candidate)."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

base = Path(__file__).parent
epec_base = json.loads((base / 'epec_results' / 'epec_sweep_1000_tau0.1.json').read_text())
epec_parm = json.loads((base / 'epec_results' / 'epec_sweep_parm_tau0.1.json').read_text())
epec_base_norm = json.loads((base / 'epec_results' / 'epec_sweep_1000_tau0.1_norm.json').read_text())
epec_parm_norm = json.loads((base / 'epec_results' / 'epec_sweep_parm_tau0.1_norm.json').read_text())
epec_base.sort(key=lambda r: r['w_help'])
epec_parm.sort(key=lambda r: r['w_help'])
epec_base_norm.sort(key=lambda r: r['w_help'])
epec_parm_norm.sort(key=lambda r: r['w_help'])

# Restrict to the same 1000 uids the PARM pool uses
parm_pool_uids = set()
for r in json.loads((base / 'scored_parm' / 'scored_candidates_PARM_all_alphas_N18.json').read_text()):
    parm_pool_uids.add(r['uid'])


def _extreme_logprob_mean(scored_path, uid_filter, fn):
    scored = json.loads(Path(scored_path).read_text())
    by_uid = {}
    for r in scored:
        if r['uid'] in uid_filter:
            by_uid.setdefault(r['uid'], []).append(r)
    helps, harms = [], []
    for uid, g in by_uid.items():
        g.sort(key=lambda r: r['candidate_id'])
        a = int(fn([r['log_prob'] for r in g]))
        helps.append(g[a]['q_help'])
        harms.append(g[a]['q_harm'])
    return float(np.mean(helps)), float(np.mean(harms)), len(helps)


def argmax_logprob_mean(scored_path, uid_filter):
    return _extreme_logprob_mean(scored_path, uid_filter, np.argmax)


def argmin_logprob_mean(scored_path, uid_filter):
    return _extreme_logprob_mean(scored_path, uid_filter, np.argmin)


bh, bm, nb = argmax_logprob_mean(
    base / 'scored' / 'scored_candidates_test_prompt_only_alpaca-7b-reproduced_N20.json',
    parm_pool_uids)
ph, pm, np_n = argmax_logprob_mean(
    base / 'scored_parm' / 'scored_candidates_PARM_all_alphas_N18.json',
    parm_pool_uids)
print(f'argmax log_prob on base-alpaca pool (n={nb}): help={bh:+.3f}  harm={bm:+.3f}')
print(f'argmax log_prob on PARM pool      (n={np_n}): help={ph:+.3f}  harm={pm:+.3f}')

# PARM curve: for each generation alpha, argmax-log_prob among the 3 samples
parm_alphas = [(0.0, 1.0), (0.2, 0.8), (0.4, 0.6), (0.6, 0.4), (0.8, 0.2), (1.0, 0.0)]
parm_curve = []
for ah, aharm in parm_alphas:
    f = base / 'scored_parm' / f'scored_candidates_PARM_{ah}help_{aharm}harm_N3.json'
    mh, mharm, n = argmax_logprob_mean(f, parm_pool_uids)
    parm_curve.append({'a_help': ah, 'a_harm': aharm, 'mean_help': mh, 'mean_harm': mharm})
    print(f'PARM α=({ah},{aharm}) argmax log_prob (n={n}): help={mh:+.3f}  harm={mharm:+.3f}')

parm_curve_min = []
for ah, aharm in parm_alphas:
    f = base / 'scored_parm' / f'scored_candidates_PARM_{ah}help_{aharm}harm_N3.json'
    mh, mharm, n = argmin_logprob_mean(f, parm_pool_uids)
    parm_curve_min.append({'a_help': ah, 'a_harm': aharm, 'mean_help': mh, 'mean_harm': mharm})
    print(f'PARM α=({ah},{aharm}) argmin log_prob (n={n}): help={mh:+.3f}  harm={mharm:+.3f}')

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([r['mean_help'] for r in epec_base], [-r['mean_harm'] for r in epec_base],
        's-', color='crimson', lw=2, ms=7, label='EPEC base pool (raw, τ=0.1)', zorder=3)
ax.plot([r['mean_help'] for r in epec_parm], [-r['mean_harm'] for r in epec_parm],
        'o-', color='steelblue', lw=2, ms=7, label='EPEC PARM pool (raw, τ=0.1)', zorder=3)
ax.plot([r['mean_help'] for r in epec_base_norm], [-r['mean_harm'] for r in epec_base_norm],
        's--', color='lightcoral', lw=1.5, ms=6, label='EPEC base pool (normalized, τ=0.1)', zorder=3)
ax.plot([r['mean_help'] for r in epec_parm_norm], [-r['mean_harm'] for r in epec_parm_norm],
        'o--', color='lightsteelblue', lw=1.5, ms=6, label='EPEC PARM pool (normalized, τ=0.1)', zorder=3)
ax.plot([bh], [-bm], '*', color='darkorange', ms=20, mec='black', mew=0.8,
        label='argmax π_base on base-alpaca pool', zorder=5)
ax.plot([ph], [-pm], 'D', color='mediumseagreen', ms=13, mec='black', mew=0.8,
        label='argmax π_base on PARM pool (all α)', zorder=5)
ax.plot([r['mean_help'] for r in parm_curve], [-r['mean_harm'] for r in parm_curve],
        '^-', color='purple', lw=2, ms=9, label='PARM (argmax π_base per α)', zorder=4)
ax.plot([r['mean_help'] for r in parm_curve_min], [-r['mean_harm'] for r in parm_curve_min],
        'v--', color='teal', lw=2, ms=9, label='PARM (argmin π_base per α)', zorder=4)
for r in parm_curve:
    ax.annotate(f'({r["a_help"]:.1f},{r["a_harm"]:.1f})', (r['mean_help'], -r['mean_harm']),
                textcoords='offset points', xytext=(6, -12), fontsize=7, color='purple')
for r in parm_curve_min:
    ax.annotate(f'({r["a_help"]:.1f},{r["a_harm"]:.1f})', (r['mean_help'], -r['mean_harm']),
                textcoords='offset points', xytext=(6, -12), fontsize=7, color='teal')

for r in epec_base:
    ax.annotate(f'({r["w_help"]:.1f},{r["w_harm"]:.1f})', (r['mean_help'], -r['mean_harm']),
                textcoords='offset points', xytext=(6, -10), fontsize=7, color='crimson')
for r in epec_parm:
    ax.annotate(f'({r["w_help"]:.1f},{r["w_harm"]:.1f})', (r['mean_help'], -r['mean_harm']),
                textcoords='offset points', xytext=(6, 4), fontsize=7, color='steelblue')

ax.set_xlabel('Helpfulness Score  (higher = better)', fontsize=12)
ax.set_ylabel('Harmlessness Score  (higher = safer)', fontsize=12)
ax.set_title('EPEC solver: pool-quality ablation  —  1000 PKU-SafeRLHF test prompts', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=11, loc='best')

plt.tight_layout()
out = base / 'pareto_pools.png'
plt.savefig(out, dpi=150)
print(f'Saved: {out}')
