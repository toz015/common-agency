"""
Plot the PARM Pareto frontier from the preference vector sweep.
Reads mean_result.json from each results/PARM_*/  folder and plots
helpfulness vs. harm cost.
"""

import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

results_dir = Path(__file__).parent / 'results'

points = []
for mean_file in sorted(results_dir.glob('PARM_*/mean_result.json')):
    folder = mean_file.parent.name  # e.g. PARM_0.3help_0.7harm
    m = re.search(r'PARM_([\d.]+)help_([\d.]+)harm', folder)
    alpha_h = float(m.group(1))
    alpha_s = float(m.group(2))
    d = json.loads(mean_file.read_text())
    points.append((alpha_h, alpha_s, d['help'], d['harm']))

points.sort(key=lambda x: x[0])  # sort by alpha_helpfulness

alpha_h_vals  = [p[0] for p in points]
help_scores   = [p[2] for p in points]
harm_scores   = [p[3] for p in points]
labels        = [f'({p[0]:.1f}, {p[1]:.1f})' for p in points]

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(harm_scores, help_scores, 'o-', color='steelblue', linewidth=2, markersize=7, zorder=3)

for x, y, lbl in zip(harm_scores, help_scores, labels):
    ax.annotate(lbl, (x, y), textcoords='offset points', xytext=(6, 4), fontsize=7.5, color='#333333')

ax.set_xlabel('Harm Cost  (lower = safer)', fontsize=12)
ax.set_ylabel('Helpfulness Score  (higher = better)', fontsize=12)
ax.set_title('PARM Pareto Frontier\n(α_help, α_harm) sweep — step 0.1', fontsize=13)
ax.grid(True, linestyle='--', alpha=0.5)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

plt.tight_layout()
out = Path(__file__).parent / 'pareto_frontier.png'
plt.savefig(out, dpi=150)
print(f'Saved: {out}')
plt.show()
