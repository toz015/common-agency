"""4-curve Pareto: ALL n=50 (apples-to-apples). PARM + GenARM, logit-sum + EPEC."""
import matplotlib.pyplot as plt

# All data is n=50 (logit-sum re-aggregated over first 50 prompts of n=100 baseline)
parm_ls = [
    {'a': 0.0, 'help': -1.1520, 'harm': -12.3397},
    {'a': 0.1, 'help':  0.2510, 'harm': -11.6822},
    {'a': 0.2, 'help': -0.7837, 'harm': -10.1220},
    {'a': 0.3, 'help':  0.2487, 'harm': -7.7647},
    {'a': 0.4, 'help':  0.3592, 'harm': -4.0094},
    {'a': 0.5, 'help':  1.3570, 'harm':  0.3106},
    {'a': 0.6, 'help':  2.9471, 'harm':  4.4219},
    {'a': 0.7, 'help':  3.3525, 'harm':  8.4657},
    {'a': 0.8, 'help':  3.7871, 'harm': 12.6788},
    {'a': 0.9, 'help':  4.5625, 'harm': 12.9152},
    {'a': 1.0, 'help':  4.5555, 'harm': 13.7916},
]
genarm_ls = [
    {'a': 0.0, 'help': -2.6604, 'harm': -13.3938},
    {'a': 0.1, 'help': -1.0914, 'harm': -12.9098},
    {'a': 0.2, 'help': -0.1725, 'harm': -10.7920},
    {'a': 0.3, 'help':  1.5685, 'harm': -7.4373},
    {'a': 0.4, 'help':  3.6100, 'harm':  1.0966},
    {'a': 0.5, 'help':  4.4359, 'harm':  7.0361},
    {'a': 0.6, 'help':  5.7278, 'harm': 10.5196},
    {'a': 0.7, 'help':  5.7273, 'harm': 12.3251},
    {'a': 0.8, 'help':  5.3851, 'harm': 12.6938},
    {'a': 0.9, 'help':  4.9501, 'harm': 13.2650},
    {'a': 1.0, 'help':  4.9630, 'harm': 14.1434},
]
parm_epec = [
    {'a': 0.0, 'help': 0.10845703125,    'harm': -6.0087109375},
    {'a': 0.2, 'help': 0.01865234375,    'harm': -7.517109375},
    {'a': 0.4, 'help': 0.2196484375,     'harm': -0.2930224609375},
    {'a': 0.6, 'help': 1.31834716796875, 'harm':  9.168125},
    {'a': 0.8, 'help': 1.634921875,      'harm': 16.2594921875},
    {'a': 1.0, 'help': 3.682421875,      'harm': 21.8378515625},
]
genarm_epec = [
    {'a': 0.0, 'help': 3.7981640625,    'harm': -12.8725},
    {'a': 0.2, 'help': 4.15306640625,   'harm': -10.7030859375},
    {'a': 0.4, 'help': 4.9874609375,    'harm': -9.5079296875},
    {'a': 0.6, 'help': 4.660380859375,  'harm': -5.3732421875},
    {'a': 0.8, 'help': 4.74908203125,   'harm':  6.2409423828125},
    {'a': 1.0, 'help': 8.528828125,     'harm': 19.7378515625},
]


def hv2d(points, ref):
    pts = sorted([(x, y) for x, y in points if x > ref[0] and y > ref[1]],
                 key=lambda p: -p[0])
    if not pts: return 0.0
    hv = 0.0
    last_y = ref[1]
    for x, y in pts:
        if y > last_y:
            hv += (x - ref[0]) * (y - last_y)
            last_y = y
    return hv


def plot_curve(ax, data, marker, color, label, label_offset=(5, 8)):
    x = [r['help'] for r in data]
    y = [-r['harm'] for r in data]
    ax.plot(x, y, marker, color=color, label=label, linewidth=2, markersize=8, alpha=0.9)
    for r in data:
        ax.annotate(f"{r['a']:.1f}", (r['help'], -r['harm']),
                    textcoords='offset points', xytext=label_offset,
                    fontsize=8, color=color, alpha=0.85)


fig, ax = plt.subplots(figsize=(11, 8))

plot_curve(ax, parm_ls,    'o--', '#4c72b0', 'PARM logit-sum',   (5, -10))
plot_curve(ax, genarm_ls,  'o--', '#55a868', 'GenARM logit-sum', (5, -10))
plot_curve(ax, parm_epec,  's-',  '#c44e52', 'PARM EPEC',        (5, 8))
plot_curve(ax, genarm_epec,'s-',  '#8172b2', 'GenARM EPEC',      (5, 8))

ax.set_xlabel('Helpfulness  (Beaver-7B reward; →higher is more helpful)', fontsize=12)
ax.set_ylabel('Safety  (–Beaver-7B cost; →higher is safer)', fontsize=12)
ax.set_title('W2S 65B EPEC vs logit-sum (PARM and GenARM), all curves on n=50\n'
             'base: alpaca-lora-65B-GPTQ  |  ARM: alpaca-7b-reproduced + LoRA  |  max_tok=512',
             fontsize=12)
ax.grid(True, alpha=0.3, linestyle=':')
ax.legend(loc='lower left', fontsize=11, framealpha=0.95)
ax.axhline(0, color='gray', linewidth=0.6, alpha=0.4)
ax.axvline(0, color='gray', linewidth=0.6, alpha=0.4)
ax.text(0.02, 0.98, 'Upper-right = better\n(more helpful AND safer)',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.85))

plt.tight_layout()
out = '/tmp/pareto_4curve_w2s_epec_n50.png'
plt.savefig(out, dpi=140, bbox_inches='tight')
print(f"Saved: {out}\n")

# HV with shared reference
all_pts = parm_ls + genarm_ls + parm_epec + genarm_epec
help_min = min(r['help'] for r in all_pts)
safety_min = min(-r['harm'] for r in all_pts)
ref = (help_min - 1, safety_min - 1)
print(f"HV reference: help={ref[0]:.2f}, safety={ref[1]:.2f}")
for name, data in [('PARM logit-sum', parm_ls), ('GenARM logit-sum', genarm_ls),
                   ('PARM EPEC', parm_epec),    ('GenARM EPEC', genarm_epec)]:
    pts = [(r['help'], -r['harm']) for r in data]
    hv = hv2d(pts, ref)
    print(f"  {name:24}  HV = {hv:.2f}")

print("\nEPEC vs logit-sum within-method:")
for method, ls, ep in [('GenARM', genarm_ls, genarm_epec), ('PARM', parm_ls, parm_epec)]:
    hv_ls = hv2d([(r['help'], -r['harm']) for r in ls], ref)
    hv_ep = hv2d([(r['help'], -r['harm']) for r in ep], ref)
    delta = hv_ep - hv_ls
    print(f"  {method}: ls HV {hv_ls:.2f} -> ep HV {hv_ep:.2f}  (Δ {delta:+.2f}, {100*delta/hv_ls:+.1f}%)")
