'''
Compute 2D safety-alignment HV and MIP for the BoN rebuttal ablation.

For each of three methods evaluated on the first M test prompts:
    - CAGE  (subset of CAGE 1000-prompt reward_result.json on the remote branch)
    - GenARM greedy  (subset of GenARM 1000-prompt reward_result.json)
    - GenARM-BoN@N   (new: reward_result.json from score_and_rerank_bon.py)

...we compute mean help, mean harmlessness (= -harm cost), then aggregate:
    HV  in the (helpfulness, harmlessness) plane
    MIP = mean over preferences of alpha . q_normalized

HV and MIP conventions mirror plot_pareto_and_metrics.py:
    - reference point: per-axis global min minus a margin (1.0 here)
    - normalization for MIP: per-axis global (min, max) across all methods
    - pymoo HV if available; MC fallback otherwise

Emits a Markdown table for direct paste into the rebuttal.
'''
import argparse
import json
import re
from pathlib import Path

import numpy as np

# Paper's α_help sweep — matches the 8 preference vectors of Table 1.
ALPHA_HELPS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def _fmt_alpha(x: float) -> str:
    '''Match the directory naming used by both the paper and this repo.'''
    s = f'{x:g}'
    return s


def _cage_dir(base: Path, ah: float) -> Path:
    return base / f'EPEC_GenARM_{_fmt_alpha(ah)}help_{_fmt_alpha(1 - ah)}harm_tau0.1_k50'


def _genarm_dir(base: Path, ah: float) -> Path:
    return base / f'GenARM_{_fmt_alpha(ah)}help_{_fmt_alpha(1 - ah)}harm'


def _bon_dir(base: Path, ah: float, n: int) -> Path:
    return base / f'GenARM_BoN_N{n}_{_fmt_alpha(ah)}help_{_fmt_alpha(1 - ah)}harm'


def load_mean_for_method(dir_getter, first_m: int):
    '''
    For each alpha, load reward_result.json, subset to the first `first_m`
    entries (sorted by eval index), and compute mean help / mean harm.
    Returns dict alpha -> {'help': float, 'harm_cost': float, 'n': int}.
    '''
    out = {}
    for ah in ALPHA_HELPS:
        d = dir_getter(ah)
        rr = d / 'reward_result.json'
        if not rr.exists():
            print(f'  MISSING: {rr}')
            continue
        entries = json.loads(rr.read_text())
        # Sort by numeric uid ordinal for stability, then take first_m.
        def uid_key(e):
            u = e['uid']
            m = re.match(r'eval(\d+)$', u)
            return int(m.group(1)) if m else u
        entries.sort(key=uid_key)
        entries = entries[:first_m]
        if len(entries) < first_m:
            print(f'  WARN: {rr} has only {len(entries)} entries (< {first_m})')
        help_scores = [e['help_score (high better)'] for e in entries]
        harm_scores = [e['harm_score (low better)'] for e in entries]
        out[ah] = {
            'help': float(np.mean(help_scores)),
            'harm_cost': float(np.mean(harm_scores)),
            'n': len(entries),
        }
    return out


def compute_hv_2d(points: np.ndarray, ref: np.ndarray) -> float:
    '''2D HV, all objectives maximized. points shape (P, 2).'''
    if len(points) == 0:
        return 0.0
    try:
        from pymoo.indicators.hv import HV
        return float(HV(ref_point=-ref)(-points))
    except ImportError:
        pass
    # Simple MC fallback (500k samples).
    non_dom = []
    for i in range(len(points)):
        dom = False
        for j in range(len(points)):
            if i == j:
                continue
            if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                dom = True
                break
        if not dom:
            non_dom.append(points[i])
    non_dom = np.array(non_dom) if non_dom else points
    bounds_high = np.max(non_dom, axis=0)
    if np.any(bounds_high <= ref):
        return 0.0
    np.random.seed(42)
    samples = np.random.uniform(ref, bounds_high, size=(500_000, 2))
    total_vol = float(np.prod(bounds_high - ref))
    is_dom = np.zeros(500_000, dtype=bool)
    for pt in non_dom:
        is_dom |= np.all(samples <= pt, axis=1)
    return total_vol * float(np.mean(is_dom))


def method_metrics(mean_by_alpha, gmin, gmax, ref):
    '''
    Given per-alpha (help, harm_cost), produce:
        pts_raw     : list of (help, -harm_cost) points in the 2D maximize plane
        HV_raw      : hypervolume in raw units
        MIP         : mean alpha . q_norm over alphas
    '''
    pts_raw = []
    mip_vals = []
    for ah in sorted(mean_by_alpha.keys()):
        m = mean_by_alpha[ah]
        h, harmless = m['help'], -m['harm_cost']
        pts_raw.append((h, harmless))
        # MIP: per-axis normalize to [0, 1] using global (min, max).
        q_raw = np.array([h, harmless])
        q_norm = (q_raw - gmin) / (gmax - gmin + 1e-12)
        alpha_vec = np.array([ah, 1 - ah])
        mip_vals.append(float(np.dot(alpha_vec, q_norm)))
    pts_raw = np.array(pts_raw)
    hv_raw = compute_hv_2d(pts_raw, ref)
    mip = float(np.mean(mip_vals)) if mip_vals else 0.0
    return pts_raw, hv_raw, mip


def parse_arguments():
    p = argparse.ArgumentParser(
        description='Compute 2D HV/MIP for CAGE vs GenARM greedy vs GenARM-BoN on the first M prompts.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--first_m', type=int, default=50)
    p.add_argument('--num_candidates', type=int, default=5,
                   help='N used in BoN (for finding the BoN result dirs).')
    p.add_argument('--cage_root',
                   default='./reference_results_1000/results_epec_1000',
                   type=str)
    p.add_argument('--genarm_root',
                   default='./reference_results_1000/results_genarm_1000',
                   type=str)
    p.add_argument('--bon_root',
                   default='./results_bon',
                   type=str)
    p.add_argument('--output_json',
                   default='./metrics/safety_bon_ablation.json',
                   type=str)
    return p.parse_args()


def main():
    args = parse_arguments()
    cage_root = Path(args.cage_root)
    genarm_root = Path(args.genarm_root)
    bon_root = Path(args.bon_root)

    print(f'Subsetting all methods to first {args.first_m} prompts (uids eval0..eval{args.first_m - 1})\n')

    print('CAGE:')
    cage = load_mean_for_method(lambda a: _cage_dir(cage_root, a), args.first_m)
    print('GenARM greedy:')
    genarm = load_mean_for_method(lambda a: _genarm_dir(genarm_root, a), args.first_m)
    print(f'GenARM-BoN@{args.num_candidates}:')
    bon = load_mean_for_method(lambda a: _bon_dir(bon_root, a, args.num_candidates), args.first_m)

    methods_all = [('CAGE', cage), ('GenARM greedy', genarm),
                   (f'GenARM-BoN@{args.num_candidates}', bon)]
    # Drop methods with zero loaded alphas (e.g. BoN before it's been run).
    methods = [(name, m) for name, m in methods_all if len(m) > 0]
    dropped = [name for name, m in methods_all if len(m) == 0]
    if dropped:
        print(f'\nWARN: no results for {dropped} -- excluding from comparison.\n')
    if not methods:
        raise SystemExit('No methods have any results; nothing to compare.')

    # Common alphas present in every retained method (like-for-like comparison).
    common_alphas = set(ALPHA_HELPS)
    for _, m in methods:
        common_alphas &= set(m.keys())
    common_alphas = sorted(common_alphas)
    print(f'\nComparing on the {len(common_alphas)} shared alpha values: {common_alphas}')
    for name, m in methods:
        for a in list(m.keys()):
            if a not in common_alphas:
                del m[a]

    # Build the raw (help, -harm_cost) point cloud for all methods jointly,
    # then derive per-axis (gmin, gmax) and the HV reference point.
    all_pts = []
    for _, m in methods:
        for a in common_alphas:
            all_pts.append((m[a]['help'], -m[a]['harm_cost']))
    all_pts = np.array(all_pts)
    gmin = all_pts.min(axis=0)
    gmax = all_pts.max(axis=0)
    ref = gmin - 1.0   # per-axis margin, same as plot_pareto_and_metrics.py

    print(f'\nGlobal axis ranges:')
    print(f'  helpfulness : [{gmin[0]:+.3f}, {gmax[0]:+.3f}]')
    print(f'  harmlessness: [{gmin[1]:+.3f}, {gmax[1]:+.3f}]')
    print(f'  HV ref point: ({ref[0]:+.3f}, {ref[1]:+.3f})')

    rows = []
    for name, m in methods:
        pts, hv, mip = method_metrics(m, gmin, gmax, ref)
        rows.append((name, pts, hv, mip))

    # Emit a Markdown table.
    print('\n\n=== Rebuttal ablation table (first {} prompts, 2D safety plane) ==='.format(args.first_m))
    print('| Method | HV | MIP |')
    print('|---|---:|---:|')
    for name, _, hv, mip in rows:
        print(f'| {name} | {hv:.2f} | {mip:.4f} |')

    # Per-alpha inspection so we can eyeball where BoN closes / doesn't close the gap.
    print('\n=== Per-alpha means (help / harm_cost) ===')
    header = 'alpha_help  ' + '  '.join(f'{name:>28s}' for name, _ in methods)
    print(header)
    for a in common_alphas:
        cells = []
        for _, m in methods:
            cells.append(f'help={m[a]["help"]:+.3f} cost={m[a]["harm_cost"]:+.3f}')
        print(f'   {a:.1f}      ' + '  '.join(f'{c:>28s}' for c in cells))

    # Persist.
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    method_payload = {}
    for (name, pts, hv, mip), (_, m_dict) in zip(rows, methods):
        method_payload[name] = {
            'HV': round(hv, 4),
            'MIP': round(mip, 6),
            'points_raw_help_harmless': pts.tolist(),
            'per_alpha': {f'{a:.1f}': m_dict[a] for a in common_alphas},
        }
    payload = {
        'first_m': args.first_m,
        'num_candidates': args.num_candidates,
        'alphas': common_alphas,
        'axis_min': gmin.tolist(),
        'axis_max': gmax.tolist(),
        'ref_point': ref.tolist(),
        'methods': method_payload,
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2))
    print(f'\nSaved JSON: {args.output_json}')


if __name__ == '__main__':
    main()
