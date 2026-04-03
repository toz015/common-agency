"""
Step 3 of Common-Agency EPEC Evaluation Pipeline
=================================================
Read the scored-candidate JSON from Step 2, compute three policy distributions
over the candidate set for each prompt, and save the augmented results.

Methods implemented:
  1. PARM baseline        – fixed preference-weighted reward aggregation
  2. Naive aggregation    – equal-weight sum of raw rewards  (q-update baseline)
  3. Common Agency EPEC   – Nash equilibrium via Nonlinear Jacobi iteration

Input  : scored_candidates_<run_tag>.json  (from score_candidates.py)
Output : solved_equilibrium_<run_tag>.json  (augmented with policy & metrics)

Usage example:
  python solve_equilibrium.py \
      --input_file scored/scored_candidates_test_prompt_only_alpaca-7b-reproduced_N20.json \
      --temperature_tau 1.0 \
      --max_iterations 50 \
      --normalize_rewards
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


# =========================================================================== #
#  Core numerical primitives
# =========================================================================== #

def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    logits = logits - np.max(logits)
    e = np.exp(logits)
    return e / e.sum()


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-30) -> float:
    """KL(p || q) = sum p_i log(p_i / q_i).  Clips to avoid log(0)."""
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return float(np.sum(p * np.log(p / q)))


def min_max_normalize(v: np.ndarray, invert: bool = False) -> np.ndarray:
    """Min-max normalise a vector to [0, 1].  Returns zeros if constant.
    If invert=True, lower values map to 1.0 (better)."""
    vmin, vmax = v.min(), v.max()
    if vmax - vmin < 1e-12:
        return np.zeros_like(v)
    norm = (v - vmin) / (vmax - vmin)
    return 1.0 - norm if invert else norm


# =========================================================================== #
#  Method 1:  PARM Baseline  (fixed preference scaling)
# =========================================================================== #

def compute_parm(q_help: np.ndarray,
                 q_harm: np.ndarray,
                 pi_base: np.ndarray,
                 tau: float,
                 weights: tuple[float, float] = (0.5, 0.5)) -> np.ndarray:
    r"""
    PARM baseline (Lin et al., 2025).

    .. math::
        \pi_{\text{parm}} \propto \pi_{\text{base}}
            \cdot \exp\!\Bigl(\frac{w_1 q^1 + w_2 q^2}{\tau}\Bigr)

    Parameters
    ----------
    q_help, q_harm : (N,) reward vectors
    pi_base        : (N,) base policy
    tau            : temperature
    weights        : (w_help, w_harm)

    Returns
    -------
    pi_parm : (N,) normalised policy
    """
    w_help, w_harm = weights
    logits = np.log(pi_base + 1e-30) + (w_help * q_help + w_harm * q_harm) / tau
    return softmax(logits)


# =========================================================================== #
#  Method 2:  Naive Aggregation  (q-update baseline)
# =========================================================================== #

def compute_naive(q_help: np.ndarray,
                  q_harm: np.ndarray,
                  pi_base: np.ndarray,
                  tau: float) -> np.ndarray:
    r"""
    Naive aggregation — sum of reward signals used directly as incentives.

    .. math::
        \pi_{\text{naive}} \propto \pi_{\text{base}}
            \cdot \exp\!\Bigl(\frac{q^1 + q^2}{\tau}\Bigr)
    """
    logits = np.log(pi_base + 1e-30) + (q_help + q_harm) / tau
    return softmax(logits)


# =========================================================================== #
#  Method 3:  Common Agency EPEC  (Nonlinear Jacobi)
# =========================================================================== #

def agent_best_response(Y: np.ndarray,
                        pi_base: np.ndarray,
                        tau: float) -> np.ndarray:
    r"""KL-regularised best response: π = softmax(log π_base + Y / τ)."""
    logits = np.log(pi_base + 1e-30) + Y / tau
    return softmax(logits)


def jacobian_pi(Y: np.ndarray,
                pi_base: np.ndarray,
                tau: float) -> np.ndarray:
    r"""Jacobian J_π(Y) = (1/τ)(diag(π) − π πᵀ)."""
    pi = agent_best_response(Y, pi_base, tau)
    return (np.diag(pi) - np.outer(pi, pi)) / tau


def solve_mpec_principal(i: int,
                         y_all: list[np.ndarray],
                         q: list[np.ndarray],
                         pi_base: np.ndarray,
                         tau: float) -> np.ndarray:
    r"""
    Solve principal *i*'s MPEC:

        max_{0 ≤ yⁱ ≤ qⁱ}  π(Y⁻ⁱ + yⁱ)ᵀ (qⁱ − yⁱ)

    with the others' transfers held fixed.  Uses L-BFGS-B with multi-start
    (same pattern as ``common_agency_simulation.py``).
    """
    N = q[i].shape[0]
    Y_minus_i = sum(y_all[j] for j in range(len(y_all)) if j != i)

    def neg_objective(yi):
        Y = Y_minus_i + yi
        pi = agent_best_response(Y, pi_base, tau)
        return -pi.dot(q[i] - yi)

    def neg_objective_grad(yi):
        Y = Y_minus_i + yi
        pi = agent_best_response(Y, pi_base, tau)
        J = jacobian_pi(Y, pi_base, tau)
        grad = J.T @ (q[i] - yi) - pi          # stationarity
        return -grad

    bounds = [(0.0, max(q[i][a], 0.0)) for a in range(N)]

    best_val = np.inf
    best_x = y_all[i].copy()

    starts = [
        y_all[i].copy(),
        np.maximum(q[i] * 0.5, 0.0),
        np.maximum(q[i] * 0.1, 0.0),
        np.maximum(q[i] * 0.9, 0.0),
    ]

    for x0 in starts:
        x0 = np.clip(x0, 0, np.maximum(q[i], 0.0))
        res = minimize(
            neg_objective, x0,
            jac=neg_objective_grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13},
        )
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x.copy()

    return best_x


def kkt_residual(y_all: list[np.ndarray],
                 q: list[np.ndarray],
                 pi_base: np.ndarray,
                 tau: float) -> float:
    """Max KKT violation across all principals and actions."""
    J_count = len(y_all)
    N = q[0].shape[0]
    Y = sum(y_all)
    pi = agent_best_response(Y, pi_base, tau)
    J_pi = jacobian_pi(Y, pi_base, tau)

    max_violation = 0.0
    for i in range(J_count):
        grad = J_pi.T @ (q[i] - y_all[i]) - pi
        for a in range(N):
            if y_all[i][a] <= 1e-12:                      # lower bound
                violation = max(0.0, grad[a])
            elif y_all[i][a] >= q[i][a] - 1e-12:          # upper bound
                violation = max(0.0, -grad[a])
            else:                                          # interior
                violation = abs(grad[a])
            max_violation = max(max_violation, violation)
    return max_violation


def jacobi_method(q: list[np.ndarray],
                  pi_base: np.ndarray,
                  tau: float,
                  max_iter: int = 50,
                  tol: float = 1e-8,
                  verbose: bool = False) -> tuple:
    """
    Nonlinear Jacobi Method for the 2-principal Common-Agency EPEC.

    Returns
    -------
    y_star  : list[np.ndarray]  – equilibrium transfer vectors
    pi_star : np.ndarray        – equilibrium policy
    info    : dict              – convergence diagnostics
    """
    J = len(q)
    N = q[0].shape[0]

    # Initialise at half of (clipped-positive) q
    y = [np.maximum(0.5 * q[j], 0.0).copy() for j in range(J)]
    Y = sum(y)
    pi = agent_best_response(Y, pi_base, tau)

    info = {
        'converged': False,
        'iterations': 0,
        'final_kkt_residual': np.inf,
        'y_err_history': [],
    }

    for t in range(max_iter):
        y_old = [yi.copy() for yi in y]

        # Jacobi update: each principal solves MPEC w.r.t. old others
        y_new = []
        for j in range(J):
            yj_new = solve_mpec_principal(j, y_old, q, pi_base, tau)
            y_new.append(yj_new)

        y = y_new
        Y = sum(y)
        pi = agent_best_response(Y, pi_base, tau)

        y_err = max(np.max(np.abs(y[j] - y_old[j])) for j in range(J))
        info['y_err_history'].append(float(y_err))

        if verbose:
            kkt_res = kkt_residual(y, q, pi_base, tau)
            print(f'    iter {t+1:3d}  |  Δy = {y_err:.3e}  |  KKT = {kkt_res:.3e}')

        if y_err < tol:
            info['converged'] = True
            break

    info['iterations'] = t + 1
    info['final_kkt_residual'] = float(kkt_residual(y, q, pi_base, tau))

    return y, pi, info


# =========================================================================== #
#  Pipeline:  process one prompt
# =========================================================================== #

def process_prompt(candidates: list[dict],
                   tau: float,
                   parm_weights: tuple[float, float],
                   max_iterations: int,
                   tolerance: float,
                   normalize: bool,
                   verbose: bool) -> dict:
    """
    Given a list of candidate dicts (all sharing the same uid/prompt),
    compute the three policy distributions and return a results dict.
    """
    N = len(candidates)
    uid = candidates[0]['uid']

    # --- Extract reward vectors ------------------------------------------- #
    q_help_raw = np.array([c['q_help'] for c in candidates], dtype=np.float64)
    q_harm_raw = np.array([c['q_harm'] for c in candidates], dtype=np.float64)

    if normalize:
        q_help = min_max_normalize(q_help_raw, invert=False)
        # q_harm is a COST (lower=safer). We invert it so higher=safer for the solver.
        q_harm = min_max_normalize(q_harm_raw, invert=True)
    else:
        q_help = q_help_raw.copy()
        q_harm = -q_harm_raw.copy()  # negate cost to form a reward

    # --- Construct base policy -------------------------------------------- #
    #   If log_prob is available (from Step 1), use softmax to get π_base.
    #   Otherwise fall back to uniform.
    log_probs = [c.get('log_prob') for c in candidates]
    if all(lp is not None for lp in log_probs):
        lp = np.array(log_probs, dtype=np.float64)
        pi_base = softmax(lp)
    else:
        pi_base = np.ones(N, dtype=np.float64) / N

    # --- Method 1:  PARM -------------------------------------------------- #
    pi_parm = compute_parm(q_help, q_harm, pi_base, tau, parm_weights)

    # --- Method 2:  Naive aggregation ------------------------------------- #
    pi_naive = compute_naive(q_help, q_harm, pi_base, tau)

    # --- Method 3:  Common Agency EPEC ------------------------------------ #
    # Weight the principals' valuation by the preference weights so EPEC traces a curve too!
    q_list = [q_help * parm_weights[0], q_harm * parm_weights[1]]       # two principals
    y_star, pi_star, info = jacobi_method(
        q_list, pi_base, tau,
        max_iter=max_iterations,
        tol=tolerance,
        verbose=verbose,
    )

    # --- Metrics ---------------------------------------------------------- #
    def expected_rewards(pi):
        return float(q_help_raw @ pi), float(q_harm_raw @ pi)

    er_parm  = expected_rewards(pi_parm)
    er_naive = expected_rewards(pi_naive)
    er_star  = expected_rewards(pi_star)

    kl_parm  = kl_divergence(pi_parm,  pi_base)
    kl_naive = kl_divergence(pi_naive, pi_base)
    kl_star  = kl_divergence(pi_star,  pi_base)

    # --- Results ---------------------------------------------------------- #
    return {
        'uid': uid,
        'num_candidates': N,
        'pi_parm':  pi_parm.tolist(),
        'pi_naive': pi_naive.tolist(),
        'pi_star':  pi_star.tolist(),
        'y_help_star': y_star[0].tolist(),
        'y_harm_star': y_star[1].tolist(),
        'q_help_used': q_help.tolist(),       # post-normalisation
        'q_harm_used': q_harm.tolist(),
        'pi_base': pi_base.tolist(),
        'metrics': {
            'kl_parm':  kl_parm,
            'kl_naive': kl_naive,
            'kl_star':  kl_star,
            'expected_q_help_parm':  er_parm[0],
            'expected_q_harm_parm':  er_parm[1],
            'expected_q_help_naive': er_naive[0],
            'expected_q_harm_naive': er_naive[1],
            'expected_q_help_star':  er_star[0],
            'expected_q_harm_star':  er_star[1],
        },
        'jacobi_info': {
            'converged':          info['converged'],
            'iterations':         info['iterations'],
            'final_kkt_residual': info['final_kkt_residual'],
        },
    }


# =========================================================================== #
#  Augment original candidate list with per-candidate policy weights
# =========================================================================== #

def augment_candidates(candidates: list[dict],
                       results: dict) -> list[dict]:
    """Add pi_parm, pi_naive, pi_star fields to each candidate dict."""
    augmented = []
    for c in candidates:
        entry = dict(c)
        idx = c['candidate_id']
        entry['pi_parm']  = results['pi_parm'][idx]
        entry['pi_naive'] = results['pi_naive'][idx]
        entry['pi_star']  = results['pi_star'][idx]
        entry['y_help_star'] = results['y_help_star'][idx]
        entry['y_harm_star'] = results['y_harm_star'][idx]
        augmented.append(entry)
    return augmented


# =========================================================================== #
#  Pretty-print summary table
# =========================================================================== #

def print_summary(all_results: list[dict]):
    """Print an aggregate summary table across all prompts."""
    n = len(all_results)
    if n == 0:
        return

    print('\n' + '=' * 80)
    print(f'  SUMMARY  ({n} prompts)')
    print('=' * 80)

    # Aggregate means
    keys = [
        ('kl_parm',              'KL(π_parm  ‖ π_base)'),
        ('kl_naive',             'KL(π_naive ‖ π_base)'),
        ('kl_star',              'KL(π_star  ‖ π_base)'),
        ('expected_q_help_parm', 'E[q_help] — PARM'),
        ('expected_q_harm_parm', 'E[q_harm] — PARM'),
        ('expected_q_help_naive','E[q_help] — Naive'),
        ('expected_q_harm_naive','E[q_harm] — Naive'),
        ('expected_q_help_star', 'E[q_help] — EPEC'),
        ('expected_q_harm_star', 'E[q_harm] — EPEC'),
    ]

    header = f'  {"Metric":<30s}  {"Mean":>12s}  {"Std":>12s}'
    print(header)
    print('  ' + '-' * 58)

    for key, label in keys:
        vals = [r['metrics'][key] for r in all_results]
        mean = np.mean(vals)
        std  = np.std(vals)
        print(f'  {label:<30s}  {mean:12.6f}  {std:12.6f}')

    # Convergence stats
    converged = sum(1 for r in all_results if r['jacobi_info']['converged'])
    avg_iter  = np.mean([r['jacobi_info']['iterations'] for r in all_results])
    avg_kkt   = np.mean([r['jacobi_info']['final_kkt_residual']
                         for r in all_results])

    print()
    print(f'  Jacobi convergence: {converged}/{n} prompts converged')
    print(f'  Avg iterations:     {avg_iter:.1f}')
    print(f'  Avg KKT residual:   {avg_kkt:.2e}')
    print('=' * 80)


# =========================================================================== #
#  CLI argument parser
# =========================================================================== #

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Step 3 — Solve equilibrium policies over scored candidate '
            'responses using PARM, Naive Aggregation, and Common Agency EPEC.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--input_file', required=True, type=str,
        help='Path to scored_candidates_<run_tag>.json from Step 2.',
    )
    parser.add_argument(
        '--output_dir', default='./solved', type=str,
        help='Directory to save the solved output file.',
    )
    parser.add_argument(
        '--temperature_tau', default=1.0, type=float,
        help='Temperature τ for the KL-regularised agent best response.',
    )
    parser.add_argument(
        '--parm_weights', nargs=2, type=float, default=[0.5, 0.5],
        metavar=('W_HELP', 'W_HARM'),
        help='Preference weights [w_help, w_harm] for PARM baseline.',
    )
    parser.add_argument(
        '--max_iterations', default=50, type=int,
        help='Maximum Jacobi iterations for Method 3 (EPEC).',
    )
    parser.add_argument(
        '--tolerance', default=1e-8, type=float,
        help='Convergence tolerance ε for Jacobi iterate change.',
    )
    parser.add_argument(
        '--normalize_rewards', action='store_true',
        help='If set, min-max normalise q_help and q_harm per prompt to [0,1].',
    )
    parser.add_argument(
        '--seed', default=42, type=int,
        help='Random seed (for reproducibility of multi-start optimisation).',
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print per-iteration Jacobi diagnostics.',
    )
    parser.add_argument(
        '--num_prompts', default=None, type=int,
        help='If set, only process the first N prompts (for debugging).',
    )
    return parser.parse_args()


# =========================================================================== #
#  Main
# =========================================================================== #

def main():
    args = parse_arguments()
    np.random.seed(args.seed)
    np.set_printoptions(precision=6, suppress=True)

    tau       = args.temperature_tau
    weights   = tuple(args.parm_weights)
    max_iter  = args.max_iterations
    tol       = args.tolerance
    normalize = args.normalize_rewards

    # ------------------------------------------------------------------ #
    #  Load scored candidates
    # ------------------------------------------------------------------ #
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f'ERROR: Input file not found: {input_path}', file=sys.stderr)
        sys.exit(1)

    with open(input_path, 'r') as f:
        all_candidates = json.load(f)
    print(f'Loaded {len(all_candidates)} candidates from {input_path}')

    # ------------------------------------------------------------------ #
    #  Group by prompt uid
    # ------------------------------------------------------------------ #
    by_uid: dict[str, list[dict]] = defaultdict(list)
    for c in all_candidates:
        by_uid[c['uid']].append(c)

    # Sort each group by candidate_id for deterministic ordering
    for uid in by_uid:
        by_uid[uid].sort(key=lambda c: c['candidate_id'])

    uids = list(by_uid.keys())
    if args.num_prompts is not None:
        uids = uids[:args.num_prompts]

    print(f'Found {len(uids)} unique prompts (processing {len(uids)})')
    print(f'Settings:  τ={tau}  weights={weights}  '
          f'max_iter={max_iter}  tol={tol}  normalize={normalize}')
    print()

    # ------------------------------------------------------------------ #
    #  Process each prompt
    # ------------------------------------------------------------------ #
    all_results = []
    augmented_candidates = []
    t0 = time.time()

    for prompt_idx, uid in enumerate(uids):
        candidates = by_uid[uid]
        N = len(candidates)
        prompt_preview = candidates[0]['prompt'][:80].replace('\n', ' ')

        print(f'[{prompt_idx+1:4d}/{len(uids)}]  uid={uid}  '
              f'N={N}  "{prompt_preview}..."')

        result = process_prompt(
            candidates, tau, weights, max_iter, tol, normalize, args.verbose,
        )
        all_results.append(result)

        # Augment original candidate entries with policy weights
        augmented = augment_candidates(candidates, result)
        augmented_candidates.extend(augmented)

        # Brief per-prompt summary
        m = result['metrics']
        ji = result['jacobi_info']
        conv_flag = '✓' if ji['converged'] else '✗'
        print(f'         EPEC: {ji["iterations"]} iter, '
              f'KKT={ji["final_kkt_residual"]:.2e} [{conv_flag}]  |  '
              f'E[q_help]: PARM={m["expected_q_help_parm"]:.4f}  '
              f'Naive={m["expected_q_help_naive"]:.4f}  '
              f'EPEC={m["expected_q_help_star"]:.4f}')

    elapsed = time.time() - t0
    print(f'\nAll prompts processed in {elapsed:.1f}s')

    # ------------------------------------------------------------------ #
    #  Summary
    # ------------------------------------------------------------------ #
    print_summary(all_results)

    # ------------------------------------------------------------------ #
    #  Save output
    # ------------------------------------------------------------------ #
    os.makedirs(args.output_dir, exist_ok=True)
    run_tag = input_path.stem.replace('scored_candidates_', '')
    output_path = Path(args.output_dir) / f'solved_equilibrium_{run_tag}.json'

    output_data = {
        'settings': {
            'temperature_tau':   tau,
            'parm_weights':      list(weights),
            'max_iterations':    max_iter,
            'tolerance':         tol,
            'normalize_rewards': normalize,
            'input_file':        str(input_path),
        },
        'per_prompt_results': all_results,
        'candidates':         augmented_candidates,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f'\nSaved to: {output_path}')
    print(f'Total candidates: {len(augmented_candidates)}')


if __name__ == '__main__':
    main()
