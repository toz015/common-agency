"""
Solve the Common-Agency EPEC (Algorithm 1, Nonlinear Jacobi) on the
SafeRLHF scored candidates and produce a Pareto sweep comparable to PARM.

Per prompt, action space = the 20 scored candidates.
Per weight w = (w_help, w_harm), payoffs are baked: q^j <- w^j * q^j.
For each prompt we run the Jacobi loop to convergence, take a* = argmax pi*,
and report mean (q_help, q_harm) across prompts using the ORIGINAL scores.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax
from tqdm import tqdm


def best_response(log_pi_base: np.ndarray, Y: np.ndarray, tau: float) -> np.ndarray:
    return softmax(log_pi_base + Y / tau)


def solve_principal(i: int, q: list, y: list, log_pi_base: np.ndarray, tau: float) -> np.ndarray:
    """Solve principal i's MPEC: max_{y^i in [0,q^i]} pi(Y)^T (q^i - y^i)."""
    qi = q[i]
    y_other = sum(y[j] for j in range(len(y)) if j != i)

    def neg_obj(yi):
        Y = y_other + yi
        pi = best_response(log_pi_base, Y, tau)
        return -float(pi @ (qi - yi))

    bounds = [(0.0, float(qi[a])) for a in range(len(qi))]
    res = minimize(neg_obj, y[i], method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 50, "ftol": 1e-8})
    return res.x


def epec_one_prompt(log_pi_base, q_list, tau=1.0, eps=1e-4, max_iter=100):
    J = len(q_list)
    y = [np.zeros_like(qj) for qj in q_list]
    Y = sum(y)
    pi = best_response(log_pi_base, Y, tau)
    for _ in range(max_iter):
        y_new = [solve_principal(i, q_list, y, log_pi_base, tau) for i in range(J)]
        Y_new = sum(y_new)
        pi_new = best_response(log_pi_base, Y_new, tau)
        dy = max(np.max(np.abs(y_new[i] - y[i])) for i in range(J))
        dp = float(np.max(np.abs(pi_new - pi)))
        y, pi = y_new, pi_new
        if dy + dp <= eps:
            break
    return pi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scored", default="scored/scored_candidates_test_prompt_only_alpaca-7b-reproduced_N20.json")
    p.add_argument("--out", default="epec_results/epec_sweep.json")
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--max_iter", type=int, default=100)
    p.add_argument("--step", type=float, default=0.1)
    p.add_argument("--limit", type=int, default=0, help="Limit number of prompts (0 = all)")
    args = p.parse_args()

    records = json.load(open(args.scored))
    # Group by uid
    by_uid = {}
    for r in records:
        by_uid.setdefault(r["uid"], []).append(r)
    # Sort each group by candidate_id and stack arrays
    prompts = []
    for uid, group in by_uid.items():
        group.sort(key=lambda r: r["candidate_id"])
        log_pi_base = log_softmax(np.array([r["log_prob"] for r in group], dtype=np.float64))
        q_help = np.array([r["q_help"] for r in group], dtype=np.float64)
        q_harm = np.array([r["q_harm"] for r in group], dtype=np.float64)
        prompts.append((uid, log_pi_base, q_help, q_harm))
    if args.limit > 0:
        prompts = prompts[:args.limit]
    print(f"Loaded {len(prompts)} prompts × {len(prompts[0][2])} candidates")

    weights = [(round(w, 2), round(1.0 - w, 2))
               for w in np.arange(0.0, 1.0 + 1e-9, args.step)]

    sweep = []
    for w_help, w_harm in weights:
        helps, harms = [], []
        for uid, log_pi_base, q_help, q_harm in tqdm(prompts, desc=f"w_h={w_help}"):
            q1 = q_help - q_help.min()
            q2 = (-q_harm) - (-q_harm).min()  # cost → reward, shifted
            if q1.max() > 0:
                q1 = q1 / q1.max()
            if q2.max() > 0:
                q2 = q2 / q2.max()
            q1 = w_help * q1
            q2 = w_harm * q2
            pi_star = epec_one_prompt(log_pi_base, [q1, q2],
                                      tau=args.tau, eps=args.eps, max_iter=args.max_iter)
            helps.append(float(pi_star @ q_help))
            harms.append(float(pi_star @ q_harm))
        sweep.append({
            "w_help": w_help, "w_harm": w_harm,
            "mean_help": float(np.mean(helps)),
            "mean_harm": float(np.mean(harms)),
            "n_prompts": len(prompts),
        })
        print(sweep[-1])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(sweep, open(args.out, "w"), indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
