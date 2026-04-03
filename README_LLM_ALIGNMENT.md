# Common Agency in LLM Alignment - Evaluation Walkthrough

This repository branch contains the final evaluation results, visualizations, and codebase updates validating the application of EPEC (Empirical Principle Equilibrium Computation) to Large Language Model Alignment using a Common Agency theoretical framework.

## Key Contributions & Fixes

### 1. The Decentralized Pareto Frontier (EPEC Sweep)
Previously, the EPEC solution was evaluated exclusively at a static, unweighted equilibrium which artificially cast it as a single point against the fully parameterized PARM benchmark.
**Fix**: We explicitly parameterized the principals' valuation limits within the mechanism logic, enabling the dynamic discovery of the complete Nash Equilibrium frontier across varied reward sensitivities.
`solve_equilibrium.py` now correctly multiplies the principals' independent $q$-signals by the experimental preference thresholds ($w$).

**Visual Evidence**: `code/evaluation/pareto_pref_sweep.png` illustrates that at optimal regularization bounds ($\tau \le 0.15$), the distributed multi-agent game mathematically approximates the explicit scalarized PARM boundary, providing the first known empirical mapping from theoretical Common Agency mechanism design directly to real-world LLM optimization limits.

### 2. The Alignment Tax Trap (Naive Baseline Defeated)
Naive additive accumulation of unconstrained $q$-signals produces high absolute scores at the detriment of model stability.
**Fix**: Implemented an automated logging metric to map Expected Rewards to Base-Policy Information Loss.
`code/evaluation/plot_kl_divergence.py` extracts the KL( $\pi_{method} \vert\vert \pi_{base}$ ) divergence across models.

**Visual Evidence**: `code/evaluation/kl_divergence_tax.png` proves the Naive solution completely corrupts the model distribution parameters, whereas the EPEC solution maintains near-optimal constraints strictly dominating alternate baselines outside of absolute frontiers.

## Scripts Provided
1. **Simulation Math**: `common_agency_simulation.py` - Synthesizes theoretical proof distributions with $N=5$, $J=3$ dimensions demonstrating standard structural scaling limits.
2. **Alignment Sweep Orchestrator**: `code/evaluation/sweep_all_locals.sh` - Parallel, hardware-maximized bash runner to evaluate 11 distinct $w$ splits across 6 critical $\tau$ gradients locally on standard computational environments.
3. **Solver Methods**: `code/evaluation/solve_equilibrium.py` - Employs L-BFGS-B & Jacobi Block iterative updates for generating continuous probability thresholds.

*Generated automatically via execution logs spanning PKU-SafeRLHF Test Datasets via Alpaca-7B inferences.*
