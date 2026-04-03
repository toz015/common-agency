"""
Common Agency Simulation for Multi-Objective LLM Alignment
===========================================================
Implements the Nonlinear Jacobi Method (Algorithm 1) for solving the
Common-Agency EPEC, and runs experiments to validate theoretical results.

References:
  - Proposition 1: Coordination outcome (KKT conditions)
  - Theorem 1: Existence & uniqueness of equilibrium
  - Theorem 2: Convergence to first-order stationary point
  - Theorem 4: Sensitivity / local stability
  - Theorem 5: No-regret property
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'legend.fontsize': 11,
    'figure.figsize': (7, 5),
    'figure.dpi': 150,
})

# Output directory
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 1.  Core mechanics
# ============================================================

def softmax(logits):
    """Numerically stable softmax."""
    logits = logits - np.max(logits)
    e = np.exp(logits)
    return e / e.sum()


def agent_best_response(Y, pi_base, tau):
    """KL-regularised best response: pi = softmax(log pi_base + Y / tau)."""
    logits = np.log(pi_base) + Y / tau
    return softmax(logits)


def jacobian_pi(Y, pi_base, tau):
    """Jacobian  J_pi(Y) = (1/tau)(diag(pi) - pi pi^T)."""
    pi = agent_best_response(Y, pi_base, tau)
    return (np.diag(pi) - np.outer(pi, pi)) / tau


def kkt_residual(y_all, q, pi_base, tau):
    """
    Compute the KKT residual for the EPEC (independent optimality measure).

    For each principal i and action a, the stationarity condition is:
      grad_a = [J_pi(Y)^T (q^i - y^i)]_a - pi_a
    KKT requires:
      y^i_a = 0        =>  grad_a <= 0
      0 < y^i_a < q^i_a =>  grad_a = 0
      y^i_a = q^i_a     =>  grad_a >= 0
    The residual is the max violation of these conditions.
    """
    J_count = len(y_all)
    N = q[0].shape[0]
    Y = sum(y_all)
    pi = agent_best_response(Y, pi_base, tau)
    J_pi = jacobian_pi(Y, pi_base, tau)

    max_violation = 0.0
    for i in range(J_count):
        grad = J_pi.T @ (q[i] - y_all[i]) - pi
        for a in range(N):
            if y_all[i][a] <= 1e-12:         # at lower bound
                violation = max(0.0, grad[a]) # grad should be <= 0
            elif y_all[i][a] >= q[i][a] - 1e-12:  # at upper bound
                violation = max(0.0, -grad[a])     # grad should be >= 0
            else:                             # interior
                violation = abs(grad[a])
            max_violation = max(max_violation, violation)
    return max_violation


# ============================================================
# 2.  MPEC solver for a single principal
# ============================================================

def solve_mpec_principal(i, y_all, q, pi_base, tau):
    """
    Solve principal i's MPEC:
        max_{0 <= y^i <= q^i}  pi(Y^{-i} + y^i)^T (q^i - y^i)
    given others' transfers fixed.
    Returns optimal y^i.
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
        grad = J.T @ (q[i] - yi) - pi
        return -grad

    bounds = [(0.0, q[i][a]) for a in range(N)]

    # Multi-start to avoid local minima
    best_val = np.inf
    best_x = y_all[i].copy()

    starts = [
        y_all[i].copy(),
        q[i] * 0.5,
        q[i] * 0.1,
        q[i] * 0.9,
    ]

    for x0 in starts:
        x0 = np.clip(x0, 0, q[i])
        res = minimize(
            neg_objective, x0,
            jac=neg_objective_grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x.copy()

    return best_x


# ============================================================
# 3.  Algorithm 1: Nonlinear Jacobi Method
# ============================================================

def jacobi_method(q, pi_base, tau, T_max=300, tol=1e-10,
                  y_init=None, record=True):
    """
    Nonlinear Jacobi Method for the Common-Agency EPEC.

    Parameters
    ----------
    q : list of np.array, length J, each of shape (N,)
    pi_base : np.array, shape (N,)
    tau : float
    T_max : int
    tol : float
    y_init : list of np.array or None
    record : bool – whether to record full history

    Returns
    -------
    y_star : list of np.array  – equilibrium transfers
    pi_star : np.array          – equilibrium policy
    history : dict              – convergence history
    """
    J = len(q)
    N = q[0].shape[0]

    # initialise at half of q
    if y_init is None:
        y = [0.5 * q[j].copy() for j in range(J)]
    else:
        y = [yi.copy() for yi in y_init]

    Y = sum(y)
    pi = agent_best_response(Y, pi_base, tau)

    history = {'y': [], 'pi': [], 'y_err': [], 'pi_err': []}

    for t in range(T_max):
        y_old = [yi.copy() for yi in y]
        pi_old = pi.copy()

        # Jacobi: solve each principal's MPEC given *old* others
        y_new = []
        for j in range(J):
            yj_new = solve_mpec_principal(j, y_old, q, pi_base, tau)
            y_new.append(yj_new)

        y = y_new
        Y = sum(y)
        pi = agent_best_response(Y, pi_base, tau)

        y_err = max(np.max(np.abs(y[j] - y_old[j])) for j in range(J))
        pi_err = np.max(np.abs(pi - pi_old))

        if record:
            history['y'].append([yi.copy() for yi in y])
            history['pi'].append(pi.copy())
            history['y_err'].append(y_err)
            history['pi_err'].append(pi_err)

        if y_err < tol and pi_err < tol:
            break

    return y, pi, history


# ============================================================
# 4.  Utility helpers
# ============================================================

def principal_utility(i, y, q, pi_base, tau):
    """Compute f_i = pi(Y)^T (q^i - y^i)."""
    Y = sum(y)
    pi = agent_best_response(Y, pi_base, tau)
    return pi.dot(q[i] - y[i])


# ============================================================
# 5.  Experiments
# ============================================================

def generate_problem(N, J, seed=42, structured=True):
    """
    Generate a random common-agency problem instance.

    Parameters
    ----------
    structured : bool
        If True, each principal has a distinct preferred action (cleaner demo).
        If False, payoffs are fully random (harder, more realistic test).
    """
    rng = np.random.RandomState(seed)
    pi_base = np.ones(N) / N  # uniform base policy

    q = []
    if structured:
        for j in range(J):
            qj = rng.uniform(1.0, 3.0, size=N)
            # boost a different action for each principal
            preferred = j % N
            qj[preferred] += rng.uniform(4.0, 8.0)
            q.append(qj)
    else:
        # Fully random payoffs — no designed structure
        for j in range(J):
            qj = rng.uniform(0.5, 5.0, size=N)
            q.append(qj)

    return q, pi_base


def experiment_1_convergence(q, pi_base, tau=0.3):
    """Fig 1: Convergence of Jacobi iterates — uses KKT residual (independent measure)."""
    y_star, pi_star, hist = jacobi_method(q, pi_base, tau, T_max=300)

    # Compute KKT residual at each iterate (independent optimality measure)
    kkt_residuals = []
    iter_diffs = []  # also show successive iterate changes for reference
    for t in range(len(hist['y'])):
        res = kkt_residual(hist['y'][t], q, pi_base, tau)
        kkt_residuals.append(max(res, 1e-16))
        if t > 0:
            yd = max(np.linalg.norm(
                hist['y'][t][j] - hist['y'][t-1][j]) for j in range(len(q)))
        else:
            yd = max(np.linalg.norm(
                hist['y'][t][j] - 0.5 * q[j]) for j in range(len(q)))
        iter_diffs.append(max(yd, 1e-16))

    iters = np.arange(1, len(kkt_residuals) + 1)

    fig, ax = plt.subplots()
    ax.semilogy(iters, kkt_residuals, 'o-',
                label='KKT residual (optimality gap)',
                markersize=4, color='#E74C3C')
    ax.semilogy(iters, iter_diffs, 's-',
                label=r'$\|y^{(t)} - y^{(t-1)}\|_2$ (iterate change)',
                markersize=4, color='#3498DB', alpha=0.7)
    ax.set_xlabel('Iteration $t$')
    ax.set_ylabel('Residual (log scale)')
    ax.set_title('Convergence of Nonlinear Jacobi Method (Algorithm 1)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig1_convergence.pdf'))
    plt.close(fig)
    print(f'[✓] Figure 1 saved  ({len(kkt_residuals)} iterations)')
    return y_star, pi_star


def experiment_2_equilibrium(y_star, pi_star, pi_base, q, tau):
    """Fig 2: Equilibrium policy vs base policy."""
    N = len(pi_star)
    J = len(q)
    x = np.arange(N)
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))

    # naive aggregation: just sum q and do best response
    Q_w = sum(q[j] for j in range(J))
    pi_naive = agent_best_response(Q_w, pi_base, tau)

    ax.bar(x - width, pi_base, width, label=r'$\pi_{\mathrm{base}}$',
           color='#3498DB', alpha=0.85, edgecolor='white')
    ax.bar(x, pi_star, width, label=r'$\pi^\star$ (equilibrium)',
           color='#E74C3C', alpha=0.85, edgecolor='white')
    ax.bar(x + width, pi_naive, width,
           label=r'$\pi_{\mathrm{naive}}$ (naive aggregation)',
           color='#2ECC71', alpha=0.85, edgecolor='white')

    ax.set_xlabel('Action $a$')
    ax.set_ylabel('Probability')
    ax.set_title(r'Equilibrium Policy $\pi^\star$ vs Alternatives')
    ax.set_xticks(x)
    ax.set_xticklabels([f'$a_{{{a+1}}}$' for a in range(N)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig2_equilibrium_policy.pdf'))
    plt.close(fig)
    print('[✓] Figure 2 saved')

    # Verify Proposition 1 conditions
    Y_star = sum(y_star)
    J_pi = jacobian_pi(Y_star, pi_base, tau)
    violations = 0
    for i in range(J):
        lhs = J_pi.T @ (q[i] - y_star[i])
        for a in range(N):
            if y_star[i][a] < 1e-8:
                if lhs[a] > pi_star[a] + 1e-5:
                    violations += 1
            elif y_star[i][a] > q[i][a] - 1e-8:
                if lhs[a] < pi_star[a] - 1e-5:
                    violations += 1
            else:
                if abs(lhs[a] - pi_star[a]) > 1e-4:
                    violations += 1
    if violations == 0:
        print('  Proposition 1 KKT conditions: ALL SATISFIED ✓')
    else:
        print(f'  Proposition 1 KKT conditions: {violations} violations ✗')


def experiment_3_sensitivity_tau(q, pi_base):
    """Fig 3: Equilibrium policy for varying tau."""
    taus = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    N = len(q[0])

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    axes = axes.flatten()

    for idx, tau in enumerate(taus):
        y_star, pi_star, _ = jacobi_method(q, pi_base, tau, record=False)
        x = np.arange(N)
        axes[idx].bar(x - 0.15, pi_base, 0.3, label=r'$\pi_{\mathrm{base}}$',
                      color='#3498DB', alpha=0.7)
        axes[idx].bar(x + 0.15, pi_star, 0.3, label=r'$\pi^\star$',
                      color='#E74C3C', alpha=0.7)
        axes[idx].set_title(rf'$\tau = {tau}$')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels([f'$a_{{{a+1}}}$' for a in range(N)])
        axes[idx].grid(True, alpha=0.3, axis='y')
        if idx == 0:
            axes[idx].legend(fontsize=9)

    fig.suptitle(r'Sensitivity of Equilibrium Policy $\pi^\star$ to Temperature $\tau$',
                 fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT_DIR, 'fig3_sensitivity_tau.pdf'))
    plt.close(fig)
    print('[✓] Figure 3 saved')


def experiment_4_sensitivity_perturbation(q, pi_base, tau=0.3):
    """Fig 4: ||pi*(1) - pi*(2)|| vs perturbation size (Theorem 4)."""
    epsilons = np.linspace(0.0, 1.0, 15)
    J = len(q)
    N = q[0].shape[0]
    rng = np.random.RandomState(123)

    # fixed perturbation direction
    dq = [rng.randn(N) for _ in range(J)]
    for j in range(J):
        dq[j] /= np.linalg.norm(dq[j])

    d_base = rng.randn(N)
    d_base -= d_base.mean()  # zero mean
    d_base /= np.linalg.norm(d_base)

    # baseline equilibrium
    _, pi_star_base, _ = jacobi_method(q, pi_base, tau, record=False)

    pi_diffs_q = []
    pi_diffs_base = []
    q_perturb_sizes = []
    base_perturb_sizes = []

    # (a) Perturb payoff vectors q
    for eps in epsilons:
        q_pert = [q[j] + eps * dq[j] for j in range(J)]
        q_pert = [np.maximum(qp, 0.01) for qp in q_pert]
        _, pi_pert, _ = jacobi_method(q_pert, pi_base, tau, record=False)
        q_perturb_sizes.append(max(np.linalg.norm(q_pert[j] - q[j]) for j in range(J)))
        pi_diffs_q.append(np.linalg.norm(pi_pert - pi_star_base))

    # (b) Perturb base policy
    for eps in epsilons:
        log_pb_pert = np.log(pi_base) + eps * d_base
        pb_pert = softmax(log_pb_pert)
        _, pi_pert, _ = jacobi_method(q, pb_pert, tau, record=False)
        base_perturb_sizes.append(np.linalg.norm(np.log(pb_pert) - np.log(pi_base)))
        pi_diffs_base.append(np.linalg.norm(pi_pert - pi_star_base))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(q_perturb_sizes, pi_diffs_q, 'o-', color='#E74C3C', markersize=6,
             linewidth=2)
    ax1.set_xlabel(r'$\max_i \|q^{i(1)} - q^{i(2)}\|_2$')
    ax1.set_ylabel(r'$\|\pi^{\star(1)} - \pi^{\star(2)}\|_2$')
    ax1.set_title('(a) Payoff Perturbation')
    ax1.grid(True, alpha=0.3)

    ax2.plot(base_perturb_sizes, pi_diffs_base, 's-', color='#3498DB', markersize=6,
             linewidth=2)
    ax2.set_xlabel(r'$\|\log\pi_{\mathrm{base}}^{(1)} - \log\pi_{\mathrm{base}}^{(2)}\|_2$')
    ax2.set_ylabel(r'$\|\pi^{\star(1)} - \pi^{\star(2)}\|_2$')
    ax2.set_title('(b) Base Policy Perturbation')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Local Stability of Equilibrium (Theorem 4)', fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT_DIR, 'fig4_sensitivity_perturbation.pdf'))
    plt.close(fig)
    print('[✓] Figure 4 saved')


def _best_response_utility(j, Y_minus_j, q_j, pi_base, tau):
    """Solve max_{0 <= y <= q_j} pi(Y^{-j} + y)^T (q_j - y)  (true best in hindsight)."""
    N = q_j.shape[0]

    def neg_obj(y):
        pi = agent_best_response(Y_minus_j + y, pi_base, tau)
        return -pi.dot(q_j - y)

    bounds = [(0.0, q_j[a]) for a in range(N)]
    best_val = np.inf
    for x0 in [q_j * 0.5, q_j * 0.1, q_j * 0.9, np.zeros(N)]:
        x0 = np.clip(x0, 0, q_j)
        res = minimize(neg_obj, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 500, 'ftol': 1e-14})
        if res.fun < best_val:
            best_val = res.fun
    return -best_val


def experiment_5_regret(q, pi_base, tau=0.3):
    """Fig 5: Cumulative and time-averaged regret (Theorem 6).

    Uses true best-in-hindsight (solves max over [0,q^i]) at each round,
    not just the NE transfer.
    """
    J = len(q)
    y_star, pi_star, hist = jacobi_method(q, pi_base, tau, T_max=300)

    T = len(hist['y'])
    cumulative_regret = {j: np.zeros(T) for j in range(J)}

    for t in range(T):
        y_t = hist['y'][t]
        for j in range(J):
            Y_minus_j_t = sum(y_t[k] for k in range(J) if k != j)

            # True best-in-hindsight: solve max_{y \in [0,q^j]} f_j(y; Y^{-j,(t)})
            u_best = _best_response_utility(j, Y_minus_j_t, q[j], pi_base, tau)

            # actual utility
            u_actual = principal_utility(j, y_t, q, pi_base, tau)

            regret_t = max(0.0, u_best - u_actual)
            cumulative_regret[j][t] = (cumulative_regret[j][t-1] if t > 0 else 0) + regret_t

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

    for j in range(J):
        iters = np.arange(1, T + 1)
        ax1.plot(iters, cumulative_regret[j], '-', color=colors[j % len(colors)],
                 label=f'Principal {j+1}', linewidth=2)
        ax2.plot(iters, cumulative_regret[j] / iters, '-', color=colors[j % len(colors)],
                 label=f'Principal {j+1}', linewidth=2)

    ax1.set_xlabel('Iteration $T$')
    ax1.set_ylabel(r'Cumulative Regret $R_i(T)$')
    ax1.set_title(r'Cumulative Regret $R_i(T) = \mathcal{O}(1)$')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Iteration $T$')
    ax2.set_ylabel(r'Average Regret $R_i(T)/T$')
    ax2.set_title(r'Time-Averaged Regret $\to 0$')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle('No-Regret Property (Theorem 6)', fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT_DIR, 'fig5_regret.pdf'))
    plt.close(fig)
    print('[✓] Figure 5 saved')


# ============================================================
# 6.  Run all experiments
# ============================================================

def main():
    np.set_printoptions(precision=6, suppress=True)

    N, J = 5, 3
    tau = 0.3   # smaller tau => principals have more incentive to steer

    # ---- Run with STRUCTURED payoffs (designed setup) ----
    print('='*60)
    print('Part A: Structured payoffs (each principal has a preferred action)')
    print('='*60)
    q, pi_base = generate_problem(N, J, seed=42, structured=True)
    print(f'  Actions N = {N},  Principals J = {J},  tau = {tau}')
    print(f'  pi_base = {pi_base}')
    for j in range(J):
        print(f'  q[{j+1}] = {q[j]}')
    print()

    print('--- Experiment 1: Convergence (KKT residual) ---')
    y_star, pi_star = experiment_1_convergence(q, pi_base, tau)
    Y_star = sum(y_star)
    print(f'  Equilibrium pi* = {pi_star}')
    print(f'  Aggregate Y*    = {Y_star}')
    for j in range(J):
        print(f'  y*[{j+1}] = {y_star[j]}')
        print(f'  Utility[{j+1}] = {principal_utility(j, y_star, q, pi_base, tau):.6f}')
    print()

    print('--- Experiment 2: Equilibrium Policy ---')
    experiment_2_equilibrium(y_star, pi_star, pi_base, q, tau)
    print()

    print('--- Experiment 3: Sensitivity to tau ---')
    experiment_3_sensitivity_tau(q, pi_base)
    print()

    print('--- Experiment 4: Sensitivity to Perturbation ---')
    experiment_4_sensitivity_perturbation(q, pi_base, tau)
    print()

    print('--- Experiment 5: Regret Analysis (true best-in-hindsight) ---')
    experiment_5_regret(q, pi_base, tau)
    print()

    # ---- Run with RANDOM payoffs (no designed structure) ----
    print('='*60)
    print('Part B: Fully random payoffs (robustness check)')
    print('='*60)
    for seed in [7, 123, 999]:
        q_rand, pi_base_rand = generate_problem(N, J, seed=seed, structured=False)
        print(f'\n  seed={seed}:')
        for j in range(J):
            print(f'    q[{j+1}] = {q_rand[j]}')
        y_star_r, pi_star_r, hist_r = jacobi_method(
            q_rand, pi_base_rand, tau, T_max=300)
        kkt_res = kkt_residual(y_star_r, q_rand, pi_base_rand, tau)
        print(f'    Converged in {len(hist_r["y_err"])} iters, '
              f'KKT residual = {kkt_res:.2e}')
        print(f'    pi* = {pi_star_r}')
        for j in range(J):
            print(f'    y*[{j+1}] = {y_star_r[j]}')
            print(f'    Utility[{j+1}] = '
                  f'{principal_utility(j, y_star_r, q_rand, pi_base_rand, tau):.6f}')

    print()
    print('='*60)
    print('All experiments completed successfully!')
    print('='*60)


if __name__ == '__main__':
    main()
