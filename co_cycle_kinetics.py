"""
Kinetic modelling of the CO cycle (six-reaction network, R1-R6).

Reproduces the ODE integration, parameter fitting, uncertainty estimation
(Hessian-based), goodness-of-fit, and oxygen mass-balance check used in
the manuscript's kinetic modelling section (300 C ATR-FTIR time series).

Requires: numpy, scipy
"""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import differential_evolution, minimize

# ---------------------------------------------------------------------------
# 1. Experimental data
#    Integrated ATR-FTIR peak areas (a.u.) at 300 C, baseline-subtracted,
#    for the combined C-O pool (950-1200 cm-1), C=O (1660 cm-1), and
#    COO (1742 cm-1), each resolved by two-component Voigt fitting.
# ---------------------------------------------------------------------------

t_data = np.array([10, 30, 60, 180, 360, 600], dtype=float)          # min

# C-O pool = sum of the 1051 and 1101 cm-1 component areas
area_1051 = np.array([4.08822, 0.81459, 0.34860, 0.14140, 0.17308, 0.18400])
area_1101 = np.array([7.12488, 2.69293, 1.12975, 0.73863, 0.91715, 0.79520])
CO_data   = area_1051 + area_1101

CeqO_data = np.array([0.660240, 0.370156, 0.192572, 0.190703, 0.271085, 0.101201])  # C=O
COO_data  = np.array([0.016341, 0.091973, 0.136438, 0.093926, 0.130066, 0.094409])  # COO


# ---------------------------------------------------------------------------
# 2. Six-reaction network (R1-R6) -> coupled ODE system
#
#    R1: C-O  -> C=O            rate = k1*[C-O]
#    R2: C=O  -> CO(g)          rate = k2*[C=O]
#    R3: CO(g)-> C* + O*        rate = k3*[CO(g)]
#    R4: O* + CH -> C-O         rate = k4*[O*]         (CH treated as constant,
#                                                        absorbed into k4)
#    R5: C-O + CO(g) -> COO     rate = k5*[C-O]*[CO(g)] (bimolecular)
#    R6: COO -> CO2(g)          rate = k6*[COO]
#
#    State vector y = [A, B, Gas, Ostar, D, CO2]
#      A     = [C-O]
#      B     = [C=O]
#      Gas   = [CO(g)]
#      Ostar = [O*]
#      D     = [COO]
#      CO2   = cumulative CO2(g) escaped from the confined interface
#              (tracked only for the oxygen mass-balance check in
#              section 7; not fitted against experimental data)
# ---------------------------------------------------------------------------

def odes(y, t, k1, k2, k3, k4, k5, k6):
    # CO2 (state index 5) is tracked as a cumulative "lost" pool so that the
    # oxygen mass balance below can be closed exactly: R6 removes COO
    # (2 O) and releases CO2(g) (2 O), which then escapes the confined
    # system and must still be counted for mass-balance purposes.
    A, B, Gas, Ostar, D, CO2 = y
    A, B, Gas, Ostar, D, CO2 = (max(v, 0.0) for v in (A, B, Gas, Ostar, D, CO2))

    r1 = k1 * A                 # C-O  -> C=O
    r2 = k2 * B                 # C=O  -> CO(g)
    r3 = k3 * Gas                # CO(g)-> C* + O*
    r4 = k4 * Ostar              # O*   -> C-O (cycle closure)
    r5 = k5 * A * Gas             # C-O + CO(g) -> COO
    r6 = k6 * D                 # COO  -> CO2(g)  (leaves the system)

    dA     = -r1 - r5 + r4
    dB     =  r1 - r2
    dGas   =  r2 - r3 - r5
    dOstar =  r3 - r4
    dD     =  r5 - r6
    dCO2   =  r6
    return [dA, dB, dGas, dOstar, dD, dCO2]


def simulate(params, t_eval):
    """Integrate the ODE system for a given parameter set and return
    [C-O](t), [C=O](t), [COO](t) at the requested time points."""
    k1, k2, k3, k4, k5, k6, A0 = params
    y0 = [A0, 0.0, 0.0, 0.0, 0.0, 0.0]      # physically constrained initial conditions
    t_full = np.concatenate(([0.0], t_eval))
    sol = odeint(odes, y0, t_full, args=(k1, k2, k3, k4, k5, k6), mxstep=2000)
    sol = sol[1:]                           # drop the t=0 row
    A, B, _, _, D, _ = sol.T
    return A, B, D


# ---------------------------------------------------------------------------
# 3. Objective function (unweighted residual sum of squares across all
#    three fitted species: C-O pool, C=O, and COO)
# ---------------------------------------------------------------------------

def loss(params):
    if any(p <= 0 for p in params):
        return 1e12
    A_sim, B_sim, D_sim = simulate(params, t_data)

    res_A = (A_sim - CO_data)
    res_B = (B_sim - CeqO_data)
    res_D = (D_sim - COO_data)

    return np.sum(res_A**2) + np.sum(res_B**2) + np.sum(res_D**2)


# ---------------------------------------------------------------------------
# 4. Global optimisation (differential evolution) + local refinement
# ---------------------------------------------------------------------------

# parameter order: k1, k2, k3, k4, k5, k6, A0
bounds = [
    (1e-4, 2.0),      # k1  (min-1)
    (1e-4, 2.0),      # k2  (min-1)
    (1e-4, 2.0),      # k3  (min-1)
    (1e-4, 2.0),      # k4  (min-1)
    (1e-6, 1e-1),     # k5  (a.u.-1 min-1, bimolecular)
    (1e-4, 2.0),      # k6  (min-1)
    (1.0, 30.0),      # A0  (a.u.)
]

def fit_model(seed=0, maxiter=300, popsize=20):
    """Global search (differential evolution) followed by local refinement
    (Nelder-Mead). NOTE: differential_evolution is stochastic and the loss
    surface for this system is multimodal; exact fitted values can vary
    run-to-run and with the random seed. For a production fit, run
    multiple seeds (e.g. 10-20) and keep the lowest-loss result -- this is
    what was done to obtain the parameter set reported in the manuscript."""
    result_global = differential_evolution(
        loss, bounds, seed=seed, maxiter=maxiter, popsize=popsize, tol=1e-9,
        mutation=(0.5, 1.5), recombination=0.7, polish=False
    )
    # local refinement from the best global point
    result_local = minimize(
        loss, result_global.x, method='Nelder-Mead',
        options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 20000}
    )
    return result_local.x, result_local.fun


# ---------------------------------------------------------------------------
# 5. Uncertainty estimation via the numerical Hessian at the optimum
# ---------------------------------------------------------------------------

def numerical_hessian(f, x0, eps=1e-4):
    """Central-difference numerical Hessian of scalar function f at x0."""
    n = len(x0)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            h_i = eps * max(abs(x0[i]), 1.0)
            h_j = eps * max(abs(x0[j]), 1.0)

            x_pp = np.array(x0, dtype=float); x_pp[i] += h_i; x_pp[j] += h_j
            x_pm = np.array(x0, dtype=float); x_pm[i] += h_i; x_pm[j] -= h_j
            x_mp = np.array(x0, dtype=float); x_mp[i] -= h_i; x_mp[j] += h_j
            x_mm = np.array(x0, dtype=float); x_mm[i] -= h_i; x_mm[j] -= h_j

            H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h_i * h_j)
    return H


def parameter_uncertainty(params_opt):
    """±1 s.d. for each parameter from the inverse Hessian of the loss
    surface (Laplace approximation)."""
    H = numerical_hessian(loss, params_opt)
    try:
        cov = np.linalg.inv(H)
        sd = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        sd = np.full(len(params_opt), np.nan)
    return sd


# ---------------------------------------------------------------------------
# 6. Goodness of fit (R^2 per species)
# ---------------------------------------------------------------------------

def r_squared(y_data, y_model):
    ss_res = np.sum((y_data - y_model) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    return 1 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# 7. Oxygen mass-balance check
#    Stoichiometric oxygen weight per species: C-O x1, C=O x1, CO(g) x1,
#    O* x1, COO x2, CO2(g) x2. CO2 is tracked as a cumulative escaped
#    pool (see odes(), section 2) so that the balance below closes
#    exactly rather than appearing to leak oxygen once R6 becomes active.
# ---------------------------------------------------------------------------

def check_oxygen_balance(params, t_eval):
    k1, k2, k3, k4, k5, k6, A0 = params
    y0 = [A0, 0.0, 0.0, 0.0, 0.0, 0.0]
    t_full = np.concatenate(([0.0], t_eval))
    sol = odeint(odes, y0, t_full, args=(k1, k2, k3, k4, k5, k6), mxstep=2000)
    A, B, Gas, Ostar, D, CO2 = sol.T
    # include the escaped CO2 pool (2 O each) so the balance closes exactly
    total_O = A * 1 + B * 1 + Gas * 1 + Ostar * 1 + D * 2 + CO2 * 2
    return t_full, total_O   # should be constant = A0 (mass conserved)


# ---------------------------------------------------------------------------
# 8. Run everything
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Fitting six-reaction kinetic model to 300 C ATR-FTIR data...")
    print("(Multi-seed search: keeps the lowest-loss result across seeds.)")

    best_params, best_loss = None, np.inf
    for seed in range(10):
        p, lv = fit_model(seed=seed)
        print(f"  seed {seed}: loss = {lv:.4f}")
        if lv < best_loss:
            best_params, best_loss = p, lv

    params_opt, loss_val = best_params, best_loss
    k1, k2, k3, k4, k5, k6, A0 = params_opt

    print("\nOptimised parameters:")
    print(f"  k1 (C-O -> C=O)      = {k1:.5f} min-1   (t1/2 = {np.log(2)/k1:.1f} min)")
    print(f"  k2 (C=O -> CO(g))    = {k2:.5f} min-1   (t1/2 = {np.log(2)/k2:.1f} min)")
    print(f"  k3 (CO(g) -> C*+O*)  = {k3:.5f} min-1   (t1/2 = {np.log(2)/k3:.1f} min)")
    print(f"  k4 (O* -> C-O)       = {k4:.5f} min-1   (t1/2 = {np.log(2)/k4:.1f} min)")
    print(f"  k5 (C-O+CO(g)->COO)  = {k5:.6f} a.u.-1 min-1 (bimolecular)")
    print(f"  k6 (COO -> CO2(g))   = {k6:.5f} min-1   (t1/2 = {np.log(2)/k6:.1f} min)")
    print(f"  A0 (initial C-O)     = {A0:.3f} a.u.")
    print(f"  final loss           = {loss_val:.4f}")

    print("\nEstimating parameter uncertainties (Hessian-based, +/-1 s.d.)...")
    sd = parameter_uncertainty(params_opt)
    names = ['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'A0']
    for name, val, s in zip(names, params_opt, sd):
        print(f"  {name} = {val:.5f} +/- {s:.5f}")

    print("\nGoodness of fit (R^2 per species):")
    A_sim, B_sim, D_sim = simulate(params_opt, t_data)
    print(f"  C-O pool : R2 = {r_squared(CO_data, A_sim):.3f}")
    print(f"  C=O      : R2 = {r_squared(CeqO_data, B_sim):.3f}")
    print(f"  COO      : R2 = {r_squared(COO_data, D_sim):.3f}")

    print("\nOxygen mass balance check (should equal A0 at all times):")
    t_full, total_O = check_oxygen_balance(params_opt, t_data)
    max_dev_pct = 100 * np.max(np.abs(total_O - A0)) / A0
    print(f"  A0 = {A0:.6f}, max deviation = {max_dev_pct:.2e} %")
