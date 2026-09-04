# CO Cycle Kinetic Modelling

Custom Python code for kinetic modelling of the CO cycle (six-reaction network, R1–R6) supporting the kinetic modelling section of the manuscript. Reproduces the ODE integration, parameter fitting, Hessian-based uncertainty estimation, goodness-of-fit assessment, and oxygen mass-balance check reported in the paper.

## What this code does

- Defines the six-reaction kinetic network (R1–R6) as a system of coupled ordinary differential equations
- Fits the model to time-resolved ATR-FTIR integrated peak areas (C–O pool, C=O, COO) measured at 300 °C
- Estimates parameter uncertainties (±1 s.d.) from the numerical Hessian of the loss surface
- Verifies oxygen mass balance across all tracked species
- Validates the published parameter set by default (deterministic); optionally re-derives the fit from scratch via multi-seed global optimisation

## Requirements

- Python 3.9+
- numpy
- scipy

Install dependencies:

```bash
pip install numpy scipy
```

## Usage

Validate the published parameter set (default; fast, deterministic, reproduces the manuscript's reported R² and mass-balance values):

```bash
python co_cycle_kinetics.py
```

Re-derive the fit from scratch via multi-seed global optimisation (slower; not required to reproduce the manuscript's results, provided for methodological transparency):

```bash
python co_cycle_kinetics.py --refit
```

## Output

The script prints:
- Fitted rate constants (k1–k6) and the initial C–O pool value, with ±1 s.d. uncertainties
- Per-species R² values (C–O pool, C=O, COO)
- Oxygen mass-balance check (percentage deviation from exact conservation)

## Citation

If you use this code, please cite the associated manuscript [complete once published] and, for the numerical methods used, SciPy:

Virtanen, P. et al. SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nat. Methods* 17, 261–272 (2020).
