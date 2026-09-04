# co-cycle-kinetics
Custom Python code for kinetic modelling of the CO cycle (six-reaction network, R1–R6) supporting the kinetic modelling section of the manuscript. Reproduces the ODE integration, parameter fitting, Hessian-based uncertainty estimation, goodness-of-fit assessment, and oxygen mass-balance check reported in the paper.
What this code does
Defines the six-reaction kinetic network (R1–R6) as a system of coupled ordinary differential equations
Fits the model to time-resolved ATR-FTIR integrated peak areas (C–O pool, C=O, COO) measured at 300 °C
Estimates parameter uncertainties (±1 s.d.) from the numerical Hessian of the loss surface
Verifies oxygen mass balance across all tracked species
Validates the published parameter set by default (deterministic); optionally re-derives the fit from scratch via multi-seed global optimisation
