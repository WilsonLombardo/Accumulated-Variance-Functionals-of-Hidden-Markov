# Accumulated-Variance Functionals of Hidden Markov and Semi-Markov Processes: Finite-Sample Upper and Lower Bounds

**Author:** Wilson Lombardo[cite: 1]

## Overview
This repository contains the preprint and the reproducibility code for the paper *Accumulated-Variance Functionals of Hidden Markov and Semi-Markov Processes: Finite-Sample Upper and Lower Bounds*[cite: 1].

We study two accumulated variance functionals for stationary two-state hidden Markov or hidden semi-Markov processes, observed through a single trajectory[cite: 1]. The study separates identifying these observable quantities from recovering the latent parameters behind them[cite: 1]. For bounded durations, inverting block moments gives finite-sample upper bounds, worked out with fully explicit constants for a capped-geometric Gaussian family[cite: 1]. On fixed interior classes, the corresponding two-point lower bounds have the same rate[cite: 1]. 

**Keywords:** accumulated variance; hidden Markov process; hidden semi-Markov process; finite-sample concentration; minimax lower bound; persistence[cite: 1].

## Code and Reproducibility
The repository includes Python scripts to audit the constants and reproduce the Monte Carlo simulations presented in the paper[cite: 1]:

* **`step1_explicit_constants.py`**: Contains the numerical matrix audit and the exact local Jacobian calculation for the explicit capped-geometric Gaussian submodel[cite: 1].
* **`Paper/simulacion_reproducible_v2.py`**: Reproduces the simulation for the persistent-path rates and the saturation audit discussed in Section 9[cite: 1].
* **Numerical Output**: The complete numerical output from the simulations is stored in two versioned CSV files located in the `Paper` directory[cite: 1].

## Reference
If you use this code or find this work helpful, please refer to the main preprint file included in this repository: `Preprint.pdf`[cite: 1].
