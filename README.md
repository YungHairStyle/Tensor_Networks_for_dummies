# Ground-state benchmarks with matrix product states

Exploratory numerical work comparing exact diagonalization with a finite matrix-product-state implementation using imaginary-time time-evolving block decimation (TEBD). The benchmark driver covers transverse-field Ising, XXZ, and XY spin chains.

## Implementation

- `core/`: Hamiltonian definitions and shared result structures.
- `classical_solver.py`: exact-diagonalization reference calculations.
- `TN_solver.py`: MPS representation, imaginary-time evolution, and diagnostics.
- `main.py`: parameter sweeps and repeated tensor-network runs.
- `benchmark_plotter.py`: energy errors, fidelity, runtime, variance, residual, and magnetization plots.
- `figures/`: saved benchmark figures; `old/` retains earlier experiments.

## Running

Run `python main.py` from the repository root. The benchmark code uses NumPy, SciPy, and Matplotlib; the existing requirements file also contains dependencies from earlier experiments. Inspect the sweep settings in `main.py` before running the complete set.

The current TEBD implementation targets open, nearest-neighbour spin-1/2 chains. Bond dimension, imaginary-time step schedule, and convergence affect the result. Dense reference calculations grow exponentially with system size. The saved figures are exploratory benchmarks, rather than a general performance or convergence guarantee.

This repository is a numerical learning project under development. The current root-level workflow is a Python benchmark program; earlier web-interface experiments are separate from it.

---
[Asaad Hanna — research, CV, and contact](https://yunghairstyle.github.io/)

