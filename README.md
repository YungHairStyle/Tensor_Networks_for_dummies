AI generated README:

# Tensor Network Lab (TFIM + MPS/DMRG) — quimb + FastAPI + HTML

## What this does
- Simulates the 1D Transverse-Field Ising Model (TFIM) ground state using DMRG (MPS) in quimb.
- Serves an HTML frontend where you can adjust N, J, h, bond dimension, cutoff, etc.
- Plots energy, magnetization, entanglement entropy profile, and correlators.

## Install
```bash
python -m venv .venv
source .venv/bin/activate   # mac/linux
# .venv\Scripts\activate    # windows
pip install -r requirements.txt