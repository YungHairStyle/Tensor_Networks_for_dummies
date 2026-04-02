import numpy as np

from old.backend.physics.hamiltonians import build_tfim_mpo
from old.backend.physics.dmrg import run_dmrg_ground_state
from old.backend.tensor.mpo_utils import mpo_to_dense, ground_energy_exact


def test_dmrg_matches_exact_small_N():
    # Small enough for exact diagonalization
    N = 8
    J = 1.0
    h = 0.9

    H = build_tfim_mpo(N, J, h)

    # DMRG
    dmrg = run_dmrg_ground_state(H, chi_max=64, cutoff=1e-12, max_sweeps=20)

    # Exact
    Hd = mpo_to_dense(H)
    e0_exact = ground_energy_exact(Hd)

    # DMRG should be very close for this small case
    assert abs(dmrg.energy - e0_exact) < 1e-6