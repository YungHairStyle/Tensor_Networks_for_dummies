import numpy as np

from old.backend.physics.hamiltonians import build_tfim_mpo
from old.backend.physics.dmrg import run_dmrg_ground_state
from old.backend.physics.observables import magnetizations, entanglement_profile, correlator_zz_center


def test_observables_sane_ranges():
    N = 10
    J = 1.0
    h = 0.6

    H = build_tfim_mpo(N, J, h)
    dmrg = run_dmrg_ground_state(H, chi_max=64, cutoff=1e-12, max_sweeps=15)
    psi = dmrg.psi

    mz, mx = magnetizations(psi)
    assert -1.0001 <= mz <= 1.0001
    assert -1.0001 <= mx <= 1.0001

    cuts, ent = entanglement_profile(psi)
    assert len(cuts) == N - 1
    assert len(ent) == N - 1
    assert all(e >= -1e-10 for e in ent)  # numeric noise allowed

    rs, corr = correlator_zz_center(psi, max_r=5)
    assert len(rs) == len(corr) == 5
    # correlator values should be in [-1, 1] (loose tolerance)
    assert all(-1.0001 <= c <= 1.0001 for c in corr)