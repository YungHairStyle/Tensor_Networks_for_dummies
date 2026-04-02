import numpy as np

from old.backend.physics.hamiltonians import build_tfim_mpo
from old.backend.tensor.mpo_utils import mpo_to_dense, is_hermitian_dense


def test_tfim_mpo_is_hermitian_small():
    N = 6
    J = 1.0
    h = 0.7
    H = build_tfim_mpo(N, J, h)
    Hd = mpo_to_dense(H)
    assert Hd.shape == (2**N, 2**N)
    assert is_hermitian_dense(Hd, tol=1e-9)