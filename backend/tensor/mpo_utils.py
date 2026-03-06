import numpy as np
import quimb.tensor as qtn


def mpo_to_dense(H_mpo) -> np.ndarray:
    """
    Convert an MPO to a dense matrix (only feasible for small N).
    Returns a numpy array of shape (2**N, 2**N).
    """
    if hasattr(H_mpo, "to_dense"):
        dense = H_mpo.to_dense()
        return np.asarray(dense)
    if hasattr(H_mpo, "to_qarray"):
        dense = H_mpo.to_qarray()
        return np.asarray(dense)

    # Fallback: contract tensor network representation (can be expensive)
    # This path might not exist for all MPO objects; if it errors, use to_dense().
    tn = H_mpo.to_tensor_network()
    dense = tn.contract(all, optimize="auto-hq")
    return np.asarray(dense)


def is_hermitian_dense(H: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Check if H is Hermitian within tolerance.
    """
    return np.max(np.abs(H - H.conj().T)) < tol


def ground_energy_exact(H_dense: np.ndarray) -> float:
    """
    Exact ground energy from a dense Hermitian Hamiltonian.
    """
    evals = np.linalg.eigvalsh(H_dense)
    return float(np.min(evals))