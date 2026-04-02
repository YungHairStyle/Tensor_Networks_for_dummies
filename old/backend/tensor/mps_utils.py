import numpy as np
import quimb as qu
import quimb.tensor as qtn


def product_state_mps(N: int, state: str = "0") -> qtn.MatrixProductState:

    if state == "0":
        v = np.array([1.0, 0.0])
    elif state == "1":
        v = np.array([0.0, 1.0])
    elif state == "+":
        v = np.array([1.0, 1.0]) / np.sqrt(2)
    elif state == "-":
        v = np.array([1.0, -1.0]) / np.sqrt(2)
    else:
        raise ValueError(state)

    return qtn.MPS_product_state([v for _ in range(N)], site_tag_id="k{}")


def mps_to_statevector(psi: qtn.MatrixProductState) -> np.ndarray:
    """
    Convert an MPS to a dense statevector (only feasible for small N).
    """
    if hasattr(psi, "to_dense"):
        # some versions: returns qarray
        dense = psi.to_dense()
        return np.asarray(dense).reshape(-1)
    if hasattr(psi, "to_qarray"):
        dense = psi.to_qarray()
        return np.asarray(dense).reshape(-1)

    # fallback: contract and reshape
    tn = psi.to_tensor_network()
    x = tn.contract(all, optimize="auto-hq")
    return np.asarray(x).reshape(-1)


def mps_norm(psi: qtn.MatrixProductState) -> float:
    """
    Compute ||psi||.
    """
    if hasattr(psi, "norm"):
        return float(psi.norm())
    # fallback: inner product
    return float(abs((psi.H @ psi)) ** 0.5)  # may not exist in all versions


def canonicalize(psi: qtn.MatrixProductState, form: str = "left") -> qtn.MatrixProductState:
    """
    Bring MPS into a canonical form. This is useful for stable entropy and expectation values.

    form: "left" or "right"
    """
    if form not in ("left", "right"):
        raise ValueError("form must be 'left' or 'right'")

    # quimb has different naming across versions:
    if form == "left":
        if hasattr(psi, "left_canonize"):
            psi.left_canonize()
            return psi
        if hasattr(psi, "left_canonicalize"):
            psi.left_canonicalize()
            return psi
    else:
        if hasattr(psi, "right_canonize"):
            psi.right_canonize()
            return psi
        if hasattr(psi, "right_canonicalize"):
            psi.right_canonicalize()
            return psi

    # If missing, just return unchanged (still usable)
    return psi


def bond_dimensions(psi: qtn.MatrixProductState) -> list[int]:
    """
    Return bond dimensions across each cut (between sites k and k+1).
    """
    N = psi.nsites
    dims = []
    # Try common internal data layout
    if hasattr(psi, "bond_sizes"):
        return list(map(int, psi.bond_sizes()))
    # Fallback: infer from tensor shapes (best-effort)
    for k in range(N - 1):
        # tensor at site k typically has shape (chiL, d, chiR)
        A = psi[k].data
        dims.append(int(A.shape[-1]))
    return dims