import numpy as np
import quimb as qu
import quimb.tensor as qtn

_DEFAULT_OPTIMIZE = "auto-hq"


def _mps_expec_mpo(psi, mpo, optimize=_DEFAULT_OPTIMIZE) -> float:
    """
    Compute <psi|MPO|psi> by explicitly matching:
      bra indices -> MPO upper indices
      ket indices -> MPO lower indices

    This avoids relying on version-specific behavior of psi.expec(...).
    """
    import numpy as np

    # bra uses the original site indices, e.g. k0, k1, ...
    bra = psi.H

    # ket must be reindexed to match the MPO lower indices, e.g. b0, b1, ...
    ket = psi.copy()

    # infer the current MPS site index pattern
    # in your project this should be k{}
    reindex_map = {f"k{i}": f"b{i}" for i in range(psi.nsites)}
    ket.reindex_(reindex_map)

    # now contract bra | mpo | ket
    tn = bra | mpo | ket
    val = tn.contract(all, optimize=optimize)

    # robust scalar extraction
    if hasattr(val, "data"):
        arr = np.asarray(val.data)
    else:
        arr = np.asarray(val)

    if arr.shape == ():
        return float(np.real_if_close(arr.item()))
    if arr.size == 1:
        return float(np.real_if_close(arr.reshape(-1)[0]))

    raise TypeError(
        f"Expected scalar contraction result, got type={type(val)} shape={arr.shape}"
    )


def _local_mpo(op, N: int, i: int):
    """
    Build an MPO representing a single-site operator op acting on site i.
    """
    d = op.shape[0]
    arrays = []

    for k in range(N):
        if k == i:
            A = op.reshape(1, 1, d, d)
        else:
            A = qu.eye(d).reshape(1, 1, d, d)

        arrays.append(A)

    return qtn.MatrixProductOperator(
        arrays,
        shape="lrud",
        upper_ind_id="k{}",   # must match MPS site indices
        lower_ind_id="b{}",   # new index set
        site_tag_id="I{}",
    )


def _two_site_mpo(opA, opB, N: int, i: int, j: int):
    """
    MPO for opA at site i and opB at site j (i != j).
    """
    if i == j:
        raise ValueError("i and j must be different for two-site MPO.")
    d = opA.shape[0]
    arrays = []
    for k in range(N):
        if k == i:
            A = opA.reshape(1, 1, d, d)
        elif k == j:
            A = opB.reshape(1, 1, d, d)
        else:
            A = qu.eye(d).reshape(1, 1, d, d)
        arrays.append(A)
    return qtn.MatrixProductOperator(arrays, site_tag_id="k{}")


def magnetizations(psi, optimize=_DEFAULT_OPTIMIZE) -> tuple[float, float]:
    """
    Return average magnetizations mZ and mX using MPO expectations.
    """

    N = psi.nsites
    Z = qu.pauli("Z")
    X = qu.pauli("X")

    mz = np.mean([_mps_expec_mpo(psi, _local_mpo(Z, N, i), optimize=optimize) for i in range(N)])
    mx = np.mean([_mps_expec_mpo(psi, _local_mpo(X, N, i), optimize=optimize) for i in range(N)])
    return float(mz), float(mx)


def entanglement_profile(psi):
    """
    Return entropy across each bond cut (between sites k and k+1).
    This part is usually stable across versions.
    """
    N = psi.nsites
    cuts = list(range(1, N))
    ent = []

    if hasattr(psi, "entropy"):
        for k in cuts:
            ent.append(float(psi.entropy(k)))
        return cuts, ent

    if hasattr(psi, "entanglement_entropy"):
        for k in cuts:
            ent.append(float(psi.entanglement_entropy(k)))
        return cuts, ent

    raise AttributeError("Couldn't find an entropy method on this quimb MPS object.")


def correlator_zz_center(psi, max_r: int, optimize=_DEFAULT_OPTIMIZE):
    """
    C(r) = <Z_i Z_{i+r}> with i chosen as center site, using MPO expectations.
    """
    N = psi.nsites
    i0 = N // 2
    max_r = int(min(max_r, N - 1 - i0))

    Z = qu.pauli("Z")
    rs = list(range(1, max_r + 1))
    vals = []
    for r in rs:
        mpo = _two_site_mpo(Z, Z, N, i0, i0 + r)
        vals.append(_mps_expec_mpo(psi, mpo, optimize=optimize))
    return rs, list(map(float, vals))