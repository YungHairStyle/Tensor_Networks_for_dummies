import quimb.tensor as qtn
import quimb as qu

def build_tfim_mpo(N: int, J: float, h: float):
    """
    TFIM: H = -J sum Z_i Z_{i+1} - h sum X_i
    Open boundary conditions.
    """
    ham = qtn.SpinHam1D(S=1/2)
    # two-site term
    ham += (-J, "Z", "Z")
    # one-site term
    ham += (-h, "X")
    # build MPO
    H = ham.build_mpo(N)
    return H

def build_tfim_localham(N: int, J: float, h: float):
    """
    Build the TFIM as a LocalHam1D object for TEBD.

    H = -J sum_i Z_i Z_{i+1} - h sum_i X_i
    """
    Z = qu.pauli("Z")
    X = qu.pauli("X")

    H2 = -J * (Z & Z)   # same as kron for quimb operator objects
    H1 = -h * X

    ham = qtn.LocalHam1D(
        L=N,
        H2=H2,
        H1=H1,
        cyclic=False,
    )

    return ham