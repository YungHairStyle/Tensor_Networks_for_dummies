from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import quimb as qu
import quimb.tensor as qtn

from backend.physics.hamiltonians import build_tfim_localham
from backend.tensor.mps_utils import product_state_mps


@dataclass
class TEBDResult:
    times: list[float]
    mz: list[float]
    mx: list[float]
    entropy_mid: list[float]
    backend_ms: float


def _local_expect(psi, op, i, max_bond=None, optimize="auto-hq"):
    """
    Compute <psi|op_i|psi> from the one-site reduced density matrix.
    This avoids quimb's broken local_expectation -> partial_trace path.
    """
    import numpy as np

    # preferred path for your quimb version
    if hasattr(psi, "partial_trace_to_dense_canonical"):
        rho = psi.partial_trace_to_dense_canonical((i,))
        rho = np.asarray(rho).reshape(2, 2)
        return float(np.trace(rho @ np.asarray(op)).real)

    if hasattr(psi, "partial_trace_to_mpo"):
        rho_mpo = psi.partial_trace_to_mpo((i,))
        if hasattr(rho_mpo, "to_dense"):
            rho = np.asarray(rho_mpo.to_dense()).reshape(2, 2)
        elif hasattr(rho_mpo, "to_qarray"):
            rho = np.asarray(rho_mpo.to_qarray()).reshape(2, 2)
        else:
            tn = rho_mpo.to_tensor_network()
            rho = np.asarray(tn.contract(all, optimize=optimize)).reshape(2, 2)

        return float(np.trace(rho @ np.asarray(op)).real)

    raise AttributeError(
        "Could not find a usable reduced density matrix method on this MPS object."
    )


def _entropy_mid(psi) -> float:
    N = psi.nsites
    k = N // 2
    if hasattr(psi, "entropy"):
        return float(psi.entropy(k))
    if hasattr(psi, "entanglement_entropy"):
        return float(psi.entanglement_entropy(k))
    return float("nan")


def tebd_quench_tfim(
    N: int,
    J: float,
    h: float,
    dt: float,
    steps: int,
    chi_max: int,
    cutoff: float,
    init_state: Literal["0", "1", "+", "-"] = "+",
    measure_every: int = 1,
) -> TEBDResult:
    """
    Run TEBD time evolution for the TFIM Hamiltonian.

    - Start in a product state
    - Evolve under the TFIM using quimb's TEBD engine
    - Track mZ, mX, and mid-chain entanglement entropy
    """
    t0 = time.perf_counter()

    # initial state
    psi = product_state_mps(N, state=init_state)

    # observables
    Z = qu.pauli("Z")
    X = qu.pauli("X")

    # build Hamiltonian in local form for TEBD
    H = build_tfim_localham(N, J, h)

    # initialize TEBD engine
    tebd = qtn.TEBD(psi, H, dt=dt, progbar=False)

    # truncation settings if supported
    if hasattr(tebd, "split_opts"):
        tebd.split_opts["cutoff"] = cutoff
        tebd.split_opts["max_bond"] = chi_max
    elif hasattr(tebd, "set_truncation"):
        tebd.set_truncation(cutoff=cutoff, max_bond=chi_max)

    times = []
    mzs = []
    mxs = []
    ents = []

    def measure(t: float):
        # tebd.pt is usually the current MPS state in quimb TEBD
        state = tebd.pt if hasattr(tebd, "pt") else psi

        times.append(float(t))
        mzs.append(float(np.mean([_local_expect(state, Z, i, chi_max) for i in range(N)])))
        mxs.append(float(np.mean([_local_expect(state, X, i) for i in range(N)])))
        ents.append(_entropy_mid(state))

    # initial measurement
    measure(0.0)

    current_t = 0.0

    for s in range(1, steps + 1):
        # evolve by one TEBD time step
        if hasattr(tebd, "step"):
            tebd.step()
        elif hasattr(tebd, "update_to"):
            current_t += dt
            tebd.update_to(current_t)
        else:
            raise AttributeError(
                "Could not find a TEBD stepping method on this quimb version."
            )

        current_t = s * dt

        if s % measure_every == 0:
            measure(current_t)

    t1 = time.perf_counter()

    return TEBDResult(
        times=times,
        mz=mzs,
        mx=mxs,
        entropy_mid=ents,
        backend_ms=1000.0 * (t1 - t0),
    )