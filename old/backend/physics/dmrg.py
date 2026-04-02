from dataclasses import dataclass
import time
import quimb.tensor as qtn

import numpy as np


def build_dmrg_schedule(chi_max: int, cutoff: float, max_sweeps: int):
    """
    Build bond_dims and cutoffs arrays of length max_sweeps.
    """

    # --- bond dimension ramp ---
    chi_start = max(4, chi_max // 8)

    # logarithmic ramp toward chi_max
    bond_dims = np.geomspace(chi_start, chi_max, max_sweeps)

    bond_dims = np.round(bond_dims).astype(int)

    # ensure monotonic
    bond_dims = np.maximum.accumulate(bond_dims)

    # cap exactly at chi_max
    bond_dims = np.clip(bond_dims, None, chi_max)

    bond_dims = bond_dims.tolist()

    # --- cutoff ramp ---
    cutoff_loose = max(cutoff * 1e4, 1e-4)

    cutoffs = np.geomspace(cutoff_loose, cutoff, max_sweeps)
    cutoffs = cutoffs.tolist()

    return bond_dims, cutoffs

@dataclass
class DMRGResult:
    energy: float
    psi: qtn.MatrixProductState
    dmrg_ms: float

def run_dmrg_ground_state(H_mpo, chi_max: int, cutoff: float, max_sweeps: int) -> DMRGResult:
    """
    Run DMRG using quimb's DMRG2 driver.
    """
    t0 = time.perf_counter()    #start timer for DMRG run

    # bond_dims/cutoffs can be lists or single values; lists allow sweep schedules
    bond_dims, cutoffs = build_dmrg_schedule(
        chi_max,
        cutoff,
        max_sweeps
    )

    dmrg = qtn.DMRG2(
        H_mpo,
        bond_dims=bond_dims,
        cutoffs=cutoffs
    )
    energy = dmrg.solve(tol=1e-9, max_sweeps=max_sweeps)
    psi = dmrg.state
   
    
    t1 = time.perf_counter()
    return DMRGResult(energy=float(energy), psi=psi, dmrg_ms=1000.0 * (t1 - t0))




