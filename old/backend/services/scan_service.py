import time
import numpy as np
from old.backend.api.schemas import ScanRequest
from old.backend.physics.hamiltonians import build_tfim_mpo
from old.backend.physics.dmrg import run_dmrg_ground_state
from old.backend.physics.observables import magnetizations
from old.backend.utils.validation import require_model_supported

def run_scan(req: ScanRequest) -> dict:
    require_model_supported(req.model)

    t0 = time.perf_counter()

    hs = np.linspace(req.h_min, req.h_max, req.points).tolist()
    energies = []
    mzs = []

    for h in hs:
        H = build_tfim_mpo(req.N, req.J, float(h))
        dmrg_res = run_dmrg_ground_state(H, chi_max=req.chi_max, cutoff=req.cutoff, max_sweeps=req.max_sweeps)
        energies.append(dmrg_res.energy)
        mz, _ = magnetizations(dmrg_res.psi)
        mzs.append(mz)

    t1 = time.perf_counter()
    backend_ms = 1000.0 * (t1 - t0)

    return {
        "model": req.model,
        "N": req.N,
        "J": req.J,
        "chi_max": req.chi_max,
        "cutoff": req.cutoff,
        "max_sweeps": req.max_sweeps,
        "h_values": hs,
        "energy_values": energies,
        "mz_values": mzs,
        "backend_ms": backend_ms,
    }