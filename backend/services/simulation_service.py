import time
from backend.api.schemas import GroundStateRequest
from backend.physics.hamiltonians import build_tfim_mpo
from backend.physics.dmrg import run_dmrg_ground_state
from backend.physics.observables import magnetizations, entanglement_profile, correlator_zz_center
from backend.utils.validation import require_model_supported

def run_ground_state(req: GroundStateRequest) -> dict:
    #require_model_supported(req.model) #check if model is supported
        #no need to implement this check for now since only one model is implemented, but can be used in the future when more models are added.
    t0 = time.perf_counter()

    H = build_tfim_mpo(req.N, req.J, req.h)
    dmrg_res = run_dmrg_ground_state(H, chi_max=req.chi_max, cutoff=req.cutoff, max_sweeps=req.max_sweeps)

    psi = dmrg_res.psi
    
    mz, mx = magnetizations(psi)
    cuts, ent = entanglement_profile(psi) 
    rs, corr = correlator_zz_center(psi, max_r=req.corr_max_r)

    t1 = time.perf_counter()
    backend_ms = 1000.0 * (t1 - t0)

    return {
        "model": req.model,
        "N": req.N,
        "J": req.J,
        "h": req.h,
        "chi_max": req.chi_max,
        "cutoff": req.cutoff,
        "max_sweeps": req.max_sweeps,
        "energy": dmrg_res.energy,
        "mz": mz,
        "mx": mx,
        "entropy_cut_indices": cuts,
        "entropy": ent,
        "corr_r": rs,
        "corr_zz": corr,
        "backend_ms": backend_ms,
        "dmrg_ms": dmrg_res.dmrg_ms,
    }