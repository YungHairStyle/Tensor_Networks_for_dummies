from backend.api.schemas import TEBDRequest
from backend.utils.validation import require_model_supported
from backend.physics.tebd import tebd_quench_tfim

def run_tebd(req: TEBDRequest) -> dict:
    require_model_supported(req.model)

    res = tebd_quench_tfim(
        N=req.N,
        J=req.J,
        h=req.h,
        dt=req.dt,
        steps=req.steps,
        chi_max=req.chi_max,
        cutoff=req.cutoff,
        init_state=req.init_state,
        measure_every=req.measure_every,
    )

    return {
        "model": req.model,
        "N": req.N,
        "J": req.J,
        "h": req.h,
        "chi_max": req.chi_max,
        "cutoff": req.cutoff,
        "dt": req.dt,
        "steps": req.steps,
        "init_state": req.init_state,
        "measure_every": req.measure_every,
        "times": res.times,
        "mz": res.mz,
        "mx": res.mx,
        "entropy_mid": res.entropy_mid,
        "backend_ms": res.backend_ms,
    }