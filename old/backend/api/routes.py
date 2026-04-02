from fastapi import APIRouter
from old.backend.api.schemas import GroundStateRequest, ScanRequest, TEBDRequest
from old.backend.services.simulation_service import run_ground_state
from old.backend.services.scan_service import run_scan
from old.backend.services.tebd_service import run_tebd

router = APIRouter()

@router.post("/ground_state")
def ground_state(req: GroundStateRequest):
    return run_ground_state(req)

@router.post("/scan")
def scan(req: ScanRequest):
    return run_scan(req)

@router.post("/tebd")
def tebd(req: TEBDRequest):
    return run_tebd(req)