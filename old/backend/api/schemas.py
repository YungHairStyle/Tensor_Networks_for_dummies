from pydantic import BaseModel, Field
from typing import Literal
class GroundStateRequest(BaseModel):
    model: str = Field(default="tfim")
    N: int = Field(ge=2, le=200)    #number of sites
    J: float    #exchange coupling
    h: float    #transverse field strength
    chi_max: int = Field(ge=2, le=2048) #maximum bond dimension for DMRG
    cutoff: float = Field(gt=0.0)   #truncation cutoff for DMRG
    max_sweeps: int = Field(ge=2, le=200)   #maximum number of DMRG sweeps
    corr_max_r: int = Field(ge=2, le=200)   #maximum distance for correlator calculation

class ScanRequest(BaseModel):
    model: str = Field(default="tfim")
    N: int = Field(ge=2, le=200)    #number of sites
    J: float    #exchange coupling
    chi_max: int = Field(ge=2, le=2048) #maximum bond dimension for DMRG
    cutoff: float = Field(gt=0.0)   #truncation cutoff for DMRG
    max_sweeps: int = Field(ge=2, le=200)   #maximum number of DMRG sweeps
    h_min: float    #minimum transverse field strength
    h_max: float    #maximum transverse field strength
    points: int = Field(ge=3, le=400)   #number of points in the scan

class TEBDRequest(BaseModel):
    model: str = Field(default="tfim")
    N: int = Field(ge=2, le=200)    #number of sites
    J: float    #exchange coupling
    h: float    #transverse field strength
    chi_max: int = Field(ge=2, le=2048) #maximum bond dimension for TEBD
    cutoff: float = Field(gt=0.0)   #truncation cutoff for TEBD
    dt: float = Field(gt=0.0, le=1.0)   #time step for TEBD
    steps: int = Field(ge=1, le=20000)   #number of TEBD steps
    init_state: Literal["0", "1", "+", "-"] = Field(default="+")
    measure_every: int = Field(ge=1, le=1000)   #measure every nth step