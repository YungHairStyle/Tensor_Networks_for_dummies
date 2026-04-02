from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SolverDiagnostics:
    solver_name: str
    method_family: str 
    converged: bool = True
    runtime_seconds: Optional[float] = None
    iterations: Optional[int] = None
    message: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TensorNetworkDiagnostics:
    """
    Optional diagnostics only relevant to TN solvers.
    Keep this nested so the same result object works for both solver types.
    """
    ansatz_type: Optional[str] = None               # e.g. "MPS", "PEPS"
    contraction_scheme: Optional[str] = None        # e.g. "left-to-right", "balanced", "greedy"
    optimizer: Optional[str] = None                 # e.g. "DMRG", "variational"
    max_bond_dimension: Optional[int] = None
    achieved_bond_dimension: Optional[int] = None
    truncation_cutoff: Optional[float] = None
    discarded_weight: Optional[float] = None
    sweeps: Optional[int] = None
    environment_strategy: Optional[str] = None
    canonical_form: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GroundStateResult:
    """
    Common result object returned by BOTH the classical solver
    and the tensor-network solver.

    This should make direct comparison easy.
    """
    problem_summary: Dict[str, Any]

    # Main physics output
    ground_energy: float
    energy_per_site: float
    state_vector: Optional[np.ndarray] = None       # for exact small-system solver
    state_metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional observables
    expectation_values: Dict[str, Any] = field(default_factory=dict)
    correlations: Dict[str, Any] = field(default_factory=dict)
    entanglement: Dict[str, Any] = field(default_factory=dict)

    # Error / residual style metrics
    energy_variance: Optional[float] = None
    residual_norm: Optional[float] = None

    # Diagnostics
    diagnostics: SolverDiagnostics = field(
        default_factory=lambda: SolverDiagnostics(
            solver_name="unknown",
            method_family="unknown",
        )
    )
    tn_diagnostics: Optional[TensorNetworkDiagnostics] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        # Convert numpy arrays for easier logging / serialization
        if self.state_vector is not None:
            data["state_vector"] = np.asarray(self.state_vector).tolist()

        return data

    def comparison_view(self) -> Dict[str, Any]:
        """
        A lightweight dictionary with the main numbers you usually want to compare.
        """
        out = {
            "model": self.problem_summary.get("model"),
            "n_sites": self.problem_summary.get("n_sites"),
            "boundary": self.problem_summary.get("boundary"),
            "solver_name": self.diagnostics.solver_name,
            "method_family": self.diagnostics.method_family,
            "converged": self.diagnostics.converged,
            "ground_energy": self.ground_energy,
            "energy_per_site": self.energy_per_site,
            "energy_variance": self.energy_variance,
            "residual_norm": self.residual_norm,
            "runtime_seconds": self.diagnostics.runtime_seconds,
            "iterations": self.diagnostics.iterations,
        }

        if self.tn_diagnostics is not None:
            out.update(
                {
                    "ansatz_type": self.tn_diagnostics.ansatz_type,
                    "contraction_scheme": self.tn_diagnostics.contraction_scheme,
                    "optimizer": self.tn_diagnostics.optimizer,
                    "max_bond_dimension": self.tn_diagnostics.max_bond_dimension,
                    "achieved_bond_dimension": self.tn_diagnostics.achieved_bond_dimension,
                    "discarded_weight": self.tn_diagnostics.discarded_weight,
                    "sweeps": self.tn_diagnostics.sweeps,
                }
            )

        return out


def compare_results(
    reference: GroundStateResult,
    candidate: GroundStateResult,
) -> Dict[str, Any]:
    """
    Compare two solver outputs, usually:
    - reference = classical exact diagonalization
    - candidate = tensor network result
    """
    comparison = {
        "same_problem": reference.problem_summary == candidate.problem_summary,
        "reference_solver": reference.diagnostics.solver_name,
        "candidate_solver": candidate.diagnostics.solver_name,
        "reference_energy": reference.ground_energy,
        "candidate_energy": candidate.ground_energy,
        "abs_energy_error": abs(reference.ground_energy - candidate.ground_energy),
        "abs_energy_per_site_error": abs(
            reference.energy_per_site - candidate.energy_per_site
        ),
        "reference_variance": reference.energy_variance,
        "candidate_variance": candidate.energy_variance,
        "reference_runtime_seconds": reference.diagnostics.runtime_seconds,
        "candidate_runtime_seconds": candidate.diagnostics.runtime_seconds,
        "reference_converged": reference.diagnostics.converged,
        "candidate_converged": candidate.diagnostics.converged,
    }

    return comparison