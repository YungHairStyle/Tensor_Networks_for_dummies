from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from core.problem import QuantumProblem
from core.result import GroundStateResult, SolverDiagnostics


@dataclass
class ClassicalSolverConfig:
    """
    Configuration for the classical solver.

    store_state_vector:
        If True, include the full ground-state vector in the result.
        For large systems this may be memory expensive.

    compute_observables:
        If True, compute one-site expectation values and nearest-neighbor correlations.

    check_hermiticity:
        If True, verify the Hamiltonian is Hermitian before diagonalizing.

    hermiticity_tol:
        Numerical tolerance for Hermiticity checks.
    """
    store_state_vector: bool = True
    compute_observables: bool = True
    check_hermiticity: bool = True
    hermiticity_tol: float = 1e-10


class ClassicalGroundStateSolver:
    """
    Dense exact diagonalization solver for small spin systems.

    This solver assumes the problem object provides:
        - full_hamiltonian()
        - summary()
        - n_sites
        - one_site_observables()
        - two_site_observables()
        - interaction_edges()

    It returns a GroundStateResult compatible with the tensor-network solver output.
    """

    def __init__(self, config: Optional[ClassicalSolverConfig] = None) -> None:
        self.config = config or ClassicalSolverConfig()

    def solve(self, problem: QuantumProblem) -> GroundStateResult:
        start_time = time.perf_counter()

        hamiltonian = problem.full_hamiltonian()

        if self.config.check_hermiticity:
            self._validate_hermitian(
                hamiltonian,
                tol=self.config.hermiticity_tol,
            )

        evals, evecs = np.linalg.eigh(hamiltonian)

        ground_energy = float(np.real(evals[0]))
        ground_state = evecs[:, 0]

        variance = self._energy_variance(hamiltonian, ground_state, ground_energy)
        residual_norm = self._residual_norm(hamiltonian, ground_state, ground_energy)

        expectation_values: Dict[str, object] = {}
        correlations: Dict[str, object] = {}
        entanglement: Dict[str, object] = {}

        if self.config.compute_observables:
            expectation_values = self._compute_one_site_expectations(problem, ground_state)
            correlations = self._compute_two_site_correlations(problem, ground_state)
            entanglement = self._compute_basic_entanglement(problem, ground_state)

        runtime_seconds = time.perf_counter() - start_time

        result = GroundStateResult(
            problem_summary=problem.summary(),
            ground_energy=ground_energy,
            energy_per_site=ground_energy / problem.n_sites,
            state_vector=ground_state if self.config.store_state_vector else None,
            state_metadata={
                "normalized": bool(np.isclose(np.vdot(ground_state, ground_state), 1.0)),
                "dimension": int(len(ground_state)),
            },
            expectation_values=expectation_values,
            correlations=correlations,
            entanglement=entanglement,
            energy_variance=variance,
            residual_norm=residual_norm,
            diagnostics=SolverDiagnostics(
                solver_name="ExactDiagonalization",
                method_family="classical",
                converged=True,
                runtime_seconds=runtime_seconds,
                iterations=1,
                message="Dense exact diagonalization completed successfully.",
                extra={
                    "hamiltonian_shape": hamiltonian.shape,
                    "dtype": str(hamiltonian.dtype),
                },
            ),
            tn_diagnostics=None,
        )

        return result

    def _validate_hermitian(self, hamiltonian: np.ndarray, tol: float = 1e-10) -> None:
        diff = np.linalg.norm(hamiltonian - hamiltonian.conj().T)
        if diff > tol:
            raise ValueError(
                f"Hamiltonian is not Hermitian within tolerance {tol}. "
                f"||H - H^†|| = {diff}"
            )

    def _energy_variance(
        self,
        hamiltonian: np.ndarray,
        state: np.ndarray,
        energy: float,
    ) -> float:
        """
        var(H) = <H^2> - <H>^2
        Should be ~0 for an exact eigenstate, up to numerical precision.
        """
        hpsi = hamiltonian @ state
        h2psi = hamiltonian @ hpsi

        exp_h = np.vdot(state, hpsi)
        exp_h2 = np.vdot(state, h2psi)

        variance = np.real(exp_h2 - exp_h * exp_h)
        variance = max(0.0, float(variance))
        return variance

    def _residual_norm(
        self,
        hamiltonian: np.ndarray,
        state: np.ndarray,
        energy: float,
    ) -> float:
        """
        ||H|psi> - E|psi>||
        """
        residual = hamiltonian @ state - energy * state
        return float(np.linalg.norm(residual))

    def _compute_one_site_expectations(
        self,
        problem: QuantumProblem,
        state: np.ndarray,
    ) -> Dict[str, object]:
        """
        Returns:
            {
                "X": [<X_0>, <X_1>, ...],
                "Y": [<Y_0>, <Y_1>, ...],
                "Z": [<Z_0>, <Z_1>, ...],
                "magnetization_x": ...,
                "magnetization_y": ...,
                "magnetization_z": ...
            }
        """
        observables = problem.one_site_observables()

        site_values: Dict[str, list] = {}
        for label, op in observables.items():
            if label == "I":
                continue
            vals = []
            for site in range(problem.n_sites):
                full_op = problem._embed_one_site_operator(site, op)
                val = np.vdot(state, full_op @ state)
                vals.append(float(np.real_if_close(val)))
            site_values[label] = vals

        out: Dict[str, object] = dict(site_values)

        if "X" in site_values:
            out["magnetization_x"] = float(np.mean(site_values["X"]))
        if "Y" in site_values:
            out["magnetization_y"] = float(np.mean(site_values["Y"]))
        if "Z" in site_values:
            out["magnetization_z"] = float(np.mean(site_values["Z"]))

        return out

    def _compute_two_site_correlations(
        self,
        problem: QuantumProblem,
        state: np.ndarray,
    ) -> Dict[str, object]:
        """
        Computes nearest-neighbor correlations on the interaction graph.

        Returns:
            {
                "XX": {"(0,1)": value, "(1,2)": value, ...},
                "YY": {...},
                "ZZ": {...},
                "average_XX": ...,
                "average_YY": ...,
                "average_ZZ": ...
            }
        """
        two_site_ops = problem.two_site_observables()

        correlations: Dict[str, Dict[str, float]] = {}
        averages: Dict[str, float] = {}

        for label, op in two_site_ops.items():
            edge_vals: Dict[str, float] = {}
            vals = []

            for i, j in problem.interaction_edges():
                full_op = problem._embed_two_site_operator(i, j, op)
                value = np.vdot(state, full_op @ state)
                real_value = float(np.real_if_close(value))
                edge_vals[f"({i},{j})"] = real_value
                vals.append(real_value)

            correlations[label] = edge_vals
            averages[f"average_{label}"] = float(np.mean(vals)) if vals else 0.0

        out: Dict[str, object] = dict(correlations)
        out.update(averages)
        return out

    def _compute_basic_entanglement(
        self,
        problem: QuantumProblem,
        state: np.ndarray,
    ) -> Dict[str, object]:
        """
        Computes a simple bipartite entanglement entropy for a half-chain cut,
        if possible.

        For n_sites = N, we reshape the state as:
            (2^(N_left), 2^(N_right))
        and compute the von Neumann entropy of the reduced density matrix.
        """
        n_left = problem.n_sites // 2
        n_right = problem.n_sites - n_left

        dim_left = 2 ** n_left
        dim_right = 2 ** n_right

        psi_matrix = state.reshape(dim_left, dim_right)
        singular_values = np.linalg.svd(psi_matrix, compute_uv=False)

        probs = singular_values**2
        probs = probs[probs > 1e-15]

        entropy = -np.sum(probs * np.log(probs))
        entropy_base2 = float(entropy / np.log(2.0))

        return {
            "half_chain_entropy_vn": entropy_base2,
            "schmidt_rank_numerical": int(len(probs)),
            "cut": {
                "left_sites": list(range(n_left)),
                "right_sites": list(range(n_left, problem.n_sites)),
            },
        }


def solve(problem: QuantumProblem, config: Optional[ClassicalSolverConfig] = None) -> GroundStateResult:
    """
    Convenience function so you can just do:

        result = solve(problem)

    instead of manually constructing the solver.
    """
    solver = ClassicalGroundStateSolver(config=config)
    return solver.solve(problem)