from __future__ import annotations

# Import timing utilities so we can report runtime in the result object.
import time

# Import dataclass so we can define a clean configuration object.
from dataclasses import dataclass

# Import typing helpers for readability and type hints.
from typing import Dict, List, Optional, Tuple

# Import NumPy for array and tensor operations.
import numpy as np

# Import expm so we can build imaginary-time evolution gates exp(-tau H_local).
from scipy.linalg import expm

# Import the problem definition and boundary enum from your problem file.
from core.problem import QuantumProblem, BoundaryCondition

# Import the shared result/diagnostic classes so this solver matches your interface.
from core.result import (
    GroundStateResult,
    SolverDiagnostics,
    TensorNetworkDiagnostics,
)


@dataclass
class TNSolverConfig:
    # Maximum MPS bond dimension kept after each two-site truncation.
    max_bond_dimension: int = 32

    # List of imaginary-time steps; large tau first, then smaller tau to refine.
    tau_schedule: Tuple[float, ...] = (1e-1, 5e-2, 1e-2, 5e-3, 1e-3)

    # Number of TEBD steps to perform for each tau value in tau_schedule.
    steps_per_tau: int = 20

    # Convergence tolerance based on the change in total energy between checks.
    energy_tolerance: float = 1e-8

    # Singular values smaller than this are discarded after each two-site update.
    svd_cutoff: float = 1e-12

    # Seed used when initializing a random product state.
    random_seed: Optional[int] = None

    # Initialization style for the MPS.
    init_state: str = "product_up"  # options: "product_up", "product_down", "random"

    # Whether to compute observables after the evolution finishes.
    compute_observables: bool = True

    # Whether to reconstruct and store the full state vector if the system is small enough.
    store_state_vector_if_small: bool = False

    # Maximum Hilbert-space dimension for reconstructing a dense state for diagnostics.
    dense_state_max_dim: int = 2 ** 14

    # Verbosity level: 0 silent, 1 progress prints.
    verbosity: int = 0


class TensorNetworkGroundStateSolver:
    """
    Finite MPS imaginary-time TEBD solver for 1D open spin-1/2 chains.

    Supported Hamiltonian structure:
        H = sum_i h_i + sum_i h_{i,i+1}

    Supported problem assumptions:
        - open boundary conditions
        - local dimension 2
        - one-site terms and nearest-neighbor two-site terms only
    """

    def __init__(self, config: Optional[TNSolverConfig] = None) -> None:
        # Store the user-provided config or create a default one.
        self.config = config or TNSolverConfig()

        # Create a reproducible random number generator for random initialization.
        self.rng = np.random.default_rng(self.config.random_seed)

    def solve(self, problem: QuantumProblem) -> GroundStateResult:
        # Record the start time so we can report runtime later.
        start_time = time.perf_counter()

        # Check that the problem satisfies the assumptions of this TEBD implementation.
        self._validate_problem(problem)

        # Group all one-site terms into a single operator per site,
        # and all nearest-neighbor two-site terms into a single operator per bond.
        onsite_terms, bond_terms = self._group_local_terms(problem)

        # Build an initial MPS representing the starting trial state.
        mps = self._initialize_mps(problem.n_sites, problem.local_dim, self.config.init_state)

        # Normalize the initial MPS so the state norm starts at 1.
        self._normalize_mps_inplace(mps)

        # Precompute all one-site and two-site imaginary-time gates for every tau in the schedule.
        onsite_gate_table, even_gate_table, odd_gate_table = self._build_gate_tables(
            problem=problem,
            onsite_terms=onsite_terms,
            bond_terms=bond_terms,
        )

        # This list will store the total energy after each set of TEBD updates.
        energy_history: List[float] = []

        # Track whether we converged according to the energy tolerance.
        converged = False

        # Track the total discarded weight from SVD truncations.
        total_discarded_weight = 0.0

        # Track the maximum discarded weight seen in any single two-site update.
        max_discarded_weight = 0.0

        # Track the largest bond dimension that actually appears during the run.
        achieved_max_bond_dimension = 1

        # Loop over progressively smaller imaginary-time steps.
        for tau in self.config.tau_schedule:
            # Fetch the precomputed one-site half-step gates for this tau.
            onsite_half_gates = onsite_gate_table[tau]

            # Fetch the precomputed even-bond half-step gates for this tau.
            even_half_gates = even_gate_table[tau]

            # Fetch the precomputed odd-bond full-step gates for this tau.
            odd_full_gates = odd_gate_table[tau]

            # Repeat TEBD updates several times for this tau value.
            for step in range(self.config.steps_per_tau):
                # Apply exp(-tau/2 * H_onsite) to every site tensor.
                self._apply_all_one_site_gates(mps, onsite_half_gates)

                # Apply exp(-tau/2 * H_even) to every even bond.
                discarded_even_1, max_bond_even_1 = self._apply_bond_gate_layer(
                    mps=mps,
                    gates=even_half_gates,
                    bond_starts=range(0, problem.n_sites - 1, 2),
                    max_bond_dim=self.config.max_bond_dimension,
                    svd_cutoff=self.config.svd_cutoff,
                )

                # Apply exp(-tau * H_odd) to every odd bond.
                discarded_odd, max_bond_odd = self._apply_bond_gate_layer(
                    mps=mps,
                    gates=odd_full_gates,
                    bond_starts=range(1, problem.n_sites - 1, 2),
                    max_bond_dim=self.config.max_bond_dimension,
                    svd_cutoff=self.config.svd_cutoff,
                )

                # Apply exp(-tau/2 * H_even) again for second-order Trotter symmetry.
                discarded_even_2, max_bond_even_2 = self._apply_bond_gate_layer(
                    mps=mps,
                    gates=even_half_gates,
                    bond_starts=range(0, problem.n_sites - 1, 2),
                    max_bond_dim=self.config.max_bond_dimension,
                    svd_cutoff=self.config.svd_cutoff,
                )

                # Apply exp(-tau/2 * H_onsite) again to complete the second-order step.
                self._apply_all_one_site_gates(mps, onsite_half_gates)

                # Renormalize the MPS because imaginary-time evolution is non-unitary.
                self._normalize_mps_inplace(mps)

                # Update discarded-weight bookkeeping.
                total_discarded_weight += discarded_even_1 + discarded_odd + discarded_even_2

                # Update the maximum single-step discarded weight seen so far.
                max_discarded_weight = max(
                    max_discarded_weight,
                    discarded_even_1,
                    discarded_odd,
                    discarded_even_2,
                )

                # Update the largest bond dimension reached so far.
                achieved_max_bond_dimension = max(
                    achieved_max_bond_dimension,
                    max_bond_even_1,
                    max_bond_odd,
                    max_bond_even_2,
                    self._current_max_bond_dimension(mps),
                )

                # Compute the current total energy after this TEBD step.
                energy = self._compute_total_energy(problem, mps, onsite_terms, bond_terms)

                # Append the current energy to the history list.
                energy_history.append(energy)

                # Print progress if verbosity is enabled.
                if self.config.verbosity >= 1:
                    print(
                        f"[tau={tau:.3e} step={step + 1:>3d}/{self.config.steps_per_tau}] "
                        f"E = {energy:.12f}"
                    )

                # If we have at least two energy values, check for convergence.
                if len(energy_history) >= 2:
                    # Compute the absolute change in energy since the previous check.
                    delta_e = abs(energy_history[-1] - energy_history[-2])

                    # If the change is smaller than the tolerance, mark as converged and stop.
                    if delta_e < self.config.energy_tolerance:
                        converged = True
                        break

            # If convergence happened inside the inner loop, stop the outer tau loop too.
            if converged:
                break

        # Measure the total runtime.
        runtime_seconds = time.perf_counter() - start_time

        # Compute observables only if requested.
        if self.config.compute_observables:
            # Compute one-site expectation values such as magnetizations.
            expectation_values = self._compute_one_site_expectations(problem, mps)

            # Compute nearest-neighbor correlations like XX, YY, ZZ.
            correlations = self._compute_two_site_correlations(problem, mps)

            # Compute a simple entanglement summary.
            entanglement = self._compute_entanglement_summary(problem, mps)
        else:
            # If observables are disabled, return empty dictionaries.
            expectation_values = {}
            correlations = {}
            entanglement = {}

        # Initialize the dense-state-related outputs.
        state_vector = None
        energy_variance = None
        residual_norm = None

        # Start building metadata about the returned state.
        state_metadata = {
            "representation": "MPS",
            "algorithm": "imaginary_time_TEBD",
        }

        # If the Hilbert space is small enough, reconstruct the dense state for diagnostics.
        if problem.hilbert_dim <= self.config.dense_state_max_dim:
            # Convert the MPS into a full dense wavefunction.
            dense_state = self._mps_to_dense_state(mps)

            # Normalize the dense state explicitly to avoid numerical drift.
            dense_state = dense_state / np.linalg.norm(dense_state)

            # Store the dense state only if the user asked for it.
            if self.config.store_state_vector_if_small:
                state_vector = dense_state

            # Build the full dense Hamiltonian for exact diagnostics on small systems.
            h_dense = problem.full_hamiltonian()

            # Compute H|psi>.
            hpsi = h_dense @ dense_state

            # Compute <psi|H|psi>.
            exp_h = np.vdot(dense_state, hpsi)

            # Compute H^2|psi>.
            h2psi = h_dense @ hpsi

            # Compute <psi|H^2|psi>.
            exp_h2 = np.vdot(dense_state, h2psi)

            # Compute the energy variance var(H) = <H^2> - <H>^2.
            energy_variance = max(0.0, float(np.real(exp_h2 - exp_h * exp_h)))

            # Compute the residual norm ||H|psi> - E|psi>||.
            residual_norm = float(np.linalg.norm(hpsi - np.real(exp_h) * dense_state))

            # Add dense-state diagnostic info to metadata.
            state_metadata["dense_state_reconstructed"] = True
            state_metadata["dense_state_dimension"] = int(problem.hilbert_dim)
        else:
            # If the system is too large, note that dense reconstruction was skipped.
            state_metadata["dense_state_reconstructed"] = False
            state_metadata["dense_state_dimension"] = None

        # Use the last recorded energy as the final ground-state estimate.
        final_energy = energy_history[-1] if energy_history else self._compute_total_energy(
            problem, mps, onsite_terms, bond_terms
        )

        # Build a human-readable convergence message.
        if converged:
            message = "Imaginary-time TEBD converged by energy tolerance."
        else:
            message = "Imaginary-time TEBD finished the full tau schedule."

        # Create and return the shared result object expected by the rest of your project.
        return GroundStateResult(
            problem_summary=problem.summary(),
            ground_energy=float(np.real(final_energy)),
            energy_per_site=float(np.real(final_energy)) / problem.n_sites,
            state_vector=state_vector,
            state_metadata=state_metadata,
            expectation_values=expectation_values,
            correlations=correlations,
            entanglement=entanglement,
            energy_variance=energy_variance,
            residual_norm=residual_norm,
            diagnostics=SolverDiagnostics(
                solver_name="MPS_TEBD",
                method_family="tensor_network",
                converged=converged,
                runtime_seconds=runtime_seconds,
                iterations=len(energy_history),
                message=message,
                extra={
                    "energy_history": energy_history,
                    "tau_schedule": list(self.config.tau_schedule),
                    "steps_per_tau": self.config.steps_per_tau,
                },
            ),
            tn_diagnostics=TensorNetworkDiagnostics(
                ansatz_type="MPS",
                contraction_scheme="sequential_MPS_TEBD_updates",
                optimizer="imaginary_time_TEBD",
                max_bond_dimension=self.config.max_bond_dimension,
                achieved_bond_dimension=achieved_max_bond_dimension,
                truncation_cutoff=self.config.svd_cutoff,
                discarded_weight=total_discarded_weight,
                sweeps=len(energy_history),
                environment_strategy="local_two_site_TEBD_gates",
                canonical_form="not_explicitly_canonicalized_every_step",
                extra={
                    "max_single_step_discarded_weight": max_discarded_weight,
                },
            ),
        )

    def _validate_problem(self, problem: QuantumProblem) -> None:
        # Enforce open boundaries because this TEBD implementation assumes an open 1D chain.
        if problem.boundary != BoundaryCondition.OPEN:
            raise NotImplementedError("This TEBD solver supports open boundary conditions only.")

        # Enforce local dimension 2 because the current code is for spin-1/2 systems.
        if problem.local_dim != 2:
            raise NotImplementedError("This TEBD solver currently supports local_dim = 2 only.")

        # Check every Hamiltonian term for supported shapes and locality.
        for term in problem.local_terms():
            # Handle one-site operators.
            if len(term.sites) == 1:
                # For spin-1/2, one-site operators must be 2x2 matrices.
                if term.operator.shape != (2, 2):
                    raise ValueError(f"One-site term must have shape (2,2), got {term.operator.shape}")
            # Handle two-site operators.
            elif len(term.sites) == 2:
                # Unpack the two site indices.
                i, j = term.sites

                # For spin-1/2, two-site operators must be 4x4 matrices.
                if term.operator.shape != (4, 4):
                    raise ValueError(f"Two-site term must have shape (4,4), got {term.operator.shape}")

                # TEBD here only supports nearest-neighbor two-site interactions.
                if abs(i - j) != 1:
                    raise NotImplementedError("This TEBD solver supports nearest-neighbor two-site terms only.")
            # Reject higher-body terms because this implementation does not handle them.
            else:
                raise NotImplementedError("This TEBD solver supports only one-site and two-site terms.")

    def _group_local_terms(
        self,
        problem: QuantumProblem,
    ) -> Tuple[Dict[int, np.ndarray], Dict[Tuple[int, int], np.ndarray]]:
        # Create a dictionary that will store the summed one-site operator for each site.
        onsite_terms: Dict[int, np.ndarray] = {}

        # Create a dictionary that will store the summed two-site operator for each bond.
        bond_terms: Dict[Tuple[int, int], np.ndarray] = {}

        # Loop over every local term produced by the problem object.
        for term in problem.local_terms():
            # Multiply the local operator by its coefficient so we store the full contribution.
            weighted_op = term.coefficient * term.operator

            # If the term acts on a single site, accumulate it into onsite_terms.
            if len(term.sites) == 1:
                # Extract the site index.
                i = term.sites[0]

                # If this is the first term on site i, initialize a zero 2x2 matrix.
                if i not in onsite_terms:
                    onsite_terms[i] = np.zeros((2, 2), dtype=complex)

                # Add the weighted operator to the summed onsite operator.
                onsite_terms[i] += weighted_op

            # If the term acts on two sites, accumulate it into bond_terms.
            elif len(term.sites) == 2:
                # Extract the site indices.
                i, j = term.sites

                # Order the pair so the key is always (smaller, larger).
                if j < i:
                    i, j = j, i

                # If this is the first term on bond (i,j), initialize a zero 4x4 matrix.
                if (i, j) not in bond_terms:
                    bond_terms[(i, j)] = np.zeros((4, 4), dtype=complex)

                # Add the weighted operator to the summed bond operator.
                bond_terms[(i, j)] += weighted_op

        # Return the grouped onsite and bond Hamiltonian pieces.
        return onsite_terms, bond_terms

    def _initialize_mps(self, n_sites: int, local_dim: int, init_state: str) -> List[np.ndarray]:
        # Create an empty list that will hold one MPS tensor per site.
        mps: List[np.ndarray] = []

        # If the user wants the all-up product state, use the local basis vector |0>.
        if init_state == "product_up":
            # Define the local spin-up state.
            vec = np.array([1.0, 0.0], dtype=complex)

            # Create a rank-3 tensor of shape (1, d, 1) for each site.
            for _ in range(n_sites):
                # Allocate the tensor.
                A = np.zeros((1, local_dim, 1), dtype=complex)

                # Insert the local product-state coefficients.
                A[0, :, 0] = vec

                # Append the tensor to the MPS list.
                mps.append(A)

            # Return the completed MPS.
            return mps

        # If the user wants the all-down product state, use the local basis vector |1>.
        if init_state == "product_down":
            # Define the local spin-down state.
            vec = np.array([0.0, 1.0], dtype=complex)

            # Create one rank-3 tensor per site.
            for _ in range(n_sites):
                # Allocate the tensor.
                A = np.zeros((1, local_dim, 1), dtype=complex)

                # Insert the local product-state coefficients.
                A[0, :, 0] = vec

                # Append the tensor to the MPS list.
                mps.append(A)

            # Return the completed MPS.
            return mps

        # If the user wants a random product state, sample a random local vector at each site.
        if init_state == "random":
            # Loop over all sites.
            for _ in range(n_sites):
                # Draw a random complex vector of length local_dim.
                vec = self.rng.normal(size=local_dim) + 1j * self.rng.normal(size=local_dim)

                # Normalize the local vector.
                vec = vec / np.linalg.norm(vec)

                # Wrap the local vector into a rank-3 tensor of shape (1, d, 1).
                A = np.zeros((1, local_dim, 1), dtype=complex)

                # Store the local coefficients.
                A[0, :, 0] = vec

                # Append the site tensor.
                mps.append(A)

            # Return the completed MPS.
            return mps

        # Reject unknown initialization strings.
        raise ValueError(f"Unknown init_state: {init_state}")

    def _build_gate_tables(
        self,
        problem: QuantumProblem,
        onsite_terms: Dict[int, np.ndarray],
        bond_terms: Dict[Tuple[int, int], np.ndarray],
    ):
        # Create a table mapping tau -> list of one-site half-step gates for each site.
        onsite_gate_table: Dict[float, List[np.ndarray]] = {}

        # Create a table mapping tau -> dictionary of even-bond half-step gates.
        even_gate_table: Dict[float, Dict[int, np.ndarray]] = {}

        # Create a table mapping tau -> dictionary of odd-bond full-step gates.
        odd_gate_table: Dict[float, Dict[int, np.ndarray]] = {}

        # Loop over every tau value in the schedule.
        for tau in self.config.tau_schedule:
            # Build and store half-step one-site gates exp(-tau/2 * h_i).
            onsite_gate_table[tau] = [
                expm(-0.5 * tau * onsite_terms.get(i, np.zeros((2, 2), dtype=complex)))
                for i in range(problem.n_sites)
            ]

            # Initialize the even-bond dictionary for this tau.
            even_gate_table[tau] = {}

            # Initialize the odd-bond dictionary for this tau.
            odd_gate_table[tau] = {}

            # Loop over all nearest-neighbor bonds.
            for i in range(problem.n_sites - 1):
                # Fetch the local two-site Hamiltonian for bond (i, i+1), or zero if absent.
                h_bond = bond_terms.get((i, i + 1), np.zeros((4, 4), dtype=complex))

                # Build the half-step imaginary-time gate exp(-tau/2 * h_bond).
                gate_half = expm(-0.5 * tau * h_bond).reshape(2, 2, 2, 2)

                # Build the full-step imaginary-time gate exp(-tau * h_bond).
                gate_full = expm(-tau * h_bond).reshape(2, 2, 2, 2)

                # Store half-step gates on even bonds and full-step gates on odd bonds.
                if i % 2 == 0:
                    even_gate_table[tau][i] = gate_half
                else:
                    odd_gate_table[tau][i] = gate_full

        # Return all precomputed gate tables.
        return onsite_gate_table, even_gate_table, odd_gate_table

    def _apply_all_one_site_gates(self, mps: List[np.ndarray], one_site_gates: List[np.ndarray]) -> None:
        # Loop over all sites and their corresponding one-site gate.
        for i, gate in enumerate(one_site_gates):
            # Apply the gate to the physical index of the MPS tensor.
            # A[alpha, s, beta] -> sum_t gate[s,t] A[alpha,t,beta]
            mps[i] = np.einsum("st,atb->asb", gate, mps[i], optimize=True)

    def _apply_bond_gate_layer(
        self,
        mps: List[np.ndarray],
        gates: Dict[int, np.ndarray],
        bond_starts,
        max_bond_dim: int,
        svd_cutoff: float,
    ) -> Tuple[float, int]:
        # Initialize the total discarded weight for this bond layer.
        total_discarded = 0.0

        # Track the largest bond dimension created in this bond layer.
        max_bond_seen = 1

        # Loop over all bond starting indices in the layer.
        for i in bond_starts:
            # Skip bonds that have no Hamiltonian term and thus no gate.
            if i not in gates:
                continue

            # Apply the two-site gate to sites i and i+1, then truncate.
            discarded_weight, kept_dim = self._apply_two_site_gate(
                A_left=mps[i],
                A_right=mps[i + 1],
                gate=gates[i],
                max_bond_dim=max_bond_dim,
                svd_cutoff=svd_cutoff,
            )

            # Replace the old MPS tensors with the updated tensors returned by the gate application.
            mps[i], mps[i + 1] = discarded_weight["A_left"], discarded_weight["A_right"]  # type: ignore

            # Add this update's discarded weight to the layer total.
            total_discarded += discarded_weight["discarded_weight"]  # type: ignore

            # Update the maximum bond dimension seen in this layer.
            max_bond_seen = max(max_bond_seen, kept_dim)

        # Return the total discarded weight and largest bond dimension from this layer.
        return total_discarded, max_bond_seen

    def _apply_two_site_gate(
        self,
        A_left: np.ndarray,
        A_right: np.ndarray,
        gate: np.ndarray,
        max_bond_dim: int,
        svd_cutoff: float,
    ):
        # Extract the left virtual dimension, physical dimension, and middle bond dimension from the left tensor.
        Dl, d1, Dm = A_left.shape

        # Extract the middle bond dimension, physical dimension, and right virtual dimension from the right tensor.
        Dm2, d2, Dr = A_right.shape

        # Check that the shared bond dimensions of the two neighboring tensors match.
        if Dm != Dm2:
            raise ValueError("Neighboring MPS tensors have incompatible bond dimensions.")

        # Merge the two neighboring site tensors into a single four-index tensor theta.
        theta = np.einsum("aib,bjc->aijc", A_left, A_right, optimize=True)

        # Apply the two-site gate to the physical indices of theta.
        # gate[s1', s2', s1, s2] acts on theta[a, s1, s2, c].
        theta = np.einsum("xyij,aijc->axyc", gate, theta, optimize=True)

        # Reshape theta into a matrix for SVD:
        # left block = (Dl * d1), right block = (d2 * Dr)
        theta_matrix = theta.reshape(Dl * d1, d2 * Dr)

        # Perform the singular value decomposition.
        U, S, Vh = np.linalg.svd(theta_matrix, full_matrices=False)

        # Determine how many singular values are above the cutoff.
        keep_by_cutoff = int(np.sum(S > svd_cutoff))

        # Enforce that we keep at least one singular value.
        keep_by_cutoff = max(1, keep_by_cutoff)

        # Enforce the user-specified maximum bond dimension.
        keep_dim = min(max_bond_dim, keep_by_cutoff)

        # Compute the discarded weight from the singular values we throw away.
        discarded_weight = float(np.sum(S[keep_dim:] ** 2)) if keep_dim < len(S) else 0.0

        # Keep only the leading singular vectors and singular values.
        U = U[:, :keep_dim]
        S = S[:keep_dim]
        Vh = Vh[:keep_dim, :]

        # Absorb sqrt(S) symmetrically into the left and right tensors.
        sqrtS = np.sqrt(S)

        # Build the updated left tensor with shape (Dl, d1, keep_dim).
        A_left_new = (U * sqrtS[np.newaxis, :]).reshape(Dl, d1, keep_dim)

        # Build the updated right tensor with shape (keep_dim, d2, Dr).
        A_right_new = (sqrtS[:, np.newaxis] * Vh).reshape(keep_dim, d2, Dr)

        # Return both updated tensors plus discarded-weight bookkeeping.
        return {
            "A_left": A_left_new,
            "A_right": A_right_new,
            "discarded_weight": discarded_weight,
        }, keep_dim

    def _normalize_mps_inplace(self, mps: List[np.ndarray]) -> None:
        # Compute the current norm squared of the MPS.
        norm_sq = self._mps_norm_squared(mps)

        # Take the square root to get the norm.
        norm = np.sqrt(np.real(norm_sq))

        # Reject a zero or invalid norm.
        if norm <= 0:
            raise ValueError("Encountered non-positive MPS norm during normalization.")

        # Divide the first tensor by the norm; this rescales the whole state.
        mps[0] = mps[0] / norm

    def _mps_norm_squared(self, mps: List[np.ndarray]) -> complex:
        # Start the contraction with the 1x1 scalar identity.
        env = np.array([[1.0 + 0.0j]])

        # Contract each site tensor with its conjugate from left to right.
        for A in mps:
            env = np.einsum("ab,asr,bst->rt", env, A, np.conj(A), optimize=True)

        # After finishing all sites, the result is a 1x1 array containing <psi|psi>.
        return env[0, 0]

    def _current_max_bond_dimension(self, mps: List[np.ndarray]) -> int:
        # The internal bond dimension is the right bond of each tensor except the last.
        if len(mps) <= 1:
            return 1

        # Return the maximum internal bond dimension currently present.
        return max(A.shape[2] for A in mps[:-1])

    def _compute_total_energy(
        self,
        problem: QuantumProblem,
        mps: List[np.ndarray],
        onsite_terms: Dict[int, np.ndarray],
        bond_terms: Dict[Tuple[int, int], np.ndarray],
    ) -> float:
        # Build left environments needed for local expectation values.
        left_envs = self._build_left_identity_environments(mps)

        # Build right environments needed for local expectation values.
        right_envs = self._build_right_identity_environments(mps)

        # Start the total energy accumulator at zero.
        energy = 0.0 + 0.0j

        # Add all one-site expectation values.
        for i, op in onsite_terms.items():
            energy += self._expectation_one_site(
                mps=mps,
                site=i,
                op=op,
                left_env=left_envs[i],
                right_env=right_envs[i + 1],
            )

        # Add all nearest-neighbor two-site expectation values.
        for (i, j), op in bond_terms.items():
            energy += self._expectation_two_site(
                mps=mps,
                site=i,
                op=op,
                left_env=left_envs[i],
                right_env=right_envs[j + 1],
            )

        # Return the real part as a Python float.
        return float(np.real_if_close(energy))

    def _build_left_identity_environments(self, mps: List[np.ndarray]) -> List[np.ndarray]:
        # Allocate a list of length n_sites + 1 for left environments.
        left_envs: List[np.ndarray] = [None] * (len(mps) + 1)  # type: ignore

        # The empty contraction to the left of the first site is the scalar 1.
        left_envs[0] = np.array([[1.0 + 0.0j]])

        # Build environments progressively from left to right.
        for i, A in enumerate(mps):
            left_envs[i + 1] = np.einsum(
                "ab,asr,bst->rt",
                left_envs[i],
                A,
                np.conj(A),
                optimize=True,
            )

        # Return the full list of left environments.
        return left_envs

    def _build_right_identity_environments(self, mps: List[np.ndarray]) -> List[np.ndarray]:
        # Allocate a list of length n_sites + 1 for right environments.
        right_envs: List[np.ndarray] = [None] * (len(mps) + 1)  # type: ignore

        # The empty contraction to the right of the last site is the scalar 1.
        right_envs[len(mps)] = np.array([[1.0 + 0.0j]])

        # Build environments progressively from right to left.
        for i in reversed(range(len(mps))):
            A = mps[i]
            right_envs[i] = np.einsum(
                "asr,bst,rt->ab",
                A,
                np.conj(A),
                right_envs[i + 1],
                optimize=True,
            )

        # Return the full list of right environments.
        return right_envs

    def _expectation_one_site(
        self,
        mps: List[np.ndarray],
        site: int,
        op: np.ndarray,
        left_env: np.ndarray,
        right_env: np.ndarray,
    ) -> complex:
        # Fetch the site tensor.
        A = mps[site]

        # Contract left environment, operator, site tensor, conjugate tensor, and right environment.
        return np.einsum(
            "ab,asr,st,btu,ru->",
            left_env,
            A,
            op,
            np.conj(A),
            right_env,
            optimize=True,
        )

    def _expectation_two_site(
        self,
        mps: List[np.ndarray],
        site: int,
        op: np.ndarray,
        left_env: np.ndarray,
        right_env: np.ndarray,
    ) -> complex:
        # Fetch the two neighboring tensors.
        A = mps[site]
        B = mps[site + 1]

        # Merge them into a four-index tensor theta[a, s1, s2, c].
        theta = np.einsum("aib,bjc->aijc", A, B, optimize=True)

        # Reshape the operator into rank-4 form op[s1', s2', s1, s2].
        op4 = op.reshape(2, 2, 2, 2)

        # Contract everything to compute <psi|op_{site,site+1}|psi>.
        return np.einsum(
            "ab,axyc,xyij,bijv,cv->",
            left_env,
            np.conj(theta),
            op4,
            theta,
            right_env,
            optimize=True,
        )

    def _compute_one_site_expectations(self, problem: QuantumProblem, mps: List[np.ndarray]) -> Dict[str, object]:
        # Get the standard one-site observables from the problem object.
        observables = problem.one_site_observables()

        # Build left environments once so we can reuse them.
        left_envs = self._build_left_identity_environments(mps)

        # Build right environments once so we can reuse them.
        right_envs = self._build_right_identity_environments(mps)

        # Prepare the output dictionary.
        out: Dict[str, object] = {}

        # Loop over each observable label and matrix.
        for label, op in observables.items():
            # Skip the identity because plotting it is not useful.
            if label == "I":
                continue

            # Compute <op_i> for each site.
            vals = [
                float(np.real_if_close(
                    self._expectation_one_site(
                        mps=mps,
                        site=i,
                        op=op,
                        left_env=left_envs[i],
                        right_env=right_envs[i + 1],
                    )
                ))
                for i in range(problem.n_sites)
            ]

            # Store the full site-resolved list.
            out[label] = vals

        # If X expectations were computed, store their mean as magnetization_x.
        if "X" in out:
            out["magnetization_x"] = float(np.mean(out["X"]))  # type: ignore

        # If Y expectations were computed, store their mean as magnetization_y.
        if "Y" in out:
            out["magnetization_y"] = float(np.mean(out["Y"]))  # type: ignore

        # If Z expectations were computed, store their mean as magnetization_z.
        if "Z" in out:
            out["magnetization_z"] = float(np.mean(out["Z"]))  # type: ignore

        # Return the dictionary of one-site observables.
        return out

    def _compute_two_site_correlations(self, problem: QuantumProblem, mps: List[np.ndarray]) -> Dict[str, object]:
        # Get the standard nearest-neighbor two-site observables from the problem object.
        observables = problem.two_site_observables()

        # Build left environments once.
        left_envs = self._build_left_identity_environments(mps)

        # Build right environments once.
        right_envs = self._build_right_identity_environments(mps)

        # Prepare the output dictionary.
        out: Dict[str, object] = {}

        # Loop over labels like XX, YY, ZZ.
        for label, op in observables.items():
            # Store correlation values for each nearest-neighbor edge.
            edge_vals: Dict[str, float] = {}

            # Store the list of values so we can compute an average too.
            vals: List[float] = []

            # Loop over the interaction edges defined by the problem.
            for i, j in problem.interaction_edges():
                # Compute the nearest-neighbor expectation value on this bond.
                val = float(np.real_if_close(
                    self._expectation_two_site(
                        mps=mps,
                        site=i,
                        op=op,
                        left_env=left_envs[i],
                        right_env=right_envs[j + 1],
                    )
                ))

                # Store the bond-specific value using a readable key.
                edge_vals[f"({i},{j})"] = val

                # Also append it for averaging.
                vals.append(val)

            # Store all bond-resolved values for this observable.
            out[label] = edge_vals

            # Store the average nearest-neighbor value for this observable.
            out[f"average_{label}"] = float(np.mean(vals)) if vals else 0.0

        # Return the correlation dictionary.
        return out

    def _compute_entanglement_summary(self, problem: QuantumProblem, mps: List[np.ndarray]) -> Dict[str, object]:
        # If the dense state is small enough, reconstruct it and compute exact half-chain entropy.
        if problem.hilbert_dim <= self.config.dense_state_max_dim:
            # Rebuild the full dense state vector from the MPS.
            psi = self._mps_to_dense_state(mps)

            # Normalize the dense state explicitly.
            psi = psi / np.linalg.norm(psi)

            # Split the chain into two halves.
            n_left = problem.n_sites // 2
            n_right = problem.n_sites - n_left

            # Compute the Hilbert-space dimensions of the two halves.
            dim_left = 2 ** n_left
            dim_right = 2 ** n_right

            # Reshape the state into a matrix for a Schmidt decomposition.
            psi_matrix = psi.reshape(dim_left, dim_right)

            # Compute singular values across the bipartition.
            svals = np.linalg.svd(psi_matrix, compute_uv=False)

            # Convert singular values into Schmidt probabilities.
            probs = svals ** 2

            # Drop tiny numerical noise.
            probs = probs[probs > 1e-15]

            # Compute the von Neumann entropy in bits.
            entropy_bits = float(-np.sum(probs * np.log2(probs)))

            # Return a compact summary.
            return {
                "half_chain_entropy_vn": entropy_bits,
                "schmidt_rank_numerical": int(len(probs)),
                "cut": {
                    "left_sites": list(range(n_left)),
                    "right_sites": list(range(n_left, problem.n_sites)),
                },
            }

        # If the state is too large to reconstruct, return placeholders.
        return {
            "half_chain_entropy_vn": None,
            "schmidt_rank_numerical": None,
            "cut": None,
        }

    def _mps_to_dense_state(self, mps: List[np.ndarray]) -> np.ndarray:
        # Start from the first tensor.
        tensor = mps[0]

        # Contract tensors from left to right along matching virtual bonds.
        for i in range(1, len(mps)):
            tensor = np.tensordot(tensor, mps[i], axes=([-1], [0]))

        # Remove the left and right dummy dimension-1 boundary indices.
        tensor = np.squeeze(tensor, axis=0)
        tensor = np.squeeze(tensor, axis=-1)

        # Flatten into a 1D state vector of length 2^N.
        return tensor.reshape(-1)


def solve(problem: QuantumProblem, config: Optional[TNSolverConfig] = None) -> GroundStateResult:
    # Create the solver object with the requested configuration.
    solver = TensorNetworkGroundStateSolver(config=config)

    # Run the solver and return the shared result object.
    return solver.solve(problem)