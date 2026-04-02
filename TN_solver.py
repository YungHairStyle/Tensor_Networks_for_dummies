from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.problem import QuantumProblem, LocalTerm, BoundaryCondition
from core.result import (
    GroundStateResult,
    SolverDiagnostics,
    TensorNetworkDiagnostics,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TNSolverConfig:
    """
    Configuration for the tensor-network ground state solver.

    max_bond_dimension:
        Maximum bond dimension kept after each two-site update.

    n_sweeps:
        Maximum number of full left-right-right-left sweeps.
    energy_tolerance:
        Convergence tolerance on the total energy between sweeps.

    random_seed:
        Seed for reproducible random initialization.

    init_state:
        Initialization strategy:
            - "random"
            - "product_up"
            - "product_down"

    local_solver:
        Currently only "dense_eigh" is implemented.

    contraction_scheme:
        A label stored in diagnostics so you can later compare different
        contraction implementations. The actual implementation here is explicit
        dense local contraction.

    compute_observables:
        If True, compute local observables and correlations from the final MPS.

    store_state_vector_if_small:
        If True, reconstruct and store the full state vector when the Hilbert
        space is not too large.

    dense_state_max_dim:
        Maximum Hilbert-space dimension allowed for reconstructing the full
        state vector and exact variance/residual diagnostics.

    svd_cutoff:
        Singular values below this threshold are discarded.

    verbosity:
        0 = silent
        1 = sweep energies
        2 = sweep energies + per-bond updates
    """
    max_bond_dimension: int = 32
    n_sweeps: int = 10
    energy_tolerance: float = 1e-8
    random_seed: Optional[int] = None
    init_state: str = "random"
    local_solver: str = "dense_eigh"
    contraction_scheme: str = "explicit_dense_local_contraction"
    compute_observables: bool = True
    store_state_vector_if_small: bool = False
    dense_state_max_dim: int = 2 ** 14
    svd_cutoff: float = 1e-12
    verbosity: int = 0


# =============================================================================
# Main Solver
# =============================================================================

class TensorNetworkGroundStateSolver:
    """
    Explicit finite-MPS / two-site DMRG-style solver for 1D open spin chains.

    Important assumptions:
    - open boundary conditions only
    - local dimension = 2 (spin-1/2)
    - Hamiltonian terms are only:
        * one-site terms
        * nearest-neighbor two-site terms
    """

    def __init__(self, config: Optional[TNSolverConfig] = None) -> None:
        self.config = config or TNSolverConfig()
        self.rng = np.random.default_rng(self.config.random_seed)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def solve(self, problem: QuantumProblem) -> GroundStateResult:
        start_time = time.perf_counter()

        self._validate_problem(problem)

        onsite_terms, bond_terms = self._group_local_terms(problem)

        mps = self._initialize_mps(
            n_sites=problem.n_sites,
            local_dim=problem.local_dim,
            max_bond_dim=self.config.max_bond_dimension,
            init_state=self.config.init_state,
        )

        # Start in right-canonical form so a left-to-right sweep is well-defined
        self._right_canonicalize_inplace(mps)
        self._normalize_mps_inplace(mps)

        sweep_energies: List[float] = []
        total_discarded_weight = 0.0
        max_discarded_weight = 0.0
        achieved_max_bond_dimension = max(t.shape[2] for t in mps[:-1]) if len(mps) > 1 else 1

        converged = False
        message = "Maximum number of sweeps reached."

        for sweep in range(self.config.n_sweeps):
            # -------------------------------------------------------------
            # Left-to-right sweep
            # -------------------------------------------------------------
            for i in range(problem.n_sites - 1):
                heff = self._build_two_site_effective_hamiltonian(
                    mps=mps,
                    problem=problem,
                    onsite_terms=onsite_terms,
                    bond_terms=bond_terms,
                    bond_index=i,
                )

                theta = self._solve_local_ground_state(heff)

                discarded_weight, kept_bond_dim = self._split_theta_left_to_right(
                    mps=mps,
                    bond_index=i,
                    theta=theta,
                    max_bond_dim=self.config.max_bond_dimension,
                    svd_cutoff=self.config.svd_cutoff,
                )

                total_discarded_weight += discarded_weight
                max_discarded_weight = max(max_discarded_weight, discarded_weight)
                achieved_max_bond_dimension = max(achieved_max_bond_dimension, kept_bond_dim)

                if self.config.verbosity >= 2:
                    print(
                        f"[sweep {sweep + 1} | L->R | bond {i}] "
                        f"kept chi={kept_bond_dim}, discarded={discarded_weight:.3e}"
                    )

            # -------------------------------------------------------------
            # Right-to-left sweep
            # -------------------------------------------------------------
            for i in reversed(range(problem.n_sites - 1)):
                heff = self._build_two_site_effective_hamiltonian(
                    mps=mps,
                    problem=problem,
                    onsite_terms=onsite_terms,
                    bond_terms=bond_terms,
                    bond_index=i,
                )

                theta = self._solve_local_ground_state(heff)

                discarded_weight, kept_bond_dim = self._split_theta_right_to_left(
                    mps=mps,
                    bond_index=i,
                    theta=theta,
                    max_bond_dim=self.config.max_bond_dimension,
                    svd_cutoff=self.config.svd_cutoff,
                )

                total_discarded_weight += discarded_weight
                max_discarded_weight = max(max_discarded_weight, discarded_weight)
                achieved_max_bond_dimension = max(achieved_max_bond_dimension, kept_bond_dim)

                if self.config.verbosity >= 2:
                    print(
                        f"[sweep {sweep + 1} | R->L | bond {i}] "
                        f"kept chi={kept_bond_dim}, discarded={discarded_weight:.3e}"
                    )

            self._normalize_mps_inplace(mps)

            energy = self._compute_total_energy(
                mps=mps,
                problem=problem,
                onsite_terms=onsite_terms,
                bond_terms=bond_terms,
            )
            sweep_energies.append(energy)

            if self.config.verbosity >= 1:
                print(f"[sweep {sweep + 1}] energy = {energy:.12f}")

            if len(sweep_energies) >= 2:
                delta_e = abs(sweep_energies[-1] - sweep_energies[-2])
                if delta_e < self.config.energy_tolerance:
                    converged = True
                    message = (
                        f"Converged after {sweep + 1} sweeps with "
                        f"|ΔE| = {delta_e:.3e} < {self.config.energy_tolerance:.3e}."
                    )
                    break

        ground_energy = sweep_energies[-1]
        runtime_seconds = time.perf_counter() - start_time

        expectation_values: Dict[str, object] = {}
        correlations: Dict[str, object] = {}
        entanglement: Dict[str, object] = {}

        if self.config.compute_observables:
            expectation_values = self._compute_one_site_expectations(problem, mps)
            correlations = self._compute_two_site_correlations(problem, mps)
            entanglement = self._compute_entanglement_summary(problem, mps)

        state_vector = None
        state_metadata: Dict[str, object] = {
            "representation": "MPS",
            "canonical_notes": (
                "Final MPS is in a gauge produced by alternating two-site sweeps; "
                "not guaranteed to be centered at a specific bond."
            ),
        }

        energy_variance = None
        residual_norm = None

        if problem.hilbert_dim <= self.config.dense_state_max_dim:
            dense_state = self._mps_to_dense_state(mps)
            dense_state = dense_state / np.linalg.norm(dense_state)

            if self.config.store_state_vector_if_small:
                state_vector = dense_state

            # exact variance and residual against the full Hamiltonian
            h = problem.full_hamiltonian()
            hpsi = h @ dense_state
            exp_h = np.vdot(dense_state, hpsi)
            h2psi = h @ hpsi
            exp_h2 = np.vdot(dense_state, h2psi)

            var = np.real(exp_h2 - exp_h * exp_h)
            energy_variance = max(0.0, float(var))

            residual = hpsi - np.real(exp_h) * dense_state
            residual_norm = float(np.linalg.norm(residual))

            state_metadata.update(
                {
                    "dense_state_reconstructed": True,
                    "dense_state_dimension": int(problem.hilbert_dim),
                    "state_normalized": bool(np.isclose(np.linalg.norm(dense_state), 1.0)),
                }
            )
        else:
            state_metadata.update(
                {
                    "dense_state_reconstructed": False,
                    "dense_state_dimension": None,
                }
            )

        result = GroundStateResult(
            problem_summary=problem.summary(),
            ground_energy=float(np.real(ground_energy)),
            energy_per_site=float(np.real(ground_energy)) / problem.n_sites,
            state_vector=state_vector,
            state_metadata=state_metadata,
            expectation_values=expectation_values,
            correlations=correlations,
            entanglement=entanglement,
            energy_variance=energy_variance,
            residual_norm=residual_norm,
            diagnostics=SolverDiagnostics(
                solver_name="ExplicitTwoSiteDMRG",
                method_family="tensor_network",
                converged=converged,
                runtime_seconds=runtime_seconds,
                iterations=len(sweep_energies),
                message=message,
                extra={
                    "sweep_energies": sweep_energies,
                    "local_solver": self.config.local_solver,
                    "total_discarded_weight": total_discarded_weight,
                    "max_discarded_weight": max_discarded_weight,
                },
            ),
            tn_diagnostics=TensorNetworkDiagnostics(
                ansatz_type="MPS",
                contraction_scheme=self.config.contraction_scheme,
                optimizer="two_site_dmrg_explicit",
                max_bond_dimension=self.config.max_bond_dimension,
                achieved_bond_dimension=achieved_max_bond_dimension,
                truncation_cutoff=self.config.svd_cutoff,
                discarded_weight=total_discarded_weight,
                sweeps=len(sweep_energies),
                environment_strategy="explicit_boundary_operator_contraction",
                canonical_form="mixed-by-sweep",
                extra={
                    "supported_problem_family": (
                        "1D open boundary spin-1/2 models with one-site "
                        "and nearest-neighbor two-site terms"
                    )
                },
            ),
        )

        return result

    # -------------------------------------------------------------------------
    # Validation and preprocessing
    # -------------------------------------------------------------------------

    def _validate_problem(self, problem: QuantumProblem) -> None:
        if problem.boundary != BoundaryCondition.OPEN:
            raise NotImplementedError(
                "This TN solver currently supports open boundary conditions only."
            )

        if problem.local_dim != 2:
            raise NotImplementedError(
                "This TN solver currently supports local_dim = 2 only."
            )

        for term in problem.local_terms():
            if len(term.sites) == 1:
                if term.operator.shape != (2, 2):
                    raise ValueError(
                        f"One-site term must have shape (2,2), got {term.operator.shape}"
                    )
            elif len(term.sites) == 2:
                i, j = term.sites
                if term.operator.shape != (4, 4):
                    raise ValueError(
                        f"Two-site term must have shape (4,4), got {term.operator.shape}"
                    )
                if abs(i - j) != 1:
                    raise NotImplementedError(
                        "This TN solver supports nearest-neighbor two-site terms only."
                    )
            else:
                raise NotImplementedError(
                    "This TN solver supports only one-site and two-site terms."
                )

    def _group_local_terms(
        self,
        problem: QuantumProblem,
    ) -> Tuple[Dict[int, np.ndarray], Dict[Tuple[int, int], np.ndarray]]:
        """
        Group all local terms into summed one-site and two-site operators.

        Returns:
            onsite_terms[site] = summed 2x2 operator
            bond_terms[(i, i+1)] = summed 4x4 operator
        """
        onsite_terms: Dict[int, np.ndarray] = {}
        bond_terms: Dict[Tuple[int, int], np.ndarray] = {}

        for term in problem.local_terms():
            coeff_op = term.coefficient * term.operator

            if len(term.sites) == 1:
                i = term.sites[0]
                if i not in onsite_terms:
                    onsite_terms[i] = np.zeros((2, 2), dtype=complex)
                onsite_terms[i] += coeff_op

            elif len(term.sites) == 2:
                i, j = term.sites
                if j < i:
                    i, j = j, i

                if (j - i) != 1:
                    raise NotImplementedError(
                        "Only nearest-neighbor two-site terms are supported."
                    )

                key = (i, j)
                if key not in bond_terms:
                    bond_terms[key] = np.zeros((4, 4), dtype=complex)
                bond_terms[key] += coeff_op

        return onsite_terms, bond_terms

    # -------------------------------------------------------------------------
    # MPS initialization and normalization
    # -------------------------------------------------------------------------

    def _initialize_mps(
        self,
        n_sites: int,
        local_dim: int,
        max_bond_dim: int,
        init_state: str,
    ) -> List[np.ndarray]:
        """
        Initialize an open-boundary MPS as a list of tensors A[i] with shape:
            (Dl_i, d, Dr_i)
        """
        bond_dims = [1]
        for i in range(1, n_sites):
            left_cap = local_dim ** min(i, n_sites - i)
            bond_dims.append(min(max_bond_dim, left_cap))
        bond_dims.append(1)

        mps: List[np.ndarray] = []

        if init_state == "product_up":
            up = np.array([1.0, 0.0], dtype=complex)
            for i in range(n_sites):
                A = np.zeros((bond_dims[i], local_dim, bond_dims[i + 1]), dtype=complex)
                A[0, :, 0] = up
                mps.append(A)
            return mps

        if init_state == "product_down":
            down = np.array([0.0, 1.0], dtype=complex)
            for i in range(n_sites):
                A = np.zeros((bond_dims[i], local_dim, bond_dims[i + 1]), dtype=complex)
                A[0, :, 0] = down
                mps.append(A)
            return mps

        if init_state == "random":
            for i in range(n_sites):
                Dl = bond_dims[i]
                Dr = bond_dims[i + 1]
                A = (
                    self.rng.normal(size=(Dl, local_dim, Dr))
                    + 1j * self.rng.normal(size=(Dl, local_dim, Dr))
                )
                A = A.astype(complex)
                mps.append(A)
            return mps

        raise ValueError(f"Unknown init_state: {init_state}")

    def _normalize_mps_inplace(self, mps: List[np.ndarray]) -> None:
        norm = np.sqrt(np.real(self._mps_norm_squared(mps)))
        if norm <= 0:
            raise ValueError("MPS norm is zero or invalid.")
        mps[0] = mps[0] / norm

    def _mps_norm_squared(self, mps: List[np.ndarray]) -> complex:
        right_envs = self._build_right_identity_environments(mps)
        left = np.array([[1.0 + 0.0j]])
        for i in range(len(mps)):
            A = mps[i]
            left = np.einsum("ab,asr,bst->rt", left, A, np.conj(A), optimize=True)
        return left[0, 0]

    def _right_canonicalize_inplace(self, mps: List[np.ndarray]) -> None:
        """
        Bring the MPS approximately into right-canonical form by sweeping from
        right to left using QR on the transposed site matrix.

        For a site tensor A with shape (Dl, d, Dr), reshape it as a matrix
        M = A.reshape(Dl, d*Dr). We want rows of M to become orthonormal.
        """
        n = len(mps)
        for i in reversed(range(1, n)):
            A = mps[i]
            Dl, d, Dr = A.shape

            M = A.reshape(Dl, d * Dr)
            Q, R = np.linalg.qr(M.T)   # M.T = Q R
            A_right = Q.T.reshape(Q.shape[1], d, Dr)
            mps[i] = A_right

            prev = mps[i - 1]
            # absorb R.T into previous tensor's right bond
            mps[i - 1] = np.tensordot(prev, R.T, axes=(2, 0))

    # -------------------------------------------------------------------------
    # Two-site optimization
    # -------------------------------------------------------------------------

    def _build_two_site_effective_hamiltonian(
        self,
        mps: List[np.ndarray],
        problem: QuantumProblem,
        onsite_terms: Dict[int, np.ndarray],
        bond_terms: Dict[Tuple[int, int], np.ndarray],
        bond_index: int,
    ) -> np.ndarray:
        """
        Build the explicit dense effective Hamiltonian for sites (i, i+1).

        The key simplification is that, because the Hamiltonian contains only
        one-site and nearest-neighbor two-site terms, the only terms that affect
        the two-site optimization are:
            - onsite terms on i and i+1
            - bond term (i, i+1)
            - bond term (i-1, i), contracted with site i-1
            - bond term (i+1, i+2), contracted with site i+2

        Terms completely outside the optimization window contribute only a
        constant and do not affect the minimizing eigenvector.
        """
        i = bond_index
        j = i + 1
        d = problem.local_dim

        A_i = mps[i]
        A_j = mps[j]
        Dl = A_i.shape[0]
        Dr = A_j.shape[2]

        I_Dl = np.eye(Dl, dtype=complex)
        I_Dr = np.eye(Dr, dtype=complex)
        I_d = np.eye(d, dtype=complex)

        dim = Dl * d * d * Dr
        heff = np.zeros((dim, dim), dtype=complex)

        # -------------------------------------------------------------
        # onsite term on i
        # -------------------------------------------------------------
        if i in onsite_terms:
            hi = onsite_terms[i]
            heff += np.kron(np.kron(np.kron(I_Dl, hi), I_d), I_Dr)

        # -------------------------------------------------------------
        # onsite term on j
        # -------------------------------------------------------------
        if j in onsite_terms:
            hj = onsite_terms[j]
            heff += np.kron(np.kron(np.kron(I_Dl, I_d), hj), I_Dr)

        # -------------------------------------------------------------
        # bond term on (i, j)
        # -------------------------------------------------------------
        if (i, j) in bond_terms:
            hij = bond_terms[(i, j)]
            heff += np.kron(np.kron(I_Dl, hij), I_Dr)

        # -------------------------------------------------------------
        # left boundary bond term (i-1, i)
        # -------------------------------------------------------------
        if i > 0 and (i - 1, i) in bond_terms:
            A_left = mps[i - 1]
            h_left = bond_terms[(i - 1, i)]
            left_op = self._contract_left_boundary_operator(A_left, h_left)
            heff += np.kron(left_op, np.eye(d * Dr, dtype=complex))

        # -------------------------------------------------------------
        # right boundary bond term (j, j+1)
        # -------------------------------------------------------------
        if j < problem.n_sites - 1 and (j, j + 1) in bond_terms:
            A_right = mps[j + 1]
            h_right = bond_terms[(j, j + 1)]
            right_op = self._contract_right_boundary_operator(A_right, h_right)
            heff += np.kron(np.eye(Dl * d, dtype=complex), right_op)

        # make explicitly Hermitian to remove small numerical asymmetries
        heff = 0.5 * (heff + heff.conj().T)
        return heff

    def _contract_left_boundary_operator(
        self,
        A_left: np.ndarray,
        h_left: np.ndarray,
    ) -> np.ndarray:
        """
        Contract a nearest-neighbor term h_left acting on (site_left, site_center)
        with the already-fixed tensor A_left.

        A_left shape:
            (Dl_prev, d, Dl)

        Returns an operator acting on:
            (bond Dl) ⊗ (physical site_center)

        Output shape:
            (Dl*d, Dl*d)

        Explicitly, if h_left has matrix elements
            <q,t| h_left |p,s>
        then the boundary operator has matrix elements
            <b,t| O_left |a,s>
            = sum_{l,p,q} conj(A_left[l,q,b]) h_left[(q,t),(p,s)] A_left[l,p,a]
        """
        Dl_prev, d, Dl = A_left.shape
        h4 = h_left.reshape(d, d, d, d)  # row=(q,t), col=(p,s)

        # O[b, t, a, s]
        O4 = np.einsum(
            "lqb,qtps,lpa->btas",
            np.conj(A_left),
            h4,
            A_left,
            optimize=True,
        )

        return O4.reshape(Dl * d, Dl * d)

    def _contract_right_boundary_operator(
        self,
        A_right: np.ndarray,
        h_right: np.ndarray,
    ) -> np.ndarray:
        """
        Contract a nearest-neighbor term h_right acting on (site_center, site_right)
        with the already-fixed tensor A_right.

        A_right shape:
            (Dr, d, Dr_next)

        Returns an operator acting on:
            (physical site_center) ⊗ (bond Dr)

        Output shape:
            (d*Dr, d*Dr)

        Explicitly, if h_right has matrix elements
            <t,q| h_right |s,p>
        then the boundary operator has matrix elements
            <t,b| O_right |s,a>
            = sum_{r,p,q} conj(A_right[b,q,r]) h_right[(t,q),(s,p)] A_right[a,p,r]
        """
        Dr, d, Dr_next = A_right.shape
        h4 = h_right.reshape(d, d, d, d)  # row=(t,q), col=(s,p)

        # O[t, b, s, a]
        O4 = np.einsum(
            "bqr,tqsp,apr->tbsa",
            np.conj(A_right),
            h4,
            A_right,
            optimize=True,
        )

        return O4.reshape(d * Dr, d * Dr)

    def _solve_local_ground_state(self, heff: np.ndarray) -> np.ndarray:
        """
        Solve the local ground state of the dense effective Hamiltonian.

        Returns:
            theta with shape (Dl, d, d, Dr)
        """
        if self.config.local_solver != "dense_eigh":
            raise NotImplementedError(
                f"Unsupported local_solver: {self.config.local_solver}"
            )

        evals, evecs = np.linalg.eigh(heff)
        ground_vec = evecs[:, 0]
        ground_vec = ground_vec / np.linalg.norm(ground_vec)
        return ground_vec

    def _merge_two_sites(self, A_i: np.ndarray, A_j: np.ndarray) -> np.ndarray:
        """
        Merge two neighboring MPS tensors.

        A_i shape: (Dl, d, Dm)
        A_j shape: (Dm, d, Dr)

        Returns:
            theta shape: (Dl, d, d, Dr)
        """
        return np.einsum("asb,btc->astc", A_i, A_j, optimize=True)

    def _split_theta_left_to_right(
        self,
        mps: List[np.ndarray],
        bond_index: int,
        theta: np.ndarray,
        max_bond_dim: int,
        svd_cutoff: float,
    ) -> Tuple[float, int]:
        """
        Split theta by SVD so that the left tensor becomes left-canonical.

        theta vector is expected in flattened ordering:
            (Dl, d_i, d_j, Dr)

        Left-to-right update:
            theta -> U S V†
            A_i   = U
            A_j   = S V†
        """
        i = bond_index
        A_i = mps[i]
        A_j = mps[i + 1]

        Dl = A_i.shape[0]
        d = A_i.shape[1]
        Dr = A_j.shape[2]

        theta4 = theta.reshape(Dl, d, d, Dr)
        M = theta4.reshape(Dl * d, d * Dr)

        U, S, Vh = np.linalg.svd(M, full_matrices=False)

        keep = min(max_bond_dim, np.sum(S > svd_cutoff))
        keep = max(1, int(keep))

        discarded_weight = float(np.sum(S[keep:] ** 2)) if keep < len(S) else 0.0

        U = U[:, :keep]
        S = S[:keep]
        Vh = Vh[:keep, :]

        A_left = U.reshape(Dl, d, keep)
        A_right = (np.diag(S) @ Vh).reshape(keep, d, Dr)

        mps[i] = A_left
        mps[i + 1] = A_right

        return discarded_weight, keep

    def _split_theta_right_to_left(
        self,
        mps: List[np.ndarray],
        bond_index: int,
        theta: np.ndarray,
        max_bond_dim: int,
        svd_cutoff: float,
    ) -> Tuple[float, int]:
        """
        Split theta by SVD so that the right tensor becomes right-canonical.

        Right-to-left update:
            theta -> U S V†
            A_i   = U S
            A_j   = V†
        """
        i = bond_index
        A_i = mps[i]
        A_j = mps[i + 1]

        Dl = A_i.shape[0]
        d = A_i.shape[1]
        Dr = A_j.shape[2]

        theta4 = theta.reshape(Dl, d, d, Dr)
        M = theta4.reshape(Dl * d, d * Dr)

        U, S, Vh = np.linalg.svd(M, full_matrices=False)

        keep = min(max_bond_dim, np.sum(S > svd_cutoff))
        keep = max(1, int(keep))

        discarded_weight = float(np.sum(S[keep:] ** 2)) if keep < len(S) else 0.0

        U = U[:, :keep]
        S = S[:keep]
        Vh = Vh[:keep, :]

        A_left = (U @ np.diag(S)).reshape(Dl, d, keep)
        A_right = Vh.reshape(keep, d, Dr)

        mps[i] = A_left
        mps[i + 1] = A_right

        return discarded_weight, keep

    # -------------------------------------------------------------------------
    # Identity environments
    # -------------------------------------------------------------------------

    def _build_left_identity_environments(self, mps: List[np.ndarray]) -> List[np.ndarray]:
        """
        left_envs[i] is the identity contraction of sites [0, ..., i-1]
        and acts on the left bond of site i.

        Shapes:
            left_envs[0] has shape (1,1)
            left_envs[i] has shape (Dl_i, Dl_i)
        """
        n = len(mps)
        left_envs: List[np.ndarray] = [None] * (n + 1)  # type: ignore
        left_envs[0] = np.array([[1.0 + 0.0j]])

        for i in range(n):
            A = mps[i]
            left_envs[i + 1] = np.einsum(
                "ab,asr,bst->rt",
                left_envs[i],
                A,
                np.conj(A),
                optimize=True,
            )

        return left_envs

    def _build_right_identity_environments(self, mps: List[np.ndarray]) -> List[np.ndarray]:
        """
        right_envs[i] is the identity contraction of sites [i, ..., n-1]
        and acts on the right bond of site i-1 / the left bond of site i.

        Shapes:
            right_envs[n] has shape (1,1)
            right_envs[i] has shape (Dl_i, Dl_i)
        """
        n = len(mps)
        right_envs: List[np.ndarray] = [None] * (n + 1)  # type: ignore
        right_envs[n] = np.array([[1.0 + 0.0j]])

        for i in reversed(range(n)):
            A = mps[i]
            right_envs[i] = np.einsum(
                "asr,bst,rt->ab",
                A,
                np.conj(A),
                right_envs[i + 1],
                optimize=True,
            )

        return right_envs

    # -------------------------------------------------------------------------
    # Energy and observables
    # -------------------------------------------------------------------------

    def _compute_total_energy(
        self,
        mps: List[np.ndarray],
        problem: QuantumProblem,
        onsite_terms: Dict[int, np.ndarray],
        bond_terms: Dict[Tuple[int, int], np.ndarray],
    ) -> float:
        energy = 0.0 + 0.0j

        left_envs = self._build_left_identity_environments(mps)
        right_envs = self._build_right_identity_environments(mps)

        for i, op in onsite_terms.items():
            energy += self._expectation_one_site(
                mps=mps,
                site=i,
                op=op,
                left_env=left_envs[i],
                right_env=right_envs[i + 1],
            )

        for (i, j), op in bond_terms.items():
            energy += self._expectation_two_site(
                mps=mps,
                site=i,
                op=op,
                left_env=left_envs[i],
                right_env=right_envs[j + 1],
            )

        return float(np.real_if_close(energy))

    def _expectation_one_site(
        self,
        mps: List[np.ndarray],
        site: int,
        op: np.ndarray,
        left_env: np.ndarray,
        right_env: np.ndarray,
    ) -> complex:
        """
        Compute <psi| op_site |psi> explicitly.
        """
        A = mps[site]
        val = np.einsum(
            "ab,asr,st,btu,ru->",
            left_env,
            A,
            op,
            np.conj(A),
            right_env,
            optimize=True,
        )
        return val

    def _expectation_two_site(
        self,
        mps: List[np.ndarray],
        site: int,
        op: np.ndarray,
        left_env: np.ndarray,
        right_env: np.ndarray,
    ) -> complex:
        """
        Compute <psi| op_{site,site+1} |psi> explicitly.
        """
        A = mps[site]
        B = mps[site + 1]

        theta = np.einsum("asr,rtu->astu", A, B, optimize=True)
        op4 = op.reshape(2, 2, 2, 2)  # row=(s,t), col=(x,y)

        val = np.einsum(
            "ab,astu,stxy,bxyv,uv->",
            left_env,
            theta,
            op4,
            np.conj(theta),
            right_env,
            optimize=True,
        )
        return val

    def _compute_one_site_expectations(
        self,
        problem: QuantumProblem,
        mps: List[np.ndarray],
    ) -> Dict[str, object]:
        observables = problem.one_site_observables()
        left_envs = self._build_left_identity_environments(mps)
        right_envs = self._build_right_identity_environments(mps)

        site_values: Dict[str, List[float]] = {}
        for label, op in observables.items():
            if label == "I":
                continue

            vals = []
            for i in range(problem.n_sites):
                val = self._expectation_one_site(
                    mps=mps,
                    site=i,
                    op=op,
                    left_env=left_envs[i],
                    right_env=right_envs[i + 1],
                )
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
        mps: List[np.ndarray],
    ) -> Dict[str, object]:
        ops = problem.two_site_observables()
        left_envs = self._build_left_identity_environments(mps)
        right_envs = self._build_right_identity_environments(mps)

        out: Dict[str, object] = {}

        for label, op in ops.items():
            edge_vals: Dict[str, float] = {}
            vals: List[float] = []

            for i, j in problem.interaction_edges():
                if j != i + 1:
                    raise NotImplementedError(
                        "This TN solver currently computes only nearest-neighbor "
                        "correlations for open chains."
                    )

                val = self._expectation_two_site(
                    mps=mps,
                    site=i,
                    op=op,
                    left_env=left_envs[i],
                    right_env=right_envs[j + 1],
                )
                real_val = float(np.real_if_close(val))
                edge_vals[f"({i},{j})"] = real_val
                vals.append(real_val)

            out[label] = edge_vals
            out[f"average_{label}"] = float(np.mean(vals)) if vals else 0.0

        return out

    def _compute_entanglement_summary(
        self,
        problem: QuantumProblem,
        mps: List[np.ndarray],
    ) -> Dict[str, object]:
        """
        For simplicity and explicitness, if the system is small enough we reconstruct
        the full state and compute a half-chain von Neumann entropy exactly.

        For larger systems, return a minimal summary.
        """
        if problem.hilbert_dim <= self.config.dense_state_max_dim:
            psi = self._mps_to_dense_state(mps)
            psi = psi / np.linalg.norm(psi)

            n_left = problem.n_sites // 2
            n_right = problem.n_sites - n_left

            dim_left = 2 ** n_left
            dim_right = 2 ** n_right

            psi_matrix = psi.reshape(dim_left, dim_right)
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
                "method": "exact_from_reconstructed_dense_state",
            }

        return {
            "half_chain_entropy_vn": None,
            "schmidt_rank_numerical": None,
            "cut": None,
            "method": "not_computed_system_too_large",
        }

    # -------------------------------------------------------------------------
    # Dense reconstruction
    # -------------------------------------------------------------------------

    def _mps_to_dense_state(self, mps: List[np.ndarray]) -> np.ndarray:
        """
        Reconstruct the full dense state vector from the MPS.

        Returns:
            state vector of shape (2^N,)
        """
        tensor = mps[0]  # shape (1, d, D1)
        for i in range(1, len(mps)):
            tensor = np.tensordot(tensor, mps[i], axes=([-1], [0]))

        # tensor has shape (1, d, d, ..., d, 1)
        tensor = np.squeeze(tensor, axis=0)
        tensor = np.squeeze(tensor, axis=-1)
        return tensor.reshape(-1)


# =============================================================================
# Convenience function
# =============================================================================

def solve(problem: QuantumProblem, config: Optional[TNSolverConfig] = None) -> GroundStateResult:
    """
    Convenience function so you can do:

        result = solve(problem)

    directly.
    """
    solver = TensorNetworkGroundStateSolver(config=config)
    return solver.solve(problem)