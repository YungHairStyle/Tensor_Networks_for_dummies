from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class ModelType(str, Enum):
    TFIM_1D = "tfim_1d"
    XXZ_1D = "xxz_1d"
    XY_1D = "xy_1d"
    CUSTOM_SPIN_1D = "custom_spin_1d"


class BoundaryCondition(str, Enum):
    OPEN = "open"
    PERIODIC = "periodic"


def pauli_x() -> np.ndarray:
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)


def pauli_y() -> np.ndarray:
    return np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)


def pauli_z() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def identity() -> np.ndarray:
    return np.eye(2, dtype=complex)


def kron_n(ops: Sequence[np.ndarray]) -> np.ndarray:
    """Kronecker product of a sequence of operators."""
    out = np.array([[1.0 + 0.0j]])
    for op in ops:
        out = np.kron(out, op)
    return out


@dataclass(frozen=True)
class LocalTerm:
    """
    Generic local Hamiltonian term.

    sites:
        Tuple of lattice sites this term acts on, e.g. (i,) or (i, j)

    operator:
        Dense local operator acting on the Hilbert space of those sites.
        For spin-1/2:
            1-site term  -> shape (2, 2)
            2-site term  -> shape (4, 4)

    coefficient:
        Scalar prefactor.
    """
    sites: Tuple[int, ...]
    operator: np.ndarray
    coefficient: complex = 1.0 + 0.0j
    label: str = ""


@dataclass
class QuantumProblem:
    """
    Single problem object passed to both classical and tensor-network solvers.

    This object is intentionally solver-agnostic:
    - exact solvers can use full_hamiltonian()
    - tensor-network solvers can use local_terms()
    - custom TN contraction schemes can use tn_metadata()
    """
    model: ModelType
    n_sites: int
    boundary: BoundaryCondition = BoundaryCondition.OPEN

    # Common model parameters
    coupling_j: float = 1.0
    field_h: float = 0.0
    gamma: float = 0.0              # XY anisotropy
    delta: float = 1.0              # XXZ anisotropy
    hz: float = 0.0                 # optional longitudinal Z field

    # For custom models
    custom_terms: Optional[List[LocalTerm]] = None

    # Extra room for future extensions
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_sites < 2:
            raise ValueError("n_sites must be at least 2 for meaningful comparisons.")
        if self.model == ModelType.CUSTOM_SPIN_1D and not self.custom_terms:
            raise ValueError("CUSTOM_SPIN_1D requires custom_terms.")

    @property
    def local_dim(self) -> int:
        return 2  # spin-1/2

    @property
    def hilbert_dim(self) -> int:
        return self.local_dim ** self.n_sites

    def interaction_edges(self) -> List[Tuple[int, int]]:
        edges = [(i, i + 1) for i in range(self.n_sites - 1)]
        if self.boundary == BoundaryCondition.PERIODIC:
            edges.append((self.n_sites - 1, 0))
        return edges

    def local_terms(self) -> List[LocalTerm]:
        """
        Return the Hamiltonian as a sum of local terms.
        This is the main interface tensor-network solvers should use.
        """
        if self.model == ModelType.TFIM_1D:
            return self._tfim_terms()
        if self.model == ModelType.XXZ_1D:
            return self._xxz_terms()
        if self.model == ModelType.XY_1D:
            return self._xy_terms()
        if self.model == ModelType.CUSTOM_SPIN_1D:
            return list(self.custom_terms or [])
        raise NotImplementedError(f"Unsupported model: {self.model}")

    def one_site_observables(self) -> Dict[str, np.ndarray]:
        """
        Standard observables you might compare after solving.
        """
        return {
            "X": pauli_x(),
            "Y": pauli_y(),
            "Z": pauli_z(),
            "I": identity(),
        }

    def two_site_observables(self) -> Dict[str, np.ndarray]:
        sx, sy, sz = pauli_x(), pauli_y(), pauli_z()
        return {
            "XX": np.kron(sx, sx),
            "YY": np.kron(sy, sy),
            "ZZ": np.kron(sz, sz),
        }

    def full_hamiltonian(self) -> np.ndarray:
        """
        Build the full dense Hamiltonian matrix.
        Intended for exact diagonalization / classical reference solvers.
        """
        dim = self.hilbert_dim
        h = np.zeros((dim, dim), dtype=complex)
        for term in self.local_terms():
            h += self.embed_local_term(term)
        return h

    def embed_local_term(self, term: LocalTerm) -> np.ndarray:
        """
        Embed a 1-site or 2-site local operator into the full Hilbert space.
        """
        if len(term.sites) == 1:
            return term.coefficient * self._embed_one_site_operator(
                site=term.sites[0],
                op=term.operator,
            )

        if len(term.sites) == 2:
            i, j = term.sites
            if j < i:
                i, j = j, i
            return term.coefficient * self._embed_two_site_operator(
                site_i=i,
                site_j=j,
                op_ij=term.operator,
            )

        raise NotImplementedError(
            "Only 1-site and 2-site terms are currently supported."
        )

    def tn_metadata(self) -> Dict[str, Any]:
        """
        Helpful metadata for tensor-network solvers.
        Add anything here that different contraction schemes may want to inspect.
        """
        return {
            "n_sites": self.n_sites,
            "local_dim": self.local_dim,
            "boundary": self.boundary.value,
            "model": self.model.value,
            "edges": self.interaction_edges(),
            "preferred_ansatz": "mps",
            "supports_periodic_mps": self.boundary == BoundaryCondition.PERIODIC,
            "notes": self.metadata.get("notes", ""),
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "model": self.model.value,
            "n_sites": self.n_sites,
            "local_dim": self.local_dim,
            "hilbert_dim": self.hilbert_dim,
            "boundary": self.boundary.value,
            "coupling_j": self.coupling_j,
            "field_h": self.field_h,
            "gamma": self.gamma,
            "delta": self.delta,
            "hz": self.hz,
            "n_local_terms": len(self.local_terms()),
            "metadata": self.metadata,
        }

    # -------------------------------------------------------------------------
    # Built-in model term generators
    # -------------------------------------------------------------------------

    def _tfim_terms(self) -> List[LocalTerm]:
        """
        H = -J sum_i Z_i Z_{i+1} - h sum_i X_i - hz sum_i Z_i
        """
        sx, sz = pauli_x(), pauli_z()
        terms: List[LocalTerm] = []

        for i, j in self.interaction_edges():
            terms.append(
                LocalTerm(
                    sites=(i, j),
                    operator=np.kron(sz, sz),
                    coefficient=-self.coupling_j,
                    label="ZZ",
                )
            )

        for i in range(self.n_sites):
            if abs(self.field_h) > 0:
                terms.append(
                    LocalTerm(
                        sites=(i,),
                        operator=sx,
                        coefficient=-self.field_h,
                        label="X",
                    )
                )
            if abs(self.hz) > 0:
                terms.append(
                    LocalTerm(
                        sites=(i,),
                        operator=sz,
                        coefficient=-self.hz,
                        label="Z",
                    )
                )

        return terms

    def _xxz_terms(self) -> List[LocalTerm]:
        """
        H = J sum_i [X_i X_{i+1} + Y_i Y_{i+1} + delta Z_i Z_{i+1}] + hz sum_i Z_i
        """
        sx, sy, sz = pauli_x(), pauli_y(), pauli_z()
        terms: List[LocalTerm] = []

        xx = np.kron(sx, sx)
        yy = np.kron(sy, sy)
        zz = np.kron(sz, sz)

        for i, j in self.interaction_edges():
            terms.append(LocalTerm((i, j), xx, self.coupling_j, "XX"))
            terms.append(LocalTerm((i, j), yy, self.coupling_j, "YY"))
            terms.append(LocalTerm((i, j), zz, self.coupling_j * self.delta, "ZZ"))

        if abs(self.hz) > 0:
            for i in range(self.n_sites):
                terms.append(LocalTerm((i,), sz, self.hz, "Z"))

        return terms

    def _xy_terms(self) -> List[LocalTerm]:
        """
        H = J sum_i [(1+gamma) X_i X_{i+1} + (1-gamma) Y_i Y_{i+1}] - h sum_i Z_i
        """
        sx, sy, sz = pauli_x(), pauli_y(), pauli_z()
        terms: List[LocalTerm] = []

        for i, j in self.interaction_edges():
            terms.append(
                LocalTerm(
                    sites=(i, j),
                    operator=np.kron(sx, sx),
                    coefficient=self.coupling_j * (1.0 + self.gamma),
                    label="XX",
                )
            )
            terms.append(
                LocalTerm(
                    sites=(i, j),
                    operator=np.kron(sy, sy),
                    coefficient=self.coupling_j * (1.0 - self.gamma),
                    label="YY",
                )
            )

        if abs(self.field_h) > 0:
            for i in range(self.n_sites):
                terms.append(
                    LocalTerm(
                        sites=(i,),
                        operator=sz,
                        coefficient=-self.field_h,
                        label="Z",
                    )
                )

        return terms

    # -------------------------------------------------------------------------
    # Embedding helpers
    # -------------------------------------------------------------------------
    def embed_one_site_operator(self, site: int, op: np.ndarray) -> np.ndarray:
        return self._embed_one_site_operator(site, op)

    def embed_two_site_operator(self, site_i: int, site_j: int, op_ij: np.ndarray) -> np.ndarray:
        return self._embed_two_site_operator(site_i, site_j, op_ij)

    def _embed_one_site_operator(self, site: int, op: np.ndarray) -> np.ndarray:
        if op.shape != (2, 2):
            raise ValueError(f"Expected one-site operator shape (2,2), got {op.shape}")

        ops = []
        for i in range(self.n_sites):
            ops.append(op if i == site else identity())
        return kron_n(ops)

    def _embed_two_site_operator(
        self,
        site_i: int,
        site_j: int,
        op_ij: np.ndarray,
    ) -> np.ndarray:
        if op_ij.shape != (4, 4):
            raise ValueError(
                f"Expected two-site operator shape (4,4), got {op_ij.shape}"
            )
        if site_i == site_j:
            raise ValueError("site_i and site_j must be different.")

        # Reshape local 2-site operator into rank-4 tensor
        op4 = op_ij.reshape(2, 2, 2, 2)

        # Build full operator by explicit basis action
        dim = self.hilbert_dim
        full = np.zeros((dim, dim), dtype=complex)

        for ket in range(dim):
            bits_ket = self._int_to_bits(ket, self.n_sites)

            a = bits_ket[site_i]
            b = bits_ket[site_j]

            for ap in range(2):
                for bp in range(2):
                    coeff = op4[ap, bp, a, b]
                    if abs(coeff) < 1e-15:
                        continue

                    bits_bra = bits_ket.copy()
                    bits_bra[site_i] = ap
                    bits_bra[site_j] = bp
                    bra = self._bits_to_int(bits_bra)

                    full[bra, ket] += coeff

        return full

    @staticmethod
    def _int_to_bits(x: int, n: int) -> List[int]:
        return [(x >> (n - 1 - i)) & 1 for i in range(n)]

    @staticmethod
    def _bits_to_int(bits: Sequence[int]) -> int:
        x = 0
        for b in bits:
            x = (x << 1) | int(b)
        return x


# -----------------------------------------------------------------------------
# Convenience constructors
# -----------------------------------------------------------------------------

def make_tfim_1d(
    n_sites: int,
    j: float = 1.0,
    h: float = 1.0,
    hz: float = 0.0,
    boundary: BoundaryCondition = BoundaryCondition.OPEN,
    metadata: Optional[Dict[str, Any]] = None,
) -> QuantumProblem:
    return QuantumProblem(
        model=ModelType.TFIM_1D,
        n_sites=n_sites,
        coupling_j=j,
        field_h=h,
        hz=hz,
        boundary=boundary,
        metadata=metadata or {},
    )


def make_xxz_1d(
    n_sites: int,
    j: float = 1.0,
    delta: float = 1.0,
    hz: float = 0.0,
    boundary: BoundaryCondition = BoundaryCondition.OPEN,
    metadata: Optional[Dict[str, Any]] = None,
) -> QuantumProblem:
    return QuantumProblem(
        model=ModelType.XXZ_1D,
        n_sites=n_sites,
        coupling_j=j,
        delta=delta,
        hz=hz,
        boundary=boundary,
        metadata=metadata or {},
    )


def make_xy_1d(
    n_sites: int,
    j: float = 1.0,
    gamma: float = 0.5,
    h: float = 0.0,
    boundary: BoundaryCondition = BoundaryCondition.OPEN,
    metadata: Optional[Dict[str, Any]] = None,
) -> QuantumProblem:
    return QuantumProblem(
        model=ModelType.XY_1D,
        n_sites=n_sites,
        coupling_j=j,
        gamma=gamma,
        field_h=h,
        boundary=boundary,
        metadata=metadata or {},
    )