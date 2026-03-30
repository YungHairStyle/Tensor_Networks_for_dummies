from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def product_of_dims(indices, size_dict) -> int:
    """
    Return the product of dimensions associated with `indices`.

    If an index is missing from `size_dict`, dimension 2 is assumed.
    """
    cost = 1
    for idx in indices:
        cost *= size_dict.get(idx, 2)
    return cost


def contraction_cost(tn, sequence) -> int:
    """
    Estimate the total FLOP cost of contracting a quimb TensorNetwork
    by eliminating inner indices in the given order.

    Notes
    -----
    - This function does not modify the original tensor network.
    - The cost model follows the same logic as the original script.
    """
    size_dict = tn.ind_sizes()
    tid_to_inds = {tid: set(t.inds) for tid, t in tn.tensor_map.items()}
    ind_to_tids = {}

    for tid, inds in tid_to_inds.items():
        for ind in inds:
            ind_to_tids.setdefault(ind, set()).add(tid)

    total_cost = 0

    for ind in sequence:
        tids = ind_to_tids.get(ind, set())

        if len(tids) == 0:
            continue

        if len(tids) == 1:
            tid = next(iter(tids))
            total_cost += product_of_dims(tid_to_inds[tid], size_dict)
            tid_to_inds[tid].discard(ind)
            ind_to_tids.pop(ind, None)
            continue

        if len(tids) == 2:
            tid1, tid2 = list(tids)
            merged_inds = tid_to_inds[tid1] | tid_to_inds[tid2]

            total_cost += product_of_dims(merged_inds, size_dict)

            tid_to_inds[tid1] = merged_inds - {ind}
            del tid_to_inds[tid2]

            for other_tids in ind_to_tids.values():
                if tid2 in other_tids:
                    other_tids.discard(tid2)
                    other_tids.add(tid1)

            ind_to_tids.pop(ind, None)
            continue

        # Degree > 2 case: preserve the original script behavior.
        tid_list = list(tids)
        merged_tid = tid_list[0]

        for other_tid in tid_list[1:]:
            merged_inds = tid_to_inds[merged_tid] | tid_to_inds[other_tid]
            total_cost += product_of_dims(merged_inds, size_dict)

            tid_to_inds[merged_tid] = merged_inds - {ind}
            del tid_to_inds[other_tid]

            for other_tids in ind_to_tids.values():
                if other_tid in other_tids:
                    other_tids.discard(other_tid)
                    other_tids.add(merged_tid)

        ind_to_tids.pop(ind, None)

    return total_cost


@dataclass(slots=True)
class SAConfig:
    """
    Configuration for simulated annealing.
    """
    n_iterations: int = 100_000
    temp_init: float = 1.0
    temp_min: float = 1e-4
    n_perturbations: int = 2
    seed: Optional[int] = None

    def validate(self) -> None:
        if self.n_iterations <= 0:
            raise ValueError("n_iterations must be > 0.")
        if self.temp_init <= 0:
            raise ValueError("temp_init must be > 0.")
        if self.temp_min <= 0:
            raise ValueError("temp_min must be > 0.")
        if self.n_perturbations <= 0:
            raise ValueError("n_perturbations must be > 0.")


class SimulatedAnnealing:
    """
    Simulated annealing optimizer using continuous encoding.

    Each inner index gets a random float in [0, 1].
    Sorting those floats produces a contraction sequence.
    """

    def __init__(self, tn, config: Optional[SAConfig] = None):
        self.tn = tn
        self.config = config or SAConfig()
        self.config.validate()

        self.edge_list = list(tn.inner_inds())
        self.n_edges = len(self.edge_list)
        self.rng = np.random.default_rng(self.config.seed)

        self.best_cost = float("inf")
        self.best_sequence = None
        self.history = []
        self.n_evaluations = 0

    def _encoding_to_sequence(self, encoding: np.ndarray):
        return [self.edge_list[i] for i in np.argsort(encoding)]

    def _perturb(self, encoding: np.ndarray) -> np.ndarray:
        if self.n_edges == 0:
            return encoding.copy()

        new_encoding = encoding.copy()
        n_changes = min(
            self.rng.integers(1, self.config.n_perturbations + 1),
            self.n_edges,
        )
        indices = self.rng.choice(self.n_edges, size=n_changes, replace=False)
        new_encoding[indices] = self.rng.random(n_changes)
        return new_encoding

    def _temperature(self, iteration: int) -> float:
        c = self.config
        return c.temp_init * (c.temp_min / c.temp_init) ** (iteration / c.n_iterations)

    def optimize(self, verbose: bool = False):
        """
        Run simulated annealing and return:
        (best_sequence, best_cost, history)
        """
        if self.n_edges == 0:
            self.best_sequence = []
            self.best_cost = 0
            self.history = [(0, 0, 0)]
            self.n_evaluations = 0
            return self.best_sequence, self.best_cost, self.history

        encoding = self.rng.random(self.n_edges)
        sequence = self._encoding_to_sequence(encoding)
        cost = contraction_cost(self.tn, sequence)

        self.n_evaluations = 1
        self.best_cost = cost
        self.best_sequence = sequence
        self.history = [(0, cost, cost)]

        log_interval = max(1, self.config.n_iterations // 20)

        for iteration in range(self.config.n_iterations):
            temperature = self._temperature(iteration)

            new_encoding = self._perturb(encoding)
            new_sequence = self._encoding_to_sequence(new_encoding)
            new_cost = contraction_cost(self.tn, new_sequence)

            self.n_evaluations += 1
            delta = (new_cost - cost) / max(cost, 1)

            accept = delta <= 0 or self.rng.random() < np.exp(-delta / temperature)

            if accept:
                encoding = new_encoding
                sequence = new_sequence
                cost = new_cost

                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_sequence = sequence

            if (iteration + 1) % log_interval == 0:
                self.history.append((iteration + 1, cost, self.best_cost))

                if verbose:
                    print(
                        f"[{iteration + 1:>7d}/{self.config.n_iterations}] "
                        f"T={temperature:.6f}  current={cost:,.0f}  best={self.best_cost:,.0f}"
                    )

        return self.best_sequence, self.best_cost, self.history


class GreedySearch:
    """
    Greedy heuristic for selecting a contraction order.
    """

    def __init__(self, tn):
        self.tn = tn

    def optimize(self):
        size_dict = self.tn.ind_sizes()
        tid_to_inds = {tid: set(t.inds) for tid, t in self.tn.tensor_map.items()}
        ind_to_tids = {}

        for tid, inds in tid_to_inds.items():
            for ind in inds:
                ind_to_tids.setdefault(ind, set()).add(tid)

        remaining = set(self.tn.inner_inds())
        sequence = []
        total_cost = 0

        while remaining:
            best_ind = None
            best_step_cost = float("inf")

            for ind in remaining:
                tids = ind_to_tids.get(ind, set())

                if not tids:
                    all_inds = set()
                elif len(tids) == 1:
                    all_inds = tid_to_inds[next(iter(tids))]
                else:
                    all_inds = set()
                    for tid in tids:
                        all_inds |= tid_to_inds[tid]

                step_cost = product_of_dims(all_inds, size_dict) if all_inds else 0

                if step_cost < best_step_cost:
                    best_step_cost = step_cost
                    best_ind = ind

            sequence.append(best_ind)
            total_cost += best_step_cost

            tids = ind_to_tids.get(best_ind, set())

            if len(tids) == 2:
                tid1, tid2 = list(tids)
                tid_to_inds[tid1] = (tid_to_inds[tid1] | tid_to_inds[tid2]) - {best_ind}
                del tid_to_inds[tid2]

                for other_tids in ind_to_tids.values():
                    if tid2 in other_tids:
                        other_tids.discard(tid2)
                        other_tids.add(tid1)

            elif len(tids) == 1:
                tid_to_inds[next(iter(tids))].discard(best_ind)

            remaining.discard(best_ind)
            ind_to_tids.pop(best_ind, None)

        return sequence, total_cost