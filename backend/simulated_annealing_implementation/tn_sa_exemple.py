from __future__ import annotations

import time

import numpy as np
import quimb.tensor as qtn

from tn_sa_core import GreedySearch, SAConfig, SimulatedAnnealing


def make_square_tn(L: int, chi: int = 2, seed: int | None = None):
    """
    Build an L x L square lattice tensor network.
    """
    rng = np.random.default_rng(seed)
    tensors = []

    for row in range(L):
        for col in range(L):
            idx = row * L + col

            inds = [f"x{idx}_0", f"x{idx}_1"]
            shape = [chi, chi]

            if col < L - 1:
                inds.append(f"h{row}_{col}")
                shape.append(chi)
            if col > 0:
                inds.append(f"h{row}_{col - 1}")
                shape.append(chi)
            if row < L - 1:
                inds.append(f"v{row}_{col}")
                shape.append(chi)
            if row > 0:
                inds.append(f"v{row - 1}_{col}")
                shape.append(chi)

            tensors.append(
                qtn.Tensor(
                    data=rng.standard_normal(shape),
                    inds=inds,
                    tags={f"T{idx}"},
                )
            )

    return qtn.TensorNetwork(tensors)


def make_erdos_renyi_tn(
    n: int,
    p: float = 0.8,
    chi: int = 2,
    seed: int | None = None,
):
    """
    Build a random Erdős-Rényi tensor network.
    """
    rng = np.random.default_rng(seed)
    edges = []
    edge_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append((i, j, f"e{edge_count}"))
                edge_count += 1

    tensors = []
    for i in range(n):
        inds = [f"x{i}_0", f"x{i}_1"]
        shape = [chi, chi]

        for ti, tj, edge_name in edges:
            if ti == i or tj == i:
                inds.append(edge_name)
                shape.append(chi)

        tensors.append(
            qtn.Tensor(
                data=rng.standard_normal(shape),
                inds=inds,
                tags={f"T{i}"},
            )
        )

    return qtn.TensorNetwork(tensors)


def make_2x2_tn(chi: int = 2):
    """
    Build the fixed 2x2 example network from the original script.
    """
    rng = np.random.default_rng(42)

    return qtn.TensorNetwork([
        qtn.Tensor(
            data=rng.standard_normal([chi] * 4),
            inds=["xA1", "xA2", "e1", "e3"],
            tags={"A"},
        ),
        qtn.Tensor(
            data=rng.standard_normal([chi] * 4),
            inds=["xB1", "xB2", "e1", "e4"],
            tags={"B"},
        ),
        qtn.Tensor(
            data=rng.standard_normal([chi] * 4),
            inds=["xC1", "xC2", "e3", "e2"],
            tags={"C"},
        ),
        qtn.Tensor(
            data=rng.standard_normal([chi] * 4),
            inds=["xD1", "xD2", "e4", "e2"],
            tags={"D"},
        ),
    ])


def compare(tn, label: str, sa_iterations: int = 50_000, n_sa_runs: int = 5):
    """
    Compare greedy search, simulated annealing, and quimb's built-in optimizer.
    """
    inner = list(tn.inner_inds())
    sizes = tn.ind_sizes()
    max_chi = max((sizes[i] for i in inner), default=0)

    print("\n" + "=" * 60)
    print(f"{label}")
    print(f"{tn.num_tensors} tensors | {len(inner)} inner indices | max χ = {max_chi}")
    print("=" * 60)

    start = time.perf_counter()
    greedy_sequence, greedy_cost = GreedySearch(tn).optimize()
    greedy_elapsed = time.perf_counter() - start

    print(f"Greedy : {greedy_cost:>18,.0f}   ({greedy_elapsed:.3f}s)")
    print(f"         sequence length = {len(greedy_sequence)}")

    best_sa_cost = float("inf")
    best_sa_sequence = None

    start = time.perf_counter()
    for run in range(n_sa_runs):
        sequence, cost, _ = SimulatedAnnealing(
            tn,
            SAConfig(n_iterations=sa_iterations, seed=run),
        ).optimize()

        if cost < best_sa_cost:
            best_sa_cost = cost
            best_sa_sequence = sequence

    sa_elapsed = time.perf_counter() - start

    print(f"SA     : {best_sa_cost:>18,.0f}   ({sa_elapsed:.3f}s, {n_sa_runs} runs)")
    print(f"         sequence length = {len(best_sa_sequence) if best_sa_sequence else 0}")

    try:
        start = time.perf_counter()
        path_info = tn.contract(all, optimize="auto-hq", get="path-info")
        quimb_elapsed = time.perf_counter() - start
        print(f"quimb  : {path_info.opt_cost:>18,.0f}   ({quimb_elapsed:.3f}s)")
    except Exception as exc:
        print(f"quimb  : failed ({exc})")

    if best_sa_cost > 0:
        print(f"Greedy / SA = {greedy_cost / best_sa_cost:.2f}x")
    else:
        print("Greedy / SA = inf")