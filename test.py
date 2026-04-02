from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.problem import (
    BoundaryCondition,
    LocalTerm,
    ModelType,
    QuantumProblem,
    make_tfim_1d,
    pauli_x,
    pauli_z,
)
from classical_solver import solve as solve_classical, ClassicalSolverConfig
from TN_solver import solve as solve_tn, TNSolverConfig
from benchmark_plotter import BenchmarkPlotter


# =============================================================================
# Helpers
# =============================================================================

def basis_label(index: int, n_sites: int) -> str:
    return format(index, f"0{n_sites}b")


def remove_global_phase(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=complex).copy()
    if state.size == 0:
        return state

    k = int(np.argmax(np.abs(state)))
    if np.abs(state[k]) < 1e-15:
        return state

    phase = np.angle(state[k])
    return state * np.exp(-1j * phase)


def state_fidelity(psi: Optional[np.ndarray], phi: Optional[np.ndarray]) -> Optional[float]:
    if psi is None or phi is None:
        return None

    psi = np.asarray(psi, dtype=complex)
    phi = np.asarray(phi, dtype=complex)

    npsi = np.linalg.norm(psi)
    nphi = np.linalg.norm(phi)

    if npsi < 1e-15 or nphi < 1e-15:
        return None

    psi = psi / npsi
    phi = phi / nphi
    return float(np.abs(np.vdot(psi, phi)) ** 2)


def pretty_print_state(state: Optional[np.ndarray], n_sites: int, top_k: int = 8) -> str:
    if state is None:
        return "state vector not available"

    state = remove_global_phase(np.asarray(state, dtype=complex))
    probs = np.abs(state) ** 2
    order = np.argsort(probs)[::-1]

    lines = []
    shown = 0
    for idx in order:
        if probs[idx] < 1e-12:
            continue
        amp = state[idx]
        lines.append(
            f"|{basis_label(int(idx), n_sites)}> : "
            f"{amp.real:+.6f}{amp.imag:+.6f}j   "
            f"prob={probs[idx]:.6f}"
        )
        shown += 1
        if shown >= top_k:
            break

    return "\n".join(lines) if lines else "all amplitudes numerically zero"


def safe_get_runtime(result) -> float:
    rt = result.diagnostics.runtime_seconds
    return float(rt) if rt is not None else float("nan")


def safe_get_mag(result, key: str) -> float:
    val = result.expectation_values.get(key, np.nan)
    return float(val) if val is not None else float("nan")


# =============================================================================
# Analytical answers for simple benchmark cases
# =============================================================================

def analytical_tfim_all_x_energy(n_sites: int, h: float) -> float:
    return -float(n_sites) * float(abs(h))


def analytical_tfim_all_z_energy(n_sites: int, j: float) -> float:
    return -(n_sites - 1) * float(abs(j))


def analytical_single_x_site_energy(h: float) -> float:
    return -float(abs(h))


# =============================================================================
# Custom benchmark problem builders
# =============================================================================

def make_single_site_x_problem(h: float) -> QuantumProblem:
    return QuantumProblem(
        model=ModelType.CUSTOM_SPIN_1D,
        n_sites=2,
        boundary=BoundaryCondition.OPEN,
        custom_terms=[
            LocalTerm(
                sites=(0,),
                operator=pauli_x(),
                coefficient=-h,
                label="X0",
            )
        ],
        metadata={"benchmark": "single_site_x"},
    )


def make_single_bond_zz_problem(j: float) -> QuantumProblem:
    return QuantumProblem(
        model=ModelType.CUSTOM_SPIN_1D,
        n_sites=2,
        boundary=BoundaryCondition.OPEN,
        custom_terms=[
            LocalTerm(
                sites=(0, 1),
                operator=np.kron(pauli_z(), pauli_z()),
                coefficient=-j,
                label="ZZ01",
            )
        ],
        metadata={"benchmark": "single_bond_zz"},
    )


# =============================================================================
# Benchmark case definitions
# =============================================================================

@dataclass
class BenchmarkCase:
    name: str
    problem: QuantumProblem
    analytical_energy: Optional[float] = None
    notes: str = ""


def build_benchmark_cases() -> List[BenchmarkCase]:
    cases: List[BenchmarkCase] = []

    h1 = 1.0
    cases.append(
        BenchmarkCase(
            name="single_site_x_h=1.0",
            problem=make_single_site_x_problem(h=h1),
            analytical_energy=analytical_single_x_site_energy(h1),
            notes="Exact product-state benchmark.",
        )
    )

    j2 = 1.0
    cases.append(
        BenchmarkCase(
            name="single_bond_zz_j=1.0",
            problem=make_single_bond_zz_problem(j=j2),
            analytical_energy=-abs(j2),
            notes="Exact 2-site ferromagnetic ZZ benchmark.",
        )
    )

    n3 = 6
    h3 = 1.3
    cases.append(
        BenchmarkCase(
            name="tfim_J0_field_only_N6_h1.3",
            problem=make_tfim_1d(
                n_sites=n3,
                j=0.0,
                h=h3,
                hz=0.0,
                boundary=BoundaryCondition.OPEN,
                metadata={"benchmark": "tfim_field_only"},
            ),
            analytical_energy=analytical_tfim_all_x_energy(n_sites=n3, h=h3),
            notes="Exact product-state TFIM limit.",
        )
    )

    n4 = 6
    j4 = 1.0
    cases.append(
        BenchmarkCase(
            name="tfim_h0_zz_only_N6_j1.0",
            problem=make_tfim_1d(
                n_sites=n4,
                j=j4,
                h=0.0,
                hz=0.0,
                boundary=BoundaryCondition.OPEN,
                metadata={"benchmark": "tfim_zz_only"},
            ),
            analytical_energy=analytical_tfim_all_z_energy(n_sites=n4, j=j4),
            notes="Exact ordered TFIM limit.",
        )
    )

    n5 = 6
    j5 = 1.0
    h5 = 1.0
    cases.append(
        BenchmarkCase(
            name="tfim_nontrivial_N6_j1.0_h1.0",
            problem=make_tfim_1d(
                n_sites=n5,
                j=j5,
                h=h5,
                hz=0.0,
                boundary=BoundaryCondition.OPEN,
                metadata={"benchmark": "tfim_nontrivial"},
            ),
            analytical_energy=None,
            notes="Compare TN against exact classical result only.",
        )
    )

    return cases


# =============================================================================
# Core repeated-run evaluation
# =============================================================================

def evaluate_problem_multiple_times(
    problem: QuantumProblem,
    classical_config: ClassicalSolverConfig,
    tn_base_config: TNSolverConfig,
    n_tn_runs: int = 5,
    analytical_energy: Optional[float] = None,
    print_states: bool = False,
) -> Dict[str, Any]:
    classical_result = solve_classical(problem, config=classical_config)

    tn_results = []
    tn_energies = []
    tn_energy_errors_vs_classical = []
    tn_energy_errors_vs_analytic = []
    tn_fidelities = []
    tn_runtimes = []
    tn_vars = []
    tn_residuals = []
    tn_magx = []
    tn_magz = []

    for run_id in range(n_tn_runs):
        seed = None if tn_base_config.random_seed is None else tn_base_config.random_seed + run_id

        tn_config = TNSolverConfig(
            max_bond_dimension=tn_base_config.max_bond_dimension,
            tau_schedule=tn_base_config.tau_schedule,
            steps_per_tau=tn_base_config.steps_per_tau,
            energy_tolerance=tn_base_config.energy_tolerance,
            svd_cutoff=tn_base_config.svd_cutoff,
            random_seed=seed,
            init_state=tn_base_config.init_state,
            compute_observables=tn_base_config.compute_observables,
            store_state_vector_if_small=tn_base_config.store_state_vector_if_small,
            dense_state_max_dim=tn_base_config.dense_state_max_dim,
            verbosity=tn_base_config.verbosity,
        )

        tn_result = solve_tn(problem, config=tn_config)
        tn_results.append(tn_result)

        energy = tn_result.ground_energy
        err_vs_classical = abs(energy - classical_result.ground_energy)

        tn_energies.append(energy)
        tn_energy_errors_vs_classical.append(err_vs_classical)
        tn_runtimes.append(safe_get_runtime(tn_result))
        tn_vars.append(float(tn_result.energy_variance) if tn_result.energy_variance is not None else np.nan)
        tn_residuals.append(float(tn_result.residual_norm) if tn_result.residual_norm is not None else np.nan)
        tn_magx.append(safe_get_mag(tn_result, "magnetization_x"))
        tn_magz.append(safe_get_mag(tn_result, "magnetization_z"))

        if analytical_energy is not None:
            tn_energy_errors_vs_analytic.append(abs(energy - analytical_energy))

        fid = state_fidelity(classical_result.state_vector, tn_result.state_vector)
        if fid is not None:
            tn_fidelities.append(fid)

    out: Dict[str, Any] = {
        "problem_summary": problem.summary(),
        "classical_result": classical_result,
        "tn_results": tn_results,
        "tn_energy_mean": float(np.nanmean(tn_energies)),
        "tn_energy_std": float(np.nanstd(tn_energies)),
        "tn_error_vs_classical_mean": float(np.nanmean(tn_energy_errors_vs_classical)),
        "tn_error_vs_classical_std": float(np.nanstd(tn_energy_errors_vs_classical)),
        "tn_runtime_mean": float(np.nanmean(tn_runtimes)),
        "tn_runtime_std": float(np.nanstd(tn_runtimes)),
        "tn_variance_mean": float(np.nanmean(tn_vars)),
        "tn_residual_mean": float(np.nanmean(tn_residuals)),
        "tn_magx_mean": float(np.nanmean(tn_magx)),
        "tn_magz_mean": float(np.nanmean(tn_magz)),
        "analytical_energy": analytical_energy,
    }

    if tn_energy_errors_vs_analytic:
        out["tn_error_vs_analytic_mean"] = float(np.nanmean(tn_energy_errors_vs_analytic))
        out["tn_error_vs_analytic_std"] = float(np.nanstd(tn_energy_errors_vs_analytic))

    if tn_fidelities:
        out["tn_fidelity_mean"] = float(np.nanmean(tn_fidelities))
        out["tn_fidelity_std"] = float(np.nanstd(tn_fidelities))

    if print_states:
        print("\nCLASSICAL DOMINANT AMPLITUDES")
        print(pretty_print_state(classical_result.state_vector, problem.n_sites, top_k=8))
        print("\nTN RUN 1 DOMINANT AMPLITUDES")
        print(pretty_print_state(tn_results[0].state_vector, problem.n_sites, top_k=8))

    return out


# =============================================================================
# Pretty printing
# =============================================================================

def print_case_summary(name: str, result: Dict[str, Any]) -> None:
    classical_result = result["classical_result"]
    print("=" * 80)
    print(name)
    print("=" * 80)
    print(f"E_exact/classical               = {classical_result.ground_energy:+.12f}")
    if result["analytical_energy"] is not None:
        print(f"E_analytical                    = {result['analytical_energy']:+.12f}")
        print(
            f"|E_classical - E_analytical|    = "
            f"{abs(classical_result.ground_energy - result['analytical_energy']):.12e}"
        )
    print(f"E_tn_mean                       = {result['tn_energy_mean']:+.12f}")
    print(f"E_tn_std                        = {result['tn_energy_std']:.12e}")
    print(f"mean |E_tn - E_classical|       = {result['tn_error_vs_classical_mean']:.12e}")
    print(f"std  |E_tn - E_classical|       = {result['tn_error_vs_classical_std']:.12e}")
    if "tn_error_vs_analytic_mean" in result:
        print(f"mean |E_tn - E_analytical|      = {result['tn_error_vs_analytic_mean']:.12e}")
        print(f"std  |E_tn - E_analytical|      = {result['tn_error_vs_analytic_std']:.12e}")
    if "tn_fidelity_mean" in result:
        print(f"mean fidelity(TN, exact)        = {result['tn_fidelity_mean']:.12f}")
        print(f"std  fidelity(TN, exact)        = {result['tn_fidelity_std']:.12e}")
    print(f"mean TN runtime [s]             = {result['tn_runtime_mean']:.6f}")
    print(f"mean TN variance                = {result['tn_variance_mean']:.12e}")
    print(f"mean TN residual                = {result['tn_residual_mean']:.12e}")
    print(f"mean TN magnetization_x         = {result['tn_magx_mean']:.12f}")
    print(f"mean TN magnetization_z         = {result['tn_magz_mean']:.12f}")


def print_compact_sweep_table(title: str, results: List[Dict[str, Any]]) -> None:
    print("\n" + "#" * 80)
    print(title)
    print("#" * 80)

    for res in results:
        line = (
            f"{res['sweep_parameter']}={res['sweep_value']:<8} | "
            f"E_exact={res['classical_result'].ground_energy:+.8f} | "
            f"E_tn_mean={res['tn_energy_mean']:+.8f} | "
            f"mean_err={res['tn_error_vs_classical_mean']:.3e}"
        )
        if "tn_fidelity_mean" in res:
            line += f" | mean_fid={res['tn_fidelity_mean']:.6f}"
        print(line)


# =============================================================================
# Sweep routines
# =============================================================================

def sweep_bond_dimension(
    problem: QuantumProblem,
    chi_values: List[int],
    classical_config: ClassicalSolverConfig,
    tn_base_config: TNSolverConfig,
    n_tn_runs: int = 5,
    analytical_energy: Optional[float] = None,
) -> List[Dict[str, Any]]:
    results = []

    print("\n" + "#" * 80)
    print("BOND-DIMENSION SWEEP")
    print("#" * 80)

    for chi in chi_values:
        print(f"\nRunning chi = {chi}")

        tn_config = TNSolverConfig(
            max_bond_dimension=chi,
            tau_schedule=tn_base_config.tau_schedule,
            steps_per_tau=tn_base_config.steps_per_tau,
            energy_tolerance=tn_base_config.energy_tolerance,
            svd_cutoff=tn_base_config.svd_cutoff,
            random_seed=tn_base_config.random_seed,
            init_state=tn_base_config.init_state,
            compute_observables=tn_base_config.compute_observables,
            store_state_vector_if_small=tn_base_config.store_state_vector_if_small,
            dense_state_max_dim=tn_base_config.dense_state_max_dim,
            verbosity=tn_base_config.verbosity,
        )

        result = evaluate_problem_multiple_times(
            problem=problem,
            classical_config=classical_config,
            tn_base_config=tn_config,
            n_tn_runs=n_tn_runs,
            analytical_energy=analytical_energy,
            print_states=False,
        )
        result["sweep_parameter"] = "chi"
        result["sweep_value"] = chi
        results.append(result)

        print_case_summary(f"chi = {chi}", result)

    return results


def sweep_tfim_field(
    n_sites: int,
    j: float,
    h_values: List[float],
    classical_config: ClassicalSolverConfig,
    tn_base_config: TNSolverConfig,
    n_tn_runs: int = 5,
) -> List[Dict[str, Any]]:
    results = []

    print("\n" + "#" * 80)
    print("TFIM FIELD SWEEP")
    print("#" * 80)

    for h in h_values:
        print(f"\nRunning h = {h}")
        problem = make_tfim_1d(
            n_sites=n_sites,
            j=j,
            h=h,
            hz=0.0,
            boundary=BoundaryCondition.OPEN,
            metadata={"sweep": "field"},
        )

        analytical_energy = None
        if abs(j) < 1e-14:
            analytical_energy = analytical_tfim_all_x_energy(n_sites=n_sites, h=h)

        result = evaluate_problem_multiple_times(
            problem=problem,
            classical_config=classical_config,
            tn_base_config=tn_base_config,
            n_tn_runs=n_tn_runs,
            analytical_energy=analytical_energy,
            print_states=False,
        )
        result["sweep_parameter"] = "h"
        result["sweep_value"] = h
        results.append(result)

        print_case_summary(f"h = {h}", result)

    return results


def sweep_tfim_size(
    n_values: List[int],
    j: float,
    h: float,
    classical_config: ClassicalSolverConfig,
    tn_base_config: TNSolverConfig,
    n_tn_runs: int = 5,
) -> List[Dict[str, Any]]:
    results = []

    print("\n" + "#" * 80)
    print("TFIM SIZE SWEEP")
    print("#" * 80)

    for n_sites in n_values:
        print(f"\nRunning N = {n_sites}")
        problem = make_tfim_1d(
            n_sites=n_sites,
            j=j,
            h=h,
            hz=0.0,
            boundary=BoundaryCondition.OPEN,
            metadata={"sweep": "size"},
        )

        analytical_energy = None
        if abs(j) < 1e-14:
            analytical_energy = analytical_tfim_all_x_energy(n_sites=n_sites, h=h)
        elif abs(h) < 1e-14:
            analytical_energy = analytical_tfim_all_z_energy(n_sites=n_sites, j=j)

        result = evaluate_problem_multiple_times(
            problem=problem,
            classical_config=classical_config,
            tn_base_config=tn_base_config,
            n_tn_runs=n_tn_runs,
            analytical_energy=analytical_energy,
            print_states=False,
        )
        result["sweep_parameter"] = "N"
        result["sweep_value"] = n_sites
        results.append(result)

        print_case_summary(f"N = {n_sites}", result)

    return results


# =============================================================================
# Plot wrappers
# =============================================================================

def generate_benchmark_plots(
    chi_results: List[Dict[str, Any]],
    field_results: List[Dict[str, Any]],
    size_results: List[Dict[str, Any]],
    save: bool = True,
) -> None:
    output_dir = Path("figures") / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    plotter = BenchmarkPlotter(
        save=save,
        output_dir=str(output_dir),
        dpi=180,
        figsize=(8, 5),
        file_format="svg",
    )

    plotter.plot_minimal_set(chi_results, title_prefix="TFIM chi sweep")
    plotter.plot_minimal_set(field_results, title_prefix="TFIM field sweep")
    plotter.plot_minimal_set(size_results, title_prefix="TFIM size sweep")

    # Full set too, in case you want all diagnostic plots
    plotter.plot_full_set(chi_results, title_prefix="TFIM chi sweep")
    plotter.plot_full_set(field_results, title_prefix="TFIM field sweep")
    plotter.plot_full_set(size_results, title_prefix="TFIM size sweep")


# =============================================================================
# Main
# =============================================================================

def main(
    save_plots: bool = True,
    n_tn_runs: int = 5,
    print_states: bool = False,
) -> None:
    classical_config = ClassicalSolverConfig(
        store_state_vector=True,
        compute_observables=True,
        check_hermiticity=True,
        hermiticity_tol=1e-10,
    )

    tn_config = TNSolverConfig(
        max_bond_dimension=16,
        tau_schedule=(1e-1, 5e-2, 1e-2, 5e-3, 1e-3),
        steps_per_tau=20,
        energy_tolerance=1e-8,
        svd_cutoff=1e-12,
        random_seed=1234,
        init_state="random",
        compute_observables=True,
        store_state_vector_if_small=True,
        dense_state_max_dim=2**14,
        verbosity=0,
    )

    # ---------------------------------------------------------
    # 1. Original benchmark cases
    # ---------------------------------------------------------
    cases = build_benchmark_cases()
    all_case_results = []

    for case in cases:
        result = evaluate_problem_multiple_times(
            problem=case.problem,
            classical_config=classical_config,
            tn_base_config=tn_config,
            n_tn_runs=n_tn_runs,
            analytical_energy=case.analytical_energy,
            print_states=print_states,
        )
        all_case_results.append((case.name, result))
        print_case_summary(case.name, result)

    print("\n" + "#" * 80)
    print("FINAL COMPACT TABLE")
    print("#" * 80)
    for case_name, res in all_case_results:
        line = (
            f"{case_name:<32} | "
            f"E_exact = {res['classical_result'].ground_energy:+.8f} | "
            f"E_tn_mean = {res['tn_energy_mean']:+.8f} | "
            f"mean_err = {res['tn_error_vs_classical_mean']:.3e}"
        )
        if "tn_fidelity_mean" in res:
            line += f" | mean_fid = {res['tn_fidelity_mean']:.6f}"
        print(line)

    # ---------------------------------------------------------
    # 2. Bond-dimension sweep
    # ---------------------------------------------------------
    nontrivial_problem = make_tfim_1d(
        n_sites=6,
        j=1.0,
        h=1.0,
        hz=0.0,
        boundary=BoundaryCondition.OPEN,
        metadata={"sweep": "chi_nontrivial"},
    )

    chi_results = sweep_bond_dimension(
        problem=nontrivial_problem,
        chi_values=[2, 4, 8, 12, 16, 24, 32],
        classical_config=classical_config,
        tn_base_config=tn_config,
        n_tn_runs=n_tn_runs,
        analytical_energy=None,
    )
    print_compact_sweep_table("COMPACT CHI SWEEP TABLE", chi_results)

    # ---------------------------------------------------------
    # 3. TFIM field sweep
    # ---------------------------------------------------------
    field_results = sweep_tfim_field(
        n_sites=6,
        j=1.0,
        h_values=[0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
        classical_config=classical_config,
        tn_base_config=tn_config,
        n_tn_runs=n_tn_runs,
    )
    print_compact_sweep_table("COMPACT FIELD SWEEP TABLE", field_results)

    # ---------------------------------------------------------
    # 4. TFIM size sweep
    # ---------------------------------------------------------
    size_results = sweep_tfim_size(
        n_values=[2, 4, 6, 8, 10],
        j=1.0,
        h=1.0,
        classical_config=classical_config,
        tn_base_config=tn_config,
        n_tn_runs=n_tn_runs,
    )
    print_compact_sweep_table("COMPACT SIZE SWEEP TABLE", size_results)

    # ---------------------------------------------------------
    # 5. Plot benchmark figures
    # ---------------------------------------------------------
    generate_benchmark_plots(
        chi_results=chi_results,
        field_results=field_results,
        size_results=size_results,
        save=save_plots,
    )

    print("\nBenchmark plotting complete.")


if __name__ == "__main__":
    main(save_plots=False, n_tn_runs=5, print_states=False)