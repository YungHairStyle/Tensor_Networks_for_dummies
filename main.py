from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.problem import (
    BoundaryCondition,
    make_tfim_1d,
    make_xxz_1d,
    make_xy_1d,
)
from classical_solver import solve as solve_classical, ClassicalSolverConfig
from TN_solver import solve as solve_tn, TNSolverConfig
from benchmark_plotter import BenchmarkPlotter


# =============================================================================
# Core repeated-run evaluation
# =============================================================================

def evaluate_problem_multiple_times(
    problem,
    classical_config: ClassicalSolverConfig,
    tn_base_config: TNSolverConfig,
    n_tn_runs: int = 5,
    analytical_energy: Optional[float] = None,
) -> Dict[str, Any]:
    classical_result = solve_classical(problem, config=classical_config)

    tn_energies: List[float] = []
    tn_energy_errors_vs_classical: List[float] = []
    tn_energy_errors_vs_analytic: List[float] = []
    tn_fidelities: List[float] = []
    tn_runtimes: List[float] = []
    tn_vars: List[float] = []
    tn_residuals: List[float] = []
    tn_magx: List[float] = []
    tn_magz: List[float] = []

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

        tn_energies.append(tn_result.ground_energy)
        tn_energy_errors_vs_classical.append(
            abs(tn_result.ground_energy - classical_result.ground_energy)
        )
        if analytical_energy is not None:
            tn_energy_errors_vs_analytic.append(
                abs(tn_result.ground_energy - analytical_energy)
            )

        if classical_result.state_vector is not None and tn_result.state_vector is not None:
            psi = classical_result.state_vector / np.linalg.norm(classical_result.state_vector)
            phi = tn_result.state_vector / np.linalg.norm(tn_result.state_vector)
            tn_fidelities.append(float(np.abs(np.vdot(psi, phi)) ** 2))

        runtime = tn_result.diagnostics.runtime_seconds
        tn_runtimes.append(float(runtime) if runtime is not None else np.nan)

        tn_vars.append(
            float(tn_result.energy_variance)
            if tn_result.energy_variance is not None
            else np.nan
        )
        tn_residuals.append(
            float(tn_result.residual_norm)
            if tn_result.residual_norm is not None
            else np.nan
        )

        tn_magx.append(float(tn_result.expectation_values.get("magnetization_x", np.nan)))
        tn_magz.append(float(tn_result.expectation_values.get("magnetization_z", np.nan)))

    out: Dict[str, Any] = {
        "problem_summary": problem.summary(),
        "classical_result": classical_result,
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

    return out


# =============================================================================
# Generic sweep routines
# =============================================================================

def sweep_parameter(
    model_name: str,
    sweep_parameter: str,
    sweep_values: List[float],
    problem_builder,
    classical_config: ClassicalSolverConfig,
    tn_base_config: TNSolverConfig,
    n_tn_runs: int = 5,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    print("\n" + "#" * 80)
    print(f"{model_name.upper()} {sweep_parameter.upper()} SWEEP")
    print("#" * 80)

    for value in sweep_values:
        print(f"Running {sweep_parameter} = {value}")

        problem = problem_builder(value)

        result = evaluate_problem_multiple_times(
            problem=problem,
            classical_config=classical_config,
            tn_base_config=tn_base_config,
            n_tn_runs=n_tn_runs,
            analytical_energy=None,
        )

        result["sweep_parameter"] = sweep_parameter
        result["sweep_value"] = value
        results.append(result)

        print(
            f"{sweep_parameter}={value:<8} | "
            f"E_exact={result['classical_result'].ground_energy:+.8f} | "
            f"E_tn_mean={result['tn_energy_mean']:+.8f} | "
            f"mean_err={result['tn_error_vs_classical_mean']:.3e}"
        )

    return results


# =============================================================================
# Plotting wrapper
# =============================================================================

def plot_model_sweeps(
    model_folder: str,
    sweep_name_to_results: Dict[str, List[Dict[str, Any]]],
    save: bool = True,
) -> None:
    outdir = Path("figures") / model_folder
    outdir.mkdir(parents=True, exist_ok=True)

    plotter = BenchmarkPlotter(
        save=save,
        output_dir=str(outdir),
        dpi=180,
        figsize=(8, 5),
        file_format="svg",
    )

    for sweep_name, results in sweep_name_to_results.items():
        title_prefix = f"{model_folder.upper()} {sweep_name}"
        plotter.plot_minimal_set(results, title_prefix=title_prefix)
        plotter.plot_full_set(results, title_prefix=title_prefix)


# =============================================================================
# Main real experiments
# =============================================================================

def main(save_plots: bool = True, n_tn_runs: int = 5) -> None:
    Path("figures").mkdir(parents=True, exist_ok=True)

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

    # -------------------------------------------------------------------------
    # TFIM
    # -------------------------------------------------------------------------
    tfim_field_results = sweep_parameter(
        model_name="tfim",
        sweep_parameter="h",
        sweep_values=[0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
        problem_builder=lambda h: make_tfim_1d(
            n_sites=6,
            j=1.0,
            h=h,
            hz=0.0,
            boundary=BoundaryCondition.OPEN,
            metadata={"model": "tfim", "sweep": "field"},
        ),
        classical_config=classical_config,
        tn_base_config=tn_config,
        n_tn_runs=n_tn_runs,
    )

    tfim_size_results = sweep_parameter(
        model_name="tfim",
        sweep_parameter="N",
        sweep_values=[2, 4, 6, 8, 10],
        problem_builder=lambda n: make_tfim_1d(
            n_sites=int(n),
            j=1.0,
            h=1.0,
            hz=0.0,
            boundary=BoundaryCondition.OPEN,
            metadata={"model": "tfim", "sweep": "size"},
        ),
        classical_config=classical_config,
        tn_base_config=tn_config,
        n_tn_runs=n_tn_runs,
    )

    tfim_chi_results: List[Dict[str, Any]] = []
    print("\n" + "#" * 80)
    print("TFIM CHI SWEEP")
    print("#" * 80)
    for chi in [2, 4, 8, 12, 16, 24, 32]:
        print(f"Running chi = {chi}")
        local_tn_config = TNSolverConfig(
            max_bond_dimension=chi,
            tau_schedule=tn_config.tau_schedule,
            steps_per_tau=tn_config.steps_per_tau,
            energy_tolerance=tn_config.energy_tolerance,
            svd_cutoff=tn_config.svd_cutoff,
            random_seed=tn_config.random_seed,
            init_state=tn_config.init_state,
            compute_observables=tn_config.compute_observables,
            store_state_vector_if_small=tn_config.store_state_vector_if_small,
            dense_state_max_dim=tn_config.dense_state_max_dim,
            verbosity=tn_config.verbosity,
        )

        result = evaluate_problem_multiple_times(
            problem=make_tfim_1d(
                n_sites=6,
                j=1.0,
                h=1.0,
                hz=0.0,
                boundary=BoundaryCondition.OPEN,
                metadata={"model": "tfim", "sweep": "chi"},
            ),
            classical_config=classical_config,
            tn_base_config=local_tn_config,
            n_tn_runs=n_tn_runs,
            analytical_energy=None,
        )
        result["sweep_parameter"] = "chi"
        result["sweep_value"] = chi
        tfim_chi_results.append(result)

    plot_model_sweeps(
        model_folder="tfim",
        sweep_name_to_results={
            "field_sweep": tfim_field_results,
            "size_sweep": tfim_size_results,
            "chi_sweep": tfim_chi_results,
        },
        save=save_plots,
    )

    # -------------------------------------------------------------------------
    # XXZ
    # -------------------------------------------------------------------------
    xxz_delta_results = sweep_parameter(
        model_name="xxz",
        sweep_parameter="delta",
        sweep_values=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
        problem_builder=lambda delta: make_xxz_1d(
            n_sites=6,
            j=1.0,
            delta=delta,
            hz=0.0,
            boundary=BoundaryCondition.OPEN,
            metadata={"model": "xxz", "sweep": "delta"},
        ),
        classical_config=classical_config,
        tn_base_config=tn_config,
        n_tn_runs=n_tn_runs,
    )

    xxz_size_results = sweep_parameter(
        model_name="xxz",
        sweep_parameter="N",
        sweep_values=[2, 4, 6, 8, 10],
        problem_builder=lambda n: make_xxz_1d(
            n_sites=int(n),
            j=1.0,
            delta=1.0,
            hz=0.0,
            boundary=BoundaryCondition.OPEN,
            metadata={"model": "xxz", "sweep": "size"},
        ),
        classical_config=classical_config,
        tn_base_config=tn_config,
        n_tn_runs=n_tn_runs,
    )

    xxz_chi_results: List[Dict[str, Any]] = []
    print("\n" + "#" * 80)
    print("XXZ CHI SWEEP")
    print("#" * 80)
    for chi in [2, 4, 8, 12, 16, 24, 32]:
        print(f"Running chi = {chi}")
        local_tn_config = TNSolverConfig(
            max_bond_dimension=chi,
            tau_schedule=tn_config.tau_schedule,
            steps_per_tau=tn_config.steps_per_tau,
            energy_tolerance=tn_config.energy_tolerance,
            svd_cutoff=tn_config.svd_cutoff,
            random_seed=tn_config.random_seed,
            init_state=tn_config.init_state,
            compute_observables=tn_config.compute_observables,
            store_state_vector_if_small=tn_config.store_state_vector_if_small,
            dense_state_max_dim=tn_config.dense_state_max_dim,
            verbosity=tn_config.verbosity,
        )

        result = evaluate_problem_multiple_times(
            problem=make_xxz_1d(
                n_sites=6,
                j=1.0,
                delta=1.0,
                hz=0.0,
                boundary=BoundaryCondition.OPEN,
                metadata={"model": "xxz", "sweep": "chi"},
            ),
            classical_config=classical_config,
            tn_base_config=local_tn_config,
            n_tn_runs=n_tn_runs,
            analytical_energy=None,
        )
        result["sweep_parameter"] = "chi"
        result["sweep_value"] = chi
        xxz_chi_results.append(result)

    plot_model_sweeps(
        model_folder="xxz",
        sweep_name_to_results={
            "delta_sweep": xxz_delta_results,
            "size_sweep": xxz_size_results,
            "chi_sweep": xxz_chi_results,
        },
        save=save_plots,
    )

    # -------------------------------------------------------------------------
    # XY
    # -------------------------------------------------------------------------
    xy_gamma_results = sweep_parameter(
        model_name="xy",
        sweep_parameter="gamma",
        sweep_values=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        problem_builder=lambda gamma: make_xy_1d(
            n_sites=6,
            j=1.0,
            gamma=gamma,
            h=1.0,
            boundary=BoundaryCondition.OPEN,
            metadata={"model": "xy", "sweep": "gamma"},
        ),
        classical_config=classical_config,
        tn_base_config=tn_config,
        n_tn_runs=n_tn_runs,
    )

    xy_field_results = sweep_parameter(
        model_name="xy",
        sweep_parameter="h",
        sweep_values=[0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
        problem_builder=lambda h: make_xy_1d(
            n_sites=6,
            j=1.0,
            gamma=0.5,
            h=h,
            boundary=BoundaryCondition.OPEN,
            metadata={"model": "xy", "sweep": "field"},
        ),
        classical_config=classical_config,
        tn_base_config=tn_config,
        n_tn_runs=n_tn_runs,
    )

    xy_size_results = sweep_parameter(
        model_name="xy",
        sweep_parameter="N",
        sweep_values=[2, 4, 6, 8, 10],
        problem_builder=lambda n: make_xy_1d(
            n_sites=int(n),
            j=1.0,
            gamma=0.5,
            h=1.0,
            boundary=BoundaryCondition.OPEN,
            metadata={"model": "xy", "sweep": "size"},
        ),
        classical_config=classical_config,
        tn_base_config=tn_config,
        n_tn_runs=n_tn_runs,
    )

    xy_chi_results: List[Dict[str, Any]] = []
    print("\n" + "#" * 80)
    print("XY CHI SWEEP")
    print("#" * 80)
    for chi in [2, 4, 8, 12, 16, 24, 32]:
        print(f"Running chi = {chi}")
        local_tn_config = TNSolverConfig(
            max_bond_dimension=chi,
            tau_schedule=tn_config.tau_schedule,
            steps_per_tau=tn_config.steps_per_tau,
            energy_tolerance=tn_config.energy_tolerance,
            svd_cutoff=tn_config.svd_cutoff,
            random_seed=tn_config.random_seed,
            init_state=tn_config.init_state,
            compute_observables=tn_config.compute_observables,
            store_state_vector_if_small=tn_config.store_state_vector_if_small,
            dense_state_max_dim=tn_config.dense_state_max_dim,
            verbosity=tn_config.verbosity,
        )

        result = evaluate_problem_multiple_times(
            problem=make_xy_1d(
                n_sites=6,
                j=1.0,
                gamma=0.5,
                h=1.0,
                boundary=BoundaryCondition.OPEN,
                metadata={"model": "xy", "sweep": "chi"},
            ),
            classical_config=classical_config,
            tn_base_config=local_tn_config,
            n_tn_runs=n_tn_runs,
            analytical_energy=None,
        )
        result["sweep_parameter"] = "chi"
        result["sweep_value"] = chi
        xy_chi_results.append(result)

    plot_model_sweeps(
        model_folder="xy",
        sweep_name_to_results={
            "gamma_sweep": xy_gamma_results,
            "field_sweep": xy_field_results,
            "size_sweep": xy_size_results,
            "chi_sweep": xy_chi_results,
        },
        save=save_plots,
    )

    print("\nAll model sweeps completed.")


if __name__ == "__main__":
    main(save_plots=True, n_tn_runs=5)