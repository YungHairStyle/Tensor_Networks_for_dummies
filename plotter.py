from __future__ import annotations

# Import pathlib so we can create and manage the figures directory cleanly.
from pathlib import Path

# Import typing helpers for clear function signatures.
from typing import Dict, List

# Import matplotlib for plotting.
import matplotlib.pyplot as plt

# Import NumPy for sorting, array conversion, and simple numeric helpers.
import numpy as np


class ResultPlotter:
    """
    Plot comparison figures from experiment records.

    Expected record structure:
        {
            "model": str,
            "n_sites": int,
            "sweep_parameter": str,
            "sweep_value": float,
            "classical_result": GroundStateResult,
            "tn_result": GroundStateResult,
            "metadata": dict,
        }
    """

    def __init__(
        self,
        save: bool = False,
        output_dir: str = "figures",
        dpi: int = 180,
        figsize: tuple = (8, 5),
        file_format: str = "svg",
    ) -> None:
        # Store whether plots should be saved instead of shown.
        self.save = save

        # Store the output directory path.
        self.output_dir = Path(output_dir)

        # Store the resolution used when saving figures.
        self.dpi = dpi

        # Store the default figure size.
        self.figsize = figsize

        # Store the output file format.
        self.file_format = file_format

        # Create the output directory if saving is enabled.
        if self.save:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _finalize(self, fig: plt.Figure, filename: str) -> None:
        # Make the layout neat before saving or showing.
        fig.tight_layout()

        # If save=True, write the figure to disk.
        if self.save:
            fig.savefig(self.output_dir / f"{filename}.{self.file_format}", dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
        else:
            # Otherwise display the figure interactively.
            plt.show()

    def _sorted_records(self, records: List[Dict]) -> List[Dict]:
        # Return records sorted by the sweep parameter value so all curves are ordered correctly.
        return sorted(records, key=lambda r: r["sweep_value"])

    def _x_label(self, sweep_parameter: str) -> str:
        # Map common sweep parameters to readable axis labels with units.
        mapping = {
            "N": "System size N [sites]",
            "h": "Transverse field h [arb. units]",
            "J": "Coupling J [arb. units]",
            "chi": "Maximum bond dimension χ [dimensionless]",
            "delta": "Anisotropy Δ [dimensionless]",
            "gamma": "Anisotropy γ [dimensionless]",
        }

        # Return a specialized label if known, otherwise fall back to the raw name.
        return mapping.get(sweep_parameter, f"{sweep_parameter} [arb. units]")

    def _title_suffix(self, records: List[Dict]) -> str:
        # If there are no records, return an empty suffix.
        if not records:
            return ""

        # Use the first record to summarize the dataset.
        first = records[0]

        # Return a compact title suffix.
        return f"{first['model']}, N={first['n_sites']}"

    def _safe_filename(self, text: str) -> str:
        # Replace spaces and punctuation so the string becomes a safe filename.
        return (
            text.replace(" ", "_")
            .replace("/", "_")
            .replace(",", "")
            .replace("=", "_")
            .replace("(", "")
            .replace(")", "")
        )

    def plot_ground_energy(self, records: List[Dict]) -> None:
        # Stop immediately if there is nothing to plot.
        if not records:
            return

        # Sort the records by the sweep parameter value.
        records = self._sorted_records(records)

        # Extract the sweep parameter name from the first record.
        sweep_parameter = records[0]["sweep_parameter"]

        # Build the x-values.
        x = np.array([r["sweep_value"] for r in records], dtype=float)

        # Build the classical energy curve.
        y_classical = np.array([r["classical_result"].ground_energy for r in records], dtype=float)

        # Build the TN energy curve.
        y_tn = np.array([r["tn_result"].ground_energy for r in records], dtype=float)

        # Create the figure and axes.
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot the classical curve.
        ax.plot(x, y_classical, marker="o", linewidth=2, label="Classical exact diagonalization")

        # Plot the TN curve.
        ax.plot(x, y_tn, marker="s", linewidth=2, label="Tensor network (MPS-TEBD)")

        # Label the x-axis.
        ax.set_xlabel(self._x_label(sweep_parameter))

        # Label the y-axis with units.
        ax.set_ylabel("Ground-state energy [arb. units]")

        # Set the title.
        ax.set_title(f"Ground-state energy comparison ({self._title_suffix(records)})")

        # Show a legend.
        ax.legend()

        # Turn on a light grid.
        ax.grid(True, alpha=0.3)

        # Save or show the figure.
        self._finalize(fig, self._safe_filename(f"ground_energy_{self._title_suffix(records)}_{sweep_parameter}"))

    def plot_energy_per_site(self, records: List[Dict]) -> None:
        # Stop if there is nothing to plot.
        if not records:
            return

        # Sort the records.
        records = self._sorted_records(records)

        # Read the sweep parameter name.
        sweep_parameter = records[0]["sweep_parameter"]

        # Extract the x-values.
        x = np.array([r["sweep_value"] for r in records], dtype=float)

        # Extract classical energy per site values.
        y_classical = np.array([r["classical_result"].energy_per_site for r in records], dtype=float)

        # Extract TN energy per site values.
        y_tn = np.array([r["tn_result"].energy_per_site for r in records], dtype=float)

        # Create the figure.
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot the classical line.
        ax.plot(x, y_classical, marker="o", linewidth=2, label="Classical exact diagonalization")

        # Plot the TN line.
        ax.plot(x, y_tn, marker="s", linewidth=2, label="Tensor network (MPS-TEBD)")

        # Label axes with units.
        ax.set_xlabel(self._x_label(sweep_parameter))
        ax.set_ylabel("Energy per site [arb. units/site]")

        # Set the title.
        ax.set_title(f"Energy per site comparison ({self._title_suffix(records)})")

        # Add a legend.
        ax.legend()

        # Add a grid.
        ax.grid(True, alpha=0.3)

        # Save or show.
        self._finalize(fig, self._safe_filename(f"energy_per_site_{self._title_suffix(records)}_{sweep_parameter}"))

    def plot_runtime(self, records: List[Dict], logy: bool = True) -> None:
        # Stop if empty.
        if not records:
            return

        # Sort the records.
        records = self._sorted_records(records)

        # Read the sweep parameter.
        sweep_parameter = records[0]["sweep_parameter"]

        # Extract x-values.
        x = np.array([r["sweep_value"] for r in records], dtype=float)

        # Extract classical runtimes.
        y_classical = np.array([
            r["classical_result"].diagnostics.runtime_seconds
            if r["classical_result"].diagnostics.runtime_seconds is not None else np.nan
            for r in records
        ], dtype=float)

        # Extract TN runtimes.
        y_tn = np.array([
            r["tn_result"].diagnostics.runtime_seconds
            if r["tn_result"].diagnostics.runtime_seconds is not None else np.nan
            for r in records
        ], dtype=float)

        # Create the figure.
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot classical runtime.
        ax.plot(x, y_classical, marker="o", linewidth=2, label="Classical exact diagonalization")

        # Plot TN runtime.
        ax.plot(x, y_tn, marker="s", linewidth=2, label="Tensor network (MPS-TEBD)")

        # Label axes.
        ax.set_xlabel(self._x_label(sweep_parameter))
        ax.set_ylabel("Runtime [s]")

        # Set the title.
        ax.set_title(f"Runtime comparison ({self._title_suffix(records)})")

        # Use log scale if requested.
        if logy:
            ax.set_yscale("log")

        # Add legend and grid.
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save or show.
        self._finalize(fig, self._safe_filename(f"runtime_{self._title_suffix(records)}_{sweep_parameter}"))

    def plot_absolute_energy_error(self, records: List[Dict], logy: bool = True) -> None:
        # Stop if empty.
        if not records:
            return

        # Sort the records.
        records = self._sorted_records(records)

        # Read the sweep parameter.
        sweep_parameter = records[0]["sweep_parameter"]

        # Extract x-values.
        x = np.array([r["sweep_value"] for r in records], dtype=float)

        # The classical solver compared to itself has zero error by construction.
        y_classical = np.zeros(len(records), dtype=float)

        # TN error is measured relative to the classical reference energy.
        y_tn = np.array([
            abs(r["tn_result"].ground_energy - r["classical_result"].ground_energy)
            for r in records
        ], dtype=float)

        # Create the figure.
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot the classical zero-reference line.
        ax.plot(x, y_classical, marker="o", linewidth=2, label="Classical reference error = 0")

        # Plot the TN absolute error.
        ax.plot(x, y_tn, marker="s", linewidth=2, label="TN absolute energy error")

        # Label axes.
        ax.set_xlabel(self._x_label(sweep_parameter))
        ax.set_ylabel("Absolute energy error [arb. units]")

        # Set the title.
        ax.set_title(f"Absolute energy error relative to classical reference ({self._title_suffix(records)})")

        # Avoid log scale when zeros are present on the classical line.
        if logy:
            # Replace exact zeros on the classical line by NaN so the TN curve can still use a log axis cleanly.
            ax.lines[0].set_ydata(np.full_like(y_classical, np.nan, dtype=float))
            ax.set_yscale("log")

        # Add legend and grid.
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save or show.
        self._finalize(fig, self._safe_filename(f"absolute_energy_error_{self._title_suffix(records)}_{sweep_parameter}"))

    def plot_magnetization(self, records: List[Dict], component: str = "x") -> None:
        # Stop if empty.
        if not records:
            return

        # Sort the records.
        records = self._sorted_records(records)

        # Read the sweep parameter.
        sweep_parameter = records[0]["sweep_parameter"]

        # Build the magnetization key.
        key = f"magnetization_{component.lower()}"

        # Extract x-values.
        x = np.array([r["sweep_value"] for r in records], dtype=float)

        # Extract classical magnetization values.
        y_classical = np.array([
            r["classical_result"].expectation_values.get(key, np.nan)
            for r in records
        ], dtype=float)

        # Extract TN magnetization values.
        y_tn = np.array([
            r["tn_result"].expectation_values.get(key, np.nan)
            for r in records
        ], dtype=float)

        # Create the figure.
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot classical magnetization.
        ax.plot(x, y_classical, marker="o", linewidth=2, label="Classical exact diagonalization")

        # Plot TN magnetization.
        ax.plot(x, y_tn, marker="s", linewidth=2, label="Tensor network (MPS-TEBD)")

        # Label axes.
        ax.set_xlabel(self._x_label(sweep_parameter))
        ax.set_ylabel(f"Magnetization ⟨{component.upper()}⟩ [dimensionless]")

        # Set the title.
        ax.set_title(f"Magnetization comparison in {component.upper()} direction ({self._title_suffix(records)})")

        # Add legend and grid.
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save or show.
        self._finalize(fig, self._safe_filename(f"magnetization_{component}_{self._title_suffix(records)}_{sweep_parameter}"))

    def plot_energy_variance(self, records: List[Dict], logy: bool = True) -> None:
        # Stop if empty.
        if not records:
            return

        # Sort the records.
        records = self._sorted_records(records)

        # Read the sweep parameter.
        sweep_parameter = records[0]["sweep_parameter"]

        # Extract x-values.
        x = np.array([r["sweep_value"] for r in records], dtype=float)

        # Extract classical energy variances.
        y_classical = np.array([
            r["classical_result"].energy_variance if r["classical_result"].energy_variance is not None else np.nan
            for r in records
        ], dtype=float)

        # Extract TN energy variances.
        y_tn = np.array([
            r["tn_result"].energy_variance if r["tn_result"].energy_variance is not None else np.nan
            for r in records
        ], dtype=float)

        # Create the figure.
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot classical variance.
        ax.plot(x, y_classical, marker="o", linewidth=2, label="Classical exact diagonalization")

        # Plot TN variance.
        ax.plot(x, y_tn, marker="s", linewidth=2, label="Tensor network (MPS-TEBD)")

        # Label axes.
        ax.set_xlabel(self._x_label(sweep_parameter))
        ax.set_ylabel("Energy variance [arb. units²]")

        # Set the title.
        ax.set_title(f"Energy variance comparison ({self._title_suffix(records)})")

        # Use log scale if requested.
        if logy:
            ax.set_yscale("log")

        # Add legend and grid.
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save or show.
        self._finalize(fig, self._safe_filename(f"energy_variance_{self._title_suffix(records)}_{sweep_parameter}"))

    def plot_residual_norm(self, records: List[Dict], logy: bool = True) -> None:
        # Stop if empty.
        if not records:
            return

        # Sort the records.
        records = self._sorted_records(records)

        # Read the sweep parameter.
        sweep_parameter = records[0]["sweep_parameter"]

        # Extract x-values.
        x = np.array([r["sweep_value"] for r in records], dtype=float)

        # Extract classical residual norms.
        y_classical = np.array([
            r["classical_result"].residual_norm if r["classical_result"].residual_norm is not None else np.nan
            for r in records
        ], dtype=float)

        # Extract TN residual norms.
        y_tn = np.array([
            r["tn_result"].residual_norm if r["tn_result"].residual_norm is not None else np.nan
            for r in records
        ], dtype=float)

        # Create the figure.
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot the classical residual curve.
        ax.plot(x, y_classical, marker="o", linewidth=2, label="Classical exact diagonalization")

        # Plot the TN residual curve.
        ax.plot(x, y_tn, marker="s", linewidth=2, label="Tensor network (MPS-TEBD)")

        # Label axes.
        ax.set_xlabel(self._x_label(sweep_parameter))
        ax.set_ylabel("Residual norm ||Hψ - Eψ|| [arb. units]")

        # Set the title.
        ax.set_title(f"Residual norm comparison ({self._title_suffix(records)})")

        # Use log scale if requested.
        if logy:
            ax.set_yscale("log")

        # Add legend and grid.
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save or show.
        self._finalize(fig, self._safe_filename(f"residual_norm_{self._title_suffix(records)}_{sweep_parameter}"))

    def plot_half_chain_entropy(self, records: List[Dict]) -> None:
        # Stop if empty.
        if not records:
            return

        # Sort the records.
        records = self._sorted_records(records)

        # Read the sweep parameter.
        sweep_parameter = records[0]["sweep_parameter"]

        # Extract x-values.
        x = np.array([r["sweep_value"] for r in records], dtype=float)

        # Extract classical half-chain entropy values.
        y_classical = np.array([
            r["classical_result"].entanglement.get("half_chain_entropy_vn", np.nan)
            for r in records
        ], dtype=float)

        # Extract TN half-chain entropy values.
        y_tn = np.array([
            r["tn_result"].entanglement.get("half_chain_entropy_vn", np.nan)
            for r in records
        ], dtype=float)

        # Create the figure.
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot classical entropy.
        ax.plot(x, y_classical, marker="o", linewidth=2, label="Classical exact diagonalization")

        # Plot TN entropy.
        ax.plot(x, y_tn, marker="s", linewidth=2, label="Tensor network (MPS-TEBD)")

        # Label axes.
        ax.set_xlabel(self._x_label(sweep_parameter))
        ax.set_ylabel("Half-chain entanglement entropy [bits]")

        # Set the title.
        ax.set_title(f"Half-chain entanglement entropy comparison ({self._title_suffix(records)})")

        # Add legend and grid.
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save or show.
        self._finalize(fig, self._safe_filename(f"half_chain_entropy_{self._title_suffix(records)}_{sweep_parameter}"))

    def plot_all_standard_comparisons(self, records: List[Dict]) -> None:
        # Plot the total energy comparison.
        self.plot_ground_energy(records)

        # Plot energy per site comparison.
        self.plot_energy_per_site(records)

        # Plot absolute energy error relative to the classical reference.
        self.plot_absolute_energy_error(records, logy=True)

        # Plot runtime comparison.
        self.plot_runtime(records, logy=True)

        # Plot magnetization in the X direction.
        self.plot_magnetization(records, component="x")

        # Plot magnetization in the Z direction.
        self.plot_magnetization(records, component="z")

        # Plot energy variance comparison.
        self.plot_energy_variance(records, logy=True)

        # Plot residual norm comparison.
        self.plot_residual_norm(records, logy=True)

        # Plot half-chain entropy comparison.
        self.plot_half_chain_entropy(records)