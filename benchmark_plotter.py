from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


class BenchmarkPlotter:
    """
    Plot compact benchmark/sweep summaries from the dictionaries produced by test.py.

    Expected record format for a sweep item:
        {
            "sweep_parameter": "chi" | "h" | "N" | ...,
            "sweep_value": float | int,
            "classical_result": GroundStateResult,
            "tn_energy_mean": float,
            "tn_energy_std": float,
            "tn_error_vs_classical_mean": float,
            "tn_error_vs_classical_std": float,
            "tn_runtime_mean": float,
            "tn_runtime_std": float,
            "tn_variance_mean": float,
            "tn_residual_mean": float,
            "tn_magx_mean": float,
            "tn_magz_mean": float,
            "tn_fidelity_mean": float,          # optional
            "tn_fidelity_std": float,           # optional
            "analytical_energy": float | None,  # optional
        }
    """

    def __init__(
        self,
        save: bool = False,
        output_dir: str = "figures/benchmarks",
        dpi: int = 180,
        figsize: tuple = (8, 5),
        file_format: str = "svg",
    ) -> None:
        self.save = save
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        self.figsize = figsize
        self.file_format = file_format

        if self.save:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # basic helpers
    # -------------------------------------------------------------------------

    def _finalize(self, fig: plt.Figure, filename: str) -> None:
        fig.tight_layout()
        if self.save:
            fig.savefig(
                self.output_dir / f"{filename}.{self.file_format}",
                dpi=self.dpi,
                bbox_inches="tight",
            )
            plt.close(fig)
        else:
            plt.show()

    def _sorted_records(self, records: List[Dict]) -> List[Dict]:
        return sorted(records, key=lambda r: r["sweep_value"])

    def _safe_filename(self, text: str) -> str:
        return (
            text.replace(" ", "_")
            .replace("/", "_")
            .replace(",", "")
            .replace("=", "_")
            .replace("(", "")
            .replace(")", "")
        )

    def _x_label(self, sweep_parameter: str) -> str:
        mapping = {
            "chi": "Maximum bond dimension χ [dimensionless]",
            "h": "Transverse field h [arb. units]",
            "N": "System size N [sites]",
            "J": "Coupling J [arb. units]",
            "delta": "Anisotropy Δ [dimensionless]",
            "gamma": "Anisotropy γ [dimensionless]",
        }
        return mapping.get(sweep_parameter, f"{sweep_parameter} [arb. units]")

    def _extract_x(self, records: List[Dict]) -> np.ndarray:
        return np.array([r["sweep_value"] for r in records], dtype=float)

    def _extract_y(self, records: List[Dict], key: str) -> np.ndarray:
        return np.array([r.get(key, np.nan) for r in records], dtype=float)

    def _extract_classical_energy(self, records: List[Dict]) -> np.ndarray:
        return np.array(
            [r["classical_result"].ground_energy for r in records],
            dtype=float,
        )

    def _extract_classical_energy_per_site(self, records: List[Dict]) -> np.ndarray:
        return np.array(
            [r["classical_result"].energy_per_site for r in records],
            dtype=float,
        )

    def _extract_classical_runtime(self, records: List[Dict]) -> np.ndarray:
        vals = []
        for r in records:
            rt = r["classical_result"].diagnostics.runtime_seconds
            vals.append(np.nan if rt is None else float(rt))
        return np.array(vals, dtype=float)

    def _extract_classical_variance(self, records: List[Dict]) -> np.ndarray:
        vals = []
        for r in records:
            v = r["classical_result"].energy_variance
            vals.append(np.nan if v is None else float(v))
        return np.array(vals, dtype=float)

    def _extract_classical_residual(self, records: List[Dict]) -> np.ndarray:
        vals = []
        for r in records:
            v = r["classical_result"].residual_norm
            vals.append(np.nan if v is None else float(v))
        return np.array(vals, dtype=float)

    def _extract_classical_mag(self, records: List[Dict], key: str) -> np.ndarray:
        vals = []
        for r in records:
            v = r["classical_result"].expectation_values.get(key, np.nan)
            vals.append(np.nan if v is None else float(v))
        return np.array(vals, dtype=float)

    def _plot_two_lines(
        self,
        x: np.ndarray,
        y1: np.ndarray,
        y2: np.ndarray,
        xlabel: str,
        ylabel: str,
        title: str,
        label1: str,
        label2: str,
        filename: str,
        logy: bool = False,
    ) -> None:
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(x, y1, marker="o", linewidth=2, label=label1)
        ax.plot(x, y2, marker="s", linewidth=2, label=label2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if logy:
            ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self._finalize(fig, filename)

    def _plot_single_line(
        self,
        x: np.ndarray,
        y: np.ndarray,
        xlabel: str,
        ylabel: str,
        title: str,
        filename: str,
        logy: bool = False,
    ) -> None:
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(x, y, marker="o", linewidth=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        self._finalize(fig, filename)

    # -------------------------------------------------------------------------
    # core benchmark plots
    # -------------------------------------------------------------------------

    def plot_energy_comparison(self, records: List[Dict], title_prefix: str = "") -> None:
        if not records:
            return

        records = self._sorted_records(records)
        sweep_parameter = records[0]["sweep_parameter"]

        x = self._extract_x(records)
        y_classical = self._extract_classical_energy(records)
        y_tn = self._extract_y(records, "tn_energy_mean")

        title = f"{title_prefix} Ground-state energy comparison".strip()
        filename = self._safe_filename(f"{title}_{sweep_parameter}")

        self._plot_two_lines(
            x=x,
            y1=y_classical,
            y2=y_tn,
            xlabel=self._x_label(sweep_parameter),
            ylabel="Ground-state energy [arb. units]",
            title=title,
            label1="Classical exact diagonalization",
            label2="TN mean energy",
            filename=filename,
            logy=False,
        )

    def plot_energy_per_site_comparison(self, records: List[Dict], title_prefix: str = "") -> None:
        if not records:
            return

        records = self._sorted_records(records)
        sweep_parameter = records[0]["sweep_parameter"]

        x = self._extract_x(records)
        y_classical = self._extract_classical_energy_per_site(records)
        y_tn = self._extract_y(records, "tn_energy_mean") / x

        # For sweeps where x is not N, use problem_summary n_sites instead.
        if sweep_parameter != "N":
            n_sites = np.array(
                [r["classical_result"].problem_summary["n_sites"] for r in records],
                dtype=float,
            )
            y_tn = self._extract_y(records, "tn_energy_mean") / n_sites

        title = f"{title_prefix} Energy per site comparison".strip()
        filename = self._safe_filename(f"{title}_{sweep_parameter}")

        self._plot_two_lines(
            x=x,
            y1=y_classical,
            y2=y_tn,
            xlabel=self._x_label(sweep_parameter),
            ylabel="Energy per site [arb. units/site]",
            title=title,
            label1="Classical exact diagonalization",
            label2="TN mean energy/site",
            filename=filename,
            logy=False,
        )

    def plot_absolute_energy_error(self, records: List[Dict], title_prefix: str = "", logy: bool = True) -> None:
        if not records:
            return

        records = self._sorted_records(records)
        sweep_parameter = records[0]["sweep_parameter"]

        x = self._extract_x(records)
        y = self._extract_y(records, "tn_error_vs_classical_mean")

        title = f"{title_prefix} Absolute TN energy error".strip()
        filename = self._safe_filename(f"{title}_{sweep_parameter}")

        self._plot_single_line(
            x=x,
            y=y,
            xlabel=self._x_label(sweep_parameter),
            ylabel=r"$|E_{\mathrm{TN}} - E_{\mathrm{exact}}|$ [arb. units]",
            title=title,
            filename=filename,
            logy=logy,
        )

    def plot_fidelity(self, records: List[Dict], title_prefix: str = "") -> None:
        if not records or "tn_fidelity_mean" not in records[0]:
            return

        records = self._sorted_records(records)
        sweep_parameter = records[0]["sweep_parameter"]

        x = self._extract_x(records)
        y = self._extract_y(records, "tn_fidelity_mean")

        title = f"{title_prefix} TN vs exact fidelity".strip()
        filename = self._safe_filename(f"{title}_{sweep_parameter}")

        self._plot_single_line(
            x=x,
            y=y,
            xlabel=self._x_label(sweep_parameter),
            ylabel="Fidelity [dimensionless]",
            title=title,
            filename=filename,
            logy=False,
        )

    def plot_runtime_comparison(self, records: List[Dict], title_prefix: str = "", logy: bool = True) -> None:
        if not records:
            return

        records = self._sorted_records(records)
        sweep_parameter = records[0]["sweep_parameter"]

        x = self._extract_x(records)
        y_classical = self._extract_classical_runtime(records)
        y_tn = self._extract_y(records, "tn_runtime_mean")

        title = f"{title_prefix} Runtime comparison".strip()
        filename = self._safe_filename(f"{title}_{sweep_parameter}")

        self._plot_two_lines(
            x=x,
            y1=y_classical,
            y2=y_tn,
            xlabel=self._x_label(sweep_parameter),
            ylabel="Runtime [s]",
            title=title,
            label1="Classical exact diagonalization",
            label2="TN mean runtime",
            filename=filename,
            logy=logy,
        )

    def plot_variance_comparison(self, records: List[Dict], title_prefix: str = "", logy: bool = True) -> None:
        if not records:
            return

        records = self._sorted_records(records)
        sweep_parameter = records[0]["sweep_parameter"]

        x = self._extract_x(records)
        y_classical = self._extract_classical_variance(records)
        y_tn = self._extract_y(records, "tn_variance_mean")

        title = f"{title_prefix} Energy variance comparison".strip()
        filename = self._safe_filename(f"{title}_{sweep_parameter}")

        self._plot_two_lines(
            x=x,
            y1=y_classical,
            y2=y_tn,
            xlabel=self._x_label(sweep_parameter),
            ylabel="Energy variance [arb. units²]",
            title=title,
            label1="Classical exact diagonalization",
            label2="TN mean variance",
            filename=filename,
            logy=logy,
        )

    def plot_residual_comparison(self, records: List[Dict], title_prefix: str = "", logy: bool = True) -> None:
        if not records:
            return

        records = self._sorted_records(records)
        sweep_parameter = records[0]["sweep_parameter"]

        x = self._extract_x(records)
        y_classical = self._extract_classical_residual(records)
        y_tn = self._extract_y(records, "tn_residual_mean")

        title = f"{title_prefix} Residual norm comparison".strip()
        filename = self._safe_filename(f"{title}_{sweep_parameter}")

        self._plot_two_lines(
            x=x,
            y1=y_classical,
            y2=y_tn,
            xlabel=self._x_label(sweep_parameter),
            ylabel=r"Residual norm $\|H\psi - E\psi\|$ [arb. units]",
            title=title,
            label1="Classical exact diagonalization",
            label2="TN mean residual",
            filename=filename,
            logy=logy,
        )

    def plot_magnetization_x_comparison(self, records: List[Dict], title_prefix: str = "") -> None:
        if not records:
            return

        records = self._sorted_records(records)
        sweep_parameter = records[0]["sweep_parameter"]

        x = self._extract_x(records)
        y_classical = self._extract_classical_mag(records, "magnetization_x")
        y_tn = self._extract_y(records, "tn_magx_mean")

        title = f"{title_prefix} Magnetization X comparison".strip()
        filename = self._safe_filename(f"{title}_{sweep_parameter}")

        self._plot_two_lines(
            x=x,
            y1=y_classical,
            y2=y_tn,
            xlabel=self._x_label(sweep_parameter),
            ylabel=r"$\langle X \rangle$ [dimensionless]",
            title=title,
            label1="Classical exact diagonalization",
            label2="TN mean magnetization X",
            filename=filename,
            logy=False,
        )

    def plot_magnetization_z_comparison(self, records: List[Dict], title_prefix: str = "") -> None:
        if not records:
            return

        records = self._sorted_records(records)
        sweep_parameter = records[0]["sweep_parameter"]

        x = self._extract_x(records)
        y_classical = self._extract_classical_mag(records, "magnetization_z")
        y_tn = self._extract_y(records, "tn_magz_mean")

        title = f"{title_prefix} Magnetization Z comparison".strip()
        filename = self._safe_filename(f"{title}_{sweep_parameter}")

        self._plot_two_lines(
            x=x,
            y1=y_classical,
            y2=y_tn,
            xlabel=self._x_label(sweep_parameter),
            ylabel=r"$\langle Z \rangle$ [dimensionless]",
            title=title,
            label1="Classical exact diagonalization",
            label2="TN mean magnetization Z",
            filename=filename,
            logy=False,
        )

    def plot_error_and_fidelity_panel(self, records: List[Dict], title_prefix: str = "") -> None:
        """
        Make a compact 2-panel summary:
            left  = absolute error
            right = fidelity (if available)
        """
        if not records:
            return

        records = self._sorted_records(records)
        sweep_parameter = records[0]["sweep_parameter"]
        x = self._extract_x(records)
        err = self._extract_y(records, "tn_error_vs_classical_mean")

        has_fidelity = "tn_fidelity_mean" in records[0]

        if has_fidelity:
            fidelity = self._extract_y(records, "tn_fidelity_mean")
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

            axes[0].plot(x, err, marker="o", linewidth=2)
            axes[0].set_xlabel(self._x_label(sweep_parameter))
            axes[0].set_ylabel(r"$|E_{\mathrm{TN}} - E_{\mathrm{exact}}|$ [arb. units]")
            axes[0].set_title("Absolute energy error")
            axes[0].set_yscale("log")
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(x, fidelity, marker="o", linewidth=2)
            axes[1].set_xlabel(self._x_label(sweep_parameter))
            axes[1].set_ylabel("Fidelity [dimensionless]")
            axes[1].set_title("TN vs exact fidelity")
            axes[1].grid(True, alpha=0.3)

            fig.suptitle(f"{title_prefix} Error/Fidelity summary".strip())
            self._finalize(
                fig,
                self._safe_filename(f"{title_prefix}_error_fidelity_panel_{sweep_parameter}"),
            )
        else:
            self.plot_absolute_energy_error(records, title_prefix=title_prefix, logy=True)

    def plot_main_dashboard(self, records: List[Dict], title_prefix: str = "") -> None:
        """
        A compact 2x2 dashboard:
            top-left     energy comparison
            top-right    runtime comparison
            bottom-left  absolute error
            bottom-right fidelity or magnetization X
        """
        if not records:
            return

        records = self._sorted_records(records)
        sweep_parameter = records[0]["sweep_parameter"]
        x = self._extract_x(records)

        y_energy_classical = self._extract_classical_energy(records)
        y_energy_tn = self._extract_y(records, "tn_energy_mean")

        y_runtime_classical = self._extract_classical_runtime(records)
        y_runtime_tn = self._extract_y(records, "tn_runtime_mean")

        y_err = self._extract_y(records, "tn_error_vs_classical_mean")

        use_fidelity = "tn_fidelity_mean" in records[0]
        if use_fidelity:
            y_last = self._extract_y(records, "tn_fidelity_mean")
            last_ylabel = "Fidelity [dimensionless]"
            last_title = "TN vs exact fidelity"
        else:
            y_last = self._extract_y(records, "tn_magx_mean")
            last_ylabel = r"$\langle X \rangle$ [dimensionless]"
            last_title = "TN mean magnetization X"

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        axes[0, 0].plot(x, y_energy_classical, marker="o", linewidth=2, label="Classical")
        axes[0, 0].plot(x, y_energy_tn, marker="s", linewidth=2, label="TN")
        axes[0, 0].set_xlabel(self._x_label(sweep_parameter))
        axes[0, 0].set_ylabel("Ground-state energy [arb. units]")
        axes[0, 0].set_title("Energy comparison")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(x, y_runtime_classical, marker="o", linewidth=2, label="Classical")
        axes[0, 1].plot(x, y_runtime_tn, marker="s", linewidth=2, label="TN")
        axes[0, 1].set_xlabel(self._x_label(sweep_parameter))
        axes[0, 1].set_ylabel("Runtime [s]")
        axes[0, 1].set_title("Runtime comparison")
        axes[0, 1].set_yscale("log")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(x, y_err, marker="o", linewidth=2)
        axes[1, 0].set_xlabel(self._x_label(sweep_parameter))
        axes[1, 0].set_ylabel(r"$|E_{\mathrm{TN}} - E_{\mathrm{exact}}|$ [arb. units]")
        axes[1, 0].set_title("Absolute energy error")
        axes[1, 0].set_yscale("log")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(x, y_last, marker="o", linewidth=2)
        axes[1, 1].set_xlabel(self._x_label(sweep_parameter))
        axes[1, 1].set_ylabel(last_ylabel)
        axes[1, 1].set_title(last_title)
        axes[1, 1].grid(True, alpha=0.3)

        fig.suptitle(f"{title_prefix} Benchmark dashboard".strip())
        self._finalize(
            fig,
            self._safe_filename(f"{title_prefix}_dashboard_{sweep_parameter}"),
        )

    # -------------------------------------------------------------------------
    # convenience groups
    # -------------------------------------------------------------------------

    def plot_minimal_set(self, records: List[Dict], title_prefix: str = "") -> None:
        """
        Recommended small set of plots for readability.
        """
        self.plot_main_dashboard(records, title_prefix=title_prefix)
        self.plot_error_and_fidelity_panel(records, title_prefix=title_prefix)

    def plot_full_set(self, records: List[Dict], title_prefix: str = "") -> None:
        """
        Full benchmark plot collection.
        """
        self.plot_energy_comparison(records, title_prefix=title_prefix)
        self.plot_energy_per_site_comparison(records, title_prefix=title_prefix)
        self.plot_absolute_energy_error(records, title_prefix=title_prefix, logy=True)
        self.plot_runtime_comparison(records, title_prefix=title_prefix, logy=True)
        self.plot_variance_comparison(records, title_prefix=title_prefix, logy=True)
        self.plot_residual_comparison(records, title_prefix=title_prefix, logy=True)
        self.plot_magnetization_x_comparison(records, title_prefix=title_prefix)
        self.plot_magnetization_z_comparison(records, title_prefix=title_prefix)
        self.plot_fidelity(records, title_prefix=title_prefix)
        self.plot_main_dashboard(records, title_prefix=title_prefix)
        self.plot_error_and_fidelity_panel(records, title_prefix=title_prefix)