from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle


def load_results(csv_file: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load sparsity and F-score values from a CSV file.

    Expected CSV columns:
        - inactive_percentage
        - f_score_ext

    Parameters
    ----------
    csv_file:
        Path to the CSV file.

    Returns
    -------
    inactive_percentage:
        1D array with sparsity / inactive neuron percentage values.
    f_score_ext:
        1D array with corresponding F-scores.
    """
    inactive_percentage = []
    f_score_ext = []

    with open(csv_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inactive_percentage.append(float(row["inactive_percentage"]))
            f_score_ext.append(float(row["f_score_ext"]))

    return np.asarray(inactive_percentage), np.asarray(f_score_ext)


def get_binned_best_f1(
    csv_file: str | Path,
    n_bins: int = 35,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the best F-score within each sparsity bin.

    The sparsity range is split into `n_bins` equally spaced bins. For each bin,
    the point with the maximum F-score is selected.

    Parameters
    ----------
    csv_file:
        Path to the CSV file containing the experiment results.
    n_bins:
        Number of bins across the sparsity range.

    Returns
    -------
    best_inactive_per_bin:
        Sparsity values of the best point in each non-empty bin.
    best_f1_per_bin:
        Best F-score found in each non-empty bin.
    """
    inactive_percentage, f_score_ext = load_results(csv_file)

    bin_edges = np.linspace(
        np.min(inactive_percentage),
        np.max(inactive_percentage),
        n_bins + 1,
    )

    best_inactive_per_bin = []
    best_f1_per_bin = []

    for i in range(n_bins):
        # Include the right edge in the final bin so the maximum value is not lost.
        if i == n_bins - 1:
            mask = (
                (inactive_percentage >= bin_edges[i])
                & (inactive_percentage <= bin_edges[i + 1])
            )
        else:
            mask = (
                (inactive_percentage >= bin_edges[i])
                & (inactive_percentage < bin_edges[i + 1])
            )

        if not np.any(mask):
            continue

        local_inactive = inactive_percentage[mask]
        local_f1 = f_score_ext[mask]
        best_idx = np.argmax(local_f1)

        best_inactive_per_bin.append(local_inactive[best_idx])
        best_f1_per_bin.append(local_f1[best_idx])

    return np.asarray(best_inactive_per_bin), np.asarray(best_f1_per_bin)


def plot_spinr_curve(
    ax: plt.Axes,
    csv_file: str | Path,
    label: str,
    color,
    n_bins: int = 35,
) -> None:
    """
    Plot the binned best-F1 curve for one result file.

    Parameters
    ----------
    ax:
        Matplotlib axis to draw on.
    csv_file:
        Path to the CSV file.
    label:
        Legend label.
    color:
        Line color.
    n_bins:
        Number of sparsity bins.
    """
    inactive, f1 = get_binned_best_f1(csv_file, n_bins=n_bins)
    ax.plot(inactive, f1, linestyle="-", linewidth=2.2, color=color, label=label)


def add_region_annotation(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    facecolor: str = "red",
    alpha: float = 0.2,
    text_color: str = "darkred",
    fontsize: int = 10,
) -> None:
    """
    Add a shaded rectangular region with a text annotation.
    """
    rect = Rectangle(
        (x, y),
        width,
        height,
        facecolor=facecolor,
        alpha=alpha,
        zorder=1,
    )
    ax.add_patch(rect)

    ax.text(
        x + 0.5,
        y + height - 0.01,
        text,
        ha="left",
        va="top",
        color=text_color,
        fontsize=fontsize,
        zorder=2,
    )


def configure_plot_style() -> None:
    """
    Configure a clean plotting style.
    """
    sns.set_theme(style="whitegrid", font_scale=1.0)
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "legend.fontsize": 11,
            "lines.linewidth": 1.8,
            "axes.edgecolor": "0.2",
            "axes.linewidth": 0.8,
            "grid.alpha": 0.3,
            "grid.linestyle": "-",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    """
    Create the F-score vs. sparsity comparison plot.
    """
    configure_plot_style()
    colors = sns.color_palette("deep")

    fig, ax = plt.subplots(figsize=(8, 4))

    plot_spinr_curve(
        ax=ax,
        csv_file="results/ags/detection_performance/spinr_fixed_m_gridseach_c1_c2.csv",
        label=r"SpiNR + AGS ($\Theta_{c_1/c_2}$ fixed, $\Theta_m$ optimized)",
        color=colors[0],
        n_bins=35,
    )
    plot_spinr_curve(
        ax=ax,
        csv_file="results/ags/detection_performance/spinr_os_cfar_float_fixed_m_gridsearch_c1_c2.csv",
        label=r"SpiNR + AGS + sOS-CFAR ($\Theta_{c_1/c_2}$ optimized, $\Theta_m$ off)",
        color=colors[1],
        n_bins=35,
    )
    plot_spinr_curve(
        ax=ax,
        csv_file="results/ags/detection_performance/spinr_turnoff_float_heatmap.csv",
        label=r"SpiNR + AGS ($\Theta_{c_1/c_2}$ optimized, $\Theta_m$ off)",
        color=colors[2],
        n_bins=35,
    )

    # Reference baselines.
    ax.axhline(0.79, color="grey", linestyle="--", linewidth=1.5, label="FFT + OS-CFAR")
    ax.axhline(0.54, color="grey", linestyle=":", linewidth=1.5, label="FFT + Const. threshold")

    # Highlight operating regions.
    add_region_annotation(
        ax=ax,
        x=0.5,
        y=0.005,
        width=20,
        height=0.99,
        text="sOS-CFAR\ndetection\ndominates",
    )
    add_region_annotation(
        ax=ax,
        x=40,
        y=0.005,
        width=59,
        height=0.99,
        text="AGS detection dominates",
    )

    ax.set_xlabel("Sparsity [%]")
    ax.set_ylabel("F-Score")
    ax.set_title("Optimized F-Score vs Sparsity", fontweight="bold")

    ax.tick_params(axis="x", labelbottom=True, labelsize=9)
    ax.tick_params(axis="y", labelsize=9, length=0)

    ax.grid(True, axis="y", linestyle="-", color="0.85")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)

    ax.legend(
        frameon=True,
        facecolor="white",
        fontsize=9,
        loc="lower right",
        framealpha=1.0,
    )

    fig.tight_layout()
    fig.savefig(
        "results/ags/plots/f1_vs_activity_latest.pdf",
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )


if __name__ == "__main__":
    main()