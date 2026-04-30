"""
Visualize SpiNR neuron dynamics with 
Activity-Gated Sparsity (AGS) on a simulated radar signal.

This script generates a simple synthetic radar sequence, evaluates two neuron
configurations (one resonant and one slightly detuned), and produces a figure
showing:

1. The simulated radar input signal.
2. The neuron magnitude dynamics and adaptive bounds.
3. The internal counters used for activity / shutdown decisions.

The script is intended as a publication-quality example for illustrating the
behavior of the neuron model rather than as a reusable library component.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import nrsp.datasets.dummy
import nrsp.utils.log


logger = nrsp.utils.log.get_logger("logs")
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------
# Plot configuration
# ---------------------------------------------------------------------
sns.set_theme(style="ticks", context="paper")
plt.rcParams.update(
    {
        "text.usetex": True,
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
    }
)


# ---------------------------------------------------------------------
# Configuration containers
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class RadarConfig:
    """Configuration for the synthetic radar example."""

    n_samples: int = 128
    n_chirps: int = 8
    target_range_bin: int = 10
    target_velocity_bin: int = 4
    noise_std: float = 0.8


@dataclass(frozen=True)
class NeuronThresholds:
    """Thresholds controlling the neuron counters and shutdown logic."""

    theta_c1: int = 30
    theta_c2: int = 8
    theta_grad: float = -10.0


@dataclass(frozen=True)
class NeuronSpec:
    """Frequency configuration and display name for one neuron."""

    omega_r: float
    phi_v: float
    name: str


@dataclass
class NeuronTrace:
    """Stores all time-series produced by one neuron simulation."""

    state_magnitude: np.ndarray
    magnitude_proxy: np.ndarray
    upper_bound: np.ndarray
    lower_bound: np.ndarray
    counter_1: np.ndarray
    counter_2: np.ndarray
    bin_gradient: np.ndarray
    shutdown_index: int


# ---------------------------------------------------------------------
# Core neuron model
# ---------------------------------------------------------------------
def resonate_and_fire(
    signal: np.ndarray,
    samples_per_chirp: int,
    omega_r: float,
    phi_v: float,
    thresholds: NeuronThresholds,
) -> NeuronTrace:
    """
    Simulate the resonate-and-fire neuron dynamics for one complex input signal.

    The neuron integrates the input with a rotating complex state. At chirp
    boundaries, an additional Doppler phase term is applied. Several auxiliary
    counters track whether the neuron is resonating strongly enough; these
    counters are later used to decide when the neuron can be turned off.

    Parameters
    ----------
    signal:
        Input radar time series.
    samples_per_chirp:
        Number of fast-time samples per chirp.
    omega_r:
        Normalized range resonance frequency.
    phi_v:
        Normalized inter-chirp phase progression for velocity.
    thresholds:
        Threshold configuration for the counters.

    Returns
    -------
    NeuronTrace
        Complete simulation traces for plotting and inspection.
    """
    n_total = len(signal)

    state = np.zeros(n_total, dtype=np.complex128)

    state_t = 0.0 + 0.0j
    upper_t = 0.0
    lower_t = 0.0
    counter_1_t = 0.0
    counter_2_t = 0.0
    bin_gradient_t = 0.0

    shutdown_index = n_total - 1

    magnitude_proxy = np.zeros(n_total, dtype=np.float64)
    upper = np.zeros(n_total, dtype=np.float64)
    lower = np.zeros(n_total, dtype=np.float64)
    counter_1 = np.zeros(n_total, dtype=np.float64)
    counter_2 = np.zeros(n_total, dtype=np.float64)
    bin_gradient = np.zeros(n_total, dtype=np.float64)

    range_rotation = np.exp(1j * 2 * np.pi * omega_r)
    chirp_rotation = np.exp(1j * 2 * np.pi * phi_v)

    for n in range(n_total):
        # Apply the Doppler phase increment only at chirp boundaries.
        if n % samples_per_chirp == 0 and n > 0:
            state_t = range_rotation * chirp_rotation * state_t + signal[n]
        else:
            state_t = range_rotation * state_t + signal[n]

        # The original implementation uses |Re(s)| + |Im(s)| as a cheap
        # magnitude proxy instead of the Euclidean magnitude |s|.
        mag_t = abs(state_t.real) + abs(state_t.imag)

        # Update the adaptive upper and lower envelopes and related counters.
        if mag_t > upper_t:
            delta = mag_t - upper_t
            upper_t = mag_t
            lower_t += delta
            counter_1_t = 0
            bin_gradient_t += 1

        if mag_t <= lower_t:
            lower_t = mag_t
            bin_gradient_t -= 1

        if lower_t < mag_t < upper_t:
            counter_1_t += 1
            bin_gradient_t -= 0.05

        # Promote c1 into c2 once enough intermediate samples have occurred.
        if counter_1_t > thresholds.theta_c1:
            counter_2_t += 1
            counter_1_t = 0

        # Determine the first point at which the neuron is considered inactive.
        if counter_2_t == thresholds.theta_c2 and shutdown_index == n_total - 1:
            shutdown_index = n
        elif (
            bin_gradient_t < thresholds.theta_grad
            and shutdown_index == n_total - 1
            and n > 128
        ):
            shutdown_index = n

        state[n] = state_t
        magnitude_proxy[n] = mag_t
        upper[n] = upper_t
        lower[n] = lower_t
        counter_1[n] = counter_1_t
        counter_2[n] = counter_2_t
        bin_gradient[n] = bin_gradient_t

    return NeuronTrace(
        state_magnitude=np.abs(state),
        magnitude_proxy=magnitude_proxy,
        upper_bound=upper,
        lower_bound=lower,
        counter_1=counter_1,
        counter_2=counter_2,
        bin_gradient=bin_gradient,
        shutdown_index=shutdown_index,
    )


# ---------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------
def generate_radar_data(config: RadarConfig, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a single-channel synthetic radar example and quantize it lightly.

    The quantization step is preserved from the original script to mimic a
    bounded integer-like signal representation while still storing the result
    as floating point values.
    """
    raw_data = nrsp.datasets.dummy.get_radar_data(
        amps=[1],
        ranges=[config.target_range_bin],
        velocities=[config.target_velocity_bin],
        angles=[0],
        n_samples=config.n_samples,
        n_chirps=config.n_chirps,
        n_channels=1,
        noise_std=config.noise_std,
        rng=rng,
    )

    raw_data = np.asarray(raw_data, dtype=np.float64)

    max_abs = np.max(np.abs(raw_data))
    if max_abs == 0:
        logger.warning("Generated radar data is identically zero.")
        return raw_data

    quantized = (raw_data / max_abs * 127).astype(np.int32) / 127.0
    return quantized


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def compute_counter_axis_limits(
    traces: Sequence[NeuronTrace],
    thresholds: NeuronThresholds,
) -> tuple[tuple[float, float], np.ndarray]:
    """Compute shared y-axis limits for counter subplots."""
    max_counter_val = int(np.ceil(max(thresholds.theta_c1 + 1, thresholds.theta_c2 + 1)))
    ylim = (0.0, max_counter_val + 1.0)
    yticks = np.arange(0, max_counter_val + 1, 5)
    return ylim, yticks


def plot_radar_signal(ax: plt.Axes, radar_signal: np.ndarray, config: RadarConfig) -> None:
    """Plot the simulated radar input signal."""
    ax.plot(radar_signal, color="black")
    ax.set_title(
        rf"\textbf{{Simulated Radar Signal with Noise and Target at "
        rf"($r={config.target_range_bin}$, $v={config.target_velocity_bin}$)}}"
    )
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    sns.despine(ax=ax, trim=True)


def plot_neuron_magnitude(
    ax: plt.Axes,
    trace: NeuronTrace,
    neuron: NeuronSpec,
    radar_config: RadarConfig,
    thresholds: NeuronThresholds,
    colors: Sequence,
    legend_loc: str,
) -> None:
    """Plot magnitude-related neuron dynamics."""
    t = np.arange(len(trace.magnitude_proxy))
    idx = min(trace.shutdown_index, len(t) - 1)

    active = slice(None, idx)
    inactive = slice(idx, None)

    ax.plot(t[active], trace.magnitude_proxy[active], label=r"$|s(t)|$", color=colors[0])
    ax.plot(t[active], trace.upper_bound[active], label="Upper", color=colors[1])
    ax.plot(t[active], trace.lower_bound[active], label="Lower", color=colors[2])
    ax.plot(
        t[active],
        trace.bin_gradient[active],
        label=rf"$m$ with $\Theta_m = {thresholds.theta_grad}$",
        color=colors[3],
    )

    # Fade the curves after the inferred shutdown point to indicate inactivity.
    ax.plot(t[inactive], trace.magnitude_proxy[inactive], color=colors[0], alpha=0.1)
    ax.plot(t[inactive], trace.upper_bound[inactive], color=colors[1], alpha=0.1)
    ax.plot(t[inactive], trace.lower_bound[inactive], color=colors[2], alpha=0.1)
    ax.plot(t[inactive], trace.bin_gradient[inactive], color=colors[3], alpha=0.1)

    ax.set_title(
        rf"\textbf{{{neuron.name}}} Neuron "
        rf"($\omega_r={neuron.omega_r * radar_config.n_samples:.1f}, "
        rf"\phi_v={neuron.phi_v * radar_config.n_chirps:.1f}$)"
    )
    ax.set_ylabel("Magnitude")
    ax.grid(True)
    ax.legend(
        loc=legend_loc,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
    )
    sns.despine(ax=ax, trim=True)


def plot_neuron_counters(
    ax: plt.Axes,
    trace: NeuronTrace,
    thresholds: NeuronThresholds,
    colors: Sequence,
    counter_ylim: tuple[float, float],
    counter_yticks: np.ndarray,
) -> None:
    """Plot the internal counters used by the neuron."""
    t = np.arange(len(trace.counter_1))
    idx = min(trace.shutdown_index, len(t) - 1)

    active = slice(None, idx)
    inactive = slice(idx, None)

    ax.plot(
        t[active],
        trace.counter_1[active],
        label=fr"$c_1$ with $\Theta_{{c_1}}={thresholds.theta_c1}$",
        color=colors[4],
    )
    ax.plot(
        t[active],
        trace.counter_2[active],
        label=fr"$c_2$ with $\Theta_{{c_2}}={thresholds.theta_c2}$",
        color=colors[5],
    )

    ax.plot(t[inactive], trace.counter_1[inactive], color=colors[4], alpha=0.1)
    ax.plot(t[inactive], trace.counter_2[inactive], color=colors[5], alpha=0.1)

    ax.set_ylabel("Counter")
    ax.set_ylim(counter_ylim)
    ax.set_yticks(counter_yticks)
    ax.grid(True)
    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
    )
    sns.despine(ax=ax, trim=True)


# ---------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------
def main() -> None:
    """Run the full simulation and save the resulting figure."""
    radar_config = RadarConfig()
    thresholds = NeuronThresholds()

    omega_base = radar_config.target_range_bin / radar_config.n_samples
    phi_base = radar_config.target_velocity_bin / radar_config.n_chirps

    neuron_specs = [
        NeuronSpec(omega_r=omega_base, phi_v=phi_base, name="Resonating"),
        NeuronSpec(
            omega_r=omega_base + 0.5 / radar_config.n_samples,
            phi_v=phi_base,
            name="Non-resonating",
        ),
    ]

    radar_data = generate_radar_data(radar_config, RNG)[0].flatten()

    traces = [
        resonate_and_fire(
            signal=radar_data,
            samples_per_chirp=radar_config.n_samples,
            omega_r=spec.omega_r,
            phi_v=spec.phi_v,
            thresholds=thresholds,
        )
        for spec in neuron_specs
    ]

    counter_ylim, counter_yticks = compute_counter_axis_limits(traces, thresholds)

    n_neurons = len(neuron_specs)
    fig = plt.figure(figsize=(10, 3 + n_neurons * 3))
    gs = gridspec.GridSpec(
        1 + 2 * n_neurons,
        1,
        height_ratios=[1] + [1 if i % 2 == 1 else 2 for i in range(2 * n_neurons)],
    )
    axes = [fig.add_subplot(gs[i, 0]) for i in range(1 + 2 * n_neurons)]
    colors = sns.color_palette("deep")

    plot_radar_signal(axes[0], radar_data, radar_config)

    for i, (spec, trace) in enumerate(zip(neuron_specs, traces)):
        ax_mag = axes[1 + i * 2]
        ax_cnt = axes[1 + i * 2 + 1]

        legend_loc = "upper left" if i == 0 else "lower right"

        plot_neuron_magnitude(
            ax=ax_mag,
            trace=trace,
            neuron=spec,
            radar_config=radar_config,
            thresholds=thresholds,
            colors=colors,
            legend_loc=legend_loc,
        )
        plot_neuron_counters(
            ax=ax_cnt,
            trace=trace,
            thresholds=thresholds,
            colors=colors,
            counter_ylim=counter_ylim,
            counter_yticks=counter_yticks,
        )

    axes[-1].set_xlabel(r"Time Index $t$")
    sns.despine(ax=axes[-1], trim=True)

    plt.tight_layout()

    output_path = Path("out/spinr_dynamics_with_ags.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)

    logger.info("Saved figure to %s", output_path)


if __name__ == "__main__":
    main()