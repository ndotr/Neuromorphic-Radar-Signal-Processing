import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def compute_tick_interval(x):
    """
    Compute a human-friendly tick interval greater than or equal to `x`,
    based on a predefined set of standard intervals. Suitable for time axes.

    Parameters
    ----------
    x : float
        Minimum desired spacing between ticks (e.g., total_range / 100)

    Returns
    -------
    float
        A computed tick interval appropriate for plotting.
    """
    standard_intervals = [
        0.1, 0.2, 0.5,
        1, 2, 5,
        10, 15, 20, 30,
        60, 120, 300, 600, 900, 1800, 3600  # up to 1 hour
    ]
    for interval in standard_intervals:
        if x <= interval:
            return interval
    # Fallback for very large ranges
    exponent = np.ceil(np.log10(x))
    return 10 ** exponent


# Power plotting
def plot_power_large_end_aligned(
    data,
    time_segments,
    proc_duration=None,        # duration of preparation segment
    exec_duration=None,        # duration of execution segment
    filename="plot.pdf",
    cutoff=1.0,
    signals=["vddm", "vddio", "vdd", "total"],
    logger=None
):
    """
    End-aligned power plot with run split into preparation, execution, processing
    based on user-provided durations. The processing segment ends at the original
    run segment end.
    """
    # ----------------------------
    # Time axis (end-aligned)
    # ----------------------------
    time_us = np.asarray(data["time"])
    time_s = time_us * 1e-6
    time_s = time_s - time_s[-1]  # last data sample at t=0

    data_start = time_s[0]
    data_end = 0.0

    # ----------------------------
    # Sanity check
    # ----------------------------
    if all(np.all(np.isnan(data[sig])) for sig in signals):
        raise ValueError("No data available in the specified signals.")

    # ----------------------------
    # Shift segments relative to last segment
    # ----------------------------
    seg_times = list(time_segments.values())
    t_seg_end = seg_times[-1]
    shifted_segments = {name: t - t_seg_end for name, t in time_segments.items()}
    items = sorted(shifted_segments.items(), key=lambda x: x[1])
    segments = {sig: [] for sig in signals}

    # ----------------------------
    # Prompt user for durations if not provided
    # ----------------------------
    while exec_duration is None:
        try:
            exec_duration = float(input("Enter execution duration (s): "))
        except ValueError:
            print("Invalid input. Must be numeric.")

    while proc_duration is None:
        try:
            proc_duration = float(input("Enter processing duration (s): "))
        except ValueError:
            print("Invalid input. Must be numeric.")

    # ----------------------------
    # Build segments
    # ----------------------------
    for i in range(len(items) - 1):
        _, start = items[i]
        prev_name, next_start = items[i + 1]

        seg_start = start + cutoff
        seg_end = next_start - cutoff

        # Force final segment to end at last data sample
        if i == len(items) - 2:
            seg_end = data_end

        # Clip to data
        raw_start, raw_end = seg_start, seg_end
        seg_start = max(seg_start, data_start)
        seg_end = min(seg_end, data_end)
        if logger:
            if seg_start != raw_start or seg_end != raw_end:
                logger.debug(f"[CLIP] Segment '{prev_name}': [{raw_start:.3f}, {raw_end:.3f}] → [{seg_start:.3f}, {seg_end:.3f}]")
            if seg_start >= seg_end:
                logger.debug(f"[SKIP] Segment '{prev_name}' discarded (invalid after clipping)")
                continue

        # ----------------------------
        # Split 'run' using user-defined durations
        # ----------------------------
        if prev_name == "run" and exec_duration != -1:
            run_total = seg_end - seg_start
            prep_duration = run_total - (exec_duration + proc_duration)
            if prep_duration < 0:
                if logger:
                    logger.debug("[WARN] Provided execution+processing durations exceed run segment!")
                prep_duration = max(0, run_total - (exec_duration + proc_duration))
                exec_duration = min(exec_duration, run_total - proc_duration)
                proc_duration = min(proc_duration, run_total - exec_duration)

            prep_end = seg_start + prep_duration
            exec_end = prep_end + exec_duration
            proc_end = exec_end + proc_duration
            run_end = seg_end

            sub_segments = [
                ("run_preparation", seg_start+cutoff, prep_end-cutoff),
                ("run_execution", prep_end+cutoff, exec_end-cutoff),
                ("run_processing", exec_end+cutoff, run_end-cutoff)
            ]

            if logger:
                logger.debug(f"[SPLIT] 'run' split into preparation ({seg_start:.3f}-{prep_end:.3f}), "
                  f"execution ({prep_end:.3f}-{exec_end:.3f}), "
                  f"processing ({exec_end:.3f}-{run_end:.3f})")

        else:
            sub_segments = [(prev_name, seg_start, seg_end)]

        # ----------------------------
        # Compute statistics
        # ----------------------------
        for sub_name, sub_start, sub_end in sub_segments:

            sub_mask = (time_s >= sub_start) & (time_s <= sub_end)
            if np.count_nonzero(sub_mask) < 2 and logger:
                logger.debug(f"[SKIP] Segment '{sub_name}' too short ({sub_end - sub_start:.3f}s)")
                continue

            for sig in signals:
                vals = np.asarray(data[sig])[sub_mask]
                segments[sig].append((
                    sub_name,
                    sub_start,
                    sub_end,
                    np.nanmean(vals),
                    np.nanstd(vals)
                ))

    # ----------------------------
    # Tick spacing
    # ----------------------------
    total_time = data_end - data_start
    tick_spacing = compute_tick_interval(abs(total_time) / 20)

    # ----------------------------
    # Plotting
    # ----------------------------
    n_signals = len(signals)
    fig, axes = plt.subplots(nrows=n_signals, figsize=(7, 3*n_signals), sharex=True)
    colors = matplotlib.cm.tab10.colors
    if n_signals == 1:
        axes = [axes]

    for ax, sig in zip(axes, signals):
        ax.plot(time_s, data[sig], color="gray", linewidth=0.8)
        ax.set_title(sig, fontsize=9)
        ax.set_ylabel("Power (W)", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacing))

        for seg, c in zip(segments[sig], colors):
            name, start, end, mean, std = seg
            ax.axvline(start, linestyle="--", color=c, linewidth=0.8)
            ax.axvline(end, linestyle="--", color=c, linewidth=0.8)
            ax.hlines(mean, start, end, colors=c, linewidth=1.5)
            ax.fill_between(
                [start, end],
                mean - std,
                mean + std,
                color=c,
                alpha=0.4,
                label=f"{name}: {mean:.2f} ± {std:.2f}, {(end-start):.1f}s"
            )
        ax.legend(fontsize=6, loc="lower right")

    axes[-1].set_xlabel("Time relative to end (s)")
    plt.suptitle("Power Telemetry (End-aligned, run split by duration)", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename)
    plt.close(fig)



def plot_power_large(data, time_segments, filename="plot.pdf", cutoff=1.0,
               signals=["vddm", "vddio", "vdd", "total"]):
    """
    Plot power telemetry with segment statistics in a row-wise layout.

    Parameters
    ----------
    data : dict
        Dictionary with 'time' (µs) and measurement arrays.
    time_segments : dict
        Mapping {event_name: start_time_in_s}. Last item ignored.
    filename : str
        Output file path (PDF recommended).
    cutoff : float
        Trim (s) at start and end of each segment.
    signals : list
        Keys in `data` to plot.
    """

    # --- Time axis in seconds ---
    time_s = (np.array(data['time']) - data['time'][0]) * 1e-6

    # --- Sanity check ---
    if all(np.all(np.isnan(data[sig])) for sig in signals):
        raise ValueError("No data available in the specified signals.")

    # --- Prepare segments ---
    items = sorted(time_segments.items(), key=lambda x: x[1])
    segments = {sig: [] for sig in signals}

    for (_, start), (name, next_start) in zip(items[:-2], items[1:-1]):
        seg_start = start + cutoff
        seg_end = next_start - cutoff
        mask = (time_s >= seg_start) & (time_s <= seg_end)

        for sig in signals:
            vals = np.array(data[sig])[mask]
            segment_mean = np.nanmean(vals)
            segment_std = np.nanstd(vals)
            segments[sig].append((name, seg_start, seg_end, segment_mean, segment_std))

    # --- Compute dynamic tick spacing ---
    total_time = time_s[-1] - time_s[0]
    raw_tick_spacing = total_time / 20
    tick_spacing = compute_tick_interval(raw_tick_spacing)

    # --- Plot horizontally (rows = signals) ---
    n_signals = len(signals)
    fig, axes = plt.subplots(nrows=n_signals, figsize=(7, 3*n_signals), sharex=True)
    colors = matplotlib.cm.tab10.colors

    if n_signals == 1:
        axes = [axes]

    for ax, sig in zip(axes, signals):
        y = data[sig]
        ax.set_ylabel("Power (W)", fontsize=9)
        ax.set_title(sig, fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.plot(time_s, y, linestyle='-', linewidth=0.8, color="gray", label=sig, zorder=0)
        ax.tick_params(axis='both', which='major', labelsize=8)

        # Apply dynamic tick spacing
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacing))

        # Overlay segment statistics
        for seg, c in zip(segments[sig], colors):
            name, start, end, mean, std = seg
            ax.axvline(x=start, linestyle='--', color=c, linewidth=0.8)
            ax.axvline(x=end, linestyle='--', color=c, linewidth=0.8)
            ax.hlines(mean, start, end, colors=c, linestyles='-', linewidth=1.5)
            ax.fill_between([start, end], mean-std, mean+std, color=c, alpha=0.5, zorder=3,
                            label=f"{name}: {mean:.2f} ± {std:.2f}, {end-start:.1f}s")
            

        ax.legend(loc='lower left',
                  fontsize=6,
                  frameon=True,                # enable the frame
                  facecolor='white',           # legend background color
                  edgecolor='black',           # optional: border color
                  framealpha=1.0               # full opacity
                )

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    plt.suptitle("Power Telemetry Over Time", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename)
    plt.close(fig)

def plot_ft1d_comparison(nx_output, np_output, mean_error, std_error, output_path, title_prefix="FT Comparison"):
    """
    Plot NxKernel DFT vs NumPy FFT outputs (1D) together with error as shaded area.

    Parameters
    ----------
    nx_output : np.ndarray
        NxKernel FFT output (1D array).
    np_output : np.ndarray
        NumPy FFT reference output (1D array, same length as nx_output).
    mean_error : float
        Mean error to display in title/label.
    std_error : float
        Std deviation of error to display.
    output_path : str
        Path to save the figure (PDF or PNG).
    title_prefix : str
        Prefix for plot titles.
    """
    eps = 1e-12
    x = np.arange(len(nx_output))

    # --- Normalize ---
    np_scaled = np.abs(np_output) / (np.max(np.abs(np_output)) + eps)
    nx_scaled = np.abs(nx_output) / (np.max(np.abs(nx_output)) + eps)

    # --- Relative error ---
    error_map = np.abs(nx_scaled - np_scaled) #/ (np.maximum(np_scaled, eps))

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=True)

    # --- Top: NxKernel vs NumPy FFT ---
    axs[0].plot(np_scaled, color='tab:orange', label='NumPy FFT', linewidth=1.0)
    axs[0].plot(nx_scaled, '-.', color='tab:blue', label='NxKernel FFT', linewidth=0.5, alpha=0.8)
    axs[0].set_ylabel("Normalized Amplitude")
    axs[0].set_title(f"{title_prefix} - FFT Comparison", fontsize=10)
    axs[0].grid(True, linestyle='--', alpha=0.5)
    axs[0].legend(loc='upper right', fontsize=8)

    # --- Bottom: Absolute error ---
    axs[1].plot(x, error_map, color='tab:orange', linewidth=1.2, label='Current error')
    # Horizontal mean line across same x-range
    axs[1].plot(x, np.full_like(x, mean_error), color='tab:red', linestyle='--', linewidth=1.5, label=f'Mean Error ({mean_error:.2e})')
    # Std deviation shaded area
    axs[1].fill_between(x, mean_error - std_error, mean_error + std_error, color='tab:red', alpha=0.2, label=f'Std ± ({std_error:.2e})')

    axs[1].set_xlabel("Sample / Bin")
    axs[1].set_ylabel("Absolute Error")
    axs[1].set_title("Error Map", fontsize=10)
    axs[1].grid(True, linestyle='--', alpha=0.5)
    axs[1].legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

def plot_ft2d_comparison(nx_output, np_output, mean_error, std_error, output_path):
    """
    Generate and save a side-by-side comparison plot of
    NumPy FFT, custom Nx FFT, their absolute error, and error std,
    with global mean and std error reflected visually via colorbar scaling.

    Parameters
    ----------
    nx_output : np.ndarray
        Scaled FFT result from Nx kernel.
    np_output : np.ndarray
        Scaled FFT result from NumPy.
    error_map : np.ndarray
        Absolute error between Nx and NumPy FFT results.
    std_map : np.ndarray
        Standard deviation of error (2D or 1D, already precomputed).
    mean_error : float
        Mean of the error map.
    std_error : float
        Standard deviation of the error map.
    output_path : str
        Path to save the output PDF.
    """
    fig, axs = plt.subplots(ncols=3, figsize=(16, 12))
    axs = axs.ravel()

    eps = 1e-12
    x = np.arange(len(nx_output))

    # --- Normalize ---
    np_scaled = np_output / (np.max(np.abs(np_output)) + eps)
    nx_scaled = nx_output / (np.max(np.abs(nx_output)) + eps)

    # --- Relative error ---
    error_map = np.abs(nx_scaled - np_scaled) #/ (np.maximum(np_scaled, eps))

    # Define plots: use vmin/vmax for error and std to relate to global metrics
    plots = [
        (np_scaled, "FFT (NumPy)", "viridis", "Normalized amplitude", None, None),
        (nx_scaled, "NxKernel rFFT+SpiNR+NCI", "viridis", "Normalized amplitude", None, None),
        (error_map, f"Absolute Error\nMean={mean_error:.3e}, Std={std_error:.3e}", "inferno", "Error",
         max(0, mean_error - 3*std_error), mean_error + 3*std_error),
    ]

    for ax, (data, title, cmap, cbar_label, vmin, vmax) in zip(axs, plots):
        im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label, fontsize=10)
        ax.set_xlabel("Velocity Bin", fontsize=10)
        ax.set_ylabel("Range Bin", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)