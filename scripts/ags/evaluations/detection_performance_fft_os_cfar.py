"""
Evaluate range-Doppler FFT detections using 2D OS-CFAR on the BBM dataset.

This script:
1. Loads a BBM dataset and its metadata.
2. Optionally applies frame / target omission filters from a JSON file.
3. Computes range-Doppler FFT responses.
4. Runs a grid search over OS-CFAR parameters:
   - order-statistic index k
   - threshold scaling alpha
   - guard / reference cell configurations
5. Evaluates detections against ground-truth target maps.
6. Saves the aggregated metrics as a CSV file.

The script is intended as a standalone evaluation entry point.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import nrsp.datasets.infineon_bbm as bbm
from nrsp.metrics.cfar_metrics import apply_detection_area
from nrsp.metrics.evaluator import CfarEvaluator
from nrsp.rsp.cfar import os_cfar_2d_detections
from nrsp.rsp.fft import range_doppler_fft
from nrsp.utils.log import get_logger


logger = get_logger("logs/")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class EvalConfig:
    """Configuration for range-Doppler FFT + OS-CFAR evaluation."""

    k_values: tuple[int, ...] = tuple(range(1, 20))
    alpha_values: tuple[float, ...] = tuple(np.linspace(1.5, 8.0, 20))

    # Each entry is ((guard_x, guard_y), (ref_x, ref_y)).
    guard_ref_cells: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = (
        ((0, 0), (4, 2)),
        ((0, 0), (8, 4)),
        ((2, 1), (4, 2)),
        ((2, 1), (8, 4)),
    )

    max_distance_bins: int = 215
    detection_area_shape: tuple[int, int] = (3, 3)

    bbm_path: str = (
        "/home/nreeb/data/BBM/paper0/"
        "level0_random_1dmax_randomrcs_8targets_32chirps_new/seed_0"
    )
    filter_file: str = (
        "nrsp/datasets/bbm/"
        "seed_0__MAX_TARGET_VALUE=1500.0_MIN_TARGET_VALUE=0.45.json"
    )
    output_csv: str = "out/rd_fft_os_cfar_fast_train.csv"

    # Preserve the original behavior: stop after the first five sequences.
    max_sequences: int = 5


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def load_filter_config(filter_file: str) -> dict[str, Any]:
    """
    Load the optional frame / target omission configuration.

    The filter file may exclude complete frames or individual targets from
    evaluation. If the file is missing, evaluation proceeds unchanged.
    """
    if not os.path.exists(filter_file):
        logger.warning(
            "Filter file %s not found. No frames or targets will be omitted.",
            filter_file,
        )
        return {}

    with open(filter_file, "r", encoding="utf-8") as f:
        filter_config = json.load(f)

    logger.info("Loaded filter configuration from %s", filter_file)
    return filter_config


def build_target_maps(
    targets: Any,
    n_samples: int,
    n_frames: int,
    n_channels: int,
    n_chirps: int,
    dataset_config: Any,
) -> np.ndarray:
    """
    Convert BBM targets into frame-wise range-Doppler target maps.

    The BBM helper first returns a target cube with dimensions:
        (n_frames, n_distances, n_angles, n_velocities)

    The angular dimension is then collapsed to obtain a binary
    range-Doppler map:
        (n_frames, n_distances, n_velocities)
    """
    target_cube = bbm.targets_to_cube(
        targets=targets,
        n_distances=n_samples // 2,
        n_frames=n_frames,
        n_angles=n_channels,
        n_velocities=n_chirps,
        config=dataset_config,
    )
    rd_target_maps = np.sum(target_cube, axis=2).astype(bool)
    return rd_target_maps


def get_valid_distance_bins(max_distance_bins: int, rd_target_maps: np.ndarray) -> range:
    """
    Construct a safe distance-bin range for the current dataset.

    This avoids out-of-bounds indexing when the configured maximum exceeds the
    number of available range bins in the target map.
    """
    n_distance_bins = rd_target_maps.shape[1]
    return range(min(max_distance_bins, n_distance_bins))


def apply_frame_target_filter(
    target_map: np.ndarray,
    frame_filter: dict[str, Any],
    distance_bins: range,
) -> np.ndarray:
    """
    Remove omitted targets from a frame-local target map.

    Parameters
    ----------
    target_map:
        Ground-truth map restricted to the evaluated distance bins.
    frame_filter:
        Frame-specific omission instructions from the filter JSON.
    distance_bins:
        Range-bin subset currently being evaluated.

    Returns
    -------
    np.ndarray
        Filtered copy of the target map.
    """
    filtered_map = target_map.copy()

    omit_targets = frame_filter.get("omit_targets", [])
    for range_idx, vel_idx in omit_targets:
        if 0 <= range_idx < len(distance_bins) and 0 <= vel_idx < filtered_map.shape[1]:
            filtered_map[range_idx, vel_idx] = False

    return filtered_map


def evaluate_frame_os_cfar(
    fft_rd_frame: np.ndarray,
    target_map: np.ndarray,
    detection_area: np.ndarray,
    distance_bins: range,
    guard_ref_cells: Iterable[tuple[tuple[int, int], tuple[int, int]]],
    k_values: tuple[int, ...],
    alpha_values: tuple[float, ...],
    evaluator: CfarEvaluator,
) -> None:
    """
    Evaluate one frame over the full OS-CFAR hyperparameter grid.

    Notes
    -----
    The FFT output is transposed before distance indexing to match the
    target-map orientation expected by the evaluator.
    """
    target_map_extended = apply_detection_area(target_map, detection_area)
    rd_response = fft_rd_frame.T[distance_bins]

    for guard_cells, ref_cells in guard_ref_cells:
        cfar_out = os_cfar_2d_detections(
            rd_response,
            guard_cells,
            ref_cells,
            k_values,
            alpha_values,
            wrap_y=True,
        )

        for k_idx, k in enumerate(k_values):
            for alpha_idx, alpha in enumerate(alpha_values):
                evaluator.eval(
                    cfar_out[:, :, alpha_idx, k_idx],
                    target_map,
                    target_map_extended,
                    k=k,
                    alpha=alpha,
                    guard_cells=guard_cells,
                    ref_cells=ref_cells,
                )


# ---------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------
def main() -> None:
    """Run the full OS-CFAR grid-search evaluation and save the results."""
    config = EvalConfig()
    detection_area = np.ones(config.detection_area_shape, dtype=int)

    filter_config = load_filter_config(config.filter_file)

    evaluator = CfarEvaluator(
        keys=["k", "alpha", "guard_cells", "ref_cells"],
        detection_area=detection_area,
    )

    dataset_config = bbm.load_config(config.bbm_path)
    bbm_iterator = bbm.BBMIterator(config.bbm_path)

    total_sequences = len(bbm_iterator.indices)
    n_sequences_to_process = min(config.max_sequences, total_sequences)

    for sequence_idx in range(n_sequences_to_process):
        radar_data, targets = next(bbm_iterator)

        n_frames, n_chirps, n_samples, n_channels = radar_data.shape

        # Preprocess the raw radar signal and keep only the first channel.
        radar_data = bbm.preprocess_data(radar_data)
        radar_data = radar_data[:, :, :, 0:1]

        # Convert annotations to frame-wise binary range-Doppler maps.
        rd_target_maps = build_target_maps(
            targets=targets,
            n_samples=n_samples,
            n_frames=n_frames,
            n_channels=n_channels,
            n_chirps=n_chirps,
            dataset_config=dataset_config,
        )

        # Clamp the evaluated distance-bin range to what is actually available.
        distance_bins = get_valid_distance_bins(
            config.max_distance_bins,
            rd_target_maps,
        )

        # Compute range-Doppler FFT responses for the selected channel.
        fft_rd = range_doppler_fft(radar_data)
        fft_rd = fft_rd[:, :, :, 0]

        # Match the current file against optional filter settings.
        current_file = os.path.basename(bbm_iterator.get_current_data_filename())
        file_filters = filter_config.get(current_file, {})

        for frame_idx in range(n_frames):
            print(
                f"\rSequence {sequence_idx + 1}/{n_sequences_to_process}, "
                f"frame {frame_idx + 1}/{n_frames}",
                end="",
                flush=True,
            )

            frame_filter = file_filters.get(str(frame_idx), {})

            # Skip the entire frame if requested by the filter configuration.
            if frame_filter.get("omit_frame", False):
                continue

            # Restrict the target map to the evaluated range interval.
            target_map = rd_target_maps[frame_idx][distance_bins]

            # Optionally remove specific ground-truth targets from evaluation.
            target_map = apply_frame_target_filter(
                target_map=target_map,
                frame_filter=frame_filter,
                distance_bins=distance_bins,
            )

            evaluate_frame_os_cfar(
                fft_rd_frame=fft_rd[frame_idx],
                target_map=target_map,
                detection_area=detection_area,
                distance_bins=distance_bins,
                guard_ref_cells=config.guard_ref_cells,
                k_values=config.k_values,
                alpha_values=config.alpha_values,
                evaluator=evaluator,
            )

        print()
        progress = 100.0 * (sequence_idx + 1) / n_sequences_to_process
        logger.info("Processed %.2f %% of selected sequences", progress)

    global_params = {
        "timestamp": int(time.time()),
        "dataset": config.bbm_path,
    }

    output_path = Path(config.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    evaluator.save(str(output_path), global_params=global_params)
    logger.info("Saved evaluation results to %s", output_path)


if __name__ == "__main__":
    main()