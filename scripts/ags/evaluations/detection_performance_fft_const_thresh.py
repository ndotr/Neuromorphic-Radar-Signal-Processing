"""
Evaluate range-Doppler FFT detections using a constant threshold on the BBM dataset.

This script:
1. Loads a BBM dataset and its metadata.
2. Optionally applies a frame / target filter from a JSON file.
3. Computes range-Doppler FFT responses.
4. Applies a constant detection threshold.
5. Evaluates detections against ground-truth target maps using CFAR-style metrics.
6. Saves the aggregated evaluation results as a CSV file.

The script is intentionally written as a standalone evaluation entry point rather
than as a general-purpose library module.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import nrsp.datasets.infineon_bbm as bbm
from nrsp.metrics.cfar_metrics import apply_detection_area
from nrsp.metrics.evaluator import CfarEvaluator
from nrsp.rsp.fft import range_doppler_fft
from nrsp.utils.log import get_logger


logger = get_logger("logs/")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class EvalConfig:
    """Configuration for constant-threshold range-Doppler FFT evaluation."""

    thresholds: tuple[float, ...] = (2.894736842105263,)
    distance_bins: range = range(215)
    detection_area: tuple[int, int] = (3, 3)

    bbm_path: str = (
        "/home/nreeb/data/BBM/paper0/"
        "level0_random_1dmax_randomrcs_8targets_32chirps_new/seed_1"
    )
    filter_file: str = (
        "nrsp/datasets/bbm/"
        "seed_1__MAX_TARGET_VALUE=1500.0_MIN_TARGET_VALUE=0.45.json"
    )
    output_csv: str = "results/ags/detection_performance/fft_const_thresh.csv"


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def load_filter_config(filter_file: str) -> dict[str, Any]:
    """
    Load the optional frame / target omission configuration.

    The filter file allows excluding entire frames or individual targets from
    evaluation. If the file does not exist, evaluation proceeds without any
    omissions.
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

    The angular dimension is then summed out to obtain a boolean
    range-Doppler target map:
        (n_frames, n_distances, n_velocities)
    """
    target_cube = bbm.targets_to_cube(
        targets=targets,
        n_distances=n_samples,
        n_frames=n_frames,
        n_angles=n_channels,
        n_velocities=n_chirps,
        config=dataset_config,
    )

    rd_target_maps = np.sum(target_cube, axis=2).astype(bool)
    return rd_target_maps


def apply_frame_target_filter(
    target_map: np.ndarray,
    frame_filter: dict[str, Any],
    distance_bins: range,
) -> np.ndarray:
    """
    Apply per-frame target omissions to a target map.

    Parameters
    ----------
    target_map:
        Boolean target map restricted to the selected distance bins.
    frame_filter:
        Dictionary that may contain an 'omit_targets' entry with
        (range_idx, vel_idx) pairs.
    distance_bins:
        Evaluated distance-bin subset.

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


def evaluate_frame(
    fft_rd_frame: np.ndarray,
    target_map: np.ndarray,
    thresholds: tuple[float, ...],
    detection_area: np.ndarray,
    distance_bins: range,
    evaluator: CfarEvaluator,
) -> None:
    """
    Evaluate one frame for all configured thresholds.

    Notes
    -----
    The FFT output is transposed before distance-bin indexing to match the
    target-map orientation expected by the evaluator.
    """
    target_map_extended = apply_detection_area(target_map, detection_area)

    for threshold in thresholds:
        detections = fft_rd_frame.T[distance_bins] > threshold
        evaluator.eval(
            detections,
            target_map,
            target_map_extended,
            threshold=threshold,
        )


# ---------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------
def main() -> None:
    """Run the full evaluation and save the aggregated metrics."""
    config = EvalConfig()
    detection_area = np.ones(config.detection_area, dtype=int)

    filter_config = load_filter_config(config.filter_file)

    evaluator = CfarEvaluator(
        keys=["threshold"],
        detection_area=detection_area,
    )

    dataset_config = bbm.load_config(config.bbm_path)
    bbm_iterator = bbm.BBMIterator(config.bbm_path)

    n_sequences = len(bbm_iterator.indices)

    for sequence_idx in range(n_sequences):
        radar_data, targets = next(bbm_iterator)

        n_frames, n_chirps, n_samples, n_channels = radar_data.shape

        # Preprocess raw radar data and restrict evaluation to the first channel.
        radar_data = bbm.preprocess_data(radar_data)
        radar_data = radar_data[:, :, :, 0:1]

        # Build binary ground-truth range-Doppler maps.
        rd_target_maps = build_target_maps(
            targets=targets,
            n_samples=n_samples//2,
            n_frames=n_frames,
            n_channels=n_channels,
            n_chirps=n_chirps,
            dataset_config=dataset_config,
        )

        # Compute the range-Doppler FFT response for the selected channel.
        fft_rd = range_doppler_fft(radar_data)
        fft_rd = fft_rd[:, :, :, 0]

        # The current source filename is used to look up optional frame filters.
        current_file = os.path.basename(bbm_iterator.get_current_data_filename())
        file_filters = filter_config.get(current_file, {})

        for frame_idx in range(n_frames):
            frame_filter = file_filters.get(str(frame_idx), {})

            # Skip entire frames if explicitly marked for omission.
            if frame_filter.get("omit_frame", False):
                continue

            # Restrict the target map to the evaluated distance bins.
            target_map = rd_target_maps[frame_idx][config.distance_bins]

            # Optionally remove selected ground-truth targets from the frame.
            target_map = apply_frame_target_filter(
                target_map=target_map,
                frame_filter=frame_filter,
                distance_bins=config.distance_bins,
            )

            evaluate_frame(
                fft_rd_frame=fft_rd[frame_idx],
                target_map=target_map,
                thresholds=config.thresholds,
                detection_area=detection_area,
                distance_bins=config.distance_bins,
                evaluator=evaluator,
            )

        progress = 100.0 * (sequence_idx + 1) / n_sequences
        logger.info("Processed %.2f %% of dataset", progress)

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