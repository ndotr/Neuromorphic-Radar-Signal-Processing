"""
Evaluate the range-Doppler SpiNR turn-off model on the BBM dataset.

This script:
1. Loads a BBM dataset and optional frame / target omission filters.
2. Converts radar data to the floating-point representation expected by the
   CUDA SpiNR implementation.
3. Runs a grid search over turn-off hyperparameters.
4. Extracts resonant neurons, gradients, and inactivity statistics.
5. Evaluates active-neuron detections against ground-truth target maps.
6. Saves aggregated metrics as a CSV file.

The script is intended as a standalone evaluation entry point for systematic
hyperparameter sweeps.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from nrsp.algs.cu.spinr_cfar.rd_log_os_cfar_float import (
    Cu_RD_SpiNR_LogOsCfar_Float_Model,
)
import nrsp.datasets.infineon_bbm as bbm
from nrsp.metrics.cfar_metrics import apply_detection_area
from nrsp.metrics.evaluator import CfarTurnoffEvaluator
from nrsp.utils.cu import complex_to_float
from nrsp.utils.log import get_logger


logger = get_logger("logs/")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class EvalConfig:
    """Configuration for the SpiNR turn-off grid search."""

    alpha_grd: float = 1e-4
    grad_threshold: float = 0.0

    thresh_silents: tuple[int, ...] = tuple(np.arange(100, 512, 15))
    thresh_silent_chirps: tuple[int, ...] = tuple(range(1, 61, 2))
    monotonicity_thresholds: tuple[int, ...] = (-19,)
    t_monotonicities: tuple[int, ...] = (0,)

    max_distance_bins: int = 215
    detection_area_shape: tuple[int, int] = (3, 3)

    bbm_path: str = (
        "/home/mgrabmann/data/BBM/"
        "level0_random_1dmax_randomrcs_8targets_32chirps_new/seed_0"
    )
    filter_file: str = (
        "nrsp/datasets/bbm/"
        "seed_0__MAX_TARGET_VALUE=1500.0_MIN_TARGET_VALUE=0.45.json"
    )
    output_csv: str = "out/spinr_v2_fixed_m_gridseach_c1_c2.csv"

    kernel_version: str = "v2"
    input_type: str = "real"

    # Preserved from the original script.
    t_enc: int = 512 * 31

    # Parameters required by the model constructor but unused in this setup.
    tau: int = 0
    thresh: int = 0

    # Disable the encoding stage exactly as in the original script.
    encoding_func: int = -1


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def load_filter_config(filter_file: str) -> dict[str, Any]:
    """
    Load the optional frame / target omission configuration.

    If the file is missing, the evaluation proceeds without filtering.
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
    Convert BBM target annotations into frame-wise range-Doppler target maps.

    The BBM helper returns a cube with dimensions:
        (n_frames, n_distances, n_angles, n_velocities)

    The angular dimension is then summed out, yielding:
        (n_frames, n_distances, n_velocities)
    """
    target_cube = bbm.targets_to_cube(
        targets=targets,
        n_distances=n_samples // 2,
        n_frames=n_frames,
        n_channels=n_channels,
        n_velocities=n_chirps,
        config=dataset_config,
    )
    return np.sum(target_cube, axis=2).astype(bool)


def get_valid_distance_bins(max_distance_bins: int, rd_target_maps: np.ndarray) -> range:
    """
    Construct a safe range-bin subset for the current dataset.

    This prevents out-of-bounds indexing when the configured maximum exceeds
    the number of available distance bins.
    """
    n_distance_bins = rd_target_maps.shape[1]
    return range(min(max_distance_bins, n_distance_bins))


def apply_frame_target_filter(
    target_map: np.ndarray,
    frame_filter: dict[str, Any],
    distance_bins: range,
) -> np.ndarray:
    """
    Remove omitted targets from a frame-local ground-truth target map.
    """
    filtered_map = target_map.copy()

    omit_targets = frame_filter.get("omit_targets", [])
    for range_idx, vel_idx in omit_targets:
        if 0 <= range_idx < len(distance_bins) and 0 <= vel_idx < filtered_map.shape[1]:
            filtered_map[range_idx, vel_idx] = False

    return filtered_map


def build_parameter_grid(config: EvalConfig) -> list[tuple[int, int, int, int]]:
    """
    Build the full grid of turn-off hyperparameter combinations.
    """
    return list(
        product(
            config.thresh_silents,
            config.thresh_silent_chirps,
            config.monotonicity_thresholds,
            config.t_monotonicities,
        )
    )


def create_spinr_model(
    *,
    config: EvalConfig,
    n_frames: int,
    n_chirps: int,
    n_samples: int,
    distance_bins: range,
    thresh_silent: int,
    thresh_silent_chirp: int,
    monotonicity_thresh: int,
    t_monotonicity: int,
) -> Cu_RD_SpiNR_LogOsCfar_Float_Model:
    """
    Create and configure one SpiNR model instance for a single parameter setting.

    Several CFAR-related constructor arguments are required by the model API but
    are not used in this evaluation setup.
    """
    model = Cu_RD_SpiNR_LogOsCfar_Float_Model(
        n_frames=n_frames,
        n_chirps=n_chirps,
        n_samples=n_samples,
        alpha_grd=config.alpha_grd,
        tau=config.tau,
        thresh=config.thresh,
        alpha_cfar=None,
        k_cfar=None,
        guard_cells=None,
        ref_cells=None,
        thresh_silent=thresh_silent,
        thresh_silent_chirp=thresh_silent_chirp,
        t_enc=config.t_enc,
        input_type=config.input_type,
        range_bins=distance_bins,
        monotonicity_thresh=monotonicity_thresh,
        t_monotonicity=t_monotonicity,
        kernel_version=config.kernel_version,
    )
    model.wrap_y = True
    model._encoding_func = config.encoding_func
    return model


def evaluate_spinr_outputs(
    *,
    evaluator: CfarTurnoffEvaluator,
    res_neurons: np.ndarray,
    grad: np.ndarray,
    mean_percent_inactive: np.ndarray,
    rd_target_maps: np.ndarray,
    file_filters: dict[str, Any],
    detection_area: np.ndarray,
    distance_bins: range,
    grad_threshold: float,
    thresh_silent: int,
    thresh_silent_chirp: int,
    monotonicity_thresh: int,
    t_monotonicity: int,
) -> None:
    """
    Evaluate one full SpiNR forward pass across all frames.

    A neuron is counted as active only if:
    1. it is marked resonant by the model, and
    2. its gradient exceeds the configured threshold.
    """
    n_frames = rd_target_maps.shape[0]

    for frame_idx in range(n_frames):
        frame_filter = file_filters.get(str(frame_idx), {})

        if frame_filter.get("omit_frame", False):
            continue

        target_map = rd_target_maps[frame_idx][distance_bins]
        target_map = apply_frame_target_filter(
            target_map=target_map,
            frame_filter=frame_filter,
            distance_bins=distance_bins,
        )

        target_map_extended = apply_detection_area(target_map, detection_area)

        active_neurons = np.logical_and(
            res_neurons[frame_idx],
            grad[frame_idx] > grad_threshold,
        )

        evaluator.eval(
            active_neurons,
            target_map,
            target_map_extended,
            inactive_percentage=mean_percent_inactive[frame_idx],
            thresh_silent=thresh_silent,
            thresh_silent_chirp=thresh_silent_chirp,
            monotonicity_thresh=monotonicity_thresh,
            t_monotonicity=t_monotonicity,
        )


# ---------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------
def main() -> None:
    """Run the full SpiNR turn-off grid search and save the results."""
    config = EvalConfig()
    detection_area = np.ones(config.detection_area_shape, dtype=int)

    filter_config = load_filter_config(config.filter_file)

    evaluator = CfarTurnoffEvaluator(
        keys=[
            "thresh_silent",
            "thresh_silent_chirp",
            "monotonicity_thresh",
            "t_monotonicity",
        ],
        detection_area=detection_area,
    )

    dataset_config = bbm.load_config(config.bbm_path)
    bbm_iterator = bbm.BBMIterator(config.bbm_path)

    parameter_grid = build_parameter_grid(config)
    n_total_combinations = len(parameter_grid)
    n_total_sequences = len(bbm_iterator.indices)

    for sequence_idx in range(n_total_sequences):
        radar_data, targets = next(bbm_iterator)

        n_frames, n_chirps, n_samples, n_channels = radar_data.shape

        # Preprocess the raw radar data, keep only the first channel, and convert
        # to the float representation expected by the CUDA implementation.
        radar_data = bbm.preprocess_data(radar_data)
        radar_data = radar_data[:, :, :, 0:1]
        spinr_input = complex_to_float(radar_data)

        # Build binary frame-wise range-Doppler ground-truth maps.
        rd_target_maps = build_target_maps(
            targets=targets,
            n_samples=n_samples,
            n_frames=n_frames,
            n_channels=n_channels,
            n_chirps=n_chirps,
            dataset_config=dataset_config,
        )

        # Clamp the evaluated range interval to the available number of bins.
        distance_bins = get_valid_distance_bins(
            config.max_distance_bins,
            rd_target_maps,
        )

        current_file = os.path.basename(bbm_iterator.get_current_data_filename())
        file_filters = filter_config.get(current_file, {})

        for combo_idx, (
            thresh_silent,
            thresh_silent_chirp,
            monotonicity_thresh,
            t_monotonicity,
        ) in enumerate(parameter_grid):
            print(
                f"\rFile {sequence_idx + 1}/{n_total_sequences}, "
                f"combo {combo_idx + 1}/{n_total_combinations}",
                end="",
                flush=True,
            )

            model = create_spinr_model(
                config=config,
                n_frames=n_frames,
                n_chirps=n_chirps,
                n_samples=n_samples,
                distance_bins=distance_bins,
                thresh_silent=thresh_silent,
                thresh_silent_chirp=thresh_silent_chirp,
                monotonicity_thresh=monotonicity_thresh,
                t_monotonicity=t_monotonicity,
            )

            # Run the full network for the current hyperparameter setting.
            model._forward_spinr(spinr_input)

            res_neurons = model.get_resonant_neurons()
            grad = model.grad()
            mean_percent_inactive = model.mean_percent_inactive()

            evaluate_spinr_outputs(
                evaluator=evaluator,
                res_neurons=res_neurons,
                grad=grad,
                mean_percent_inactive=mean_percent_inactive,
                rd_target_maps=rd_target_maps,
                file_filters=file_filters,
                detection_area=detection_area,
                distance_bins=distance_bins,
                grad_threshold=config.grad_threshold,
                thresh_silent=thresh_silent,
                thresh_silent_chirp=thresh_silent_chirp,
                monotonicity_thresh=monotonicity_thresh,
                t_monotonicity=t_monotonicity,
            )

        print()
        logger.info(
            "Processed file %d / %d.",
            sequence_idx + 1,
            n_total_sequences,
        )

    global_params = {
        "timestamp": int(time.time()),
        "dataset": config.bbm_path,
        "alpha_grd": config.alpha_grd,
        "grad_threshold": config.grad_threshold,
    }

    output_path = Path(config.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    evaluator.save(str(output_path), global_params=global_params)
    logger.info("Saved evaluation results to %s", output_path)


if __name__ == "__main__":
    main()