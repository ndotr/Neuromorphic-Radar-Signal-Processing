"""
BBM Range-Doppler Filter Dataset

This script processes radar data and identifies frames and targets that should be omitted based on FFT values:
- Frames with any target having FFT value > MAX_TARGET_VALUE are marked for omission
- Individual targets with FFT value < MIN_TARGET_VALUE are marked for omission

Output JSON structure:
{
    "data_000.mat": {                   # File name
        0: {                           # Frame index
            "omit_frame": false,       # Whether entire frame should be omitted
            "omit_targets": [          # List of (range, velocity) indices for targets to omit
                [224, 29],
                [109, 15]
            ]
        },
        1: {
            "omit_frame": true,        # This frame should be omitted entirely
            "omit_targets": []
        },
        ...
    },
    "data_001.mat": {
        ...
    }
}

Usage:
- Adjust MAX_TARGET_VALUE to control frame omission
- Adjust MIN_TARGET_VALUE to control target omission
- Run script to generate out/bbm_omit_targets.json
"""

import nrsp.datasets.infineon_bbm as bbm
import numpy as np
from nrsp.rsp.fft import range_doppler_fft
import json
import os

# --- User-defined thresholds ---
MAX_TARGET_VALUE = 1500.0  # omit frame if any target exceeds this value
MIN_TARGET_VALUE = 0.45  # omit targets with value less than this
OUT_PATH = f"out/seed_1_MAX_TARGET_VALUE={MAX_TARGET_VALUE}_MIN_TARGET_VALUE={MIN_TARGET_VALUE}.json"

BBM_PATH = "/home/mgrabmann/data/BBM/level0_random_1dmax_randomrcs_8targets_32chirps_new/seed_1"
config = bbm.load_config(BBM_PATH)
bbmIterator = bbm.BBMIterator(BBM_PATH)

results_dict = {}
for i in range(len(bbmIterator.indices)):
    print(f"{((i+1) / len(bbmIterator.indices))*100} %")

    radar_data, targets = next(bbmIterator)
    n_frames, n_chirps, n_samples, n_channels = radar_data.shape
    radar_data = bbm.preprocess_data(radar_data)
    radar_data = radar_data[:, :, :, 0:1]  # only first channel

    # targets (n_frames, n_distances, n_angles, n_velocities)
    target_cube = bbm.targets_to_cube(targets, n_samples // 2, n_frames, n_channels, n_chirps, config)
    rd_target_maps = np.sum(target_cube, axis=2).astype(bool)  # (n_frames, n_distances, n_velocities)

    # fft
    fft_rd = range_doppler_fft(radar_data)
    fft_rd = fft_rd[:, :, :, 0]  # only first channel
    fft_rd = np.swapaxes(fft_rd, 1, 2)

    current_file = os.path.basename(bbmIterator.get_current_data_filename())
    results_dict[current_file] = {}

    for frame_idx in range(n_frames):
        target_mask = rd_target_maps[frame_idx]
        # Get indices of targets (range, velocity)
        target_indices = np.argwhere(target_mask)  # shape: (num_targets, 2)
        target_values = fft_rd[frame_idx][target_mask].tolist()

        omit_frame = any(v > MAX_TARGET_VALUE for v in target_values)
        # Store (range_idx, velocity_idx) for targets below MIN_TARGET_VALUE
        omit_tuples = []
        for range_idx, vel_idx in target_indices:
            fft_value = fft_rd[frame_idx][range_idx, vel_idx]
            if fft_value < MIN_TARGET_VALUE:
                omit_tuples.append((int(range_idx), int(vel_idx)))

        results_dict[current_file][int(frame_idx)] = {"omit_frame": omit_frame, "omit_targets": omit_tuples}

# Save results_dict to a JSON file
os.makedirs("out", exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(results_dict, f, indent=2)
