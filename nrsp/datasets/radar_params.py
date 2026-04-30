#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import pathlib
import logging
from abc import ABC
from typing import Optional, Any

import yaml
import numpy as np


ROOT_DIR = pathlib.Path("/home/nreeb/code/neuromorphic-radar-processing/")
MODULE_LOGGER = logging.getLogger(__name__)


def yaml_type_check(name: str, value, expected_type: type):
    """
    Validate a YAML-loaded value against an expected type.

    Special handling:
    - float accepts int, float, and numeric strings such as "2e-6"
    - int rejects bool
    """
    if expected_type is int:
        if isinstance(value, bool):
            raise TypeError(f"Expected '{name}' to be int, got bool: {value}")
        if not isinstance(value, int):
            raise TypeError(f"Expected '{name}' to be int, got {type(value).__name__}: {value}")
        return value

    if expected_type is float:
        if isinstance(value, bool):
            raise TypeError(f"Expected '{name}' to be float, got bool: {value}")

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            try:
                return float(value)
            except ValueError as exc:
                raise TypeError(
                    f"Expected '{name}' to be float-compatible, got str: {value}"
                ) from exc

        raise TypeError(f"Expected '{name}' to be float, got {type(value).__name__}: {value}")

    if not isinstance(value, expected_type):
        raise TypeError(f"Expected '{name}' to be {expected_type.__name__}, got {type(value).__name__}: {value}")

    return value


class Naomi4RadarRadarParams(ABC):
    def __init__(self, config_path_radar: pathlib.Path):
        """
        Initialize Naomi4Radar radar parameters from YAML.
        """
        self.config_path_radar = pathlib.Path(config_path_radar)

        if not self.config_path_radar.exists():
            raise FileNotFoundError(f"Radar config file does not exist: {self.config_path_radar}")

        with self.config_path_radar.open("r", encoding="utf-8") as f:
            config_radar = yaml.safe_load(f)

        if not isinstance(config_radar, dict):
            raise TypeError(f"Radar config must load to a dict, got {type(config_radar).__name__}")

        # Required integer fields
        self.n_ctrx = yaml_type_check("n_ctrx", config_radar["n_ctrx"], int)
        self.n_tx = yaml_type_check("n_tx", config_radar["n_tx"], int)
        self.n_rx = yaml_type_check("n_rx", config_radar["n_rx"], int)
        self.n_samples = yaml_type_check("n_samples", config_radar["n_samples"], int)
        self.n_chirps = yaml_type_check("n_chirps", config_radar["n_chirps"], int)
        self.n_frames = yaml_type_check("n_frames", config_radar["n_frames"], int)
        self.n_az_vx = yaml_type_check("n_az_vx", config_radar["n_az_vx"], int)
        self.n_el_vx = yaml_type_check("n_el_vx", config_radar["n_el_vx"], int)
        self.n_padded_az = yaml_type_check("n_padded_az", config_radar["n_padded_az"], int)

        self.n_vx = self.n_az_vx * self.n_el_vx
        self.num_ctrx = int(np.ceil(self.n_rx / 4))

        # Calibration
        self.calibration_filename: Optional[pathlib.Path] = None
        self.calibration_array: Optional[np.ndarray] = None
        self.calibration_type: Optional[str] = None

        if "calibration_filename" in config_radar:
            self.calibration_filename = self.check_calibration_file(
                calibration_filename=config_radar["calibration_filename"]
            )

        if "calibration_type" in config_radar:
            self.calibration_type = str(config_radar["calibration_type"])

        if self.calibration_filename is not None:
            self.calibration_array = np.load(str(self.calibration_filename))

        # Radar waveform params
        self.F_sampling = yaml_type_check("F_sampling", config_radar["F_sampling"], int)
        self.f_start = yaml_type_check("f_start", config_radar["f_start"], int)
        self.f_delta_eff = yaml_type_check("f_delta_eff", config_radar["f_delta_eff"], int)
        self.f_step = yaml_type_check("f_step", config_radar["f_step"], int)

        self.t_pre = yaml_type_check("t_pre", config_radar["t_pre"], float)
        self.t_wait = yaml_type_check("t_wait", config_radar["t_wait"], float)
        self.t_flyback = yaml_type_check("t_flyback", config_radar["t_flyback"], float)

        # Calculated values
        self.t_ramp = self.n_samples / self.F_sampling
        self.n_chirps_per_tx = self.n_chirps // self.n_tx

        self.t_rep = self.t_ramp + self.t_pre + self.t_wait + self.t_flyback

        c0 = 299792458.0
        self.v_max = c0 / (4.0 * self.f_start * self.t_rep * self.n_tx)
        self.v_res = 2.0 * self.v_max / self.n_chirps_per_tx
        self.r_max = c0 * self.n_samples / (4.0 * abs(self.f_delta_eff))
        self.r_res = c0 / (2.0 * abs(self.f_delta_eff))

        # Misc processing params
        self.scale = 1.0 / 2 ** (11 + 4)
        self.uniform_range = (14, 64)

    def __repr__(self):
        return (
            f"<{self.__class__.__module__}.{self.__class__.__qualname__}>("
            f"radar_config={self.config_path_radar})"
        )

    def log_parameters(self, logger: Optional[logging.Logger] = None):
        """
        Log all instance attributes.
        """
        logger = logger or MODULE_LOGGER
        for name, val in self.__dict__.items():
            logger.debug("  %s = %r", name, val)

    def check_calibration_file(self, calibration_filename: str) -> pathlib.Path:
        """
        Validate calibration file path and resolve relative paths against ROOT_DIR.
        """
        path = pathlib.Path(calibration_filename)

        if not path.exists():
            MODULE_LOGGER.info(
                "Could not find calibration file %s. Trying relative to project root.",
                path,
            )
            candidate = ROOT_DIR / path
            if not candidate.exists():
                raise FileNotFoundError(f"Calibration file does not exist: {candidate}")
            path = candidate

        if path.suffix.lower() != ".npy":
            raise TypeError(
                f"Calibration file must have '.npy' suffix, got '{path.suffix}'"
            )

        return path

    # -------------------------------------------------------------------------
    # Calibration
    # -------------------------------------------------------------------------

    def update_calibration_array(
        self,
        calibration_type: str,
        calibration_array: Optional[np.ndarray] = None,
    ) -> None:
        self.calibration_array = calibration_array
        self.calibration_type = calibration_type

    def apply_calibration_on_frame(self, data: np.ndarray, calib_array: np.ndarray) -> np.ndarray:
        """
        Apply calibration by broadcasting.

        Supported shapes
        ----------------
        data.ndim == 4:
            data shape assumed: (A, B, C, D)
            calib_array shape expected: (A, B)

        data.ndim == 3:
            data shape assumed: (A, B, C)
            calib_array shape expected: (A,)

        Returns
        -------
        np.ndarray
            Calibrated data, same shape as input.
        """
        data = np.asarray(data)
        calib_array = np.asarray(calib_array)

        if data.ndim == 4:
            if calib_array.ndim != 2:
                raise ValueError(
                    f"For 4D data, expected 2D calibration array, got shape {calib_array.shape}"
                )
            if calib_array.shape != data.shape[:2]:
                raise ValueError(
                    f"Calibration shape {calib_array.shape} does not match first two data dims {data.shape[:2]}"
                )
            return data * calib_array[:, :, np.newaxis, np.newaxis]

        if data.ndim == 3:
            if calib_array.ndim != 1:
                raise ValueError(
                    f"For 3D data, expected 1D calibration array, got shape {calib_array.shape}"
                )
            if calib_array.shape[0] != data.shape[0]:
                raise ValueError(
                    f"Calibration length {calib_array.shape[0]} does not match first data dim {data.shape[0]}"
                )
            return data * calib_array[:, np.newaxis, np.newaxis]

        raise NotImplementedError(
            f"Data must be 3D or 4D, got shape {data.shape}"
        )