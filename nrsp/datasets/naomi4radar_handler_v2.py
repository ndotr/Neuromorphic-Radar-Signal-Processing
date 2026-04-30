#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pathlib
import logging
from typing import Optional, Tuple, Any

import h5py
import yaml
import numpy as np

from nrsp.datasets.radar_params import Naomi4RadarRadarParams


def yaml_type_check(name: str, value: Any, expected_type: type):
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


class Naomi4RadarDataHandler:
    def __init__(
        self,
        config_path_radar: pathlib.Path,
        config_path_handler: pathlib.Path,
        logger: logging.Logger,
    ):
        self.logger = logger
        self.hdf5_file = None
        self.measurement_dir = None
        self.meas_file_type = None
        self.crt_frame_id = 0

        self.radar_params = Naomi4RadarRadarParams(config_path_radar=config_path_radar)
        self.config_path_handler = pathlib.Path(config_path_handler)

        with self.config_path_handler.open("r", encoding="utf-8") as f:
            config_handler = yaml.safe_load(f)

        if not isinstance(config_handler, dict):
            raise TypeError("Handler config must load to a dict")

        self.input_type = self.check_input_type(config_handler["input_type"])

        if "measurement_file" not in config_handler:
            raise KeyError("Missing required handler-config key: 'measurement_file'")

        # Accept both plural and singular config styles
        filter_args = self._parse_filter_config(config_handler)

        self.filter_chirps, self.filter_channels = self.create_filter_arrays(filter_args)

        self.n_rx = len(self.filter_channels)
        self.n_tx = len(filter_args["TX_antennas"])
        self.n_chirps = len(self.filter_chirps)
        self.n_samples = self.radar_params.n_samples

        self.measurement_dir, self.meas_file_type = self.check_measurement_file(
            config_handler["measurement_file"]
        )

        if self.meas_file_type == "hdf5":
            self.hdf5_file = h5py.File(self.measurement_dir, "r")
            self._init_hdf5_handles()

    def __repr__(self):
        return (
            f"<{self.__class__.__module__}.{self.__class__.__qualname__}>\n"
            f"  radar config: {self.radar_params.config_path_radar}\n"
            f"  handler config: {self.config_path_handler}\n"
            f"  measurement: {self.measurement_dir}\n"
            f"  meas_file_type: {self.meas_file_type}"
        )

    def close(self):
        hdf5_file = getattr(self, "hdf5_file", None)
        if hdf5_file is not None:
            try:
                hdf5_file.close()
            except Exception:
                pass
            self.hdf5_file = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Setup / validation
    # -------------------------------------------------------------------------

    @staticmethod
    def check_input_type(input_type: int) -> int:
        if input_type != 1:
            raise ValueError(
                f"Only input_type=1 (from file) is supported here, but got {input_type}"
            )
        return input_type

    def _parse_filter_config(self, config_handler: dict) -> dict:
        """
        Accept both:
        - RX_antennas / TX_antennas / n_chirps
        - RX_antenna / TX_antenna / chirp_idx

        Returns a normalized dict for create_filter_arrays().
        """

        # New/plural style
        if all(k in config_handler for k in ("RX_antennas", "TX_antennas", "n_chirps")):
            rx_list = config_handler["RX_antennas"]
            tx_list = config_handler["TX_antennas"]
            n_chirps = yaml_type_check("n_chirps", config_handler["n_chirps"], int)

            if not isinstance(rx_list, list):
                raise TypeError(f"'RX_antennas' must be a list, got {type(rx_list).__name__}")
            if not isinstance(tx_list, list):
                raise TypeError(f"'TX_antennas' must be a list, got {type(tx_list).__name__}")

            return {
                "mode": "grouped",
                "RX_antennas": rx_list,
                "TX_antennas": tx_list,
                "n_chirps": n_chirps,
            }

        # Old/singular TDM style
        if all(k in config_handler for k in ("RX_antenna", "TX_antenna", "chirp_idx")):
            rx_val = yaml_type_check("RX_antenna", config_handler["RX_antenna"], int)
            tx_val = yaml_type_check("TX_antenna", config_handler["TX_antenna"], int)
            chirp_idx = yaml_type_check("chirp_idx", config_handler["chirp_idx"], int)

            return {
                "mode": "chirp_idx",
                "RX_antennas": [rx_val],
                "TX_antennas": [tx_val],
                "chirp_idx": chirp_idx,
            }

        raise KeyError(
            "Missing required handler-config keys. "
            "Expected either {'RX_antennas','TX_antennas','n_chirps'} "
            "or {'RX_antenna','TX_antenna','chirp_idx'}."
        )

    def check_measurement_file(self, file_path: str) -> Tuple[pathlib.Path, str]:
        path = pathlib.Path(file_path)

        # Case 1: direct file path
        if path.is_file():
            if path.suffix.lower() == ".hdf5":
                return path, "hdf5"
            raise FileExistsError(
                f"Unsupported measurement file type: {path}. "
                f"Only direct .hdf5 files are supported as file input."
            )

        # Case 2: directory path
        if not path.is_dir():
            raise IsADirectoryError(
                f"Expected measurement_file to be a directory or .hdf5 file, got: {file_path}"
            )

        file_list_raw = []
        file_list_hdf5 = []

        for file in os.listdir(path):
            if file.endswith(".raw"):
                file_list_raw.append(file)
            elif file.endswith(".hdf5"):
                file_list_hdf5.append(file)

        if len(file_list_raw) > 0 and len(file_list_hdf5) > 0:
            raise FileExistsError(
                f"Directory {path} contains both .raw and .hdf5 files. "
                f"Expected exactly one measurement format."
            )

        if len(file_list_raw) == 0 and len(file_list_hdf5) == 0:
            raise FileExistsError(
                f"Directory {path} contains neither .raw nor .hdf5 files."
            )

        if len(file_list_raw) > 0:
            for k in range(self.radar_params.num_ctrx):
                expected = f"ctrx{k}_bin.raw"
                if expected not in file_list_raw:
                    raise FileExistsError(
                        f"Missing expected raw file {expected} in {path}"
                    )
            return path, "raw"

        hdf5_path = path / file_list_hdf5[0]
        self.logger.warning("Using first .hdf5 file found in measurement directory.")
        return hdf5_path, "hdf5"

    def _init_hdf5_handles(self):
        if self.hdf5_file is None:
            raise RuntimeError("HDF5 file not open")

        if "measurement_00" not in self.hdf5_file:
            raise KeyError("HDF5 file does not contain group 'measurement_00'")

        meas_group = self.hdf5_file["measurement_00"]

        if "cubes" not in meas_group:
            raise KeyError("HDF5 group 'measurement_00' does not contain dataset 'cubes'")

        self.ds_cubes = meas_group["cubes"]
        self.ds_header = meas_group["header"] if "header" in meas_group else None
        self.ds_timestamp = meas_group["time"] if "time" in meas_group else None

        if not isinstance(self.ds_cubes, h5py.Dataset):
            raise TypeError(f"'measurement_00/cubes' is not a dataset but {type(self.ds_cubes)}")

        if self.ds_cubes.ndim != 4:
            raise ValueError(
                f"Expected 'measurement_00/cubes' with 4 dims "
                f"(frames, rx, chirps, samples), got shape {self.ds_cubes.shape}"
            )

        self.n_frames_available = int(self.ds_cubes.shape[0])

        self.logger.info(
            f"Using HDF5 cubes dataset: measurement_00/cubes, "
            f"shape={self.ds_cubes.shape}, dtype={self.ds_cubes.dtype}"
        )

    def create_filter_arrays(self, output_args: dict) -> Tuple[list, list]:
        self.logger.debug(f"Creating filter arrays according to {self.config_path_handler}")

        mode = output_args["mode"]
        rx_list = list(output_args["RX_antennas"])
        tx_list = list(output_args["TX_antennas"])

        for val in rx_list:
            if not 0 <= val < self.radar_params.n_rx:
                raise KeyError(
                    f"'RX_antennas' entries must be in [0, {self.radar_params.n_rx - 1}], "
                    f"but got {rx_list}"
                )

        for val in tx_list:
            if not 0 <= val < self.radar_params.n_tx:
                raise KeyError(
                    f"'TX_antennas' entries must be in [0, {self.radar_params.n_tx - 1}], "
                    f"but got {tx_list}"
                )

        if mode == "grouped":
            n_chirps = yaml_type_check("n_chirps", output_args["n_chirps"], int)

            if not 1 <= n_chirps <= self.radar_params.n_chirps_per_tx:
                raise KeyError(
                    f"'n_chirps' must be in [1, {self.radar_params.n_chirps_per_tx}], "
                    f"but is {n_chirps}"
                )

            chirp_filter = [
                tx + (x * self.radar_params.n_tx)
                for x in range(n_chirps)
                for tx in tx_list
            ]

        elif mode == "chirp_idx":
            chirp_idx = yaml_type_check("chirp_idx", output_args["chirp_idx"], int)

            if not 0 <= chirp_idx < self.radar_params.n_chirps:
                raise KeyError(
                    f"'chirp_idx' must be in [0, {self.radar_params.n_chirps - 1}], "
                    f"but is {chirp_idx}"
                )

            # In TDM, same TX means same modulo-n_tx lane.
            tx_from_chirp = chirp_idx % self.radar_params.n_tx

            # Optional consistency check with TX_antenna from config
            if len(tx_list) != 1:
                raise ValueError(
                    f"'chirp_idx' mode expects exactly one TX antenna, got {tx_list}"
                )

            if tx_list[0] != tx_from_chirp:
                self.logger.warning(
                    f"'TX_antenna'={tx_list[0]} does not match chirp_idx % n_tx = {tx_from_chirp}. "
                    f"Using chirp_idx-derived TX lane."
                )
                tx_list = [tx_from_chirp]

            chirp_filter = list(range(chirp_idx, self.radar_params.n_chirps, self.radar_params.n_tx))

        else:
            raise ValueError(f"Unsupported filter mode: {mode}")

        channel_filter = rx_list

        self.logger.debug(f"Filter mode: {mode}")
        self.logger.debug(f"Chirp filter: {chirp_filter}")
        self.logger.debug(f"Channel filter: {channel_filter}")

        return chirp_filter, channel_filter

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def set_frame(self, frame_idx: int):
        if frame_idx < 0:
            raise ValueError("frame_idx must be >= 0")
        self.crt_frame_id = frame_idx

    def load_frame(self, frame_idx: Optional[int] = None) -> Tuple[np.ndarray, int]:
        if frame_idx is None:
            frame_idx = self.crt_frame_id
            self.crt_frame_id += 1
        else:
            self.crt_frame_id = frame_idx + 1

        if self.meas_file_type == "raw":
            raw_frame = self._load_raw_frame_from_binary_files(frame_idx)
        elif self.meas_file_type == "hdf5":
            raw_frame = self._load_raw_frame_from_hdf5(frame_idx)
        else:
            raise NotImplementedError(f"Unsupported measurement type: {self.meas_file_type}")

        return raw_frame, frame_idx

    def convert_frame(self, raw_data: np.ndarray, output_type: str, calib_flag: bool = True) -> np.ndarray:
        rxtx_data = self._convert_raw_to_rxtx(raw_data)
        if (
            self.radar_params.calibration_array is not None
            and self.radar_params.calibration_type == "rxtx"
            and calib_flag
        ):
            rxtx_data = self.radar_params.apply_calibration_on_frame(
                rxtx_data, self.radar_params.calibration_array
            )
        if output_type == "rxtx":
            return rxtx_data

        vx_data = self._convert_rxtx_to_flatten_vx(rxtx_data)
        if (
            self.radar_params.calibration_array is not None
            and self.radar_params.calibration_type == "vx"
            and calib_flag
        ):
            vx_data = self.radar_params.apply_calibration_on_frame(
                vx_data, self.radar_params.calibration_array
            )
        if output_type == "vx":
            return vx_data

        az_el_vx_data = self._convert_flatten_vx_to_az_el_vx(vx_data)
        if (
            self.radar_params.calibration_array is not None
            and self.radar_params.calibration_type == "az_el_vc"
            and calib_flag
        ):
            az_el_vx_data = self.radar_params.apply_calibration_on_frame(
                az_el_vx_data, self.radar_params.calibration_array
            )
        if output_type == "az_el_vx":
            return az_el_vx_data

        nonuniform_2d_vx = self._convert_az_el_vx_to_nonuniform_2d_vx(az_el_vx_data)
        if output_type == "nonuniform":
            return nonuniform_2d_vx

        uniform_2d_vx = self._convert_nonuniform_2d_vx_to_uniform_2d_vx(nonuniform_2d_vx)
        if output_type == "uniform":
            return uniform_2d_vx

        raise ValueError(f"Unsupported output_type: {output_type}")

    # -------------------------------------------------------------------------
    # Internal loading
    # -------------------------------------------------------------------------

    def _load_raw_frame_from_binary_files(self, frame_idx: int) -> np.ndarray:
        blocks = []
        for k in range(self.radar_params.num_ctrx):
            file_path = self.measurement_dir / f"ctrx{k}_bin.raw"
            block = self._read_single_ctrx_binary(
                read_frame_id=frame_idx,
                n_samples=self.radar_params.n_samples,
                n_chirps=self.radar_params.n_chirps,
                file_path=file_path,
            )
            blocks.append(block)

        raw_full = np.concatenate(blocks, axis=0)  # (n_rx_total, n_chirps_total, n_samples)

        filtered_frame = raw_full[
            np.array(self.filter_channels, dtype=int)[:, None],
            np.array(self.filter_chirps, dtype=int)[None, :],
            :
        ].astype(np.int16, copy=False)

        return filtered_frame

    def _load_raw_frame_from_hdf5(self, frame_idx: int) -> np.ndarray:
        if frame_idx < 0 or frame_idx >= self.n_frames_available:
            raise IndexError(
                f"frame_idx={frame_idx} out of range for HDF5 with {self.n_frames_available} frames"
            )
    
        raw_full = self.ds_cubes[frame_idx, ...]
        raw_full = np.asarray(raw_full)
    
        if raw_full.ndim != 3:
            raise ValueError(
                f"Expected HDF5 frame with 3 dims (rx, chirps, samples), got {raw_full.shape}"
            )
    
        # ---------------------------------------------------------
        # Check raw frame before filtering
        # ---------------------------------------------------------
        raw_nnz = np.count_nonzero(raw_full)
        raw_absmax = np.max(np.abs(raw_full))
    
        if raw_nnz == 0:
            raise RuntimeError(
                f"HDF5 frame {frame_idx} contains only zeros before filtering "
                f"(shape={raw_full.shape})"
            )
    
        self.logger.debug(
            f"Frame {frame_idx} raw stats: shape={raw_full.shape}, "
            f"nnz={raw_nnz}, abs_max={raw_absmax}"
        )
    
        # ---------------------------------------------------------
        # Apply filtering
        # ---------------------------------------------------------
        filtered_frame = raw_full[
            np.array(self.filter_channels, dtype=int)[:, None],
            np.array(self.filter_chirps, dtype=int)[None, :],
            :
        ].astype(np.int16, copy=False)
    
        # ---------------------------------------------------------
        # Check filtered frame
        # ---------------------------------------------------------
        filt_nnz = np.count_nonzero(filtered_frame)
        filt_absmax = np.max(np.abs(filtered_frame))
    
        if filt_nnz == 0:
            raise RuntimeError(
                f"Filtered frame {frame_idx} is all zeros. "
                f"Filter RX={self.filter_channels}, "
                f"first chirps={self.filter_chirps[:8]}... "
                f"raw_abs_max={raw_absmax}"
            )
    
        self.logger.debug(
            f"Frame {frame_idx} filtered stats: shape={filtered_frame.shape}, "
            f"nnz={filt_nnz}, abs_max={filt_absmax}"
        )
    
        return filtered_frame

    @staticmethod
    def _read_single_ctrx_binary(
        read_frame_id: int,
        n_samples: int,
        n_chirps: int,
        file_path: pathlib.Path,
    ) -> np.ndarray:
        n_channels_per_file = 4
        n_bins = int(n_samples * n_channels_per_file * n_chirps)
        frame_size_bytes = n_bins * 2

        with file_path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            n_frames = file_size // frame_size_bytes

            if read_frame_id > n_frames - 1:
                raise IndexError(
                    f"Invalid frame ID {read_frame_id}, only {n_frames} frames available in {file_path}"
                )

            f.seek(read_frame_id * frame_size_bytes)
            raw = np.frombuffer(f.read(frame_size_bytes), dtype=np.uint16)

        raw = np.bitwise_and(raw, 0xFFF0)
        raw = raw.astype(np.int16, copy=False)

        # original format per file: (4, n_samples, n_chirps)
        raw = raw.reshape((n_channels_per_file, n_samples, n_chirps), order="F")

        # convert to common loader format: (4, n_chirps, n_samples)
        raw = raw.transpose(0, 2, 1)

        return raw

    # -------------------------------------------------------------------------
    # Format conversions
    # -------------------------------------------------------------------------

    def _convert_raw_to_rxtx(self, raw_data: np.ndarray) -> np.ndarray:
        n_rx, n_chirps, n_samples = raw_data.shape
        n_chirps_per_tx = n_chirps // self.radar_params.n_tx

        # (rx, chirps, samples) -> (samples, rx, chirps)
        raw_data = raw_data.transpose((2, 0, 1))

        # (samples, rx, tx, chirps_per_tx)
        time_data = raw_data.reshape(
            (n_samples, n_rx, self.radar_params.n_tx, n_chirps_per_tx),
            order="F",
        )

        # -> (rx, tx, chirps_per_tx, samples)
        time_data = np.transpose(time_data, (1, 2, 3, 0))
        return time_data

    def _convert_rxtx_to_flatten_vx(self, time_data: np.ndarray) -> np.ndarray:
        n_rx, n_tx, n_chirps_per_tx, n_samples = time_data.shape
        n_vx = n_rx * n_tx

        vx_data = np.zeros((n_vx, n_chirps_per_tx, n_samples), dtype=time_data.dtype)

        for txi in range(self.radar_params.n_tx):
            for blk in range(n_rx // 4):
                src_idx = slice(blk * 4, blk * 4 + 4)
                dst_idx = slice(blk * 64 + txi * 4, blk * 64 + txi * 4 + 4)
                vx_data[dst_idx, :, :] = time_data[src_idx, txi, :, :]

        return vx_data

    def _convert_flatten_vx_to_az_el_vx(self, vx_data: np.ndarray) -> np.ndarray:
        n_vx, n_chirps_per_tx, n_samples = vx_data.shape

        Vx_Tx_idx = np.array([24, 0, 52, 44, 28, 12, 56, 40, 20, 4, 48, 32, 16, 8, 60, 36]) - 1
        Vx_Rx_idx = np.array([1, 2, 67, 68, 65, 66, 3, 4, 193, 194, 131, 132, 129, 130, 195, 196])

        az_el_vx_data = np.zeros(
            (
                self.radar_params.n_az_vx,
                self.radar_params.n_el_vx,
                n_chirps_per_tx,
                n_samples,
            ),
            dtype=vx_data.dtype,
        )

        for tx_group in range(4):
            for tx_idx_in_group in range(4):
                full_tx_idx = tx_group * 4 + tx_idx_in_group
                start_idx = tx_group * 16
                target_indices = start_idx + np.arange(16)
                src_indices = Vx_Tx_idx[full_tx_idx] + Vx_Rx_idx
                az_el_vx_data[target_indices, tx_idx_in_group, :, :] = vx_data[src_indices, :, :]

        return az_el_vx_data

    def _convert_az_el_vx_to_nonuniform_2d_vx(self, az_el_vx_data: np.ndarray) -> np.ndarray:
        n_az_vx, n_el_vx, n_chirps_per_tx, n_samples = az_el_vx_data.shape

        lambda0 = 1
        d_tx_az = np.array([0, 4.5, 16, 20.5]) * lambda0
        d_rx_az = np.arange(16) * lambda0
        d_vir_az = np.array([tx + rx for tx in d_tx_az for rx in d_rx_az])
        fft_vx_idx = np.argsort(d_vir_az[:64])

        nonuniform_2d_vx_data = np.zeros(
            (
                self.radar_params.n_padded_az,
                n_el_vx,
                n_chirps_per_tx,
                n_samples,
            ),
            dtype=az_el_vx_data.dtype,
        )

        valid_indices = [0, 2, 4, 6] + list(range(8, 64)) + [65, 67, 69, 71]
        nonuniform_2d_vx_data[valid_indices, :, :, :] = az_el_vx_data[fft_vx_idx, :, :, :]

        return nonuniform_2d_vx_data

    def _convert_nonuniform_2d_vx_to_uniform_2d_vx(self, nonuniform_2d_vx_data: np.ndarray) -> np.ndarray:
        return nonuniform_2d_vx_data[
            self.radar_params.uniform_range[0]:self.radar_params.uniform_range[1],
            :, :, :
        ]