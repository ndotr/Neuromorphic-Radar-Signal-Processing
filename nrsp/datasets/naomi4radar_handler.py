import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.windows
import pathlib

class Naomi4RadarHandler:
    def __init__(self, config_path: pathlib.Path):
        self._load_config(config_path)

    def _load_config(self, config_path):
        """
        Load configuration from YAML file.
        
        """
        with config_path.open('r') as f:
            config = yaml.safe_load(f)

        # Output parameters
        self.n_ranges = config["n_ranges"]
        self.n_az_angles = config["n_az_angles"]
        self.n_velocities = config["n_velocities"]

        # Radar parameters
        self.scale = 1/2**(11+4)
        self.n_ctrx = config["n_ctrx"]
        self.n_tx = config["n_tx"]
        self.n_rx = config["n_rx"]
        self.n_samples = config["n_samples"]
        self.n_chirps = config["n_chirps"]
        self.n_frames = config["n_frames"]
        self.n_chirps_per_tx = self.n_chirps // self.n_tx
        self.n_az_vx = config["n_az_vx"]
        self.n_el_vx = config["n_el_vx"]
        self.n_padded_az = config["n_padded_az"]
        # Physical world
        c0 = 299792458
        self.F_sampling = config["F_sampling"]
        self.f_start = config["f_start"]
        self.f_delta_eff = config["f_delta_eff"]
        self.t_ramp = self.n_samples / self.F_sampling
        self.t_rep = self.t_ramp + float(config["t_pre"]) + float(config["t_wait"]) + float(config["t_flyback"])
        self.v_max = c0 / 4 / self.f_start / self.t_rep / self.n_tx
        self.v_res = 2 * self.v_max / self.n_chirps // self.n_tx
        self.r_max = c0 * self.n_samples / 4 / np.abs(self.f_delta_eff)
        self.r_res = c0 / 2 / np.abs(self.f_delta_eff)
        # Axes
        # Range vector
        self.range_bins = np.linspace(0, 1 - 1/(2*self.n_ranges), (2*self.n_ranges)) * self.r_max * 2
        self.range_bins = self.range_bins[:self.n_ranges]
        # Velocity vector (full symmetric FFT axis)
        self.velocity_bins = np.linspace(-0.5, 0.5 - 1 / self.n_velocities, self.n_velocities) * self.v_max * 2
        # Angle vector (full symmetric FFT axis)
        self.az_angle_bins = np.degrees(np.arcsin(np.arange(-self.n_az_angles/2, self.n_az_angles/2) / (self.n_az_angles) * 0.5))

    def load_az_el_vx_frame(self, data_path, frame):
        """
        Load and process a radar data frame from raw binary files, converting it to
        azimuth-elevation virtual array format.

        Args:
            data_dir (str): Directory path containing the raw radar data files.
            frame (int): Index of the frame to load.

        Returns:
            np.ndarray: Processed radar data with shape 
                        (n_az_channels, n_el_channels, n_samples, n_chirps_per_tx),
                        where these dimensions are defined in the class configuration.
        """
        #raw_data = self._load_raw_frame(data_path=data_path, frame=frame)
        raw_data = self.read_raw_data(data_path=data_path, read_frame_id=frame)
        time_data = self._convert_raw_to_time(raw_data=raw_data)
        vx_data = self._convert_time_to_flatten_vx(time_data=time_data) 
        az_el_vx_data = self._convert_flatten_vx_to_az_el_vx(vx_data=vx_data)

        return az_el_vx_data
    
    def load_2d_vx_frame(self, data_path, frame):
        """
        Load and process a radar data frame from raw binary files, converting it to
        2d virtual array format.

        Args:
            data_dir (str): Directory path containing the raw radar data files.
            frame (int): Index of the frame to load.

        Returns:
            np.ndarray: Processed radar data with shape 
                        (n_az_channels, n_el_channels, n_samples, n_chirps_per_tx),
                        where these dimensions are defined in the class configuration.
        """
        #raw_data = self._load_raw_frame(data_path=data_path, frame=frame)
        raw_data = self.read_raw_data(data_path=data_path, read_frame_id=frame)
        time_data = self._convert_raw_to_time(raw_data=raw_data)
        vx_data = self._convert_time_to_flatten_vx(time_data=time_data) 
        az_el_vx_data = self._convert_flatten_vx_to_az_el_vx(vx_data=vx_data)
        vx_2d_data = self._convert_az_el_vx_to_nonuniform_2d_vx(az_el_vx_data=az_el_vx_data)

        return vx_2d_data
        
    #def generate_fft1d_data(self, data, n, axis, **kwargs):
#
#        # Plain FT
#        if "at" in kwargs:
#            az_el_vx_data = self.apply_1d_chebwin(data=az_el_vx_data, axis=axis, at=kwargs["at"])
#        az_el_vx_fft_data = np.fft.fft(az_el_vx_data, n=n, axis=axis)
#
#        return az_el_vx_fft_data
    
    def generate_fftnd_data(self, data, n, axis, **kwargs):

        for i in range(len(axis)):
            if "at" in kwargs:
                if kwargs["at"][i] is not None:
                    data = self.apply_1d_chebwin(data=data, axis=axis[i], at=kwargs["at"][i])
            data = np.fft.fft(data, n=n[i], axis=axis[i])

        return data


    def convert_az_el_vx_to_uniform_2d_vx(self, az_el_vx_data):

        vx_2d_data = self._convert_az_el_vx_to_nonuniform_2d_vx(az_el_vx_data=az_el_vx_data)
        uniform_2d_vx_data = self._convert_nonuniform_2d_vx_to_uniform_2d_vx(vx_2d_data)

        return uniform_2d_vx_data


    def generate_calibration_vector(self, calib_data_path, idx=None, **kwargs):
        """
        Generate a calibration vector for antenna array correction using radar data.

        This function computes the calibration vector from range FFT data of a given frame.
        It supports flexible configuration via keyword arguments.

        Args:
            calib_data_path (pathlib.Path): Directory containing the calibration radar data.
            pipeline (int, optional): Selects the processing pipeline. 
            idx (int, optional): Index along the specified axis to use for calibration. If None, the index with the
                                 highest average magnitude is used automatically.
            **kwargs:
                axis (int): Axis along which the range FFT is computed and calibration vector is extracted.
                frame (int): Frame index from which to load the calibration data.
                save_to_file (bool or str): If a string (filename) is provided, saves the calibration vector to file.
                                            If False or not provided, skipping save.

        Returns:
            np.ndarray: Calibration vector of shape (n_az_channels, n_el_channels), ready to be broadcasted
                        over range and chirp dimensions.

        Raises:
            ValueError: If required `kwargs` like `axis` or `frame` are missing when pipeline 0 is selected.
        """
        # Plain FT
        axis=2
        # Average over multiple frame (different averagings are possible)
        if "n_frames" in kwargs:
            start_frame = 0
            end_frame = kwargs["n_frames"]
            norm = kwargs["n_frames"]
        # Pick one frame
        elif "frame" in kwargs:
            start_frame = kwargs["frame"]
            end_frame = kwargs["frame"] + 1
            norm = 1

        calib_vec = np.zeros((self.n_az_vx, self.n_el_vx)).astype('complex128')
        for frame in range(start_frame, end_frame, 1):
            az_el_vx_data = self.load_az_el_vx_frame(calib_data_path, frame=frame)
            if "at" in kwargs:
                az_el_vx_data = self.apply_1d_chebwin(data=az_el_vx_data, axis=axis, at=kwargs["at"])
            az_el_vx_range_data = np.fft.fft(az_el_vx_data, n=2*self.n_ranges, axis=2)
            az_el_vx_range_data = az_el_vx_range_data[:,:,:self.n_ranges, :]

            axes = tuple(i for i in range(az_el_vx_range_data.ndim) if i != axis)
            if idx is None:
                idx = np.argmax(np.mean(np.abs(az_el_vx_range_data), axis=axes))

            range_data = np.mean(az_el_vx_range_data, axis=3)

            calib_vec += self._generate_calibration_vector_from_range_data(range_data=range_data,
                                                                            axis=axis,
                                                                            idx=idx,
                                                                            calib_filename=kwargs["calib_filename"])
            calib_vec /= norm
            
        return calib_vec

    
    def apply_calibration(self, az_el_vx_data: np.ndarray, calib_vec: np.ndarray = None, calib_filename=None) -> np.ndarray:
        """
        Apply antenna array calibration to the azimuth-elevation virtual array data.
        The calibration vector is broadcasted and multiplied with the input data to perform calibration.

        Args:
            az_el_vx_data (np.ndarray): Input radar data of shape 
                                        (n_az_channels, n_el_channels, n_samples, n_chirps_per_tx).

        Returns:
            np.ndarray: Calibrated radar data with the same shape as input.
        """

        if calib_vec is None:
            if calib_filename is not None:
                calib_vec = np.load(calib_filename)
            else:
                raise AssertionError

        # Expand dims and tile calibration matrix to match az_el_vx_data dimensions
        calib_mat = np.tile(calib_vec[:, :, np.newaxis, np.newaxis], 
                            (1, 1, self.n_samples, self.n_chirps_per_tx))
        # Apply calibration
        calib_az_el_vx_data = az_el_vx_data * calib_mat

        return calib_az_el_vx_data

    def apply_1d_chebwin(self, data, axis, at=100):
        """
        Apply a 1D Chebyshev window to data along a specified axis.

        Args:
            data (np.ndarray): Input data array.
            axis (int): Axis along which to apply the window.
            at (float): Attenuation parameter for Chebyshev window.

        Returns:
            np.ndarray: Windowed data.
        """
        n_samples = data.shape[axis]
        win = scipy.signal.windows.chebwin(n_samples, at=at)
        shape = [1] * data.ndim
        shape[axis] = -1
        return (data * win.reshape(shape)) * (1.0 / np.sum(win))


    def db(self, x, ref=1.0, floor_db=-100):
        """
        Convert linear magnitude to decibel (dB) scale with flooring.

        Args:
            x (np.ndarray): Input linear magnitude.
            ref (float): Reference magnitude for 0 dB.
            floor_db (float): Minimum dB value.

        Returns:
            np.ndarray: Magnitude in dB scale.
        """
        with np.errstate(divide='ignore'):
            db_val = 20 * np.log10(np.maximum(x / ref, 1e-12))
        return np.maximum(db_val, floor_db)

    ###########################################################################
    # Private
    ###########################################################################

    def _load_raw_frame(self, data_path: pathlib.Path, frame: int) -> np.ndarray:
        """
        Load one frame of raw radar data from binary files using Pathlib for path handling.

        Parameters:
        -----------
        data_dir : Path
            Path object pointing to the directory containing radar data files.
        frame : int
            Index of the frame to load.

        Returns:
        --------
        np.ndarray
            Loaded raw radar data of shape (n_samples, total_channels, n_chirps).
        """

        # Total number of bins in one frame and total bytes
        nbins = self.n_samples * self.n_chirps * 4
        frame_size = nbins * 2  # each bin = 2 bytes (16-bit)

        raw_blocks = []

        for fid in range(self.n_ctrx):
            file_path = data_path / f"ctrx{fid}_bin.raw"
            with file_path.open("rb") as f:
                f.seek(frame_size * frame)
                data = f.read(frame_size)

            # Read and reshape one CTRX block
            block = np.frombuffer(data, dtype=np.uint16)
            block = block.reshape((4, self.n_samples, self.n_chirps), order='F')
            raw_blocks.append(block)

        # Concatenate all CTRX blocks along the channel axis
        raw = np.concatenate(raw_blocks, axis=0)

        # Rearrange dimensions: (n_samples, n_channels, n_chirps)
        raw = raw.transpose((1, 0, 2))

        # Remove padded bits and cast
        raw = np.bitwise_and(raw, np.uint16(0xFFF0))
        raw = raw.astype(np.int16, copy=False).astype(np.float64)

        return raw
    
    def _convert_raw_to_time(self, raw_data):
        """
        Convert raw data to time domain data.

        Input shape: (n_samples, n_rx * n_tx, n_chirps)
        Output shape: (n_rx, n_tx, n_samples, n_chirps_per_tx)

        Args:
            raw_data (np.ndarray): Raw radar data.

        Returns:
            np.ndarray: Time domain data reshaped and transposed.
        """
        n_samples, n_rx, n_chirps = raw_data.shape
        n_chirps_per_tx = n_chirps // self.n_tx
        time_data = raw_data.reshape((n_samples, n_rx, self.n_tx, n_chirps_per_tx), order='F')
        return np.transpose(time_data, (1, 2, 0, 3))

    def _convert_time_to_flatten_vx(self, time_data):
        """
        Rearrange time domain data to virtual array data (vx_data).

        Input shape: (n_rx, n_tx, n_samples, n_chirps_per_tx)
        Output shape: (n_vx, n_samples, n_chirps_per_tx)

        Returns:
            np.ndarray: Virtual array data.
        """
        n_rx, n_tx, n_samples, n_chirps_per_tx = time_data.shape
        n_vx = n_rx * n_tx
        vx_data = np.zeros((n_vx, n_samples, n_chirps_per_tx), dtype=time_data.dtype)
        # MIMO rearrangement as per original logic
        for txi in range(self.n_tx):
            for blk in range(n_rx // 4):
                src_idx = slice(blk * 4, blk * 4 + 4)
                dst_idx = slice(blk * 64 + txi * 4, blk * 64 + txi * 4 + 4)
                vx_data[dst_idx, :, :] = time_data[src_idx, txi, :, :]
        return vx_data

    def _convert_flatten_vx_to_az_el_vx(self, vx_data):
        """
        Convert flatten virtual array data to azimuth-elevation virtual array.

        Input shape: (n_vx, n_samples, n_chirps_per_tx)
        Output shape: (n_az_vx, n_el_vx, n_samples, n_chirps_per_tx)

        Returns:
            np.ndarray: Azimuth-elevation-velocity domain data (n_az_vx, n_el_vx, n_samples, n_chirps_per_tx).
        """
        n_vx, n_samples, n_chirps_per_tx = vx_data.shape
        Vx_Tx_idx = np.array([24, 0, 52, 44, 28, 12, 56, 40, 20, 4, 48, 32, 16, 8, 60, 36]) - 1
        Vx_Rx_idx = np.array([1, 2, 67, 68, 65, 66, 3, 4, 193, 194, 131, 132, 129, 130, 195, 196])# - 1

        az_el_vx_data = np.zeros((self.n_az_vx, self.n_el_vx, n_samples, n_chirps_per_tx), dtype=vx_data.dtype)

        for tx_group in range(4):
            for tx_idx_in_group in range(4):
                full_tx_idx = tx_group * 4 + tx_idx_in_group
                start_idx = tx_group * 16
                target_indices = start_idx + np.arange(16)
                src_indices = Vx_Tx_idx[full_tx_idx] + Vx_Rx_idx
                az_el_vx_data[target_indices, tx_idx_in_group, :, :] = vx_data[src_indices, :, :]

        return az_el_vx_data

    def _convert_az_el_vx_to_nonuniform_2d_vx(self, az_el_vx_data):
        """
        Pad azimuth channels to a fixed size with zeros and reorder according to FFT indices.

        Args:
            az_el_vx_data (np.ndarray): Input az-el-vx data.

        Returns:
            np.ndarray: Padded az-el-range data with shape (n_padded_az, n_el_channels, n_samples, n_chirps_per_tx).
        """
        n_az_vx, n_el_vx, n_samples, n_chirps_per_tx = az_el_vx_data.shape
        lambda0 = 1  # normalized wavelength
        d_tx_az = np.array([0, 4.5, 16, 20.5]) * lambda0
        d_rx_az = np.arange(16) * lambda0
        d_vir_az = np.array([tx + rx for tx in d_tx_az for rx in d_rx_az])
        FFT_Vx_idx = np.argsort(d_vir_az[:64])

        nonuniform_2d_vx_data = np.zeros((self.n_padded_az, n_el_vx, n_samples, n_chirps_per_tx), dtype=az_el_vx_data.dtype)
        valid_indices = [0, 2, 4, 6] + list(range(8, 64)) + [65, 67, 69, 71]
        nonuniform_2d_vx_data[valid_indices, :, :, :] = az_el_vx_data[FFT_Vx_idx, :, :, :]

        return nonuniform_2d_vx_data
    
    def _convert_nonuniform_2d_vx_to_uniform_2d_vx(self, nonuniform_2d_vx_data):
        ret = nonuniform_2d_vx_data[14:64, :, :, :]  
        return ret

    def _generate_calibration_vector_from_range_data(self, range_data, axis, idx=None, calib_filename=None):
        """
        Generate a calibration vector from range data.

        Args:
            range_data (np.ndarray): Input range data.
            axis (int): Axis over which to select the calibration vector.
            idx (int, optional): Index along the axis to compute calibration vector. Defaults to None (auto-select).
            save_to_file (bool): Whether to save the calibration vector to file.

        Returns:
            np.ndarray: Calibration vector.
        """
        calib_vec = np.take(range_data, idx, axis=axis)
        calib_angle = np.angle(calib_vec)
        calib_mag = np.abs(calib_vec)
        calib_mag = np.max(calib_mag) / calib_mag
        calib_vec = calib_mag * np.exp(-1j * calib_angle)
        if isinstance(calib_filename, pathlib.Path):
            np.save(calib_filename, calib_vec)

        return calib_vec

    def read_raw_data(self, data_path, read_frame_id):
        """
        Load raw radar time data from multiple CTRX binary files.

        Parameters
        ----------
        data_dir : str or Path
            Path to directory containing CTRX binary files.
        read_frame_id : int
            Frame index to read.
        n_samples : int
            Number of ADC samples per chirp.
        n_ramps : int
            Number of chirps per frame.
        n_channels : int
            Total number of channels (typically n_rx * n_tx).

        Returns
        -------
        np.ndarray
            Time-domain data of shape (n_samples, n_channels, n_ramps).
        """
        num_ctrx = int(np.ceil(self.n_rx / 4))

        # Read first CTRX block
        file_path = data_path / f"ctrx0_bin.raw"
        timedata = self._read_single_ctrx_binary(read_frame_id, self.n_samples, self.n_chirps, file_path)

        # Read additional CTRX blocks
        for k in range(1, num_ctrx):
            file_path = data_path / f"ctrx{k}_bin.raw"
            td_tmp = self._read_single_ctrx_binary(read_frame_id, self.n_samples, self.n_chirps, file_path)
            timedata = np.concatenate((timedata, td_tmp), axis=1)

        return timedata


    def _read_single_ctrx_binary(self, read_frame_id, n_samples, n_chirps, file_path):
        """
        Read a single CTRX binary file and extract a frame.

        Returns
        -------
        np.ndarray
            Data of shape (n_samples, 4, n_ramps)
        """
        n_channels_per_file = 4
        n_bins = n_samples * n_channels_per_file * n_chirps
        frame_size_bytes = n_bins * 2  # uint16

        with file_path.open("rb") as f:
            f.seek(0, 2)  # Move to EOF
            file_size = f.tell()
            n_frames = file_size // frame_size_bytes

            assert read_frame_id < n_frames, f"Invalid frame ID {read_frame_id}, only {n_frames} frames available"

            f.seek(read_frame_id * frame_size_bytes)
            raw = np.frombuffer(f.read(frame_size_bytes), dtype=np.uint16)

        # Mask padded bits and convert to signed int16
        raw = np.bitwise_and(raw, 0xFFF0)
        raw = raw.astype(np.int16).astype(np.float64)

        # Reshape and permute like MATLAB: reshape(Na, Ns, Nr) → permute([2 1 3]) → (Ns, 4, Nr)
        raw = raw.reshape((n_channels_per_file, n_samples, n_chirps), order='F')
        timedata = raw.transpose((1, 0, 2))  # shape: (n_samples, 4, n_ramps)

        assert timedata.shape[2] == n_chirps, "Mismatch in number of ramps"
        return timedata




    
