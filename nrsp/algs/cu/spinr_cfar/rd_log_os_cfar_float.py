import cupy as cp
import numpy as np

import nrsp.utils.cu
from nrsp.algs.cu.kernels.rd_spinr_turnoff_float import rd_spinr_turnoff_float_kernel
from nrsp.algs.cu.kernels.rd_spinr_turnoff_v2_float import rd_spinr_turnoff_v2_float_kernel
from nrsp.algs.cu.cfar.kernels.log_os_cfar import log_os_cfar_kernel
from nrsp.utils.radar import phasor_weights
from nrsp.utils.cfar import get_cfar_kernel_2d


class Cu_RD_SpiNR_LogOsCfar_Float_Model:

    def __init__(
        self,
        n_frames,
        n_chirps,
        n_samples,
        alpha_grd,
        tau,  # log encoding
        thresh,  # log encoding
        alpha_cfar,  # os cfar
        k_cfar,  # os cfar
        guard_cells,
        ref_cells,
        thresh_silent=-1,  # -1 means no silencing
        thresh_silent_chirp=1,
        t_grd=0,
        t_enc=0,
        input_type="complex",
        range_bins=None,
        monotonicity_thresh=None,
        t_monotonicity=None,
        kernel_version=None,  # "v2" or None (default)
    ):
        assert input_type in ["real", "complex"]
        # assert thresh_silent_chirp >= 1

        self.n_frames = n_frames
        self.n_chirps = n_chirps
        self.n_samples = n_samples
        self.alpha_grd = alpha_grd
        self.tau = tau
        self.thresh = thresh

        self._encoding_func = 0  # Log-Time-Encodeded spiking_function

        # Ensure alpha_cfar and k_cfar are lists, even if passed as numpy arrays or similar
        if not isinstance(alpha_cfar, list):
            self.alpha_cfar = list(alpha_cfar) if hasattr(alpha_cfar, "__iter__") and not isinstance(alpha_cfar, str) else [alpha_cfar]
        else:
            self.alpha_cfar = alpha_cfar

        if not isinstance(k_cfar, list):
            self.k_cfar = list(k_cfar) if hasattr(k_cfar, "__iter__") and not isinstance(k_cfar, str) else [k_cfar]
        else:
            self.k_cfar = k_cfar

        self.guard_cells = guard_cells
        self.ref_cells = ref_cells
        self.t_grd = t_grd
        self.t_enc = t_enc

        self.wrap_x = False  # wrap around distances
        self.wrap_y = False  # wrap around velocities

        self.thresh_silent = thresh_silent
        self.thresh_silent_chirp = thresh_silent_chirp

        self.monotonicity_thresh = monotonicity_thresh
        self.t_monotonicity = t_monotonicity
        self.kernel_version = kernel_version

        if range_bins is not None:
            self._n_distances = len(range_bins)
            self._range_bins = range_bins
        else:
            self._n_distances = self.n_samples if input_type == "complex" else self.n_samples // 2
            self._range_bins = range(self._n_distances)
        self._n_velocities = self.n_chirps

        # cupy init
        self.R = nrsp.utils.cu.complex_to_float(phasor_weights(self.n_samples)[0 : self._n_distances])
        self.D = nrsp.utils.cu.complex_to_float(cp.fft.fftshift(phasor_weights(self.n_chirps)))

        self._grad = cp.zeros((self.n_frames, self._n_distances, self._n_velocities), dtype=cp.float32)
        self._spinr_out = cp.zeros((self.n_frames, self._n_distances, self._n_velocities), dtype=cp.int32)

        self._inactive_neurons = cp.full((self.n_frames, self._n_distances, self._n_velocities), dtype=bool, fill_value=False)
        self._inactive_count = cp.zeros((self.n_frames, self.n_chirps * self.n_samples), dtype=cp.int32)  # count of inactive neurons at each timestep

        self._n_t_enc = self.n_chirps * self.n_samples - self.t_enc  # since we only spike from t=t_enc on
        self._spikes = cp.zeros((self.n_frames, self._n_t_enc, self._n_distances, self._n_velocities), dtype=bool)

        self._cfar_out = cp.zeros((self.n_frames, self._n_distances, self._n_velocities, len(self.alpha_cfar), len(self.k_cfar)), dtype=bool)

    def _spinr_params(self):
        spinr_params = cp.zeros(13, dtype=cp.float32)

        spinr_params[0] = self.alpha_grd
        spinr_params[1] = self.t_grd
        spinr_params[2] = self._encoding_func  # Log-Time-Encodeded spiking_function
        spinr_params[6] = self.tau
        spinr_params[7] = self.thresh
        spinr_params[8] = self.t_enc
        spinr_params[9] = self.thresh_silent
        spinr_params[10] = self.thresh_silent_chirp
        if self.kernel_version == "v2":
            spinr_params[11] = self.monotonicity_thresh  # for v2
            spinr_params[12] = self.t_monotonicity  # for v2

        return spinr_params

    def forward(self, x):
        # x (n_frames, n_chirps, n_samples, 2)
        self._forward_spinr(x)
        self._forward_cfar()

    def _forward_spinr(self, x):
        kernel = rd_spinr_turnoff_v2_float_kernel if self.kernel_version == "v2" else rd_spinr_turnoff_float_kernel
        kernel(
            (self._n_velocities, self.n_frames, 1),  # grid shape
            (self._n_distances, 1, 1),  # block shape
            (
                x,
                self.R,
                self.D,
                cp.int32(self.n_samples),
                cp.int32(self.n_chirps),
                self._spinr_params(),
                self._grad,
                self._spinr_out,
                self._spikes,
                self._inactive_neurons,
                self._inactive_count,
            ),
        )

    def _forward_cfar(self):
        n_t = self._spikes.shape[1]
        t_inhib = -self.tau * np.log(1 / np.array(self.alpha_cfar))
        t_inhib = np.rint(t_inhib)
        t_inhib = cp.array(t_inhib, dtype=cp.int32)
        k_cfar = cp.array(self.k_cfar, dtype=cp.int32)
        cfar_kernel = cp.array(get_cfar_kernel_2d(self.guard_cells, self.ref_cells), dtype=bool)  # neighbor connections

        log_os_cfar_kernel(
            (self._n_velocities, self.n_frames, 1),  # grid shape
            (self._n_distances, 1, 1),  # block shape
            (
                self._spikes,
                n_t,
                cfar_kernel,
                cp.array(cfar_kernel.shape, dtype=cp.int32),
                self.wrap_x,
                self.wrap_y,
                t_inhib,
                cp.int32(t_inhib.size),
                k_cfar,
                cp.int32(k_cfar.size),
                self._cfar_out,
            ),
        )

    def mean_percent_inactive(self):
        """Returns the mean percentage of inactive neurons over all frames."""
        inactive_count = self._inactive_count.get()  # shape (n_frames, n_timesteps)

        # Cumulative sum over timesteps
        inactive_cumsum = np.cumsum(inactive_count, axis=1)
        # Final timestep gives total inactive neurons per frame
        mean_inactive = np.mean(inactive_cumsum, axis=1)
        total_neurons = self._n_distances * self._n_velocities
        percent_inactive = mean_inactive / total_neurons * 100

        return percent_inactive

    def mean_active_per_timestep(self):
        """Returns the mean percentage of active neurons over all frames at each timestep."""
        inactive_count = self._inactive_count.get()  # shape (n_frames, n_timesteps)

        # Cumulative sum over timesteps
        inactive_cumsum = np.cumsum(inactive_count, axis=1)
        mean_inactive = np.mean(inactive_cumsum, axis=0)  # shape (n_timesteps,)

        total_neurons = self._n_distances * self._n_velocities
        mean_active = 1 - mean_inactive / total_neurons
        return mean_active * 100

    def grad(self):
        return self._grad.get()

    def spinr_out(self):
        return self._spinr_out.get()

    def cfar_out(self):
        """Returns a boolean array of shape (n_frames, n_distances, n_velocities, len(alpha_cfar), len(k_cfar))"""
        return self._cfar_out.get()

    def get_resonant_neurons(self):
        """Returns a boolean array of shape (n_frames, n_distances, n_velocities) indicating which neurons are resonant,
        i.e. which neurons have not turned off by the end of the input sequence because of early turnoff (no turnoff because of spike)."""
        return np.logical_not(self._inactive_neurons.get())
