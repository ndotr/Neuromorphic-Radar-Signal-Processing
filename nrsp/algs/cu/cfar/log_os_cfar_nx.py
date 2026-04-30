import cupy as cp
import numpy as np

import nrsp.utils.cu
from nrsp.algs.cu.kernels.ra_spinr_nx import ra_spinr_nx_kernel
from nrsp.algs.cu.kernels.log_os_cfar import log_os_cfar_kernel
from nrsp.utils.radar import phasor_weights, steering_weights
from nrsp.utils.cfar import get_cfar_kernel_2d


class Cu_RA_SpiNR_LogOsCfar_NX_Model:

    def __init__(
        self,
        n_frames,
        n_channels,
        n_samples,
        alpha_grd,
        tau,  # log encoding
        thresh,  # log encoding
        alpha_cfar,  # os cfar
        k_cfar,  # os cfar
        guard_cells,
        ref_cells,
        distance_bins=None,
        angle_bins=None,
        t_grd=0,
        t_enc=0,
        weight_shr=9,
        grd_shl=8,
    ):
        self.n_frames = n_frames
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.alpha_grd = alpha_grd
        self.tau = tau
        self.thresh = thresh

        if not isinstance(alpha_cfar, list):
            self.alpha_cfar = [alpha_cfar]
        else:
            self.alpha_cfar = alpha_cfar

        if not isinstance(k_cfar, list):
            self.k_cfar = [k_cfar]
        else:
            self.k_cfar = k_cfar

        self.guard_cells = guard_cells
        self.ref_cells = ref_cells
        self.t_grd = t_grd
        self.t_enc = t_enc

        self.distance_bins = range(n_samples)
        if distance_bins:
            self.distance_bins = distance_bins
        self._n_distances = len(self.distance_bins)

        self.angle_bins = range(n_channels)
        if angle_bins:
            self.angle_bins = angle_bins
        self._n_angles = len(self.angle_bins)

        self.wrap_x = False
        self.wrap_y = False  # wrap around angles

        self.weight_shr = weight_shr  # (-weight_exp)
        self.grd_shl = grd_shl

        # cupy init
        self.W = nrsp.utils.cu.complex_to_float(steering_weights(self.n_channels))
        self.W *= 127 * n_channels
        self.W = self.W.astype(cp.int32)
        self.R_range = nrsp.utils.cu.complex_to_float(phasor_weights(self.n_samples))

    def _spinr_params(self):
        spinr_params = cp.zeros(10, dtype=cp.float32)

        spinr_params[0] = self.alpha_grd
        spinr_params[1] = self.t_grd
        spinr_params[2] = 0  # Log-Time-Encodeded spiking_function
        spinr_params[6] = self.tau
        spinr_params[7] = self.thresh
        spinr_params[8] = self.t_enc

        return spinr_params

    def forward(self, data):
        assert data.dtype == "int32"

        # data (n_frames, n_samples, n_antennas, 2)
        _s = (self.n_frames, self._n_distances, self._n_angles)
        grad = cp.zeros(_s, dtype=cp.int32)
        spinr_out = cp.zeros(_s, dtype=cp.int32)
        spikes = cp.zeros((self.n_frames, self.n_samples, self._n_distances, self._n_angles), dtype=bool)
        cfar_out = cp.zeros((self.n_frames, self._n_distances, self._n_angles, len(self.alpha_cfar), len(self.k_cfar)), dtype=bool)

        self._forward_spinr(data, grad, spinr_out, spikes)
        self._forward_cfar(spikes, cfar_out)

        return grad.get(), spinr_out.get(), cfar_out.get()

    def _forward_spinr(self, data, grad, spinr_out, spikes):
        assert data.dtype == "int32"

        ra_spinr_nx_kernel(
            (self._n_angles, self.n_frames, 1),  # grid shape
            (self._n_distances, 1, 1),  # block shape
            (
                data,
                self.W,
                self.R_range,
                cp.int32(self.n_samples),
                cp.int32(self.n_channels),
                self._spinr_params(),
                grad,
                spinr_out,
                spikes,
                cp.array(self.distance_bins, dtype=cp.int32),
                cp.array(self.angle_bins, dtype=cp.int32),
                cp.int32(self.weight_shr),
                cp.int32(self.grd_shl),
            ),
        )

    def _forward_cfar(self, spikes, cfar_out):
        t_inhib = -self.tau * np.log(1 / np.array(self.alpha_cfar))
        t_inhib = np.rint(t_inhib)
        t_inhib = cp.array(t_inhib, dtype=cp.int32)

        k_cfar = cp.array(self.k_cfar, dtype=cp.int32)

        cfar_kernel = cp.array(get_cfar_kernel_2d(self.guard_cells, self.ref_cells), dtype=bool)  # neighbor connections

        log_os_cfar_kernel(
            (self._n_angles, self.n_frames, 1),  # grid shape
            (self._n_distances, 1, 1),  # block shape
            (
                spikes,
                cp.int32(self.n_samples),
                cfar_kernel,
                cp.array(cfar_kernel.shape, dtype=cp.int32),
                self.wrap_x,
                self.wrap_y,
                t_inhib,
                cp.int32(t_inhib.size),
                k_cfar,
                cp.int32(k_cfar.size),
                cfar_out,
            ),
        )
