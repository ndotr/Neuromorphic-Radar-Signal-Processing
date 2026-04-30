"""
CuRD_SpiNR_AGS_LogOSCFAR_Float_Model (single-frame forward)

Update: time-resolved spike tensor is OPTIONAL.

- Default: do NOT allocate / record spikes over time.
  You always get:
    - out  (spike time per neuron, int32)
    - grad (float32)
    - inactive bookkeeping (inactive_neurons, inactive_count)

- If record_spikes=True:
  - allocate spikes tensor [n_t, n_distances, n_velocities]
  - kernel writes to spikes (time-resolved)
  - CFAR can run on spikes (enable_cfar requires record_spikes)

Kernel ABI (single-frame) requirement:
  The CUDA kernel must accept:
      const int record_spikes
  and treat `spikes` as optional when record_spikes==0.

This class assumes your UPDATED single-frame kernel:
  void rd_spinr_ags_float_kernel(
      x: float32 [n_chirps, n_samples, 2],
      R: float32 [n_distances, 2],
      D: float32 [n_velocities, 2],
      n_samples, n_chirps,
      params: float32[13],
      grad: float32 [n_distances, n_velocities],
      out:  int32   [n_distances, n_velocities],
      spikes: bool* (optional) [n_t, n_distances, n_velocities],
      inactive_neurons: bool [n_distances, n_velocities],
      inactive_count: int32 [n_chirps*n_samples],
      record_spikes: int
  )

Launch layout:
  grid  = (n_velocities, 1, 1)
  block = (n_distances, 1, 1)

Forward pass processes exactly ONE frame at a time.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union, List

import cupy as cp
import numpy as np

import nrsp.utils.cu
from nrsp.algs.cu.kernels.log_os_cfar import log_os_cfar_kernel
from nrsp.algs.cu.kernels.rd_spinr_ags_float import rd_spinr_ags_float_kernel
from nrsp.algs.cu.kernels.rd_spinr_cfar_float import rd_spinr_cfar_float_kernel
from nrsp.utils.radar import phasor_weights
from nrsp.utils.cfar import get_cfar_kernel_2d


Number = Union[int, float, np.number]


def _as_list(x: Union[Number, Sequence[Number]]) -> List[Number]:
    if isinstance(x, (list, tuple)):
        return list(x)
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        return list(x)
    return [x]


class CuRD_SpiNR_AGS_LogOSCFAR_Float_Model:
    """
    Single-frame GPU model:
      1) SpiNR neuron simulation on Range-Doppler grid using rd_spinr_ags_float_kernel
      2) Optional Log-OS-CFAR inhibition / detection layer (requires record_spikes=True)
    """

    def __init__(
        self,
        *,
        # dimensions (single frame)
        n_chirps: int,
        n_samples: int,
        # spiNR params
        alpha_grd: float,
        tau: float,          # tau_log
        thresh: float,       # thresh_log
        # Turnoff controls
        thresh_silent: int = -1,
        thresh_silent_chirp: int = 1,
        monotonicity_thresh: float = -1.0,
        t_monotonicity: int = 0,
        # Gradient
        t_grd: int = 0,
        # Spiking
        t_enc: int = 0,
        spike_func: int = 0,
        # Range bin selection
        range_bins: Optional[Sequence[int]] = None,
        input_type: str = "complex",
        # Spike recording toggle
        record_spikes: bool = False,
        # CFAR parameters (requires record_spikes=True)
        enable_cfar: bool = False,
        alpha_cfar: Union[Number, Sequence[Number]] = (1.5,),
        k_cfar: Union[int, Sequence[int]] = (8,),
        guard_cells: Sequence[int] = (2, 2),
        ref_cells: Sequence[int] = (4, 4),
        wrap_x: bool = False,
        wrap_y: bool = False,
        # Clearing policy
        clear_buffers_each_forward: bool = True,
    ):
        if input_type not in ("real", "complex"):
            raise ValueError("input_type must be 'real' or 'complex'")
        if n_chirps <= 0 or n_samples <= 0:
            raise ValueError("n_chirps, n_samples must be positive")
        if t_enc < 0:
            raise ValueError("t_enc must be >= 0")
        if spike_func != 0:
            raise ValueError("Only spike_func==0 is supported (log-time spiking).")


        self.n_chirps = int(n_chirps)
        self.n_samples = int(n_samples)

        self.alpha_grd = float(alpha_grd)
        self.tau = float(tau)
        self.thresh = float(thresh)

        self.t_grd = int(t_grd)
        self.t_enc = int(t_enc)
        self.spike_func = int(spike_func)

        self.thresh_silent = int(thresh_silent)
        self.thresh_silent_chirp = int(thresh_silent_chirp)
        self.monotonicity_thresh = float(monotonicity_thresh)
        self.t_monotonicity = int(t_monotonicity)

        self.record_spikes = bool(record_spikes)
        self.enable_cfar = bool(enable_cfar)
        self.wrap_x = bool(wrap_x)
        self.wrap_y = bool(wrap_y)

        self.clear_buffers_each_forward = bool(clear_buffers_each_forward)

        # CFAR config
        self.alpha_cfar = [float(a) for a in _as_list(alpha_cfar)]
        self.k_cfar = [int(k) for k in _as_list(k_cfar)]
        self.guard_cells = tuple(guard_cells)
        self.ref_cells = tuple(ref_cells)

        if self.enable_cfar and not self.record_spikes:
            raise ValueError("enable_cfar=True requires record_spikes=True (CFAR operates on spikes tensor).")

        if self.enable_cfar:
            self.kernel = rd_spinr_cfar_float_kernel
        else:
            self.kernel = rd_spinr_ags_float_kernel

        # Range bins
        if range_bins is not None:
            self._range_bins = list(range_bins)
            self._n_distances = len(self._range_bins)
        else:
            self._n_distances = self.n_samples if input_type == "complex" else self.n_samples // 2
            self._range_bins = list(range(self._n_distances))

        self._n_velocities = self.n_chirps

        n_total_t = self.n_chirps * self.n_samples
        if self.t_enc > n_total_t:
            raise ValueError(f"t_enc={self.t_enc} must be <= n_chirps*n_samples={n_total_t}")

        self._n_t = n_total_t - self.t_enc

        # -----------------------------
        # Precompute phasors on GPU
        # -----------------------------
        self.R = nrsp.utils.cu.complex_to_float(
            phasor_weights(self.n_samples)[0:self._n_distances]
        )
        self.D = nrsp.utils.cu.complex_to_float(
            cp.fft.fftshift(phasor_weights(self.n_chirps))
        )

        # -----------------------------
        # Allocate GPU buffers (single-frame)
        # -----------------------------
        self._grad = cp.zeros((self._n_distances, self._n_velocities), dtype=cp.float32)
        self._out = cp.zeros((self._n_distances, self._n_velocities), dtype=cp.int32)

        self._inactive_neurons = cp.zeros((self._n_distances, self._n_velocities), dtype=cp.bool_)
        self._inactive_count = cp.zeros((self.n_chirps * self.n_samples), dtype=cp.int32)

        self._spikes = None
        if self.record_spikes:
            self._spikes = cp.zeros((self._n_t, self._n_distances, self._n_velocities), dtype=cp.bool_)

        self._cfar_out = None
        if self.enable_cfar:
            self._cfar_out = cp.zeros(
                (self._n_distances, self._n_velocities, len(self.alpha_cfar), len(self.k_cfar)),
                dtype=cp.bool_,
            )

        # Cache CFAR tensors
        self._cfar_kernel = None
        self._cfar_kernel_shape = None
        self._t_inhib = None
        self._k_cfar_gpu = None
        if self.enable_cfar:
            self._build_cfar_cache()

        # Dummy pointer for spikes when record_spikes=False
        self._dummy_spikes = cp.empty((1,), dtype=cp.bool_)

    def _spinr_params(self) -> cp.ndarray:
        params = cp.zeros(13, dtype=cp.float32)
        params[0] = self.alpha_grd
        params[1] = float(self.t_grd)
        params[2] = float(self.spike_func)
        # reserved [3..5]
        params[6] = self.tau
        params[7] = self.thresh
        params[8] = float(self.t_enc)
        params[9] = float(self.thresh_silent)
        params[10] = float(self.thresh_silent_chirp)
        params[11] = float(self.monotonicity_thresh)
        params[12] = float(self.t_monotonicity)
        return params

    def _build_cfar_cache(self) -> None:
        cfar_kernel_host = get_cfar_kernel_2d(self.guard_cells, self.ref_cells)
        self._cfar_kernel = cp.asarray(cfar_kernel_host, dtype=cp.bool_)
        self._cfar_kernel_shape = cp.asarray(self._cfar_kernel.shape, dtype=cp.int32)

        alpha_np = np.asarray(self.alpha_cfar, dtype=np.float64)
        if np.any(alpha_np <= 0):
            raise ValueError("alpha_cfar must be > 0")

        t_inhib = -1/self.tau * np.log(1.0 / alpha_np)
        t_inhib = np.rint(t_inhib).astype(np.int32)
        self._t_inhib = cp.asarray(t_inhib, dtype=cp.int32)

        k_np = np.asarray(self.k_cfar, dtype=np.int32)
        self._k_cfar_gpu = cp.asarray(k_np, dtype=cp.int32)

    def _clear_buffers(self) -> None:
        self._grad.fill(0.0)
        self._out.fill(0)
        self._inactive_neurons.fill(False)
        self._inactive_count.fill(0)

        if self._spikes is not None:
            self._spikes.fill(False)

        if self._cfar_out is not None:
            self._cfar_out.fill(False)

    def forward(self, x: cp.ndarray) -> None:
        """
        Single-frame forward.

        x must be cupy float32, shape (n_chirps, n_samples, 2).
        """
        if not isinstance(x, cp.ndarray):
            raise TypeError("x must be a cupy.ndarray")
        if x.dtype != cp.float32:
            raise TypeError(f"x must be float32, got {x.dtype}")
        if x.ndim != 3 or x.shape != (self.n_chirps, self.n_samples, 2):
            raise ValueError(
                f"x must have shape (n_chirps, n_samples, 2) = "
                f"({self.n_chirps}, {self.n_samples}, 2), got {x.shape}"
            )

        if self.clear_buffers_each_forward:
            self._clear_buffers()

        spikes_ptr = self._spikes if self._spikes is not None else self._dummy_spikes
        record_spikes_flag = cp.int32(1 if self._spikes is not None else 0)

        # Single-frame launch: NO frame dimension in grid
        self.kernel(
            (self._n_velocities, 1, 1),      # grid: (doppler, 1, 1)
            (self._n_distances, 1, 1),       # block: (distance, 1, 1)
            (
                x,
                self.R,
                self.D,
                cp.int32(self.n_samples),
                cp.int32(self.n_chirps),
                self._spinr_params(),
                self._grad,
                self._out,
                spikes_ptr,
                self._inactive_neurons,
                self._inactive_count,
                record_spikes_flag,
            ),
        )

        if self.enable_cfar:
            if self._cfar_kernel is None:
                self._build_cfar_cache()
            self._forward_cfar()

    def _forward_cfar(self) -> None:
        if self._spikes is None:
            raise RuntimeError("CFAR requested but spikes tensor is not allocated (record_spikes=False).")

        n_t = cp.int32(self._spikes.shape[0])

        # IMPORTANT: CFAR kernel ABI must match your actual log_os_cfar_kernel implementation.
        # This call assumes a spikes tensor WITHOUT frame dimension.
        log_os_cfar_kernel(
            (self._n_velocities, 1, 1),
            (self._n_distances, 1, 1),
            (
                self._spikes,
                n_t,
                self._cfar_kernel,
                self._cfar_kernel_shape,
                self.wrap_x,
                self.wrap_y,
                self._t_inhib,
                cp.int32(self._t_inhib.size),
                self._k_cfar_gpu,
                cp.int32(self._k_cfar_gpu.size),
                cp.int32(513),  # t_cut_max: max possible spike time
                self._cfar_out,
            ),
        )

    # -----------------------------
    # Getters (host)
    # -----------------------------
    def grad(self) -> np.ndarray:
        """(n_distances, n_velocities)"""
        return self._grad.get()

    def out(self) -> np.ndarray:
        """Spike time per neuron: (n_distances, n_velocities)"""
        return self._out.get()

    def spikes(self) -> Optional[np.ndarray]:
        """(n_t, n_distances, n_velocities) if record_spikes else None"""
        return None if self._spikes is None else self._spikes.get()

    def cfar_out(self) -> Optional[np.ndarray]:
        """(n_distances, n_velocities, n_alpha, n_k) if enable_cfar else None"""
        return None if self._cfar_out is None else self._cfar_out.get()

    def resonant_neurons(self) -> np.ndarray:
        """True where neuron did NOT turn off early."""
        return np.logical_not(self._inactive_neurons.get())

    def inactive_count(self) -> np.ndarray:
        """(n_chirps*n_samples,) turnoffs per timestep (single-frame)."""
        return self._inactive_count.get()

    def final_percent_inactive(self) -> float:
        """
        Percent inactive neurons at END of the sequence (single-frame scalar).
        """
        inactive_count = self._inactive_count.get()
        inactive_final = int(np.sum(inactive_count))
        total_neurons = self._n_distances * self._n_velocities
        return (inactive_final / total_neurons) * 100.0