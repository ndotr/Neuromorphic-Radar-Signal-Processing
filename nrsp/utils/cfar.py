import numpy as np
import typing as ty
import numpy.typing as npt


def get_cfar_kernel_2d(
    guard_cells: ty.Tuple[int, int],
    ref_cells: ty.Tuple[int, int],
) -> npt.NDArray[np.bool_]:
    """Build 2D CFAR kernel. Kernel is true for reference cells and false for CUT and guard cells."""

    shape = (
        2 * (guard_cells[0] + ref_cells[0]) + 1,
        2 * (guard_cells[1] + ref_cells[1]) + 1,
    )
    kernel = np.full(shape, fill_value=True, dtype=bool)

    centre_x = shape[0] // 2
    centre_y = shape[1] // 2

    kernel[
        centre_x - guard_cells[0] : centre_x + guard_cells[0] + 1,
        centre_y - guard_cells[1] : centre_y + guard_cells[1] + 1,
    ] = False

    return kernel


def get_cfar_kernel_1d(
    guard_cells: int,
    ref_cells: int,
) -> npt.NDArray[np.bool_]:
    """Build 1D CFAR kernel. Kernel is true for reference cells and false for CUT and guard cells."""

    len = 2 * (guard_cells + ref_cells) + 1
    kernel = np.full((len,), fill_value=True, dtype=bool)

    centre = len // 2
    kernel[centre - guard_cells : centre + guard_cells + 1] = False

    return kernel


def find_best_ca_cfar_weight_scale_nx(alpha: float, N: int) -> float:
    # cfar equation:
    #       n_spikes_cut - alpha * (n_spikes_neighbors / N)> 0
    # Weights would be
    #  - 1 for CUT
    #  - -alpha/N for neighbor neurons

    # weights can only be integers in range (-128, 127)
    # we want to find the best value to scale the weights, so that the abs error is minimal

    def err(scale, alpha, N):
        return np.abs(np.rint(scale) - scale) + np.abs(np.rint(scale * alpha / N) - scale * alpha / N)

    scale_range = np.linspace(max(1 / 2, N / (2 * alpha)), min(127, 127 * N / alpha), 10000)

    # Compute function values
    f_values = err(scale_range, alpha, N)

    min_index = np.argmin(f_values)
    scale = scale_range[min_index]
    return scale


def calculate_num_ref_cells(guard_cells: ty.Tuple[int, int], ref_cells: ty.Tuple[int, int]):
    """Calculate number of refence cell for 2D CFAR window. ref_cells refers to the number of reference cells in eitehr direction, starting form the end of the guard cells."""

    ref_row, ref_col = ref_cells
    guard_row, guard_col = guard_cells

    total_window = (2 * (ref_row + guard_row) + 1) * (2 * (ref_col + guard_col) + 1)
    guard_window = (2 * guard_row + 1) * (2 * guard_col + 1)

    reference_cells = total_window - guard_window
    return reference_cells

def ca_cfar_2d(data, guard_cells, ref_cells, alpha=1.0, offset=0.0, dtype=np.int32):
    """
    Apply 2D CA-CFAR detection on a 2D input array with zero-padding.

    Parameters:
    - data: 2D numpy array (e.g., range-Doppler map)
    - guard_cells: tuple (g_row, g_col) guard region half-widths
    - ref_cells: tuple (r_row, r_col) reference region half-widths
    - alpha: float, scaling factor for threshold (depends on desired Pfa)
    - offset: float, constant value added to threshold (optional)
    - dtype: numpy dtype (e.g., np.float32, np.float64) for all float calculations

    Returns:
    - detection_map: binary 2D numpy array of type uint8
    - threshold_map: 2D numpy array with dtype `dtype`
    """

    # Cast input to desired precision
    data = np.asarray(data, dtype=dtype)

    g_row, g_col = guard_cells
    r_row, r_col = ref_cells

    pad_row = g_row + r_row
    pad_col = g_col + r_col

    # Zero-pad the input with dtype consistency
    padded_data = np.pad(data, 
                         ((pad_row, pad_row), (pad_col, pad_col)), 
                         mode='constant', 
                         constant_values=0).astype(dtype)

    rows, cols = data.shape
    detection_map = np.zeros((rows, cols), dtype=np.uint8)
    threshold_map = np.zeros((rows, cols), dtype=dtype)

    alpha = dtype(alpha)
    offset = dtype(offset)

    for i in range(rows):
        for j in range(cols):
            i_p = i + pad_row
            j_p = j + pad_col

            window = padded_data[i_p - pad_row:i_p + pad_row + 1,
                                 j_p - pad_col:j_p + pad_col + 1]

            guard_mask = np.ones_like(window, dtype=bool)
            guard_start_row = pad_row - g_row
            guard_end_row = pad_row + g_row + 1
            guard_start_col = pad_col - g_col
            guard_end_col = pad_col + g_col + 1
            guard_mask[guard_start_row:guard_end_row, guard_start_col:guard_end_col] = False

            reference_cells = window[guard_mask]
            noise_level = np.sum(reference_cells, dtype=dtype)

            threshold = alpha * noise_level + offset
            threshold_map[i, j] = threshold

            if data[i, j] > threshold:
                detection_map[i, j] = 1

    return detection_map, threshold_map

from scipy.signal import convolve2d

def ca_cfar_2d_convolution(data, guard_cells, ref_cells, alpha=1.0, offset=0.0, dtype=np.float32):
    """
    Apply 2D CA-CFAR using convolution.

    Parameters:
    - data: 2D numpy array (e.g., range-Doppler map)
    - guard_cells: tuple (g_row, g_col): half-sizes of the guard region
    - ref_cells: tuple (r_row, r_col): half-sizes of the reference region (excluding guard)
    - alpha: float: scaling factor for noise threshold
    - offset: float: constant added to threshold
    - dtype: numpy float type (e.g., np.float32, np.float64)

    Returns:
    - detection_map: uint8 binary map (1: detection, 0: no detection)
    - threshold_map: float threshold map
    """

    data = np.asarray(data, dtype=dtype)

    g_row, g_col = guard_cells
    r_row, r_col = ref_cells

    # Total mask size
    win_row = r_row + g_row
    win_col = r_col + g_col
    kernel_shape = (2 * win_row + 1, 2 * win_col + 1)

    # Create convolution kernel
    kernel = np.ones(kernel_shape, dtype=dtype)

    # Mask out the guard cells + CUT
    cut_r_start = win_row - g_row
    cut_r_end   = win_row + g_row + 1
    cut_c_start = win_col - g_col
    cut_c_end   = win_col + g_col + 1
    kernel[cut_r_start:cut_r_end, cut_c_start:cut_c_end] = 0

    num_reference_cells = np.sum(kernel)

    # Compute sum over reference cells
    ref_sum = convolve2d(data, kernel, mode='same', boundary='fill', fillvalue=0)
    noise_mean = ref_sum / num_reference_cells

    # Compute threshold map
    threshold_map = dtype(alpha) * noise_mean + dtype(offset)

    # Detection map
    detection_map = (data > threshold_map).astype(np.uint8)

    return detection_map, threshold_map

from scipy.signal import correlate2d

def ca_cfar_2d_xcorr(data, guard_cells, ref_cells, alpha=1.0, offset=0.0, dtype=np.float32):
    """
    Apply 2D CA-CFAR using cross-correlation.

    Parameters:
    - data: 2D numpy array (e.g., range-Doppler map)
    - guard_cells: tuple (g_row, g_col): half-sizes of the guard region
    - ref_cells: tuple (r_row, r_col): half-sizes of the reference region (excluding guard)
    - alpha: float: scaling factor for noise threshold
    - offset: float: constant added to threshold
    - dtype: numpy float type (e.g., np.float32, np.float64)

    Returns:
    - detection_map: uint8 binary map (1: detection, 0: no detection)
    - threshold_map: float array of threshold values
    """

    data = np.asarray(data, dtype=dtype)

    g_row, g_col = guard_cells
    r_row, r_col = ref_cells

    win_row = g_row + r_row
    win_col = g_col + r_col
    kernel_shape = (2 * win_row + 1, 2 * win_col + 1)

    # Create the kernel: 1s in reference region, 0 in guard+CUT
    kernel = np.ones(kernel_shape, dtype=dtype)
    cut_r_start = win_row - g_row
    cut_r_end   = win_row + g_row + 1
    cut_c_start = win_col - g_col
    cut_c_end   = win_col + g_col + 1
    kernel[cut_r_start:cut_r_end, cut_c_start:cut_c_end] = 0

    num_reference_cells = np.sum(kernel)

    # Apply 2D cross-correlation
    ref_sum = correlate2d(data, kernel, mode='same', boundary='fill', fillvalue=0)
    noise_mean = ref_sum / num_reference_cells

    threshold_map = dtype(alpha) * noise_mean + dtype(offset)

    detection_map = (data > threshold_map).astype(np.uint8)

    return detection_map, threshold_map

from scipy.signal import convolve2d
import numpy as np


def ca_cfar_2d_conv(
    data,
    guard_cells,
    ref_cells,
    alpha=1.0,
    offset=0.0,
    dtype=np.float32,
):
    """
    Apply 2D CA-CFAR using convolution.

    Parameters
    ----------
    data : np.ndarray
        2D input array, e.g. range-Doppler map.
    guard_cells : tuple[int, int]
        (g_row, g_col), half-sizes of the guard region around the CUT.
    ref_cells : tuple[int, int]
        (r_row, r_col), half-sizes of the reference region outside the guard region.
    alpha : float
        Multiplicative threshold scaling factor.
    offset : float
        Additive threshold offset.
    dtype : np.dtype
        Floating dtype used internally, e.g. np.float32 or np.float64.

    Returns
    -------
    detection_map : np.ndarray
        uint8 binary detection map, shape equal to input.
    threshold_map : np.ndarray
        Threshold map, same shape as input, dtype=dtype.
    """
    data = np.asarray(data, dtype=dtype)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D input array, got shape {data.shape}")

    g_row, g_col = map(int, guard_cells)
    r_row, r_col = map(int, ref_cells)

    if g_row < 0 or g_col < 0 or r_row < 0 or r_col < 0:
        raise ValueError("guard_cells and ref_cells must be non-negative")

    # Full window half-size = reference extension + guard extension
    win_row = r_row + g_row
    win_col = r_col + g_col
    kernel_shape = (2 * win_row + 1, 2 * win_col + 1)

    # Reference mask: ones in reference area, zeros in guard region + CUT
    kernel = np.ones(kernel_shape, dtype=dtype)

    cut_r_start = win_row - g_row
    cut_r_end = win_row + g_row + 1
    cut_c_start = win_col - g_col
    cut_c_end = win_col + g_col + 1
    kernel[cut_r_start:cut_r_end, cut_c_start:cut_c_end] = 0

    # Sum over valid reference cells
    ref_sum = convolve2d(data, kernel, mode="same", boundary="fill", fillvalue=0)

    # Count how many valid reference cells contributed at each position
    valid_mask = np.ones_like(data, dtype=dtype)
    ref_count = convolve2d(valid_mask, kernel, mode="same", boundary="fill", fillvalue=0)

    # Edge-corrected local noise estimate
    noise_mean = np.divide(
        ref_sum,
        ref_count,
        out=np.zeros_like(ref_sum, dtype=dtype),
        where=ref_count > 0,
    )

    threshold_map = dtype(alpha) * noise_mean + dtype(offset)
    detection_map = (data > threshold_map).astype(np.uint8)

    return detection_map, threshold_map