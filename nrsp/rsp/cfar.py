import numpy as np
import typing as ty

try:
    import cupy as cp
    from cupyx.scipy.signal import convolve2d
except:
    import numpy as cp
    from scipy.signal import convolve2d

from nrsp.utils.cfar import get_cfar_kernel_2d


def ca_cfar_2d_thresholds(data, kernel, normalize=True):
    """
    Create CA - CFAR Threshold Maps by performing averaging via convolve2D.
    This map CA can by used within the peak detection formula:
        data > factor * CA + offset

    Args:
        data (np.array):    Array of RA Maps with shape (n_frames, n_distances, n_angles).
        kernel (np.array):  Kernel for convolution.

    Returns:
        (np.array):     CA Map array with shape (n_frames, n_distances, n_angles).
    """

    if normalize:
        data = cp.einsum("ijk,i->ijk", data, 1 / cp.max(data, axis=(1, 2)))

    ca_thresholds = cp.zeros_like(data)
    for i in range(data.shape[0]):
        ca_thresholds[i] = (
            convolve2d(
                cp.pad(data[i], pad_width=((kernel.shape[0], kernel.shape[0]), (kernel.shape[1], kernel.shape[1])), mode="symmetric"),
                kernel,
                mode="same",
            )
            / cp.sum(kernel)
        )[kernel.shape[0] : -kernel.shape[0], kernel.shape[1] : -kernel.shape[1]]
    return ca_thresholds


def ca_cfar_2d_detections(data, ca_thresholds, factor, offset, normalize=True, detection_area=None):
    """
    Create Peak Maps by performing CA-CFAR.
    -> data > factor * CA + offset

    Args:
        data (np.array):    Array of RA Maps with shape (n_frames, n_distances, n_angles).
        kernel (np.array):  Kernel for convolution.
        factor (int):       Factor.
        offset (int):       Offset.
        kernel (np.array):  Kernel to extend the peak maps.
        normalize (bool):   Whether to normalize the data.

    Returns:
        (np.array):     Peak Maps array with shape (n_frames, n_distances, n_angles).
    """

    # TODO: undirty it
    # data = data[...,:200,:]
    # ca_maps = ca_maps[...,:200,:]

    peak_maps = cp.zeros_like(data)

    if normalize:
        data = cp.einsum("ijk,i->ijk", data, 1 / np.max(data, axis=(1, 2)))

    # CA CFAR
    cfar_maps = (ca_thresholds + offset) * factor
    peak_maps = cp.array(data > (cfar_maps), dtype="float")

    if detection_area is not None:
        extended_peak_maps = cp.zeros_like(peak_maps)
        for i in range(peak_maps.shape[0]):
            extended_peak_maps[i] = convolve2d(peak_maps[i], detection_area, mode="same") > 0
        peak_maps = extended_peak_maps

    return peak_maps


def os_cfar_2d_detections(
    data: np.array,
    guard_cells: ty.Tuple[int, int],
    reference_cells: ty.Tuple[int, int],
    k_values: ty.List[int],
    alpha_values: ty.List[float],
    wrap_x: bool = False,
    wrap_y: bool = False,
):
    assert data.ndim == 2

    out = np.empty(data.shape + (len(alpha_values), len(k_values)))
    kernel = get_cfar_kernel_2d(guard_cells, reference_cells)
    dx = guard_cells[0] + reference_cells[0]
    dy = guard_cells[1] + reference_cells[1]

    if wrap_x and wrap_y:
        data_pad = np.pad(data, ((dx, dx), (dy, dy)), "wrap")

    if wrap_x:
        data_pad = np.pad(data, ((dx, dx), (0, 0)), "wrap")
        data_pad = np.pad(data_pad, ((0, 0), (dy, dy)), "reflect")
    if wrap_y:
        data_pad = np.pad(data, ((0, 0), (dy, dy)), "wrap")
        data_pad = np.pad(data_pad, ((dx, dx), (0, 0)), "reflect")

    if not wrap_x and not wrap_y:
        data_pad = np.pad(data, ((dx, dx), (dy, dy)), "reflect")

    for i, j in np.ndindex(data.shape):
        neighbors = data_pad[
            i : i + 2 * dx + 1,
            j : j + 2 * dy + 1,
        ]
        t = np.sort(neighbors[kernel], axis=None)
        for a_idx, a in enumerate(alpha_values):
            for k_idx, k in enumerate(k_values):
                out[i, j, a_idx, k_idx] = data[i, j] > a * t[-k]

    return out


def ca_cfar_2d_threshold(
    data: np.array,
    guard_cells: ty.Tuple[int, int],
    reference_cells: ty.Tuple[int, int],
    wrap_x: bool = False,
    wrap_y: bool = False,
):
    assert data.ndim == 2

    kernel = get_cfar_kernel_2d(guard_cells, reference_cells)
    dx = guard_cells[0] + reference_cells[0]
    dy = guard_cells[1] + reference_cells[1]

    if wrap_x and wrap_y:
        data_pad = np.pad(data, ((dx, dx), (dy, dy)), "wrap")

    if wrap_x:
        data_pad = np.pad(data, ((dx, dx), (0, 0)), "wrap")
        data_pad = np.pad(data_pad, ((0, 0), (dy, dy)), "reflect")
    if wrap_y:
        data_pad = np.pad(data, ((0, 0), (dy, dy)), "wrap")
        data_pad = np.pad(data_pad, ((dx, dx), (0, 0)), "reflect")

    if not wrap_x and not wrap_y:
        data_pad = np.pad(data, ((dx, dx), (dy, dy)), "reflect")

    thresh = convolve2d(
        cp.asarray(data_pad),
        cp.asarray(kernel),
        mode="valid",
    ).get() / np.sum(kernel)

    return thresh
