import numpy as np
import cupy as cp
import cupyx.scipy.signal

def snr_1d(data, binary_target):
    """

    Args:
        data (np.array):    .

    Returns:
        (np.array):     .
    """

    snr = data[binary_target]/(cp.sum(data)+1e-16)

    return snr