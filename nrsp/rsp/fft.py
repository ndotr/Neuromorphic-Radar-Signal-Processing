import numpy as np


def range_angle_fft(radar_data, remove_zero_modes=4):
    """
    Apply Range-Angle FFT on radar data with specific format:
    radar_data[n_frames, n_chirps, n_samples, n_vx].

    For float data:     n_range = n_samples // 2
    For complex data:   n_range = n_samples

    Args:
        radar_data (np.array):      Radar data (float or complex).
        remove_zero_modes (int):    Number of frequency bins to remove [0:n].

    Return:
        np.array:   [n_frames, n_chirps, n_range, n_angles=n_vx]
    """

    # Get shape of radar data
    n_samples = radar_data.shape[-2]
    n_rx = radar_data.shape[-1]

    # Perform Range Angle FFT on float data
    if radar_data.dtype == np.float64:
        fft_data = (np.fft.fftshift((np.abs(np.fft.fft(np.fft.fft(
                        radar_data, axis=-1), axis=-2)))[...,0:n_samples//2,:], axes=-1))
    # Perform Range Angle FFT on float data
    elif radar_data.dtype == np.complex128:
        fft_data = (np.fft.fftshift((np.abs(np.fft.fft(np.fft.fft(
                        radar_data, axis=-1), axis=-2)))[...,:,:], axes=-1))

    # Set first 'remove_zero_modes'th frequency bins to 0
    fft_data[..., :remove_zero_modes,n_rx//2] = 0

    return fft_data

def range_doppler_fft(radar_data):
    """
    Apply Range-Doppler FFT on radar data with specific format:
    radar_data[n_frames, n_chirps, n_samples, n_vx].

    For float data:     n_range = n_samples // 2
    For complex data:   n_range = n_samples

    Args:
        radar_data (np.array):      Radar data (float or complex).

    Return:
        np.array:   [n_frames, n_chirps, n_range, n_angles=n_vx]
    """

    # Get shape of radar data
    n_samples = radar_data.shape[-2]
    n_rx = radar_data.shape[-1]

    # Perform Range Angle FFT on float data
    if radar_data.dtype == np.float64:
        fft_data = (np.fft.fftshift((np.abs(np.fft.fft(np.fft.fft(
                        radar_data, axis=-3), axis=-2)))[...,0:n_samples//2,:], axes=-3))
    # Perform Range Angle FFT on float data
    elif radar_data.dtype == np.complex128:
        fft_data = (np.fft.fftshift((np.abs(np.fft.fft(np.fft.fft(
                        radar_data, axis=-3), axis=-2)))[...,:,:], axes=-3))

    return fft_data
