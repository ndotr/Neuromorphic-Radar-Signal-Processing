try:
    import cupy as cp
except:
    import numpy as cp

import scipy

def subtract_gauss_filter(radar_data, sigma, axis=-2):
    
    return radar_data - scipy.ndimage.gaussian_filter1d(radar_data, sigma=sigma, axis=-2)

def subtract_exp_filter(radar_data, alpha):

    n_samples = radar_data.shape[-2]
    n_frames = radar_data.shape[0]
    n_chirps = radar_data.shape[1]
    n_rx = radar_data.shape[-1]

    exp_filtered_radar_data = cp.zeros_like(radar_data[:,:,:,:], dtype=radar_data.dtype)

    for f in range(n_frames):
        for c in range(n_chirps):
            tmp = cp.zeros(n_rx)
            for i in range(n_samples):
                if i == 0:
                    tmp = radar_data[f,c,i]
                else:
                    tmp = alpha*radar_data[f,c,i] + (1-alpha)*tmp
                exp_filtered_radar_data[f,c,i] = tmp

    return radar_data - exp_filtered_radar_data

def subtract_lin_filter(radar_data):

    n_samples = radar_data.shape[-2]
    n_frames = radar_data.shape[0]
    n_chirps = radar_data.shape[1]
    n_rx = radar_data.shape[-1]

    x = cp.linspace(0, 1, n_samples)
    linear_radar_data = cp.zeros((n_frames, n_chirps, n_samples, n_rx))
    for f in range(n_frames):
        for c in range(n_chirps):
            for rx in range(n_rx):
                y = radar_data[f,c,:,rx]
                coef = cp.polyfit(x,y,1)
                poly1d_fn = cp.poly1d(coef) 
                linear_radar_data[f, c, :, rx] = poly1d_fn(x)

    return radar_data - linear_radar_data

def subtract_mean(radar_data):
    n_samples = radar_data.shape[-2]

    return radar_data - cp.repeat(cp.expand_dims(cp.mean(radar_data, axis=-2), axis=-2), n_samples, axis=-2)

def hann_window(radar_data):

    n_frames, n_chirps, n_samples, n_vx = cp.shape(radar_data)

    # Precalculate the window
    window = cp.hanning(n_samples)
    window = window * n_samples / cp.sum(window)

    # Apply Window
    radar_data = cp.einsum('frsa,s->frsa', radar_data, window)

    return radar_data
