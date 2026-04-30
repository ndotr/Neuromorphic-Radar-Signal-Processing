import numpy as np


def log_encoding(values, tau, thresh):
    eps = 10**-12
    enc = -tau * np.log((values + eps) / thresh)
    return enc


def log_encoding_snn(values, tau, thresh, n_timesteps):
    # enc = n_timesteps incase a neuron did not spike with n_timesteps
    enc = np.full_like(values, n_timesteps, dtype=int)
    u = np.zeros_like(values, dtype=float)

    for t in range(n_timesteps):
        u += (values + u) / tau

        spikes = np.asarray(u > (thresh - values))
        spikes = np.logical_and(enc == n_timesteps, spikes)
        enc[spikes] = t

    return enc


def log_encoding_snn_nx(values, tau, thresh, n_timesteps):
    # enc = n_timesteps incase a neuron did not spike with n_timesteps
    enc = np.full_like(values, n_timesteps, dtype=int)
    u = np.zeros_like(values, dtype=int)
    # u += 29

    for t in range(n_timesteps):
        u += ((values + u) / tau).astype(int)

        spikes = np.asarray(u > (thresh - values))
        spikes = np.logical_and(enc == n_timesteps, spikes)
        enc[spikes] = t

    return enc


def rate_encoding(values, tau, thresh, rest):
    # eps = 10**-12
    enc = -1 / (tau * np.log(1 - thresh / (values + rest)))
    return enc


def rate_encoding_snn(values, tau, thresh, rest, n_timesteps):
    enc = np.zeros_like(values, dtype=int)
    u = np.zeros_like(values, dtype=float)

    for t in range(n_timesteps):
        u += (values + rest - u) / tau

        spikes = np.asarray(u > thresh)
        u[spikes] -= thresh
        enc[spikes] += 1

    return enc / n_timesteps


def rate_encoding_snn_nx(values, tau, thresh, rest, n_timesteps):
    enc = np.zeros_like(values, dtype=int)
    u = np.zeros_like(values, dtype=int)

    for t in range(n_timesteps):
        u += ((values + rest - u) / tau).astype(int)

        spikes = np.asarray(u > thresh)
        u[spikes] -= int(thresh)
        enc[spikes] += 1

    return enc / n_timesteps
