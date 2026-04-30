import numpy as np


def get_phase_shift_weights(n_angles, n_rx):
    # Weight matrix
    # weight value: exp(i*phi) with phi = 2*pi*f_distance*d*cos(theta)*n

    # Uniformly spaced phase shifts converted to angles
    # angles = np.arcsin(2*np.linspace(-n_angles//2+1, n_angles//2, n_angles)/n_angles)

    W = np.zeros((n_angles, n_rx)).astype("complex64")
    a_ranges = np.zeros(n_angles).astype("float")
    for a in range(n_angles):
        for rx in range(n_rx):
            phi = 2 * np.pi * rx * (a - n_angles // 2) / n_angles
            if phi == 0:
                W[a, rx] = 1
            else:
                W[a, rx] = np.exp(-1j * phi)  # *np.exp(-((rx-n_rx/2)/n_rx)**2)
        a_ranges[a] = -np.arcsin(2 * (a - n_angles // 2) / n_angles)

    return W, a_ranges


def get_omegas(n_distances, n_angles, n_samples):

    omegas = np.zeros((n_distances, n_angles))

    for i in range(n_angles):
        omegas[:, i] = np.linspace(0, n_distances - 1, n_distances) / (n_distances * 2) * np.pi * 2

    return omegas.astype("float32")


def get_rotation_vector(n_distances, n_samples):

    omega = np.linspace(0, n_distances - 1, n_distances) / (n_distances * 2) * np.pi * 2
    # omega = np.linspace(0,1,n_samples+1)*np.pi*2
    # omega = omega[:n_distances]
    R = np.exp(1j * omega).astype("complex64")
    # R[0] = 0
    # R[1] = 0

    return R


def phasor_weights(
    n_samples,
    distance_bins: list = None,
):
    """Generate complex phasor weights for distance estimation.
    If distance_bins is given, onyl return the weights restricted to those indices/bins.
    """

    wgts = np.exp(1j * np.linspace(0, 1 - 1 / n_samples, n_samples) * np.pi * 2)
    if distance_bins is not None:
        wgts = wgts[distance_bins]
    return wgts


def steering_weights(
    n_channels,
    angle_bins: list = None,
):
    """Generate steering weight matrix of shape (angles, channels).
    If angle_bins is not given, the full angle bins are returned.
    The weight matrix is normalized by n_channels.
    """

    # assuming lambda/2 antenna spacing
    W = np.zeros((n_channels, n_channels), dtype="complex128")

    for o in range(n_channels):
        for i in range(n_channels):
            phi = 2 * np.pi * i * (o - n_channels // 2) / n_channels
            W[o, i] = np.exp(-1j * phi)

    # normalize by in_dims
    W /= n_channels
    # restrict to angle bins
    if angle_bins is not None:
        W = W[angle_bins]
    return W
