# nrsp/radar/signal.py
import numpy as np
import nrsp.utils.log

# Get logger for this module

def if_signal(r, theta, v, n_samples, n_channels, n_chirps):
    """
    Generate the intermediate frequency (IF) radar signal for a single target.

    Parameters
    ----------
    r : float
        Range-dependent phase term.
    theta : float
        Angle-dependent phase term.
    v : float
        Velocity-dependent phase term.
    n_samples : int
        Number of time samples per chirp.
    n_channels : int
        Number of radar receiver channels.
    n_chirps : int
        Number of chirps per frame.

    Returns
    -------
    np.ndarray
        IF signal of shape (n_channels, n_chirps, n_samples)
    """

    logger = nrsp.utils.log.get_logger(__name__)
    logger.debug(f"Generating IF signal: r={r}, theta={theta}, v={v}, "
                 f"n_samples={n_samples}, n_channels={n_channels}, n_chirps={n_chirps}")

    # Create sample arrays
    time_samples = np.linspace(0, n_samples - 1, n_samples)
    time_chirps = np.linspace(0, n_chirps - 1, n_chirps)
    idx_channel = np.linspace(0, n_channels - 1, n_channels)

    # Compute IF signal as outer product across time, channels, chirps
    signal = np.einsum(
        "t,a,c->act",
        np.exp(2j * np.pi * r * time_samples / n_samples),  # time-phase term
        np.exp(2j * np.pi * theta * idx_channel),           # channel-phase term
        np.exp(2j * np.pi * v * time_chirps / n_chirps)    # chirp-phase term
    )

    logger.debug(f"IF signal generated with shape {signal.shape}")
    return signal


def get_radar_data(amps, ranges, velocities, angles, n_samples, n_channels, n_chirps, noise_std=None, rng=None):
    """
    Combine IF signals from multiple targets and optionally add noise.

    Parameters
    ----------
    amps : list[float]
        Amplitudes of each target.
    ranges : list[float]
        Range parameters for each target.
    velocities : list[float]
        Velocity parameters for each target.
    angles : list[float]
        Angle parameters for each target.
    n_samples : int
        Number of time samples per chirp.
    n_channels : int
        Number of radar receiver channels.
    n_chirps : int
        Number of chirps per frame.
    noise_std : float, optional
        Standard deviation of Gaussian noise to add.

    Returns
    -------
    np.ndarray
        Combined radar data of shape (n_channels, n_chirps, n_samples)
    """

    logger = nrsp.utils.log.get_logger(__name__)
    logger.info("Generating radar data for multiple targets")
    radar_data = None

    for i, (amp, r, v, theta) in enumerate(zip(amps, ranges, velocities, angles)):
        logger.debug(f"Processing target {i}: amp={amp}, r={r}, v={v}, theta={theta}")
        target_signal = amp * if_signal(r=r, theta=theta, v=v,
                                        n_samples=n_samples,
                                        n_channels=n_channels,
                                        n_chirps=n_chirps)
        if radar_data is None:
            radar_data = target_signal
        else:
            radar_data += target_signal

    # Add noise if requested
    if noise_std is not None and rng is not None:
        logger.debug(f"Adding Gaussian noise with std={noise_std}")
        radar_data += rng.normal(0, noise_std, (n_channels, n_chirps, n_samples))
        radar_data += 1j * rng.normal(0, noise_std, (n_channels, n_chirps, n_samples))
    elif noise_std is not None and rng is None:
        logger.error('RNG is missing.')
    elif noise_std is None and rng is not None:
        logger.error('Std of noise is missing.')

    logger.debug(f"Radar data generated with shape {radar_data.shape}")
    return radar_data
