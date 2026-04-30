import pathlib
import re
import numpy as np
import scipy.io
import ast

from nrsp.rsp.preprocess import subtract_lin_filter


def load_data(path):
    """
    Loading BBM radar data from given filepath (../../../data.mat).

    Args:
        path (String): Path to matlab data file.

    Returns:
        numpy.array: radar data cube (frames, chirps, samples, antennas).
    """

    file_path = path
    mat = scipy.io.loadmat(file_path)
    data = mat["data"]
    data = np.swapaxes(data, 1, 2)
    data = np.swapaxes(data, 0, 1)

    return data.T


def load_targets(path):
    """
    Loading BBM target data from given filepath (../../../target.mat).

    Args:
        path (String): Path to matlab target file.

    Returns:
        list: targets (targets, frames) containg dictionaries.
    """

    file_path = path
    mat = scipy.io.loadmat(file_path)
    targets = mat["targets"]

    return targets


def load_config(path):
    """
    Loading BBM radar config from given directory path.

    Args:
        path (String): Path to directory where config file is located.

    Returns:
        dict: dictionary containing parameters of the simulated radar sensor.
    """

    file_path = path + "/config.mat"
    mat = scipy.io.loadmat(file_path, struct_as_record=True)
    config = mat["mmic"]
    keys = config.dtype.names
    vals = config[0][0]
    config = {}
    for k, v in zip(keys, vals):
        config[k] = v[0][0]

    return config


def distance_range(config, n_distances=None):
    """
    Return array of distance to map to frequency bins.

    Args:
        config (dict):      Configuration dictionary of radar sensor settings.
        n_distances (int):  Number of distance bins.

    Returns:
        (np.array):     Array containing distance values.
    """

    fs = 50e6  # Hard-coded sampling rate of ADC
    # Infineon BBM simulation cuts off samples by defining Nrange
    # => not full bandwidth is used
    # => effective bandwidth is needed
    eff_b = float(config["Nrange"]) * config["bandwidth"] / (fs * config["Tramp"] / 2 + 1)
    d_res = config["c0"] / (2 * eff_b)
    d_max = float(config["Nrange"]) // 2 * d_res
    if n_distances is None:
        distance_range = np.linspace(0, d_max, config["Nrange"] // 2, endpoint=False)
    else:
        distance_range = np.linspace(0, d_max, n_distances, endpoint=False)

    return distance_range


def angle_range(n_angles):
    """
    Return array of angles to map to frequency bins.

    Args:
        n_angles (int):  Number of angle bins.

    Returns:
        (np.array):     Array containing angle values.
    """

    angles = np.zeros(n_angles)
    for a in range(n_angles):
        angles[a] = -np.arcsin(2 * (a - np.floor(n_angles / 2)) / n_angles)

    return angles


def targets_to_map_pertarget(targets, n_distances, n_angles, config, rcs_flag=False):
    """
    Read targets from BBM target structure and create binary or rcs range-angle map to match SNN/FFT output.

    Args:
        targets (list):     Targets from BBM structure.
        n_distances:        Number of distance bins.
        n_angles:           Number of angle bins.
        config (dict):      Configuration dictionary of BBM radar sensor settings.
        rcs_flag (bool):    Boolean to indicate, whether binary or rcs map.

    Return:
        (np.array):     Target Range-Angle map

    """

    d_range = distance_range(config, n_distances)
    a_range = angle_range(n_angles)

    target_map = np.zeros((len(targets) + 1, n_distances, n_angles)).astype("float32")

    for i, target in enumerate(targets):
        # target is a tuple of name, pos, velo, rcs, (?)
        pos = target[1][0]  # pos
        d = np.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2)
        a = np.arctan(pos[1] / pos[0])
        rcs = target[3][0]

        d_idx = (np.abs(d_range - d)).argmin()
        a_idx = (np.abs(a_range - a)).argmin()

        if rcs_flag:
            target_map[i + 1, d_idx, a_idx] = rcs
            target_map[0, d_idx, a_idx] = rcs
        else:
            target_map[i + 1, d_idx, a_idx] = rcs > -100
            target_map[0, d_idx, a_idx] = rcs > -100

    return target_map


def targets_to_maps(targets, n_distances, n_angles, config, rcs_flag):
    """
    Run over all frames to create target maps.

    Args:
        targets (list):     Targets from BBM structure.
        n_distances:        Number of distance bins.
        n_angles:           Number of angle bins.
        config (dict):      Configuration dictionary of BBM radar sensor settings.
        rcs_flag (bool):    Boolean to indicate, whether binary or rcs map.

    Return:
        (np.array):     Target Range-Angle map over all frames
    """

    target_maps = []
    target_maps_per_target = []
    n_targets, n_frames = targets.shape
    for frame in range(n_frames):
        target = targets[:, frame]
        target_map = targets_to_map_pertarget(target, n_distances, n_angles, config, rcs_flag=rcs_flag)
        target_maps_per_target.append(target_map)

    return np.swapaxes(np.array(target_maps_per_target), axis1=0, axis2=1)


def velocity_bins(config, n_chirps):
    fs = 50e6  # Hard-coded sampling rate of ADC
    eff_b = float(config["Nrange"]) * config["bandwidth"] / (fs * config["Tramp"] / 2 + 1)
    center_freq = config["f0"] + eff_b / 2
    lmbda = config["c0"] / center_freq

    doppler_resolution = lmbda / (2 * n_chirps * config["Trep"])
    v_max = n_chirps * doppler_resolution

    a = np.linspace(-v_max / 2, v_max / 2, n_chirps, False)
    return a


def targets_to_cube(
    targets,
    n_distances,
    n_frames,
    n_angles,
    n_velocities,
    config,
):
    t_cube = np.full((n_frames, n_distances, n_angles, n_velocities), False)

    d_range = distance_range(config, n_distances)
    a_range = angle_range(n_angles)
    v_range = velocity_bins(config, n_velocities)

    for frame in range(n_frames):
        for target in targets[:, frame]:

            # compute distance and azimuth angle of target to radar antennas
            pos = target["r0"][0]
            d = np.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2)
            a = np.arctan(pos[1] / pos[0])

            # compute speed
            v_abs = np.sqrt(target["v"][0][0] ** 2 + target["v"][0][1] ** 2 + target["v"][0][2] ** 2)
            v_sign = np.sign(np.dot(target["r0"][0], target["v"][0]))
            v = v_sign * v_abs

            # only use actual targets
            if float(target["rcs"][0][0]) <= -100:
                continue

            d_idx = (np.abs(d_range - d)).argmin()  # TODO d > d_max
            a_idx = (np.abs(a_range - a)).argmin()  # TODO modulo? as 0 bin also represents targets at 90°-bin_size/2
            v_idx = (np.abs(v_range - v)).argmin()  # TODO modulo?
            t_cube[frame, d_idx, a_idx, v_idx] = True

    return t_cube


def preprocess_data(radar_data):
    """Remove drift from bbm data of shape (n_frames, n_chirps, n_samples, n_channels) by subtracting average drift per timestep and zero centering data along fast time."""

    # computed from level0_random_1dmax_randomrcs_8targets_32chirps_new/seed_0 data but should be constant for all other datasets as well
    avg = np.load("nrsp/datasets/bbm_drift_per_timestep.npy")
    radar_data -= avg[None, None, :, None]

    return radar_data


class BBMIterator:

    def __init__(self, path):
        self.path = path
        self._idx = 0

        # get list of indices
        # make sure there is always a pair of data and corresponding target file
        data_pathlist = pathlib.Path(path).rglob("data_*.mat")
        targets_pathlist = pathlib.Path(path).rglob("targets_*.mat")

        data_indices = [re.match(r"data_(\d+)\.mat", i.name).group(1) for i in data_pathlist]
        targets_indices = [re.match(r"targets_(\d+)\.mat", i.name).group(1) for i in targets_pathlist]
        self.indices = sorted(set(data_indices) & set(targets_indices))

    def __next__(self):
        """Return radar data of shape (n_frames, n_chirps, n_samples, n_channels) and targets."""

        radar_data = load_data(pathlib.Path(self.path).joinpath(f"data_{self.indices[self._idx]}.mat").absolute())
        targets = load_targets(pathlib.Path(self.path).joinpath(f"targets_{self.indices[self._idx]}.mat").absolute())[0]
        self._idx += 1
        return radar_data, targets

    def get_current_data_filename(self):
        """Return the filename of the current radar data file."""
        return f"data_{self.indices[self._idx - 1]}.mat"
