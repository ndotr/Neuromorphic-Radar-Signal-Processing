import numpy as np
import typing as ty
import numpy.typing as npt
from numpy.typing import NDArray
import os
import logging
import paramiko
import threading
import time
import numpy as np
import socket
import atexit
import subprocess
from datetime import datetime
import argparse
import pathlib

# import dash
# from dash import dcc, html
# from dash.dependencies import Input, Output
# import plotly.graph_objects as go

from typing import Union, Literal, List
from scipy.sparse import coo_matrix, csr_matrix

import nrsp.utils.log

host = None



def init_host(speed=None):
    global host
    host = Host(speed)


class Host:
    def __init__(self, speed=None):
        self.logger = nrsp.utils.log.get_logger(__name__)
        self.ip = None
        self.name = None
        self.speed = speed
        self.init_host()
        self.logger.info("Detected: {}".format(self.name))
        self.init_loihi_env()

    def init_host(self):
        ret = None
        # Jetson Mercedes
        if os.uname().nodename == "mb-jetson-san" and os.uname().machine == "aarch64" and "tegra" in os.uname().release:
            self.logger.info("Running on mercedes.")
            self.name = "mercedes"
            self.ip = "192.168.2.204"

        elif os.uname().nodename == "ubuntu" and os.uname().machine == "aarch64" and "tegra" in os.uname().release:
            self.logger.info("Running on TWT.")
            self.ip = "192.168.1.236"
            self.name = "twt"
        # Local NC0
        elif os.uname().nodename == "nc0" and os.uname().machine == "x86_64" and "arch" in os.uname().release:
            self.logger.info("Running on nreeb@nc0.")
            self.name = "nc0"
            self.ip = "192.168.0.102"

        elif os.uname().nodename == "ncl-edu" and os.uname().machine == "x86_64":
            self.logger.info("Running on ncl-edu.")
            self.name = "ncl"

        else:
            self.logger.info("Current environment: {}".format(os.uname()))
            self.logger.error("No suitable environment found.")

        if self.speed == "slowest":
            os.environ["NXOPTIONS"] = "--pio-cfg-chip=0x41ff"
        elif self.speed == "slow":
            os.environ["NXOPTIONS"] = "--pio-cfg-chip=0x4192"
        else:
            os.environ["NXOPTIONS"] = ""

        return ret

    def get_ethernet_params(self):

        ret = None
        # Jetson Mercedes
        if self.name == "mercedes":

            ETH_INTERFACE = os.environ.get("ETH_INTERFACE", "eth-loihi")
            ETH_MAC = os.environ.get("ETH_MAC", "0x48b02dfe68a6")  # Mercedes
            LOIHI_MAC = os.environ.get("LOIHI_MAC", "0x0015edbeefed")
            ETH_CHIP_IDX = 6  # 8 chip system

            ret = (ETH_INTERFACE, ETH_MAC, LOIHI_MAC, ETH_CHIP_IDX)

        elif self.name == "twt":

            ETH_INTERFACE = os.environ.get("ETH_INTERFACE", "eth0")
            ETH_MAC = os.environ.get("ETH_MAC", "0x3c6d6602b60f")  # TWT
            LOIHI_MAC = os.environ.get("LOIHI_MAC", "0x0015edbeefed")
            ETH_CHIP_IDX = 0  # 1 chip system

            ret = (ETH_INTERFACE, ETH_MAC, LOIHI_MAC, ETH_CHIP_IDX)

        elif self.name == "nc0":

            ETH_INTERFACE = os.environ.get("ETH_INTERFACE", "enp12s0u1c2")
            ETH_MAC = os.environ.get("ETH_MAC", "0xf8e43b90cadf") # nc0
            LOIHI_MAC = os.environ.get("LOIHI_MAC", "0x000000000000") 
            ETH_CHIP_IDX = 0  # 1 chip system

            ret = (ETH_INTERFACE, ETH_MAC, LOIHI_MAC, ETH_CHIP_IDX)
        else:
            # LOIHI_MAC = os.environ.get('LOIHI_MAC', '0xe293d2aa339a')
            # LOIHI_MAC = os.environ.get('LOIHI_MAC', '0x00d3005a00ce')
            self.logger.info("Current environment: {}".format(os.uname()))
            self.logger.info("Current name: {}".format(self.name))
            self.logger.error("No suitable environment found for ethernet streaming.")

        return ret

    def init_loihi_env(self, ncl_board_option=0, mercedes_board_option=1):

        ret = 0
        # Jetson
        if self.name == "mercedes":
            os.environ["NOSLURM"] = "1"
            if mercedes_board_option == 0:
                os.environ["NXSDKHOST"] = "192.168.2.236"
            elif mercedes_board_option == 1:
                os.environ["NXSDKHOST"] = "192.168.2.204"
            os.environ["HOST_BINARY"] = "/home/loihiuser/nxcore_power/bin/armv7l-N3C1/nx_driver_server"
            os.environ["LOIHI_GEN"] = "N3C1"

            self.logger.info("NOSLURM: {}".format(os.environ["NOSLURM"]))
            self.logger.info("NXSDKHOST: {}".format(os.environ["NXSDKHOST"]))
            self.logger.info("HOST_BINARY: {}".format(os.environ["HOST_BINARY"]))

            ret = 1

        elif self.name == "twt":
            os.environ["NOSLURM"] = "1"
            os.environ["NXSDKHOST"] = "192.168.1.236"
            os.environ["HOST_BINARY"] = "/home/loihiuser/nxcore/bin/armv7l-N3C1/nx_driver_server"
            os.environ["LOIHI_GEN"] = "N3C1"

            self.logger.info("NOSLURM: {}".format(os.environ["NOSLURM"]))
            self.logger.info("NXSDKHOST: {}".format(os.environ["NXSDKHOST"]))
            self.logger.info("HOST_BINARY: {}".format(os.environ["HOST_BINARY"]))

            ret = 1

        # Local NC0
        elif self.name == "nc0":
            self.logger.info("Running on nreeb@nc0.")

            os.environ["NOSLURM"] = "1"
            os.environ["NXSDKHOST"] = "192.168.0.102"
            os.environ["HOST_BINARY"] = "/home/loihiuser/nxcore-power/bin/armv7l-N3C1/nx_driver_server"
            os.environ["LOIHI_GEN"] = "N3C1"

            self.logger.info("NOSLURM: {}".format(os.environ["NOSLURM"]))
            self.logger.info("NXSDKHOST: {}".format(os.environ["NXSDKHOST"]))
            self.logger.info("HOST_BINARY: {}".format(os.environ["HOST_BINARY"]))
            ret = 1

        elif self.name == "ncl":
            self.logger.info("Running on ncl-edu.")

            os.environ["SLURM"] = "1"
            os.environ["PARTITION"] = "oheogulch"

            # This
            if ncl_board_option == 0:
                os.environ["LOIHI_GEN"] = "N3C1"
                os.environ["BOARD"] = "ncl-ext-og-06"

            # Or That
            if ncl_board_option == 1:
                os.environ["LOIHI_GEN"] = "N3B3"
                os.environ["BOARD"] = "ncl-ext-og-05"

            self.logger.info("SLURM: {}".format(os.environ["SLURM"]))
            self.logger.info("PARTITION: {}".format(os.environ["PARTITION"]))
            self.logger.info("BOARD: {}".format(os.environ["BOARD"]))
            ret = 1

        else:
            self.logger.info("Current environment: {}".format(os.uname()))
            self.logger.error("No suitable environment found.")

        return ret


def complex_to_float(x_complex):

    x_float = np.zeros(x_complex.shape + (2,))
    x_float[..., 0] = np.real(x_complex)
    x_float[..., 1] = np.imag(x_complex)

    return x_float


def kernel_to_weights_2d(
    shape: ty.Tuple[int, int],
    kernel: npt.NDArray,
    wrap_x: bool = False,
    wrap_y: bool = False,
) -> npt.NDArray:
    """Build linear weight matrix from 2D convolution kernel. Output shape equals input shape.
    If wrap is False, connections reflect instead.

    :param ty.Tuple[int, int] shape: 2D neuronen population shape
    :param npt.NDArray kernel: 2D convolution kernel
    :param bool wrap_x: wether weight connections should wrap around x axis
    :param bool wrap_y: wether weight connections should wrap around y axis
    :return npt.NDArray: weight matrix
    """

    n_neurons = np.prod(shape)
    weights = np.zeros(shape=(n_neurons, n_neurons), dtype=kernel.dtype)

    for i, j in np.ndindex(shape):
        idx_lin = i * shape[1] + j
        w = np.zeros(shape=shape, dtype=kernel.dtype)

        for k in range(kernel.shape[0]):
            for l in range(kernel.shape[1]):
                x = i + k - kernel.shape[0] // 2
                y = j + l - kernel.shape[1] // 2

                if wrap_x:
                    x = x % shape[0]
                else:
                    x = _reflect_index(x, shape[0])

                if wrap_y:
                    y = y % shape[1]
                else:
                    y = _reflect_index(y, shape[1])

                # if x >= 0 and x < shape[0] and y >= 0 and y < shape[1]:
                w[x, y] += kernel[k, l]

        weights[idx_lin] = w.flatten()

    return weights


def _reflect_index(i, n):
    period = 2 * (n - 1)
    i_mod = abs(i) % period
    return i_mod if i_mod < n else period - i_mod


def complex_vector_to_real_stack(
    vec: NDArray[np.complex128], mode: Literal["interleave", "block"] = "interleave", axis: int = -1
) -> NDArray[np.float64]:
    """
    Convert a complex array into a real-valued array representation along a specified axis.

    Args:
        vec: Complex input array of any shape.
        mode:
            - 'interleave': interleave real and imaginary parts along the axis.
            - 'block': concatenate real and imaginary parts along the axis.
        axis: Axis along which to apply interleaving or blocking.

    Returns:
        Real-valued array with shape adjusted per mode.
    """
    vec = np.asarray(vec)
    ndim = vec.ndim
    # Normalize axis to positive index
    if axis < 0:
        axis += ndim
    if not (0 <= axis < ndim):
        raise np.AxisError(f"axis {axis} is out of bounds for array of dimension {ndim}")

    if mode == "interleave":
        # Stack real and imag parts along a new axis next to the specified axis,
        # then reshape to interleave along original axis
        stacked = np.stack((vec.real, vec.imag), axis=axis + 1)
        shape = list(vec.shape)
        shape[axis] *= 2
        return stacked.reshape(shape)

    elif mode == "block":
        # Concatenate real and imag parts along the specified axis
        return np.concatenate([vec.real, vec.imag], axis=axis)

    else:
        raise ValueError("Invalid mode for complex_vector_to_real_stack. Use 'interleave' or 'block'.")


def complex_matrix_to_real_stack(mat: NDArray[np.complex128], mode: Literal["interleave", "block"] = "interleave") -> NDArray[np.float64]:
    """
    Convert a complex matrix into a real-valued representation.

    Args:
        mat: Complex input matrix.
        mode: If 'interleave', use 2D interleaving. If 'block', concatenate real and imaginary blocks side-by-side.

    Returns:
        Real-valued matrix.
    """
    if mode == "interleave":
        real = np.empty((mat.shape[0] * 2, mat.shape[1] * 2))
        real[0::2, 0::2] = mat.real
        real[0::2, 1::2] = -mat.imag
        real[1::2, 0::2] = mat.imag
        real[1::2, 1::2] = mat.real
        return real
    elif mode == "block":
        top = np.hstack([mat.real, -mat.imag])
        bottom = np.hstack([mat.imag, mat.real])
        return np.vstack([top, bottom])
    else:
        raise ValueError("Invalid mode for complex_to_real_stack.")


def complex_matrix_to_real_stack_sparse(mat: NDArray[np.complex128], mode: Literal["interleave", "block"] = "interleave") -> csr_matrix:
    """
    Convert a complex matrix into a real-valued sparse representation.

    Args:
        mat: Complex input matrix (dense or sparse).
        mode: 'interleave' or 'block'.

    Returns:
        Sparse csr_matrix with real-valued representation.
    """
    r, c = mat.shape

    if mode == "interleave":
        # Map each element to 4 positions
        rows = np.repeat(np.arange(r), c)
        cols = np.tile(np.arange(c), r)

        row_idx = np.hstack([2 * rows, 2 * rows, 2 * rows + 1, 2 * rows + 1])
        col_idx = np.hstack([2 * cols, 2 * cols + 1, 2 * cols, 2 * cols + 1])
        data = np.hstack([mat.real.ravel(), -mat.imag.ravel(), mat.imag.ravel(), mat.real.ravel()])

        return csr_matrix((data, (row_idx, col_idx)), shape=(2 * r, 2 * c))

    elif mode == "block":
        # Build each block separately using COO format
        def block_to_coo(block, row_offset, col_offset):
            rr, cc = np.nonzero(block)
            data = block[rr, cc]
            return rr + row_offset, cc + col_offset, data

        top_left = mat.real
        top_right = -mat.imag
        bottom_left = mat.imag
        bottom_right = mat.real

        rl, cl, dl = block_to_coo(top_left, 0, 0)
        rr, cr, dr = block_to_coo(top_right, 0, c)
        bl, bc, db = block_to_coo(bottom_left, r, 0)
        br, bc2, db2 = block_to_coo(bottom_right, r, c)

        rows = np.hstack([rl, rr, bl, br])
        cols = np.hstack([cl, cr, bc, bc2])
        data = np.hstack([dl, dr, db, db2])

        return csr_matrix((data, (rows, cols)), shape=(2 * r, 2 * c))

    else:
        raise ValueError("Invalid mode, must be 'interleave' or 'block'.")


def real_stack_to_complex_vector(
    vec_real: NDArray[np.float64], mode: Literal["interleave", "block"] = "interleave", axis: int = -1
) -> NDArray[np.complex128]:
    """
    Convert a real-valued stacked array back to a complex array along a specified axis.

    Args:
        vec_real: Real-valued input array with 2x complex length along the axis.
        mode:
            - 'interleave': [..., Re0, Im0, Re1, Im1, ...] along the axis.
            - 'block': [..., Re0, Re1, ..., Im0, Im1, ...] along the axis.
        axis: Axis along which real and imaginary parts are stacked.

    Returns:
        Complex-valued array with half the size along that axis.
    """
    vec_real = np.asarray(vec_real)
    axis = axis if axis >= 0 else vec_real.ndim + axis
    length = vec_real.shape[axis]

    if length % 2 != 0:
        raise ValueError("Stacked axis must have even length (real + imag).")

    if mode == "interleave":
        # Use strided take for even and odd indices
        real = np.take(vec_real, indices=np.arange(0, length, 2), axis=axis)
        imag = np.take(vec_real, indices=np.arange(1, length, 2), axis=axis)
    elif mode == "block":
        # Split directly along axis into real and imaginary parts
        real, imag = np.split(vec_real, 2, axis=axis)
    else:
        raise ValueError("Invalid mode. Use 'interleave' or 'block'.")

    return real + 1j * imag


def real_stack_to_complex_matrix(mat_real: NDArray[np.float64], mode: Literal["interleave", "block"] = "interleave") -> NDArray[np.complex128]:
    """
    Convert a real-valued stacked matrix back to a complex matrix.

    Args:
        mat_real: Real-valued input matrix of shape (2*M, 2*N).
        mode:
            - 'interleave': 2D interleaving format.
            - 'block': real and imaginary parts concatenated in blocks.

    Returns:
        Complex matrix of shape (M, N).
    """
    M2, N2 = mat_real.shape
    if M2 % 2 != 0 or N2 % 2 != 0:
        raise ValueError("Input shape must be even in both dimensions.")

    M = M2 // 2
    N = N2 // 2

    if mode == "interleave":
        real = mat_real[0::2, 0::2]
        imag = mat_real[1::2, 0::2]
        return real + 1j * imag
    elif mode == "block":
        top_left = mat_real[:M, :N]
        bottom_left = mat_real[M:, :N]
        return top_left + 1j * bottom_left
    else:
        raise ValueError("Invalid mode for real_stack_to_complex_matrix.")


# def kernel_to_weights_1d(
#     n_neurons: int,
#     kernel: npt.NDArray,
#     wrap: bool = False,
# ) -> npt.NDArray:
#     """Build linear weight matrix from 1D convolution kernel. Output neuron layer has same number of neurons as input layer.

#     :param int n_neurons: number of neurons
#     :param npt.NDArray kernel: 1D convolution kernel
#     :param bool wrap: wether weight connections should wrap around, defaults to False
#     :return npt.NDArray: weight matrix
#     """

#     weights = np.zeros(shape=(n_neurons, n_neurons), dtype=kernel.dtype)

#     for i in range(n_neurons):
#         w = np.zeros(shape=(n_neurons,), dtype=kernel.dtype)

#         for k in range(kernel.shape[0]):
#             x = i + k - kernel.shape[0] // 2

#             if wrap:
#                 x = x % n_neurons
#             if x >= 0 and x < n_neurons:
#                 w[x] = kernel[k]

#         weights[i] = w.flatten()

#     return weights


class PowerTelemetry:
    """
    PowerTelemetry is a class designed to manage and monitor power telemetry
    data across multiple boards. It provides functionality for SSH connections,
    data collection, real-time visualization, and data processing.

    Attributes:
    ----------
    config : str
        Specifies the configuration settings for the Power Telemetry instance.
        [Note: This attribute will be removed soon.]
    filename : str, optional
        The name of the file where telemetry data will be saved. If not
        provided, data will not be saved to a file. Defaults to None.
    console_out : bool, optional
        Determines whether output should be displayed in the console.
        Defaults to False.
    loglevel : int, optional
        Sets the logging level, with a default of logging.WARNING, to control
        the verbosity of log messages.
    loop_latency_ms : int, optional
        Defines the latency in milliseconds for the telemetry loop, with
        a default value of 5 ms.
    pac_port : int, optional
        Specifies the port number for Host-SuperHost communication,
        defaulting to 7225.
    visualize : bool, optional
        Determines whether to visualize the telemetry data in real-time.
        Defaults to False.
    dash_port : int, optional
        Sets the port for the dashboard used in visualization, with a
        default of 8050.
    visualization_buffer : int, optional
        Specifies the buffer size for visualization data, defaulting to 500.
    compile_binary : bool, optional
        Builds the power telemetry binary from the ground up. By default,
        this option is set to False since the precompiled binary
        is already available.

    Methods:
    --------
    __init__(self, config: str, filename: str = None,
            console_out: bool = False, loglevel: int = logging.WARNING,
            read_pac_all: bool = False, loop_latency_ms: int = 5,
            pac_port: int = 7225, visualize: bool = False,
            dash_port: int = 8050, visualization_buffer: int = 500,
            compile_binary: bool = False)
        Initializes a new instance of the PowerTelemetry class with
        the specified configuration and options.
    _parse_boards(self) -> None
        Parses environment variables to determine the boards to be monitored.
    _start_ssh_connection(self) -> None
        Establishes SSH connections to the boards and prepares them
        for telemetry data collection.
    _close_ssh(self) -> None
        Closes all SSH connections.
    _finalize_power_telemetry_bin(self) -> None
        Finalizes the telemetry process by terminating binaries
        and cleaning up resources.
    _read_output(self, stream) -> None
        Reads and logs output from SSH command execution.
    _start_power_telemetry_bin(self, ssh, board_id, path, board) -> None
        Starts the power telemetry binary on the remote boards.
    _read_socket(self, board) -> None
        Reads telemetry data from a socket connection to the boards.
    start_power_telemetry(self) -> None
        Initiates the power telemetry process across all boards.
    end_power_telemetry(self) -> None
        Ends the telemetry process and processes the collected data.
    get_power_data(self) -> dict
        Returns the processed telemetry data.
    _process_data_by_system(self) -> None
        Processes raw telemetry data based on the system configuration.
    """

    def __init__(
        self,
        config: str = None,
        log_filename: str = None,
        loglevel: int = logging.DEBUG,
        loop_latency_ms: int = 5,
        pac_port: int = 7225,
        build_directory: pathlib.Path = pathlib.Path("/home/loihiuser/nxcore-power/n3_apps/modules/power_telemetry/build/"),
        #build_directory: pathlib.Path = pathlib.Path("/home/loihiuser/nxcore_power/n3_apps/modules/power_telemetry/build/"),
        binary: pathlib.Path = pathlib.Path("power_telemetry_daemon.bin"),
    ) -> None:

        self.logger_ = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        # Add line number, filename, and function name
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s")
        handler.setFormatter(formatter)
        self.logger_.addHandler(handler)
        self.logger_.setLevel(loglevel)

        self.config = config
        self.pac_port = pac_port
        self.loop_latency_ms = loop_latency_ms
        self.build_directory = build_directory
        self.binary = binary
        self.acquire_data = False
        self.log_filename = log_filename

        self.keep_socket_open = True

        self.boards = []
        self.boards_ids = {}

        self.ssh_clients = []
        self._completed = {}

        if self.config is None:
            raise ValueError("config must be provided when use_ssh is True")

        self._parse_boards()
        self._start_ssh_connection()

        self._bin_power_threads = []
        self._out_threads = []
        self._raw_data = {}

        for board_ in self.boards:
            self._raw_data[board_] = {"id": [], "time": [], "acc_count": [], "vddm": [], "vddio": [], "vdd": [], "na": [], "total": []}

        self._processed_data = {}

        self.data = {}
        self._start_power_telemetry()

    def _parse_boards(self) -> None:
        def parse_host_string(host_string):
            if "@" in host_string:
                boards = [item.split("@")[0] for item in host_string.split(":")]
            else:
                boards = [host_string]
            boards_ids = {item: idx for idx, item in enumerate(boards)}
            return boards, boards_ids

        if "NXSDKHOST" in os.environ:
            self.boards, self.boards_ids = parse_host_string(os.environ["NXSDKHOST"])
        elif "BOARD" in os.environ:
            self.boards, self.boards_ids = parse_host_string(os.environ["BOARD"])
        else:
            raise Exception("NXSDKHOST or BOARD not defined")

    def _start_ssh_connection(self) -> None:

        # Load SSH config once
        ssh_config_file = os.path.expanduser("~/.ssh/config")
        ssh_config = paramiko.SSHConfig()
        if os.path.exists(ssh_config_file):
            with open(ssh_config_file) as f:
                ssh_config.parse(f)

        for idb in range(len(self.boards)):
            board_ip = self.boards[idb]

            host_config = ssh_config.lookup(board_ip)
            hostname = host_config.get("hostname", board_ip)
            port = int(host_config.get("port", 22))
            username = host_config.get("user", None)
            identityfile = host_config.get("identityfile", [None])[0]

            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(hostname, port=port, username=username, key_filename=identityfile)

            ssh.exec_command(f"pkill -9 {self.binary}")
            time.sleep(1)
            self.ssh_clients.append([ssh, self.boards_ids[board_ip], "", board_ip])

        self.logger_.debug("power_bin ssh connections completed")

    def _close_ssh(self) -> None:
        for ssh, _, _, _ in self.ssh_clients:
            if ssh is not None:
                ssh.close()
        self.logger_.debug("power_bin ssh connections closed")

    def _finalize_power_telemetry_bin(self) -> None:
        for ssh, _, _, _ in self.ssh_clients:
            if ssh is not None:
                ssh.exec_command(f"pkill -SIGTERM -f {self.binary}")
                self.logger_.debug(f"{self.binary} killed successfully")
        for thread in self._out_threads:
            thread.join()
        for thread in self._bin_power_threads:
            thread.join()
        self.logger_.debug("power_bin telemetry finalized ")

    def _read_output(self, stream) -> None:
        for line in iter(stream.readline, ""):
            line_strip = line.strip()
            self.logger_.debug(line_strip)
        stream.close()

    def _start_power_telemetry_bin(self, ssh, board_id, path, board) -> None:

        flags = f"-config {self.config}"
        flags += f" -socket_port {self.pac_port}"
        flags += f" -sleep_ms {self.loop_latency_ms}"
        if self.logger_.level == logging.DEBUG:
            flags += " -log_level DEBUG"
        elif self.logger_.level == logging.INFO:
            flags += " -log_level INFO"
        elif self.logger_.level == logging.WARNING:
            flags += " -log_level WARNING"
        elif self.logger_.level == logging.ERROR:
            flags += " -log_level ERROR"

        # remote_command = f"cd /tmp/power/; chmod +x power_telemetry_host.bin; ./power_telemetry_host.bin {flags}" # old
        remote_command = f"cd {self.build_directory}; nohup ./{self.binary} {flags} > log 2>&1 &"
        stdin, stdout, stderr = ssh.exec_command(remote_command)
        self._out_threads.append(threading.Thread(target=self._read_output, args=(stdout,)))
        self._out_threads[-1].start()
        self._out_threads.append(threading.Thread(target=self._read_output, args=(stderr,)))
        self._out_threads[-1].start()
        self._out_threads.append(threading.Thread(target=self._read_socket, args=(board,)))
        self._out_threads[-1].start()
        self._completed[board] = False

        if stderr.channel.recv_exit_status() != 0:
            self.logger_.debug(stderr.read())

        self.logger_.debug(f"{self.binary} of board {board} started successfully")

    def _read_socket(self, board) -> None:
        time.sleep(5.0)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
        s.settimeout(5.0)  # Set a timeout of 1 second

        try:
            s.connect((board, self.pac_port))
        except (socket.error, socket.timeout) as e:
            self.end_power_telemetry()
            self.logger_.error(f"Failed to connect to {board} on port {self.pac_port}: {e}")
            raise RuntimeError(f"Socket connection failed for board {board}") from e

        try:
            DATA_SIZE = 7 * 4  # 5 data points each of size float32
            while self.keep_socket_open:
                try:
                    data_rec = np.frombuffer(s.recv(DATA_SIZE, socket.MSG_WAITALL), dtype=np.float32)
                    if len(data_rec) != 7:
                        self.logger_.debug(f"Data received from {board} is not of size 7")
                        continue

                    if data_rec[0] < 0:
                        self._completed[board] = True
                        break

                    if self.acquire_data:
                        self._raw_data[board]["id"].append(int(data_rec[0]))
                        self._raw_data[board]["time"].append(int(data_rec[1]))
                        self._raw_data[board]["acc_count"].append(int(data_rec[2]))

                        self._raw_data[board]["vddm"].append(data_rec[3])
                        self._raw_data[board]["vddio"].append(data_rec[4])
                        self._raw_data[board]["vdd"].append(data_rec[5])

                        self._raw_data[board]["na"].append(data_rec[6])
                        self._raw_data[board]["total"].append(data_rec[3] + data_rec[4] + data_rec[5])

                except socket.timeout:
                    # Timeout occurred, check for external conditions or signals
                    continue

        except Exception as e:
            self.logger_.error(f"Error reading socket: {e}")
        finally:
            s.close()

    def _start_power_telemetry(self) -> None:
        for ssh, board_id, path, board in self.ssh_clients:
            thread = threading.Thread(target=self._start_power_telemetry_bin, args=(ssh, board_id, path, board))
            thread.start()

        for board in self.boards:
            self._out_threads.append(threading.Thread(target=self._read_socket, args=(board,)))
            self._out_threads[-1].start()
            self._completed[board] = False
            self.keep_socket_open = True
            self.logger_.debug(f"Thread to acquire data of board {board} initialized successfully")
        time.sleep(1)

    def start_power_telemetry(self) -> None:
        self.acquire_data = True

    def pause_telemetry(self) -> None:
        self.acquire_data = False
        for board in self.boards:
            self._raw_data[board]["id"] = []
            self._raw_data[board]["time"] = []
            self._raw_data[board]["acc_count"] = []
            self._raw_data[board]["vddm"] = []
            self._raw_data[board]["vddio"] = []
            self._raw_data[board]["vdd"] = []
            self._raw_data[board]["na"] = []
            self._raw_data[board]["total"] = []

    def _catch_log(self, ssh, remote_log, local_log):
        # Later, fetch the remote log into a local file
        sftp = ssh.open_sftp()
        self.logger_.debug(f"Getting {remote_log}")
        sftp.get(str(remote_log), str(local_log))
        sftp.close()
        self.logger_.debug(f"Fetched remote log to local file {local_log}")

    def end_power_telemetry(self) -> None:
        self.acquire_data = False
        self.keep_socket_open = False
        self.logger_.debug("Wait to finish ...")
        time.sleep(1)
        self._finalize_power_telemetry_bin()
        for ssh, _, _, board_id in self.ssh_clients:
            self._catch_log(ssh, remote_log=self.build_directory / "log", local_log=self.log_filename)
        self._close_ssh()
        self._process_data_by_system()

    def get_power_data(self) -> dict:
        return self._processed_data

    def _process_data_by_system(self) -> None:
        self._processed_data = self._raw_data.copy()


def scale_data(radar_data, data_exp):
    """Scale radar data by a factor of 2^data_exp. If the magnitude exceeds 2^15-1, scale down to fit."""
    data = radar_data * 2**data_exp
    max_magnitude = 2**15 - 1
    magnitudes = np.abs(data)
    scale = np.minimum(1, max_magnitude / magnitudes)
    data = data * scale
    return data
