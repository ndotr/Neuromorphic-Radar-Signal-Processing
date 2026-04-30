# Public packages
from pathlib import Path
import numpy as np

# Intel packages
from nrsp.algs.nx.base_nx_model import BaseNxModel
import nxkernel as nxk
import nxkernel.kernels as K
from nxkernel.kernels.sender import Sender

# Custom packages
from nrsp.utils.nx import complex_to_float
from nrsp.utils.radar import phasor_weights, steering_weights

KERNELS_PATH = Path(__file__).resolve().parent.parent / "kernels"


def angle_synapse_weights(
    n_distances,
    n_channels,
    angle_bins: list = None,
    is_complex=True,
):
    """Build synaptic weights for a range-angle neuron layer.
    If angle_bins is given, restrict neurons to those angle bins.
    is_complex TODO

    :param n_distances: number of distance neurons
    :param n_channels: number of input channels
    :param list angle_bins: list of angle bins, defaults to None
    :return: (real_weights, imag_weights) tuple of synaptic weights of dtype int
    """

    n_angles = n_channels
    if angle_bins is not None:
        n_angles = len(angle_bins)

    # unflattened weights for a neuron population of shape (n_distances, n_angles)
    weights = np.zeros(shape=(n_distances, n_angles, n_channels), dtype="complex128")

    # set individual synapric weights for each neuron
    steering_wgts = steering_weights(n_channels, angle_bins)  # (n_angles, n_channels)
    for dist, angle in np.ndindex(n_distances, n_angles):
        weights[dist, angle] = steering_wgts[angle]

    # flatten weights
    weights = weights.reshape((n_distances * n_angles, n_channels))
    # convert to float
    weights = complex_to_float(weights)  # in range [-1/n_channels, 1/n_channels]
    # scale weights to range [-127, 127] since synapse weights are stored as signed bytes ([-128, 127])
    weights *= 127 * n_channels

    real_weights_ = np.rint(weights[..., 0]).astype(int)
    imag_weights_ = np.rint(weights[..., 1]).astype(int)

    if is_complex:
        real_weights = np.hstack((real_weights_, -imag_weights_))
        imag_weights = np.hstack((imag_weights_, real_weights_))
    else:
        real_weights = real_weights_
        imag_weights = imag_weights_

    return real_weights, imag_weights


def build_ra_spinr_group(
    ucode: str,
    n_samples: int,
    n_channels: int,
    alpha_grd: float,
    distance_bins: list,
    angle_bins: list,
    t_grd: int = 0,
    grd_shl: int = 10,
    is_complex=True,
    ucode_kwargs: dict = dict,
    mem_kwargs: dict = dict,
) -> nxk.NcGroup:
    """Build Range-Angle SPINR group from ucode files that inherit form ra_spinr.dasm

    :param str ucode: ucode file name
    :param int n_samples: number of samples of radar data
    :param int n_channels: number of channels of radar data
    :param float alpha_grd: alpha parameter for exponential filtering for gradient estimation
    :param list distance_bins: TODO
    :param list angle_bins: TODO
    :param int t_grd: exp. filtering of gradient starts at t_grd, defaults to 0
    :param int grd_shl: bit shift of gradient estimates to scale it up, defaults to 10
    :param is_complex float: TODO
    :param ucode_kwargs dict: additional ucode arguments
    :param mem_kwargs dict: additional memory arguments
    :return nxk.NcGroup: nxk.NcGroup named <ucode> with synapses "real_wgt" and "imag_wgt"
    """

    ucode_path = KERNELS_PATH / f"{ucode}.dasm"

    n_distances = len(distance_bins)
    n_angles = len(angle_bins)

    real_wgt, imag_wgt = angle_synapse_weights(n_distances, n_channels, angle_bins, is_complex=is_complex)
    syn_args = dict(sparse_packing=True, use_shared_axon=True, weight_exp=-9)

    # range phasor weights
    r_phasor_wgts = phasor_weights(n_samples, distance_bins)  # (n_distances, )
    r_phasor_wgts = np.repeat(r_phasor_wgts[:, None], n_angles, axis=1)  # (n_distances, n_angles)
    r_phasor_wgts = r_phasor_wgts.flatten()
    lct = (r_phasor_wgts.real * (2**15 - 1)).astype(int)
    lst = (r_phasor_wgts.imag * (2**15 - 1)).astype(int)

    ucode_args = dict(
        alpha_grd=int(alpha_grd * 2**15),
        beta_grd=int((1 - alpha_grd) * 2**15),
        grd_sh=15 - grd_shl,
        t_grd=t_grd,
    )
    ucode_args.update(ucode_kwargs)

    mem_args = dict(lct=lct, lst=lst)
    mem_args.update(mem_kwargs)

    spinr_neurons = K.Neuron(
        shape=r_phasor_wgts.shape,
        ucode_path=ucode_path,
        ucode_args=ucode_args,
        out_spike_type="z",
        **mem_args,
    )

    # build group
    g = nxk.NcGroup(
        neuron=spinr_neurons,
        synapses=[
            K.Linear(weight=real_wgt, name="real_wgt", **syn_args),
            K.Linear(weight=imag_wgt, name="imag_wgt", **syn_args),
        ],
        name=ucode,
    )
    return g


def build_rd_spinr_group(
    ucode: str,
    n_samples: int,
    n_chirps: int,
    alpha_grd: float,
    distance_bins: list,
    grd_shl: int = 10,
    ucode_kwargs: dict = dict(),
    mem_kwargs: dict = dict(),
    index: int = 0,
) -> nxk.NcGroup:
    """Build range-Doppler SPINR group from ucode files that inherit form rd_spinr.dasm

    :param str ucode: ucode file name
    :param int n_samples: number of samples of radar data
    :param float alpha_grd: alpha parameter for exponential filtering for gradient estimation
    :param list distance_bins: TODO
    :param list n_angles: TODO
    :param int t_grd: exp. filtering of gradient starts at t_grd, defaults to 0
    :param int grd_shl: bit shift of gradient estimates to scale it up, defaults to 10
    :param ucode_kwargs dict: additional ucode arguments
    :param mem_kwargs dict: additional memory arguments
    :return K.Neuron:
    """

    ucode_path = KERNELS_PATH / f"{ucode}.dasm"

    # range phasor weights
    r_phasor_wgts = phasor_weights(n_samples, distance_bins)  # (n_distances, )
    r_phasor_wgts = np.repeat(r_phasor_wgts[:, None], n_chirps, axis=1)  # (n_distances, n_chirps)
    r_phasor_wgts = r_phasor_wgts.flatten()
    r_lct = (r_phasor_wgts.real * (2*15 - 1)).astype(int)
    r_lst = (r_phasor_wgts.imag * (2*15 - 1)).astype(int)

    # Doppler phasor weights
    d_phasor_wgts = phasor_weights(n_chirps)
    d_phasor_wgts = np.fft.fftshift(d_phasor_wgts)
    d_phasor_wgts = np.repeat(d_phasor_wgts[None, :], len(distance_bins), axis=0)  # (n_distances, n_chirps)
    d_phasor_wgts = d_phasor_wgts.flatten()
    d_lct = (d_phasor_wgts.real * (2**15 - 1)).astype(int) # prevent overflow default 2**15
    d_lst = (d_phasor_wgts.imag * (2**15 - 1)).astype(int) # prevent overflow

    ucode_args = dict(
        alpha_grd=int(alpha_grd * 2**15),
        beta_grd=int((1 - alpha_grd) * 2**15),
        grd_sh=15 - grd_shl,
        n_samples=n_samples,
    )
    ucode_args.update(ucode_kwargs)

    mem_args = dict(r_lct=r_lct, r_lst=r_lst, d_lct=d_lct, d_lst=d_lst)
    mem_args.update(mem_kwargs)

    spinr_neurons = K.Neuron(
        shape=d_phasor_wgts.shape,
        ucode_path=ucode_path,
        ucode_args=ucode_args,
        out_spike_type="z",
        **mem_args,
    )

    # build group
    syn_args = dict(sparse_packing=False, use_shared_axon=False, optimize_weights=False)
    n = np.prod(spinr_neurons.shape)
    #weights_identity = np.ones(n)[:, None]
    weights_identity = ((~mem_args['turnoff'][:, None])).astype('bool')

    g = nxk.NcGroup(
        neuron=spinr_neurons,
        synapses=[
            K.Linear(weight=weights_identity, name="real_wgt", **syn_args),
            #K.Linear(weight=weights_identity, name="imag_wgt", **syn_args),
        ],
        name=f"spinr{index}",
    )
    return g


class Nx_RD_SpiNR(BaseNxModel):

    def __init__(
        self,
        n_samples,
        n_chirps,
        alpha_grd,
        grd_shl,
        distance_bins=None,
    ):
        self.n_samples = n_samples
        self.n_chirps = n_chirps
        self.alpha_grd = alpha_grd
        self.grd_shl = grd_shl

        self.distance_bins = distance_bins
        if self.distance_bins is None:
            self.distance_bins = range(n_samples)

    def build_cpu_model(self):
        self.model = nxk.Module()

        # add CPU input
        self.model.add_group(nxk.CpuInputGroup((1,), name="cpu_input_real", out_spike_type="ws"))
        self.model.add_group(nxk.CpuInputGroup((1,), name="cpu_input_imag", out_spike_type="ws"))

        spinr_group = build_rd_spinr_group(
            "rd_spinr",
            self.n_samples,
            self.n_chirps,
            alpha_grd=self.alpha_grd,
            distance_bins=self.distance_bins,
            grd_shl=self.grd_shl,
        )
        self.model.add_group(spinr_group)

        # connect
        nxk.connect(dst=self.model.rd_spinr, src=self.model.cpu_input_real, dst_port_idx=0)
        nxk.connect(dst=self.model.rd_spinr, src=self.model.cpu_input_imag, dst_port_idx=1)

        self.model.setup()

    def build_cyclic_buffer_model(self, data):
        # sender instead of cyclic buffer as cyclic only allows for 1 byte msg
        # what else is different????
        # data shape: (1, n_timesteps)
        
        self.model = nxk.Module()

        # add NX Sender
        self.model.add_group(nxk.NcGroup(Sender(data.real, spike_type="ws"), name="input_real"))
        #self.model.add_group(nxk.NcGroup(Sender(data.imag, spike_type="ws"), name="input_imag"))

        spinr_group = build_rd_spinr_group(
            "rd_spinr",
            self.n_samples,
            self.n_chirps,
            alpha_grd=self.alpha_grd,
            distance_bins=self.distance_bins,
            grd_shl=self.grd_shl,
        )
        self.model.add_group(spinr_group)

        # connect
        nxk.connect(dst=self.model.rd_spinr.real_wgt, src=self.model.input_real)
        #nxk.connect(dst=self.model.rd_spinr.imag_wgt, src=self.model.input_imag)

        self.model.setup()
