# Public packages
from pathlib import Path
import numpy as np
import typing as ty

# Intel packages
import nxkernel as nxk
import nxkernel.kernels as K
from nxkernel.kernels.sender import Sender
from nxkernel.system.address import AddressFactory

# Custom package
from nrsp.utils.nx import kernel_to_weights_2d
from nrsp.utils.cfar import calculate_num_ref_cells, find_best_ca_cfar_weight_scale_nx, get_cfar_kernel_2d
from nrsp.algs.nx.spinr.nx_spinr import build_ra_spinr_group
from nrsp.algs.nx.cfar.base_nx_model import BaseNxModel


KERNELS_PATH = Path(__file__).resolve().parent.parent / "kernels"
UCODE_PATH_ENC = KERNELS_PATH / "rate_enc.dasm"
UCODE_PATH_CFAR = KERNELS_PATH / "rate_ca_cfar.dasm"


def build_rate_enc_group(
    shape: ty.Tuple[int, int],
    tau: float,
    thresh: int,
    rest: int,
    values: np.array,
) -> nxk.NcGroup:
    """Build rate encode neuron group.
    Neuron dynamic: u += (da - u) / tau and neuron spikes if u >= thresh

    :return: nxk.NxGroup "rate_enc" of encoder neurons
    """
    assert thresh > 0
    assert tau >= 1

    ucode_args = dict(tau_inv=int(2**15 / tau), thresh=int(thresh), rest=int(rest))
    mem_args = dict(value=values)
    enc = K.Neuron(
        shape=shape,
        ucode_path=UCODE_PATH_ENC,
        ucode_args=ucode_args,
        out_spike_type="z",  # bool
        **mem_args,
    )

    # build group
    syn_args = dict(sparse_packing=True, use_shared_axon=False)
    g = nxk.NcGroup(
        neuron=enc,
        # synapses=[
        #     K.Linear(
        #         weight=np.eye(np.prod(shape)),
        #         **syn_args,
        #     ),
        # ],
        name="rate_enc",
    )
    return g


def build_rate_ca_cfar_group(
    shape_2d: ty.Tuple[int, int],
    alpha: float,
    offset: float,
    n_timesteps: int,
    guard_cells: ty.Tuple[int, int],
    ref_cells: ty.Tuple[int, int],
    wrap_x: bool = False,
    wrap_y: bool = False,
    flatten: bool = True,
) -> nxk.NcGroup:
    """Build 2D Rate-CA-CFAR neuron group. This group is implemented with a single synapse.

    :param ty.Tuple[int, int] shape_2d: 2D neuron population shape
    :param float alpha: CA-CFAR alpha parameter
    :param float offset: CA-CFAR offset parameter
    :param int n_timesteps: total number of timesteps as neurons must check spiking condition in very last timestep
    :param ty.Tuple[int, int] guard_cells: ca-cfar guard cells
    :param ty.Tuple[int, int] ref_cells: ca-cfar reference cells
    :param bool wrap_x: wether cfar conections should wrap around x-axis, defaults to False
    :param bool wrap_y: wether cfar conections should wrap around y-axis, defaults to False
    :param bool flatten: wether neuron population is flattened, defaults to True
    :return nxk.NcGroup: nxk.NcGroup "rate_ca_cfar"
    """

    n_neurons = np.prod(shape_2d)
    neuron_shape = shape_2d
    if flatten:
        neuron_shape = (n_neurons,)

    # neighbor weights:
    # - cfar equation: n_spikes_cut - alpha * (n_spikes_neighbors / n_neighbors) > offset
    # - Weights: 1 for CUT and -alpha/n_neighbors for neighbor neurons
    # - first build weight matrix as float and determine best scale afterwards
    kernel = get_cfar_kernel_2d(guard_cells, ref_cells).astype(int)
    weights = kernel_to_weights_2d(shape_2d, kernel, wrap_x=wrap_x, wrap_y=wrap_y).astype(float)  # 1 for neighbors
    N = calculate_num_ref_cells(guard_cells, ref_cells)

    weights *= -alpha / N  # neighbor weights
    np.fill_diagonal(weights, 1)  # add CUT weight

    scale = find_best_ca_cfar_weight_scale_nx(alpha, N)
    weights *= scale
    weights = np.rint(weights).astype(int)
    offset_ = np.rint(offset * scale).astype(int)

    # build group
    ucode_args = dict(t_end=n_timesteps, offset=offset_)
    mem_args = dict()
    neurons = K.Neuron(
        shape=neuron_shape,
        ucode_path=UCODE_PATH_CFAR,
        ucode_args=ucode_args,
        out_spike_type="z",  # bool
        **mem_args,
    )

    syn_args = dict(sparse_packing=True, use_shared_axon=True)
    g = nxk.NcGroup(
        neuron=neurons,
        synapses=[K.Linear(weight=weights, **syn_args)],
        name="rate_ca_cfar",
    )
    return g


class NxRASpiNRRateCaCfarModel(BaseNxModel):

    def __init__(
        self,
        n_channels,
        n_samples,
        alpha_grd,
        tau,  # rate encoding
        thresh,  # rate encoding
        rest,  # rate encoding
        alpha_cfar,  # ca cfar
        offset_cfar,  # ca cfar
        guard_cells,
        ref_cells,
        distance_bins=None,
        angle_bins=None,
        t_grd=0,
        t_enc=0,
        grd_shl=9,
    ):
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.alpha_grd = alpha_grd

        self.tau = tau
        self.thresh = thresh
        self.rest = rest

        self.alpha_cfar = alpha_cfar
        self.offset_cfar = offset_cfar

        self.guard_cells = guard_cells

        self.ref_cells = ref_cells
        self.t_grd = t_grd
        self.t_enc = t_enc
        self.grd_shl = grd_shl

        self.distance_bins = range(n_samples)
        if distance_bins:
            self.distance_bins = distance_bins
        self.n_distances = len(self.distance_bins)

        self.angle_bins = range(n_channels)
        if angle_bins:
            self.angle_bins = angle_bins
        self.n_angles = len(self.angle_bins)

        self._n_timesteps = self.n_samples + 2  # 2 layer network, spikes need to travel through layers

    def build_model(self, data: np.array):
        """Build nxk.Module: CyclicBuffer -> Rate-Encode SPINR -> Rate CA--CFAR
        Sets self.model and calls self.model.setup()
        data is expected to be scaled to signed byte range.

        :param np.array data: radar data of shape (n_channels, n_timesteps) scaled to a signed byte.
        """

        is_complex = np.iscomplexobj(data)
        if is_complex:
            data = np.vstack((data.real, data.imag))
        data = data.astype(int)

        model = nxk.Module()

        # ---------------------------------------------------------------------- #
        # add buffer
        # model.add_group(nxk.NcGroup(K.CyclicBuffer(data), name="buf"))
        model.add_group(nxk.NcGroup(Sender(data, spike_type="ws"), name="buf"))
        # ---------------------------------------------------------------------- #
        # add spinr group
        model.add_group(
            build_ra_spinr_group(
                "ra_spinr_rate_enc",
                self.n_samples,
                self.n_channels,
                self.alpha_grd,
                self.distance_bins,
                self.angle_bins,
                t_grd=self.t_grd,
                grd_shl=self.grd_shl,
                is_complex=is_complex,
                ucode_kwargs=dict(
                    tau_inv=int(2**15 / self.tau),
                    thresh=int(self.thresh),
                    rest=int(self.rest),
                    t_enc=self.t_enc,
                ),
                mem_kwargs=dict(),
            )
        )
        nxk.connect(dst=model.ra_spinr_rate_enc.real_wgt, src=model.buf)
        nxk.connect(dst=model.ra_spinr_rate_enc.imag_wgt, src=model.buf)
        # ---------------------------------------------------------------------- #
        # add rate ca-cfar group
        model.add_group(
            build_rate_ca_cfar_group(
                shape_2d=(self.n_distances, self.n_angles),
                alpha=self.alpha_cfar,
                offset=self.offset_cfar,
                n_timesteps=self._n_timesteps,
                ref_cells=self.ref_cells,
                guard_cells=self.guard_cells,
            )
        )
        nxk.connect(dst=model.rate_ca_cfar, src=model.ra_spinr_rate_enc)

        self.model = model
        self.model.setup()

    def _make_addrs_list(self):
        # TODO
        # self.model.partition([self.n_channels, 500, 200]) # real data
        self.model.partition([self.n_channels, 250, 200])  # complex data

        addrs_list = []
        nc_offset = 0
        for group in self.model.groups:
            addrs_list.append([AddressFactory.make_addr(chip=0, core_idx=idx + nc_offset) for idx in range(group.num_cores)])
            nc_offset += group.num_cores

        return addrs_list
