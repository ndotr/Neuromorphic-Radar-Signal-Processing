# Public packages
from pathlib import Path
import numpy as np
import typing as ty

# Intel packages
import nxkernel as nxk
import nxkernel.kernels as K
from nxkernel.kernels.sender import Sender
from nxkernel.system.address import AddressFactory

# Custom packages
from nrsp.utils.nx import kernel_to_weights_2d
from nrsp.utils.cfar import get_cfar_kernel_2d
from nrsp.algs.nx.spinr.nx_spinr import build_ra_spinr_group, build_rd_spinr_group
from nrsp.algs.nx.cfar.base_nx_model import BaseNxModel as BaseNxModelOld
from nrsp.algs.nx.base_nx_model import BaseNxModel

KERNELS_PATH = Path(__file__).resolve().parent.parent / "kernels"
UCODE_PATH_ENC = KERNELS_PATH / "log_enc.dasm"
UCODE_PATH_CFAR = KERNELS_PATH / "log_os_cfar.dasm"


def build_log_enc_group(
    shape: ty.Tuple[int, int],
    tau: float,
    thresh: float,
    values: np.array,
) -> nxk.NcGroup:
    """Build log encode neuron group.
    Neuron dynamic: u += (da + u) / tau and neuron spikes if u >= thresh-da and blocks afterwards.

    :return: nxk.NxGroup "log_enc" of encoder neurons
    """
    assert thresh > 0  # TODO fits as wu
    assert tau >= 1

    ucode_args = dict(tau_inv=int((1 / tau) * 2**15), thresh=int(thresh))
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
        name="log_enc",
    )
    return g


def build_log_os_cfar_group_2d(
    shape_2d: ty.Tuple[int, int],
    tau: float,
    alpha: float,
    k: int,
    n_timesteps: int,
    guard_cells: ty.Tuple[int, int],
    ref_cells: ty.Tuple[int, int],
    wrap_x: bool = False,
    wrap_y: bool = False,
    flatten: bool = True,
    index: int = 0,
) -> nxk.NcGroup:
    """Build 2D Log-OS-CFAR neuron group.

    :param ty.Tuple[int, int] shape_2d: 2D neuron population shape
    :param float tau: tau as used in log encoding
    :param float alpha: os-cfar alpha parameter
    :param int k: os-cfar k parameter
    :param int n_timesteps: total number of timesteps as neurons must check spiking condition in very last timestep
    :param ty.Tuple[int, int] guard_cells: os-cfar guard cells
    :param ty.Tuple[int, int] ref_cells: os-cfar reference cells
    :param bool wrap_x: wether cfar conections should wrap around x-axis, defaults to False
    :param bool wrap_y: wether cfar conections should wrap around y-axis, defaults to False
    :param bool flatten: wether neuron population is flattened, defaults to True
    :return nxk.NcGroup: nxk.NcGroup "log_os_cfar" with synapses "neighb_wgt and "presyn_wgt
    """

    assert tau >= 1
    assert alpha >= 1
    n_neurons = np.prod(shape_2d)

    t_inhib = np.rint(-tau * np.log(1 / alpha)).astype(int)
    neuron_shape = shape_2d
    if flatten:
        neuron_shape = (n_neurons,)

    ucode_args = dict(t_end=n_timesteps)
    mem_args = dict(t_inhib=t_inhib)
    neurons = K.Neuron(
        shape=neuron_shape,
        ucode_path=UCODE_PATH_CFAR,
        ucode_args=ucode_args,
        out_spike_type="z",  # bool
        **mem_args,
    )

    kernel = get_cfar_kernel_2d(guard_cells, ref_cells).astype(int)
    neighb_wgt = kernel_to_weights_2d(shape_2d, kernel, wrap_x=wrap_x, wrap_y=wrap_y)
    presyn_wgt = np.eye(n_neurons, dtype=int) * k
    syn_args = dict(sparse_packing=True, use_shared_axon=True)

    # build group
    g = nxk.NcGroup(
        neuron=neurons,
        synapses=[
            K.Linear(weight=neighb_wgt, name="neighb_wgt", **syn_args),
            K.Linear(weight=presyn_wgt, name="presyn_wgt", **syn_args),
        ],
        name="log_os_cfar" + f"{index}",
    )
    return g


class NxRASpiNRLogOsCfarModel(BaseNxModelOld):
    """Range-Angle SpiNR model with log encoding of gradient estimate and Log-OS-CFAR layer"""

    def __init__(
        self,
        n_channels,
        n_samples,
        alpha_grd,
        tau,  # log encoding
        thresh,  # log encoding
        alpha_cfar,  # os cfar
        k_cfar,  # os cfar
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
        self.alpha_cfar = alpha_cfar
        self.k_cfar = k_cfar
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
        """Build nxk.Module: CyclicBuffer -> Log-Encode SPINR -> Log-OS-CFAR
        Sets self.model and calls self.model.setup().
        data is expected to be scaled to signed byte range.

        :param np.array data: radar data of shape (n_channels, n_timesteps) TODO
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
                "ra_spinr_log_enc",
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
                    t_enc=self.t_enc,
                ),
                mem_kwargs=dict(),
            )
        )
        nxk.connect(dst=model.ra_spinr_log_enc.real_wgt, src=model.buf)
        nxk.connect(dst=model.ra_spinr_log_enc.imag_wgt, src=model.buf)
        # ---------------------------------------------------------------------- #
        # add log os-cfar group
        model.add_group(
            build_log_os_cfar_group_2d(
                shape_2d=(self.n_distances, self.n_angles),
                tau=self.tau,
                alpha=self.alpha_cfar,
                k=self.k_cfar,
                n_timesteps=self._n_timesteps,
                ref_cells=self.ref_cells,
                guard_cells=self.guard_cells,
            )
        )

        nxk.connect(dst=model.log_os_cfar.neighb_wgt, src=model.ra_spinr_log_enc)
        nxk.connect(dst=model.log_os_cfar.presyn_wgt, src=model.ra_spinr_log_enc)

        self.model = model
        self.model.setup()

    def _make_addrs_list(self):
        # TODO
        # self.model.partition([self.n_channels, 500, 300]) # real data
        self.model.partition([self.n_channels, 250, 300])  # complex data

        addrs_list = []
        nc_offset = 0
        for group in self.model.groups:
            addrs_list.append([AddressFactory.make_addr(chip=0, core_idx=idx + nc_offset) for idx in range(group.num_cores)])
            nc_offset += group.num_cores

        return addrs_list


class Nx_RD_SpiNR_LogOSCfar(BaseNxModel):

    def __init__(
        self,
        n_samples,
        n_chirps,
        alpha_grd,
        tau_enc,
        thresh_enc,
        alpha_cfar,
        k_cfar,
        guard_cells,
        ref_cells,
        grd_shl,
        t_enc=0,
        distance_bins=None,
        thresh_silent=None,
        turnoff_perc=None,
        rng=np.random.default_rng(seed=42),
    ):
        super().__init__()
        self.rng = rng
        self.n_samples = n_samples
        self.n_chirps = n_chirps
        self.alpha_grd = alpha_grd
        self.tau_enc = tau_enc
        self.thresh_enc = thresh_enc
        self.alpha_cfar = alpha_cfar
        self.k_cfar = k_cfar
        self.guard_cells = guard_cells
        self.ref_cells = ref_cells
        self.grd_shl = grd_shl
        self.t_enc = t_enc

        self.thresh_silent = thresh_silent
        self.spinr_name = "rd_spinr_log_enc"
        if thresh_silent is not None:
            self.spinr_name = "rd_spinr_log_enc_turnoff_real"

        self.distance_bins = distance_bins
        if self.distance_bins is None:
            self.distance_bins = range(n_samples)
        self.n_distances = len(self.distance_bins)

        # for energy plots, disable some percentage of neurons
        self.turnoff_perc = turnoff_perc

    def build_cpu_model(self):
        self.model = nxk.Module()

        # add CPU input
        self.model.add_group(nxk.CpuInputGroup((1,), name="cpu_input_real", out_spike_type="ws"))
        self.model.add_group(nxk.CpuInputGroup((1,), name="cpu_input_imag", out_spike_type="ws"))

        self._add_main_groups()

        # connect
        nxk.connect(dst=self.spinr_group(), src=self.model.cpu_input_real, dst_port_idx=0)
        nxk.connect(dst=self.spinr_group(), src=self.model.cpu_input_imag, dst_port_idx=1)

        self.model.setup()

    def build_cyclic_buffer_model(self, data: np.array, n_models=1):
        self.model = nxk.Module()

        for i in range(n_models):
            # add NX Sender
            self.model.add_group(nxk.NcGroup(Sender(data.real, spike_type="ws"), name=f"input_real{i}"))
            #self.model.add_group(nxk.NcGroup(Sender(data.imag, spike_type="ws"), name=f"input_imag{i}"))

            self._add_main_groups(index=i)

            # connect
            spinr_group = getattr(self.model, f"spinr{i}")
            input_real_group = getattr(self.model, f"input_real{i}")
            nxk.connect(dst=spinr_group.real_wgt, src=input_real_group)
            #input_imag_group = getattr(self.model, f"input_imag{i}")
            #nxk.connect(dst=spinr_group.imag_wgt, src=input_imag_group)

        self.model.setup()

    def _add_main_groups(self, index=0):
        """Add SpiNR and OS-CFAR group and connect them."""

        mem_kwargs = dict()
        ucode_kwargs = dict(
            tau_inv=int(2**15 / self.tau_enc),
            thresh=int(self.thresh_enc),
            t_enc=self.t_enc,
        )

        if self.thresh_silent is not None:
            ucode_kwargs["thresh_silent"] = self.thresh_silent

        # set turnoff mask if needed for turnoff model
        if self.turnoff_perc is not None and self.thresh_silent is not None:
            mem_kwargs["turnoff"] = self.rng.random(self.n_distances * self.n_chirps) < self.turnoff_perc / 100
            print(f"Turning off {np.sum(mem_kwargs['turnoff'])} neurons ({self.turnoff_perc}%)")

        self.model.add_group(
            build_rd_spinr_group(
                self.spinr_name,
                self.n_samples,
                self.n_chirps,
                alpha_grd=self.alpha_grd,
                distance_bins=self.distance_bins,
                grd_shl=self.grd_shl,
                ucode_kwargs=ucode_kwargs,
                mem_kwargs=mem_kwargs,
                index=index,
            )
        )

        ## add log os-cfar group
        #self.model.add_group(
        #    build_log_os_cfar_group_2d(
        #        shape_2d=(self.n_distances, self.n_chirps),
        #        tau=self.tau_enc,
        #        alpha=self.alpha_cfar,
        #        k=self.k_cfar,
        #        n_timesteps=self.n_samples * self.n_chirps + 2,
        #        ref_cells=self.ref_cells,
        #        guard_cells=self.guard_cells,
        #        index=index,
        #    )
        #)

        ## connect
        #spinr_group = getattr(self.model, f"spinr{index}")
        #cfar_group = getattr(self.model, f"log_os_cfar{index}")
        #nxk.connect(dst=cfar_group.neighb_wgt, src=spinr_group)
        #nxk.connect(dst=cfar_group.presyn_wgt, src=spinr_group)

    def spinr_group(self):
        return getattr(self.model, self.spinr_name)

    def get_addrs_list(self, *args):
        # *args for backward compability as old implementation accepted core_types argument

        addrs_list = []
        neuron_offset = 0
        cpu_offset = 0
        chip_offset = -1

        self.logger.debug("Number of cores: {}".format(self.model.num_cores))
        for i, g in enumerate(self.model.groups):
            num_cores = g.num_cores
            group_type = type(g)

            self.logger.debug("Group: {}, # of cores: {}, core type: {}".format(g.name, num_cores, group_type))

            if group_type == nxk.groups.nc_group.NcGroup:
                if i%3==0:
                    chip_offset += 1
                    neuron_offset = 0

                addrs_list.append([nxk.system.AddressFactory.make_addr(core_idx=idx + neuron_offset, chip_idx=chip_offset) for idx in range(num_cores)])
                neuron_offset += num_cores


            elif group_type in [nxk.groups.cpu_group.CpuInputGroup, nxk.groups.cpu_group.CpuOutputGroup]:
                addrs_list.append([nxk.system.AddressFactory.make_addr(cpu_idx=idx + cpu_offset) for idx in range(num_cores)])
                cpu_offset += num_cores

            elif group_type in [
                nxk.groups.eth_group.EthernetInputGroup,
                nxk.groups.eth_group.EthernetOutputGroup,
                nxk.groups.eth_group.EthernetOutputServer,
            ]:
                ETH_INTERFACE, ETH_MAC, LOIHI_MAC, ETH_CHIP_IDX = nrsp.utils.nx.host.get_ethernet_params()
                tmp = []
                for idx in range(num_cores):
                    eth_chip_loc = self.mesh.get_chip_loc(idx=ETH_CHIP_IDX)
                    tmp.append(
                        nxk.make_addr(
                            chip_idx=ETH_CHIP_IDX,
                            chip_loc=eth_chip_loc,
                            ethernet_interface=ETH_INTERFACE,
                            ethernet_mac_address=ETH_MAC,
                            loihi_mac_address=LOIHI_MAC,
                        )
                    )
                addrs_list.append(tmp)
                if group_type in [nxk.groups.eth_group.EthernetInputGroup]:
                    chip_offset += 1
                    neuron_offset = 0

            else:
                raise (f"unknwon group type {group_type}")

        return addrs_list

