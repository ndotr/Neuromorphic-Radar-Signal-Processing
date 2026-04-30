# Public packages
import numpy as np
import logging
import importlib.resources
import time

# Intel packages
import nxkernel as nxk
import nxkernel.kernels as K

# Custom packages
import nrsp.algs.nx.base_nx_model as base_nx_model
import nrsp.utils.nx
import nrsp.utils.log


class Nx_RD_SpiNR_AGS(base_nx_model.BaseNxModel):
    """
    Range-Doppler SpiNR model with AGS-style gradient dynamics.

    The model directly maps time-domain radar samples onto a 2D range-Doppler
    resonator population of shape (n_distances, n_chirps).
    """

    def __init__(
        self,
        n_samples,
        n_chirps,
        alpha_grd,
        grd_shl,
        c1_thresh,
        c2_thresh,
        mon_thresh,
        TILE_SHAPES,
        distance_bins=None,
        rng=np.random.default_rng(seed=42),
    ):
        super().__init__()

        self.logger = nrsp.utils.log.get_logger(__name__)
        self.rng = rng
        self.TILE_SHAPES = TILE_SHAPES

        # Network parameters
        self.n_samples = n_samples
        self.n_chirps = n_chirps
        self.alpha_grd = alpha_grd
        self.grd_shl = grd_shl
        self.c1_thresh = c1_thresh
        self.c2_thresh = c2_thresh
        self.mon_thresh = mon_thresh

        self.distance_bins = distance_bins
        if self.distance_bins is None:
            self.distance_bins = list(range(n_samples//2, n_samples))
        self.n_distances = len(self.distance_bins)

        self.input_shape = (1,)
        self.output_shape = (self.n_distances* self.n_chirps,)
        self.n_spinr_neurons = self.n_distances * self.n_chirps

        # Kernel / ucode setup
        self.spinr_ucode_name = "rd_spinr_ags"

        self.spinr_ucode_path = importlib.resources.files("nrsp.algs.nx.kernels") / f"{self.spinr_ucode_name}.dasm"
        self.spinr_ucode_path = self.spinr_ucode_path.resolve()

        self.spinr_ucode_args = dict(
            alpha_grd=int(self.alpha_grd * 2**15),
            beta_grd=int((1 - self.alpha_grd) * 2**15),
            grd_sh=15 - self.grd_shl,
            n_samples=self.n_samples,
            c1_thresh=self.c1_thresh,
            c2_thresh=self.c2_thresh,
            mon_thresh=self.mon_thresh,
            spike_time=self.n_chirps * self.n_samples + 1
        )

        # Init NxKernel module
        self.model = nxk.Module()
        host = nrsp.utils.nx.host.name
        if host == "mercedes":
            self.mesh = nxk.system.mesh.KapohoPointMesh()
        elif host == "twt":
            self.mesh = nxk.system.mesh.OheoGulchMesh()
        elif host == "nc0":
            self.mesh = nxk.system.mesh.OheoGulchMesh()
        else:
            self.mesh = nxk.system.mesh.OheoGulchMesh()

        self.logger.debug("Initialized Nx_RD_SpiNR_AGS.")

    def run_sample(self, data):
        """
        Run one scalar sample through the model.

        In ethernet mode, `data` is expected to be one value or shape (1,).
        """
        output = None
        ct = None
        try:
            start_time = time.time()
            self.model.input.send(data)
            output = self.model.output.recv()
            ct = time.time() - start_time
        except Exception as e:
            self.logger.error(f"Something went wrong during run_sample: {e}", exc_info=True)
        finally:
            if ct is not None:
                self.logger.debug(f"Runtime: {ct * 1e3:.2f} ms")
            else:
                self.logger.error("Runtime could not be measured due to error.")
        return output, ct

    def _add_input(self, io_type, input_data=None, input_shape=None, name="input"):
        core_types = ()
        tile_shapes = ()

        if io_type == "buffer":
            input_group = nxk.NcGroup(
                nxk.kernels.CyclicBuffer(input_data.T, num_msg_bytes=2),
                name=name,
            )
            self.model.add_group(input_group)

            core_types += ("neuron",)
            tile_shapes += self.TILE_SHAPES['input_buf']
            self.log_added_group(self.logger, tile_shape=tile_shapes[-1])


        elif io_type == "ethernet":
            input_group = nxk.EthernetInputGroup(
                shape=input_shape,
                out_spike_type="ls",
                name=name,
            )
            self.model.add_group(input_group)

            core_types += ("ethernet",)
            tile_shapes += self.TILE_SHAPES['input_eth']
            self.log_added_group(self.logger, tile_shape=tile_shapes[-1])

        else:
            raise ValueError(f"Unsupported io_type: {io_type}")

        return tile_shapes, core_types

    def _add_output(self, io_type, src_group, output_shape=None):
        core_types = ()
        tile_shapes = ()

        if io_type == "ethernet":
            output_group = nxk.EthernetOutputGroup(shape=(np.prod(output_shape),), name="output")
            self.model.add_group(output_group)
            nxk.connect(src=src_group, dst=output_group)

            core_types += ("ethernet",)
            #tile_shapes += self.TILE_SHAPES['output_eth']
            tile_shapes += (np.prod(output_shape),)
            self.log_added_group(self.logger, tile_shape=tile_shapes[-1])

        elif io_type == "buffer":
            pass

        else:
            raise ValueError(f"Unsupported io_type: {io_type}")

        return tile_shapes, core_types

    def _add_core_model(self, input_group, index=0):
        core_types = ()
        tile_shapes = ()

        self.logger.debug("Adding RD SpiNR group ...")
        spinr_group = self._create_spinr_group(name=f"spinr{index}")
        self.model.add_group(spinr_group)
        nxk.connect(dst=spinr_group, src=input_group, dst_port_idx=0)

        core_types += ("neuron",)
        tile_shapes += self.TILE_SHAPES['spinr']
        self.log_added_group(self.logger, tile_shape=tile_shapes[-1])

        return tile_shapes, core_types

    def build_model(self, io_type, input_data=None):
        core_types = ()
        tile_shapes = ()

        self.model = nxk.Module()

        if io_type == "buffer":
            tile_shape, core_type = self._add_input(
                io_type=io_type,
                input_data=input_data,
                name="input",
            )
            tile_shapes += tile_shape
            core_types += core_type
            input_group = getattr(self.model, "input")
        else:
            tile_shape, core_type = self._add_input(
                io_type=io_type,
                input_shape=self.input_shape,
                name="input",
            )
            tile_shapes += tile_shape
            core_types += core_type
            input_group = getattr(self.model, "input")

        tile_shape, core_type = self._add_core_model(input_group=input_group, index=0)
        tile_shapes += tile_shape
        core_types += core_type

        tile_shape, core_type = self._add_output(
            io_type=io_type,
            src_group=self.model.spinr0,
            output_shape=self.output_shape,
        )
        tile_shapes += tile_shape
        core_types += core_type

        self.logger.info("Tile shapes: {}".format(tile_shapes))
        self.logger.info("Core types: {}".format(core_types))

        self.logger.info("Partition and setup ...")
        self.model.setup()
        self.model.partition(tile_shapes=tile_shapes)

        for g in self.model.groups:
            self.logger.info("{} on {} core(s)".format(g.name, g.num_cores))

        addrs_list = self.get_addrs_list()
        self.logger.info("Address list: {}".format(addrs_list))
        self.init_board(addrs_list=addrs_list)

    def _create_spinr_group(self, name="spinr"):
        """
        Create the range-Doppler SpiNR neuron group directly inside the class,
        analogous to `_create_fft_stage_group` or `_create_cacfar_group` in the
        FFT-based template.
        """
        # Range phasor weights
        r_phasor_wgts = self._phasor_weights(self.n_samples, perc=1.0, bins=self.distance_bins)
        r_phasor_wgts = np.repeat(r_phasor_wgts[:, None], self.n_chirps, axis=1)
        r_phasor_wgts = r_phasor_wgts.flatten()
        r_lct = (r_phasor_wgts.real * (2**15 )).astype(np.int32)
        r_lst = (r_phasor_wgts.imag * (2**15 )).astype(np.int32)

        # Doppler phasor weights
        d_phasor_wgts = self._phasor_weights(self.n_chirps, perc=1.0)
        d_phasor_wgts = np.fft.fftshift(d_phasor_wgts)
        d_phasor_wgts = np.repeat(d_phasor_wgts[None, :], self.n_distances, axis=0)
        d_phasor_wgts = d_phasor_wgts.flatten()
        d_lct = (d_phasor_wgts.real * (2**15 )).astype(np.int32)
        d_lst = (d_phasor_wgts.imag * (2**15 )).astype(np.int32)

        mem_args = dict(
            r_lct=r_lct,
            r_lst=r_lst,
            d_lct=d_lct,
            d_lst=d_lst,
        )

        weights_identity = np.ones((self.n_spinr_neurons, 1), dtype=bool)

        neuron_args = dict(
            out_spike_type="ls",
            ucode_path=self.spinr_ucode_path,
            ucode_args=self.spinr_ucode_args,
            **mem_args,
        )

        neuron = K.Neuron(
            shape=(self.n_spinr_neurons,),
            **neuron_args,
        )

        syn_args = dict(
            sparse_packing=False,
            use_shared_axon=False,
            optimize_weights=False,
        )

        spinr_group = nxk.NcGroup(
            neuron=neuron,
            synapses=[
                K.Linear(weight=weights_identity, name="real_wgt", **syn_args),
            ],
            name=name,
        )

        return spinr_group

    def _phasor_weights(self, n_bins, bins=None, perc=1.0):
        """
        Create complex phasor weights.

        If `bins` is None, use all bins.
        """
        if bins is None:
            bins = np.arange(n_bins)
        bins = np.asarray(list(bins))
        return np.exp(1j * (bins / n_bins) * 2 * np.pi * perc - 1j * np.pi)

    def get_addrs_list(self, *args):
        # *args for backward compatibility as old implementation accepted core_types argument
    
        addrs_list = []
        cpu_offset = 0
    
        N_CHIPS = 1
        chip_neuron_offset = [0] * N_CHIPS
    
        self.logger.debug("Number of cores: {}".format(self.model.num_cores))
        for i, g in enumerate(self.model.groups):
            num_cores = g.num_cores
            group_type = type(g)
    
            self.logger.debug(
                "Group: {}, # of cores: {}, core type: {}".format(g.name, num_cores, group_type)
            )
    
            if group_type == nxk.groups.nc_group.NcGroup:
                # -------------------------
                # INPUT
                # -------------------------
                if "input" in g.name:
                    N_CHIPS_FOR_INPUT = 1
                    chip = 0
    
                # -------------------------
                # SPINR
                # -------------------------
                elif "spinr" in g.name:
                    N_CHIPS_FOR_SPINR = 1
                    chip = 0
    
                # -------------------------
                # FALLBACK
                # -------------------------
                else:
                    chip = 0
    
                # -------------------------
                # ALLOCATION
                # -------------------------
                addrs = [
                    nxk.system.AddressFactory.make_addr(
                        core_idx=idx + chip_neuron_offset[chip],
                        chip_idx=chip,
                    )
                    for idx in range(num_cores)
                ]
    
                chip_neuron_offset[chip] += num_cores
                addrs_list.append(addrs)
    
                used_cores = [addr.core_idx for addr in addrs]
                self.logger.debug(
                    f"Group '{g.name}' assigned to chip {chip}, cores: {used_cores}"
                )
    
            elif group_type in [
                nxk.groups.cpu_group.CpuInputGroup,
                nxk.groups.cpu_group.CpuOutputGroup,
            ]:
                addrs_list.append(
                    [
                        nxk.system.AddressFactory.make_addr(cpu_idx=idx + cpu_offset)
                        for idx in range(num_cores)
                    ]
                )
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
    
            else:
                raise RuntimeError(f"unknown group type {group_type}")
    
        return addrs_list