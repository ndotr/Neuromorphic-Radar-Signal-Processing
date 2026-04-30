# Public packages
import numpy as np
from collections import OrderedDict
import logging
import importlib.resources
from scipy.sparse import csr_matrix
import time

# Intel packages
import nxkernel as nxk
import nxkernel.kernels as K
import nxcore.arch.n3b.n3board
import nxkernel.ports.port_transform as PT
from nxkernel import characterization as ch

# Custom packages
import nrsp.algs.nx.base_nx_model as base_nx_model
import nrsp.utils.fft_gpt
import nrsp.utils.nx
import nrsp.utils.cfar

logger = logging.getLogger(__name__)


class NxSpiNROSCFARModel(base_nx_model.BaseNxModel):
    """
    Doppler-SpiNR and OS-CFAR
    """
    def __init__(self, 
                 n_ranges, 
                 n_velocities,
                 spinr_alpha,
                 spinr_threshold,  
                 spinr_thresh_time,
                 spinr_thresh_silent,
                 guard_cells,
                 ref_cells,
                 cfar_k,
                 cfar_tau,
                 cfar_alpha,
                 mode='interleave'):

        #logger.debug("Initialize NxRFFTSpiNR2DModel() ...")

        self.mode = mode

        # CFAR parameters
        self.guard_cells = guard_cells
        self.ref_cells = ref_cells
        self.cfar_alpha = cfar_alpha
        self.cfar_k = cfar_k
        self.cfar_tau = cfar_tau

        # Network parameters
        self.n_ranges = n_ranges
        self.n_velocities = n_velocities # number of velocities can be choosen flexible
        self.output_shape = (self.n_velocities * self.n_ranges, )

        self.n_input_channels = n_ranges # input channels for range FFT equals number of input samples
        self.n_spinr_neurons = self.n_ranges * self.n_velocities 

        # Kernels:
        self.oscfar_ucode_path = importlib.resources.files('nrsp.algs.nx.kernels') / 'log_os_cfar.dasm'
        self.oscfar_ucode_path = self.oscfar_ucode_path.resolve()
        self.oscfar_ucode_args = dict(t_end=67)

        self.spinr_ucode_path = importlib.resources.files('nrsp.algs.nx.kernels') / 'spinr_long_tempspike_turnoff.dasm'
        self.spinr_ucode_path = self.spinr_ucode_path.resolve()
        self.spinr_ucode_args = {
                                'alpha_grd': (spinr_alpha*2**15),
                                'beta_grd': ((1-spinr_alpha)*2**15),
                                #'thresh_chirp': spinr_thresh_chirp,
                                'thresh_grd': spinr_threshold,
                                'thresh_silent': spinr_thresh_silent,
                                'thresh_time': spinr_thresh_time,
                                } 

        self.model = None
        self.mesh = None
        #self.viz = ch.visualize.WorkloadVisualizer(self.mesh)
        logger.debug("Initialized NxSpiNROSCFARModel().")

    def run_sample(self, input_data):

        try:
            logger.debug("Input Data Shape: {}".format(input_data.shape))
            ct = time.time()
            self.model.input_real.send(input_data[:, 0])
            self.model.input_imag.send(input_data[:, 1])
            output = self.model.output.recv()
            ct = time.time() - ct
        except:
            logger.error("Something went wrong.")
        finally:
            logger.info("Runtime: {} ms".format(ct*1e3))

        return output

    def prepare_input(self, input_data):

        # Convert complex to real 2d
        input_data_real = np.zeros((self.n_velocities, self.n_ranges, 2))
        input_data_real[...,0] = input_data.real.astype('int')
        input_data_real[...,1] = input_data.imag.astype('int')
        logger.debug("Input shape (timesteps, input_channels, 2): {}".format(input_data_real.shape))

        return input_data_real
    
    def _add_input(self, io_type, input_data=None, input_shape=None, tile_shape=None):

        core_types = ()
        tile_shapes = ()

        if io_type=='buffer':
            # === Cyclic Buffer as Input ===
            input_data = self.prepare_input(input_data)
            # nxkernel buffer assumes: (channels, timesteps)

            # Real Input Group
            input_group = nxk.NcGroup(nxk.kernels.CyclicBuffer(input_data[...,0].T, num_msg_bytes=2), name='input_real')
            self.model.add_group(input_group)
            core_types += ('neuron', )
            if tile_shape is not None:
                tile_shapes += tile_shape
            else:
                tile_shapes += (self.n_input_channels, )
            self.log_added_group(logger, tile_shape=tile_shapes[-1])

            # Imaginary Input Group
            input_group = nxk.NcGroup(nxk.kernels.CyclicBuffer(input_data[...,1].T, num_msg_bytes=2), name='input_imag')
            self.model.add_group(input_group)
            core_types += ('neuron', )
            if tile_shape is not None:
                tile_shapes += tile_shape
            else:
                tile_shapes += (self.n_input_channels, )
            self.log_added_group(logger, tile_shape=tile_shapes[-1])

        elif io_type=='cpu':
            # === CPU Input ===
            # Real Input Group
            input_group = nxk.CpuInputGroup(shape=input_shape, out_spike_type='ws', name='input_real')
            self.model.add_group(input_group)

            core_types += ('cpu', )
            tile_shapes += input_shape
            self.log_added_group(logger, tile_shape=tile_shapes[-1])

            # Imaginary Input Group
            input_group = nxk.CpuInputGroup(shape=input_shape, out_spike_type='ws', name='input_imag')
            self.model.add_group(input_group)

            core_types += ('cpu', )
            tile_shapes += input_shape
            self.log_added_group(logger, tile_shape=tile_shapes[-1])

        elif io_type=='ethernet':
            # === Ethernet Input ===
            # Real Input Group
            input_group = nxk.EthernetInputGroup(shape=input_shape, out_spike_type='ws', name='input_real')
            self.model.add_group(input_group)

            core_types += ('ethernet', )
            tile_shapes += input_shape
            self.log_added_group(logger, tile_shape=tile_shapes[-1])

            # Imaginary Input Group
            input_group = nxk.EthernetInputGroup(shape=input_shape, out_spike_type='ws', name='input_imag')
            self.model.add_group(input_group)

            core_types += ('ethernet', )
            tile_shapes += input_shape
            self.log_added_group(logger, tile_shape=tile_shapes[-1])
        else:
            logger.error("IO Type {} does not exist.".format(io_type))
        
        return tile_shapes, core_types

    def _add_output(self, io_type, output_shape, src_group=None, tile_shape=None):

        core_types = ()
        tile_shapes = ()

        if io_type=='cpu':
            # === Add CPU Output ===
            output_group = nxk.CpuOutputGroup(shape=(np.prod(output_shape),), name='output')

            self.model.add_group(output_group)
            if src_group is not None:
                nxk.connect(src=src_group, dst=output_group)

            core_types += ('cpu', )
            if tile_shape is not None:
                tile_shapes += tile_shape
            else:
                tile_shapes += output_shape
            self.log_added_group(logger, tile_shape=tile_shapes[-1])

        elif io_type=='ethernet':
            # === Add Ethernet Output ===
            output_group = nxk.EthernetOutputGroup(shape=(np.prod(output_shape),), name='output')

            self.model.add_group(output_group)
            if src_group is not None:
                nxk.connect(src=src_group, dst=output_group)

            core_types += ('ethernet', )
            tile_shapes += output_shape
            self.log_added_group(logger, tile_shape=tile_shapes[-1])

        elif io_type=='buffer':
            # === Cyclic Buffer as Output ===
            None
        else:
            logger.error("IO Type {} does not exist.".format(io_type))
        
        return tile_shapes, core_types

    def _add_core_model(self):

        core_types = ()
        tile_shapes = ()

        # Add SpiNR
        logger.debug("Adding SpiNR group ...")
        spinr_group = self._create_spinr_group()
        self.model.add_group(spinr_group)
        core_types += ('neuron', )
        tile_shapes += (self.n_spinr_neurons//33, )
        self.log_added_group(logger, tile_shape=tile_shapes[-1])

        # Add Linear OS-CFAR
        logger.debug("Adding OS-CFAR group ...")
        cfar_group = self._create_linear_oscfar_group(ref_cells=self.ref_cells, 
                                               guard_cells=self.guard_cells, 
                                               alpha=self.cfar_alpha, 
                                               tau=self.cfar_tau,
                                               k=self.cfar_k,
                                               output_shape=self.output_shape)
        self.model.add_group(cfar_group)
        core_types += ('neuron', )
        tile_shape = (self.n_spinr_neurons//33,)
        tile_shapes += tile_shape
        self.log_added_group(logger, tile_shape=tile_shapes[-1])

        return tile_shapes, core_types

    def build_model(self, io_type, input_data=None):

        # Init NxKernel module
        self.model = nxk.Module()
        self.mesh = nxk.system.mesh.OheoGulchMesh()

        core_types = ()
        tile_shapes = ()

        # === Add Input ===
        tile_shape, core_type = self._add_input(io_type=io_type, input_data=input_data, input_shape=(self.n_input_channels, ))
        core_types += core_type
        tile_shapes += tile_shape

        # === Add Core Model (FFT + SpiNR) ===
        tile_shape, core_type = self._add_core_model()
        core_types += core_type
        tile_shapes += tile_shape

        # === Add Output ===
        tile_shape, core_type = self._add_output(io_type=io_type, output_shape=self.output_shape)
        core_types += core_type
        tile_shapes += tile_shape

        # Connections
        # Can't connect group(CPU,Ethernet) to synapse directly?
        nxk.connect(src=self.model.input_real, dst=self.model.spinr.wgt_real, da_idx=0)
        nxk.connect(src=self.model.input_imag, dst=self.model.spinr.wgt_imag, da_idx=0)

        nxk.connect(dst=self.model.cfar.neighb_wgt, src=self.model.spinr)
        nxk.connect(dst=self.model.cfar.presyn_wgt, src=self.model.spinr)

        if io_type=='cpu' or io_type=='ethernet':
            nxk.connect(dst=self.model.output, src=self.model.cfar)

        # === Setup ===
        logger.info("Tile shapes: {}".format(tile_shapes))
        logger.info("Core types: {}".format(core_types))

        logger.info("Partition and setup ...")
        self.model.setup()
        self.model.partition(tile_shapes=tile_shapes)

        for g in self.model.groups:
            logger.info('{} on {} core(s)'.format(g.name, g.num_cores))

        addrs_list = self.get_addrs_list(core_types)
        logger.info("Address list: {}".format(addrs_list))
        self.init_board(addrs_list=addrs_list)

    def _create_spinr_group(self):

        # Define synapses
        syn_args = dict(sparse_packing=True, use_shared_axon=False, 
                        optimize_weights=True)
        # Create synapse
        W = np.zeros((self.n_ranges * self.n_velocities, self.n_ranges))
        for i in range(self.n_velocities):
            W[i*self.n_ranges:(i+1)*self.n_ranges] = np.eye(self.n_ranges)
        synapses = [K.Linear(weight=W.astype('int'), **syn_args, name='wgt_real'),
                    K.Linear(weight=W.astype('int'), **syn_args, name='wgt_imag')] 
        logger.debug("Synapse: {}".format(synapses))

        # Define neuron
        omega_range = np.repeat(self._phasor_weights(self.n_velocities, perc=1.0), self.n_ranges)
        neuron_args = dict(
            out_spike_type='ws', 
            ucode_path=self.spinr_ucode_path,
            ucode_args=self.spinr_ucode_args,
            re=0,
            im=0,
            lct=(omega_range.real * 2**15).astype(np.int32),
            lst=(omega_range.imag * 2**15).astype(np.int32),
        )

        # Create Neuron  
        neuron = K.Neuron(shape=(self.n_spinr_neurons,), 
                            **neuron_args)

        # Create Neuron group and connect
        spinr_group = nxk.NcGroup(neuron=neuron,
                                synapses=synapses,
                                name='spinr')
        
        return spinr_group
    
    
    def _create_linear_oscfar_group(self, ref_cells, guard_cells, alpha, k, tau, output_shape):

        # Define synapses
        syn_args = dict(sparse_packing=False, 
                        optimize_weights=True)

        ## Create synapse
        kernel = nrsp.utils.cfar.get_cfar_kernel_2d(guard_cells=guard_cells, ref_cells=ref_cells).astype(int)
        neighb_wgt = csr_matrix(nrsp.utils.nx.kernel_to_weights_2d((self.n_velocities, self.n_ranges), kernel))
        presyn_wgt = csr_matrix(np.eye(self.n_spinr_neurons, dtype=int) * k)

        synapses=[
            K.Linear(weight=neighb_wgt, name="neighb_wgt", **syn_args),
            K.Linear(weight=presyn_wgt, name="presyn_wgt", **syn_args),
        ]

        # Define neuron
        t_inhib = np.rint(-tau * np.log(1 / alpha)).astype(int)
        neuron_args = dict(
            out_spike_type='ws', 
            t_inhib=t_inhib,
        )

        # Create Neuron  
        neuron = K.Neuron(
            shape=output_shape,
            ucode_path=self.oscfar_ucode_path,
            ucode_args=self.oscfar_ucode_args,
            **neuron_args,
        )

        # Create Neuron group and connect
        cfar_group = nxk.NcGroup(neuron=neuron,
                                synapses=synapses,
                                name='cfar')
        
        return cfar_group    

    def _phasor_weights(self, n_bins, perc=1):

        return np.exp(1j*(np.linspace(0,perc-perc/n_bins,n_bins))*np.pi*2)  
    



        
