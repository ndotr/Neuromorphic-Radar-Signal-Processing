# Public packages
import numpy as np
from collections import OrderedDict
import logging
import importlib.resources
import time
from scipy.sparse import csr_matrix


# Intel packages
import nxkernel as nxk
import nxkernel.kernels as K
import nxcore.arch.n3b.n3board
import nxkernel.ports.port_transform as PT
from nxkernel import characterization as ch

# Custom packages
import nrsp.algs.nx.base_nx_model as base_nx_model
import nrsp.utils.nx

class NxSpiNRModel(base_nx_model.BaseNxModel):
    """

    1D real FFT

    """
    def __init__(self, 
                 n_samples, 
                 n_channels, 
                 tile_shapes_dict, 
                 spinr_alpha=0.1,
                 ):
        """
        TODO:

        """
        self.logger = nrsp.utils.log.get_logger(__name__)
        self.logger.debug("Initialize NxSpiNRModel() ...")

        # Network parameters
        self.tile_shapes_dict = tile_shapes_dict
        self.n_samples = n_samples
        self.n_ranges = n_samples //2 + 1
        self.n_channels = n_channels
        self.input_shape = (1,)
        self.output_shape = self.tile_shapes_dict['output'][0] 


        # Kernels:
        self.spinr_ucode_path = importlib.resources.files('nrsp.algs.nx.kernels') / 'rspinr_long_gradspike_simple_continuous.dasm'
        self.spinr_ucode_path = self.spinr_ucode_path.resolve()
        self.spinr_ucode_args = {
                                'alpha_grd': (spinr_alpha*2**15),
                                'beta_grd': ((1-spinr_alpha)*2**15),
                                #'thresh_chirp': spinr_thresh_chirp,
                                'thresh_grd': 0,
                                'thresh_silent': 1024,
                                'thresh_start_time': 0,
                                'thresh_spike_time': int(self.n_samples + 2),
                                'reset_time': int(self.n_samples + 2),
                                } 

        # Init NxKernel module
        self.model = nxk.Module()
        host = nrsp.utils.nx.host.name
        #self.mesh = nxk.system.mesh.OheoGulchMesh()
        if host == 'mercedes':
            self.mesh = nxk.system.mesh.KapohoPointMesh()
        elif host == 'twt':
            self.mesh = nxk.system.mesh.OheoGulchMesh()
        elif host == 'nc0':
            self.mesh = nxk.system.mesh.OheoGulchMesh()
        #self.viz = ch.visualize.WorkloadVisualizer(self.mesh)
        self.logger.debug("Initialized NxSpinRModel().")

    def run_sample(self, data):
        output = None
        ct = None
        try:
            start_time = time.time()
            for i in range(self.n_channels):
                input_obj = getattr(self.model, f"input{i}")
                input_obj.send(data[i])
            output = np.zeros((self.n_ranges, self.n_channels))
            for i in range(self.n_channels):
                output_obj = getattr(self.model, f"output_ch{i}")
                output[:, i] = output_obj.recv() #+ self.model.output_imag.recv()*1j
            ct = time.time() - start_time
        except Exception as e:
            self.logger.error(f"Something went wrong during run_sample: {e}", exc_info=True)
        #finally:
        #    if ct is not None:
        #        self.logger.debug(f"Runtime: {ct * 1e3:.2f} ms")
        #    else:
        #        self.logger.error("Runtime could not be measured due to error.")
        return output, ct  

    def _add_input(self, io_type, input_data=None, input_shape=None, name='input'):

        core_types = ()
        tile_shapes = ()

        if io_type=='buffer':
            # === Cyclic Buffer as Input ===
            input_real_group = nxk.NcGroup(nxk.kernels.CyclicBuffer(input_data.real.reshape((1,-1)), num_msg_bytes=2), name=name)
            self.model.add_group(input_real_group)
            
            core_types += ('neuron',)
            tile_shapes += self.tile_shapes_dict["input_buf"]
            self.log_added_group(self.logger, tile_shape=tile_shapes[-1])

        elif io_type=='cpu':
            # === CPU Input ===
            input_group = nxk.CpuInputGroup(shape=input_shape, out_spike_type='ws', name=name)
            self.model.add_group(input_group)

            core_types += ('cpu', )
            tile_shapes += self.tile_shapes_dict["input_cpu"]
            self.log_added_group(self.logger, tile_shape=tile_shapes[-1])

        elif io_type=='ethernet':
            # === Ethernet Input ===
            input_real_group = nxk.EthernetInputGroup(shape=input_shape, out_spike_type='ws', name=name)
            self.model.add_group(input_real_group)
            core_types += ('ethernet', )
            tile_shapes += self.tile_shapes_dict["input_eth"]
            self.log_added_group(self.logger, tile_shape=tile_shapes[-1])
            
        return tile_shapes, core_types

    def _add_output(self, io_type, src_group, output_shape=None, name='output'):

        core_types = ()
        tile_shapes = ()

        if io_type=='cpu':
            # === CPU Output ===
            output_real_group = nxk.CpuOutputGroup(shape=output_shape, name=name)

            self.model.add_group(output_real_group)
            nxk.connect(src=src_group, dst=output_real_group)

            core_types += ('cpu', )
            tile_shapes += self.tile_shapes_dict['output']
            #tile_shapes += (output_shape[0]//2, )
            self.log_added_group(self.logger, tile_shape=tile_shapes[-1])

        
        elif io_type=='ethernet':
            # === Add Ethernet Output ===
            output_real_group = nxk.EthernetOutputGroup(shape=(np.prod(output_shape),), name=name)

            self.model.add_group(output_real_group)
            nxk.connect(src=src_group, dst=output_real_group, src_port_idx=0)

            core_types += ('ethernet', )
            tile_shapes += (np.prod(output_shape), )
            self.log_added_group(self.logger, tile_shape=tile_shapes[-1])
            
            #output_imag_group = nxk.EthernetOutputGroup(shape=(np.prod(output_shape),), name='output_imag')

            #self.model.add_group(output_imag_group)
            #nxk.connect(src=src_group, dst=output_imag_group, src_port_idx=1)

            #core_types += ('ethernet', )
            #tile_shapes += self.tile_shapes_dict['output']
            #self.log_added_group(self.logger, tile_shape=tile_shapes[-1])

        elif io_type=='buffer':
            # === Cyclic Buffer as Output ===
            None
        
        return tile_shapes, core_types
    

    def build_model(self, io_type, input_data=None):
        
        self.model = nxk.Module()

        core_types = ()
        tile_shapes = ()

        for channel_idx in range(self.n_channels):
            # === Add Input ===
            if io_type=="no":
                if channel_idx==0:
                    tile_shape, core_type = self._add_input(io_type=io_type, input_data=input_data[channel_idx], 
                                                            input_shape=self.input_shape, name=f'input{channel_idx}')
                    core_types += core_type
                    tile_shapes += tile_shape
                input_group = getattr(self.model, "input0")
            elif io_type=='buffer':
                tile_shape, core_type = self._add_input(io_type=io_type, input_data=input_data[channel_idx], 
                                                        name=f'input{channel_idx}')
                core_types += core_type
                tile_shapes += tile_shape
                input_group = getattr(self.model, f"input{channel_idx}")

            elif io_type=="ethernet":
                tile_shape, core_type = self._add_input(io_type=io_type, input_shape=self.input_shape, name=f'input{channel_idx}')
                core_types += core_type
                tile_shapes += tile_shape
                input_group = getattr(self.model, f"input{channel_idx}")

            # === Add Core Model (FFT) ===
            tile_shape, core_type = self._add_spinr_model(input_group=input_group, name=f"spinr_ch{channel_idx}")
            core_types += core_type
            tile_shapes += tile_shape


        # === Add Output ===
        for channel_idx in range(self.n_channels):
            src_group = getattr(self.model, f"spinr_ch{channel_idx}", None)
            tile_shape, core_type = self._add_output(io_type=io_type, src_group=src_group, output_shape=(np.prod((self.n_ranges,)),), name=f'output_ch{channel_idx}')
            core_types += core_type
            tile_shapes += tile_shape
     
        # === Setup ===
        self.logger.info("Tile shapes: {}".format(tile_shapes))
        self.logger.info("Core types: {}".format(core_types))

        self.logger.info("Partition and setup ...")
        self.model.setup()
        self.model.partition(tile_shapes=tile_shapes)

        for g in self.model.groups:
            self.logger.info('{} on {} core(s)'.format(g.name, g.num_cores))

        self.addrs_list = self.get_addrs_list(core_types)
        self.logger.info("Address list: {}".format(self.addrs_list))
        self.init_board(addrs_list=self.addrs_list)

    def _add_spinr_model(self, input_group, name="spinr"):

        core_types = ()
        tile_shapes = ()

        # Add SpiNR 
        self.logger.debug("Adding SpiNR group: {} ...".format(name))
        spinr_group = self._create_spinr_group(name=name)
        self.model.add_group(spinr_group)
        core_types += ('neuron', )
        tile_shapes += self.tile_shapes_dict['spinr']

        nxk.connect(src=input_group, dst=spinr_group, da_idx=0, src_port_idx=0)

        self.log_added_group(self.logger, tile_shape=tile_shapes[-1])

        return tile_shapes, core_types
    
    def _create_spinr_group(self, name="spinr"):

        # Define synapses
        syn_args = dict(sparse_packing=True, use_shared_axon=True, 
                        optimize_weights=False, weight_exp=0, num_weight_bits=1)
        # Create synapse
        W = np.ones((self.n_ranges,1))
        synapses = [
                    K.Linear(weight=(W).astype('int32') , **syn_args, name='wgt_r2r'),
                    ] 
        self.logger.debug("Synapse: {}".format(synapses))

        # Define neuron
        omega_range = self._phasor_weights(self.n_samples, perc=1.0)[:self.n_ranges]
        neuron_args = dict(
            out_spike_type='ws', 
            ucode_path=self.spinr_ucode_path,
            ucode_args=self.spinr_ucode_args,
            re=0,
            im=0,
            turnoff=0,
            lct=(omega_range.real * 2**15).astype(np.int32),
            lst=(omega_range.imag * 2**15).astype(np.int32),
        )

        # Create Neuron  
        neuron = K.Neuron(shape=(self.n_ranges,), 
                            **neuron_args)

        # Create Neuron group and connect
        spinr_group = nxk.NcGroup(neuron=neuron,
                                synapses=synapses,
                                da_ports=[0],
                                name=name)
        
        return spinr_group 

    def _phasor_weights(self, n_bins, perc=1):

        return np.exp(1j*(np.linspace(0,perc-perc/n_bins,n_bins))*np.pi*2)  