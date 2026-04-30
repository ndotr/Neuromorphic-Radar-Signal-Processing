# Public packages
import numpy as np
from collections import OrderedDict
import logging
import importlib.resources


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


class NxConvCACFARModel(base_nx_model.BaseNxModel):
    """

    Conv CA-CFAR Implementation

    """
    def __init__(self, 
                n_channels_0,
                n_channels_1,
                alpha_cfar,  # ca cfar
                offset_cfar,  # ca cfar
                guard_cells,
                ref_cells,
                ):
        """
        TODO:

        """
        logger.debug("Initialize NxConvCACFARModel() ...")
        self.cycle_buffer_length = None


        self.n_channels_0 = n_channels_0
        self.n_channels_1 = n_channels_1
        self.n_neurons = 72

        self.alpha_cfar = alpha_cfar
        self.offset_cfar = offset_cfar

        self.guard_cells = guard_cells
        self.ref_cells = ref_cells

        # Plain Matrix Multiplication
        self.cfar_ucode_path = importlib.resources.files('nrsp.algs.nx.kernels') / 'threshold.dasm'
        self.cfar_ucode_path = self.cfar_ucode_path.resolve()
        self.cfar_ucode_args = {'thresh':0}

        self.model = None
        self.mesh = None
        #self.viz = ch.visualize.WorkloadVisualizer(self.mesh)
        logger.debug("Initialized NxConvCACFARModel().")

    def build_cyclic_buffer_model(self, input_data):
        
        self.model = nxk.Module()
        self.mesh = nxk.system.mesh.OheoGulchMesh()

        # Readout number of timesteps
        self.cycle_buffer_length = input_data.shape[0]

        # Convert complex to real 2d
        # nxkernel assumes: (channels_0, channels_1, timesteps)
        logger.debug("Input shape: {}".format(input_data.shape))

        # Prepare variables for distribution on chip
        tile_shapes = ()

        ##################
        # Add input group
        logger.debug("Adding input_group {}...")
        input_group = nxk.NcGroup(nxk.kernels.CyclicBuffer(input_data.astype('int'), num_msg_bytes=2), name='input')
        self.model.add_group(input_group)

        # Distribute neuron on cores
        tile_shapes += ((self.n_channels_0//2, self.n_channels_1//2, 1), )
        logger.debug("Added input_group {}.".format(self.model.groups[-1].shape))

        logger.debug("Adding conv_group ...")

        conv_group = self._create_conv_cafar_group(in_group=-1)
        # Add neuron group to model
        self.model.add_group(conv_group)

        # Distribute neuron on cores
        tile_shapes += ((self.n_channels_0//8, self.n_channels_1//2, 1),)

        logger.debug("Added conv_group {}.".format(self.model.groups[-1].shape))
            

        logger.debug("Tile shapes: {}".format(tile_shapes)) 

        self.model.setup()
        self.model.partition(tile_shapes=tile_shapes)

        for g in self.model.groups:
            logger.debug('Num cores for {}: {}'.format(g.name, g.num_cores))

        addrs_list = self.get_simple_addrs_list()
        self.init_board(addrs_list=addrs_list)

    
    def _create_conv_cafar_group(self, in_group):

        # Define synapses
        syn_args = dict(optimize_weights=True,
                        sparse_packing=False)
                        
        # Create synapse
        kernel = (-1)*nrsp.utils.cfar.get_cfar_kernel_2d(guard_cells=self.guard_cells, ref_cells=self.ref_cells)
        kernel[kernel.shape[0]//2, kernel.shape[1]//2] = 8
        kernel = kernel.reshape(1, kernel.shape[0], kernel.shape[1], 1).astype('int32')

        shape = (self.n_channels_0, self.n_channels_1, 1)
        weight = kernel
        stride = (1,1)
        padding = (self.ref_cells[0] + self.guard_cells[0], self.ref_cells[1] + self.guard_cells[1])
        synapses = [K.Conv(weight=weight, 
                           output_shape=shape,
                           stride=stride,
                           padding=padding,
                            **syn_args, 
                            name='conv_wgt')] 

        # Define neuron
        neuron_args = dict(
            out_spike_type='ls', 
            ucode_path=self.cfar_ucode_path,
            ucode_args=self.cfar_ucode_args,
            state=0,
            abs_state=0,
        )

        # Create Neuron  
        neuron = K.Neuron(shape=shape, 
                            **neuron_args)

        # Create Neuron group and connect
        conv_group = nxk.NcGroup(neuron=neuron,
                                    synapses=synapses,
                                    in_ports=[self.model.groups[in_group].id - self.model.next_id],
                                    #da_ports=[0],
                                    name='conv')
        
        return conv_group
    

        
