# Public packages
import numpy as np
from collections import OrderedDict

# Intel packages
import nxkernel as nxk
import nxkernel.kernels as K
import nxcore.arch.n3b.n3board
from nxkernel.kernels.utils.weight_utils import SignMode


# Custom packages
import nrsp.utils.nx

class NxRASpiNRRangeModel():
    """
    Range-Angle-SpiNR Fixed-Range Model.
    

    Architecture:
    (n_channels) channels are multiplied with a weight matrix W 
    to (n_neurons_0) neurons with the same dynamic.
    Each of the (n_neurons_0) neurons produce outputs.

    -->             -->         (\omega_r)      --> output
            W       -->         (\omega_r)      --> output
                    -->         (\omega_r)      --> output
    -->             -->         (\omega_r)      --> output
   (n_channels)     (n_neurons_0)               (n_neurons_0) 

    Attributes:
        n_channels (int):
        n_neurons_0 (int):
        weights (np.array):
        omega_range (float):
        alpha_mag_smooth (float):
        log (logging.log):
        TODO:
        

    Methods:
        TODO:

    """
    def __init__(self, n_channels, n_neurons, range_idx, n_ranges,
                 alpha_mag, alpha_grad, u_threshold, u_rest,
                 log=None):
        """
        TODO:

        """
        self.log = log
        self.log.debug("Initialize NxRASpiNRRangeModel() ...")

        self.n_channels = n_channels
        self.n_neurons = n_neurons
        
        self.omega_range = nrsp.utils.nx.complex_to_float(np.repeat(self._phasor_weights(n_ranges, perc=0.5)[range_idx:range_idx+1], self.n_neurons))
        self.weights = nrsp.utils.nx.complex_to_float(self._steering_weights(in_dims=n_channels, out_dims=n_neurons))*127

        self.spinr_ucode_path = '/home/nreeb/code/nrsp/algs/nx/spinr/kernels/spinr_word_rspike.dasm'
        self.spinr_ucode_args = {
                                'alpha_mag': (alpha_mag*2**15),
                                'beta_mag': ((1 - alpha_mag)*2**15),
                                'alpha_grd': (alpha_grad*2**15),
                                'beta_grd': ((1 - alpha_grad)*2**15),
                                'grd_th': u_threshold,
                                'grd_rest': u_rest,
                                } 
        
        self.model = self._build_cpu_model()
        self._partition_model(tile_shapes=[self.n_channels, self.n_channels, self.n_neurons, self.n_neurons])
        
        self.log.debug("Initialized NxRASpiNRRangeModel().")

    def _build_spinr_neuron(self):

        # Define neuron
        neuron_args = dict(
            ucode_path=self.spinr_ucode_path,
            ucode_args=self.spinr_ucode_args,
            lct=(self.omega_range[:,0] * 2**15).astype(np.int32),
            lst=(self.omega_range[:,1] * 2**15).astype(np.int32),
        )
        
        init_cxstates = {'upp':100} 
        neuron = K.Neuron(shape=(self.n_neurons,), 
                            out_spike_type='z', 
                            **neuron_args,
                            **init_cxstates)
        
        return neuron
    
    def _build_spinr_synapses(self):
        """
        TODO:
        """

        # Define synapses
        syn_args = dict(sparse_packing=True, use_shared_axon=True, optimize_weights=False,
                        num_weight_bits=8, sign_mode=SignMode.MIXED)
        #syn_args = dict(sparse_packing=True, use_shared_axon=True)
        synapses = [K.Linear(weight=self.weights[...,0], **syn_args, name='real_wgt'), 
                    K.Linear(weight=self.weights[...,1], **syn_args, name='imag_wgt')]

        return synapses
    
    def _build_cpu_model(self):

        model = nxk.Module()
        # Real input group
        real_input_group = nxk.CpuInputGroup((self.n_channels,), name='real_input_group')
        model.add_group(real_input_group) 
        # Imag input group
        imag_input_group = nxk.CpuInputGroup((self.n_channels,), name='imag_input_group')
        model.add_group(imag_input_group) 
        spinr_group = nxk.NcGroup(neuron=self._build_spinr_neuron(),
                                synapses=self._build_spinr_synapses(),
                                in_ports=[model.real_input_group.id - model.next_id,
                                          model.imag_input_group.id - model.next_id],
                                da_ports=[0,1],
                                name='spinr')
        model.add_group(spinr_group)
        output_group = nxk.CpuOutputGroup((self.n_neurons,), 
                                        in_ports=[model.spinr.id - model.next_id],
                                        name='output_group')
        model.add_group(output_group)
        model.setup()

        return model
    
    def _partition_model(self, tile_shapes, type=None):

        self.model.partition(tile_shapes=tile_shapes)

        if type is None:
            self.addrs_list = [[nxk.make_addr(chip_idx=0, cpu_idx=0)],
                                [nxk.make_addr(chip_idx=0, cpu_idx=1)],
                                [nxk.make_addr(chip_idx=0, core_idx=0)],
                                [nxk.make_addr(chip_idx=0, cpu_idx=2)]] 

    def to_loihi(self, addrs_list=None):

        self.board = nxcore.arch.n3b.n3board.N3Board()

        if self.addrs_list is not None:
            self.model.to_nxcore(self.board, self.addrs_list)
        elif addrs_list is not None:
            self.model.to_nxcore(self.board, addrs_list)
        else:
            self.log.info("Provide a address list!")

    def save_model_image(self, filename):

        self.model.connectivity('LR').save(filename)

    def forward_cpu(self, input_data):
        
        n_timesteps = input_data.shape[-2]
        output = np.zeros((n_timesteps, self.n_neurons))
        try:
            self.board.run(n_timesteps, aSync=True)
            self.log.debug('Board runnning ...')
            self.model.start()
            self.log.debug('Started model.')
            for t in range(n_timesteps):
                self.model.real_input_group.send(input_data[:,t,0])
                self.model.imag_input_group.send(input_data[:,t,1])
                output[t] = self.model.output_group.recv()
        finally:
            self.model.stop()
            self.board.stop()

        return output
    
    def _phasor_weights(self, n_bins, perc=1):

        return np.exp(1j*np.linspace(0,perc-perc/n_bins,n_bins)*np.pi*2)    
    
    def _steering_weights(self, in_dims, out_dims):

        W = np.zeros((out_dims, in_dims)).astype('complex128')
        for o in range(out_dims):
            for i in range(in_dims):
                phi = 2*np.pi*i*(o-out_dims//2)/out_dims
                #if phi == 0:
                #    W[a,rx] = 0
                #else:
                W[o,i] = np.exp(1j*phi)

        return W