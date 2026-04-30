import cupy as cp
import numpy as np
import nrsp.utils.cu

class CuSpiNRModel():
    """
    Range-Angle-Doppler-SpiNR Model.

    Architecture:
    (n_channels) channels are weighted to (n_neurons_0) neurons.
    These outputs are passed to (n_neurons_1 x n_neurons_2) neurons with different dynamics.

    Single neuron architecture:
    -->            
            w_\theta    --> (\omega_range, \omega_velocity)     --> output
                             
    -->                      
   (n_channels)         (1)                                     (1) 

   Angle neuron group architecture for one range and one velocity:
    -->             --> (\omega_range, \omega_velocity)     --> output       
            W       --> (\omega_range, \omega_velocity)     --> output
                    --> (\omega_range, \omega_velocity)     --> output        
    -->             --> (\omega_range, \omega_velocity)     --> output                
   (n_channels)     (n_neurons_0)                           (n_neurons_0) 

    Optimized weight matrix multiplication architecture:
    -->                     
            w_\theta_0  --> range-doppler group 2D (range, velocity)    --> output
                            
    -->
    (n_channels)        (n_neurons_1 x n_neurons_2)                     (n_neurons_1 x n_neurons_2)
    .
    .
    .
    -->                     
            w_\theta_N  --> range-doppler group 2D (range, velocity)    --> output
                            
    -->
    (n_channels)        (n_neurons_1 x n_neurons_2)                     (n_neurons_1 x n_neurons_2)
    
    (n_channels)        (n_neurons_0 x n_neurons_1 x n_neurons_2)       (n_neurons_0 x n_neurons_1 x n_neurons_2)


    Attributes:
        

    Methods:

    """

    def __init__(self, n_channels, n_angles, n_ranges, n_velocities,
                 n_samples, n_chirps,
                 spinr_dict,
                 log=None):

        self.log = log
        self.log.debug("Constructing CuSpiNRModel() ...")
        self.spinr_kernel = self._init_kernel()

        self.n_samples      = n_samples
        self.n_chirps       = n_chirps
        self.n_timesteps    = n_samples * n_chirps

        self.n_channels     = n_channels 
        self.n_ranges       = n_ranges
        self.n_angles       = n_angles
        self.n_velocities   = n_velocities

        # Spinr group
        self.weights    = nrsp.utils.cu.complex_to_float(self._steering_weights(n_in_dims=n_channels, n_out_dims=n_angles))
        self.sample_R   = nrsp.utils.cu.complex_to_float(self._phasor_weights(self.n_ranges, perc=0.5))
        self.chirp_R    = nrsp.utils.cu.complex_to_float(self._phasor_weights(self.n_velocities))
                       
        # Global spinr neuron params
        self.spinr_params = self._init_params(spinr_dict=spinr_dict)

        # States
        self.states = None
        self.spikes = None
        self.reset()

        self.log.debug("Constructed CuSpiNRModel().")

    def _init_params(self, spinr_dict):

        spinr_params = []
        spinr_keys = ['alpha_l', 'alpha_mag', 'alpha_grad', 't_start', 'alpha_u', 'u_threshold', 'u_rest']

        for key in spinr_keys:
            spinr_params.append(spinr_dict[key])
        
        spinr_params     = cp.array(spinr_params, dtype='float32')

        return spinr_params

    def reset(self, init_states=None):
        self.spikes = cp.zeros((self.n_velocities, self.n_ranges, self.n_angles, self.n_timesteps, 1), dtype='bool')
        self.states = cp.zeros((self.n_velocities, self.n_ranges, self.n_angles, 2)).astype('float32')
        if init_states is None:
            self.states = cp.zeros((self.n_velocities, self.n_ranges, self.n_angles, 2), dtype='float32')
        else:
            self.states = cp.array(init_states)

    def forward(self, input_data, debug):

        self.debug = cp.array([debug])

        self.log.debug("Run SpiNR kernel on GPU ...")
        self.spinr_kernel((int(self.n_ranges), int(self.n_velocities), 1),  # grid shape
                          (int(self.n_angles), 1, 1),                       # block shape
                    (input_data, 
                     self.weights,
                     self.sample_R, 
                     self.chirp_R, 
                     self.states, 
                     self.spikes, 
                     cp.array([self.n_samples]), 
                     cp.array([self.n_chirps]), 
                     cp.array([self.n_channels]),
                     self.spinr_params,
                     self.debug)
                    )
        self.log.debug("Finished running SpiNR kernel on GPU.")

        return self.spikes

    def __call__(self, input_data, debug=False):

        output_gpu = self.forward(input_data, debug=debug)

        return output_gpu

    def _init_kernel(self):

        self.log.debug("Initialize spinr kernel ...")
        import nrsp.algs.cu.spinr.kernels.float_spinr as spinr
        self.log.debug("Initialized spinr kernel.")

        return spinr.kernel

    def _phasor_weights(self, n_bins, perc=1):
        return np.exp(1j*np.linspace(0,perc-perc/n_bins,n_bins)*np.pi*2)
    
    def _steering_weights(self, n_in_dims, n_out_dims):

        W = np.zeros((n_out_dims, n_in_dims)).astype('complex128')
        for o in range(n_out_dims):
            for i in range(n_in_dims):
                phi = 2*np.pi*i*(o-n_out_dims//2)/n_out_dims
                #if phi == 0:
                #    W[a,rx] = 0
                #else:
                W[o,i] = np.exp(1j*phi)

        return W