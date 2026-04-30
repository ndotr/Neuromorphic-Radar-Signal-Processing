import numpy as np
import nrsp.utils.log

def create_dft_weights(n_inputs, n_outputs):

    n = np.linspace(0, n_inputs-1, n_inputs)
    k = np.linspace(0, n_outputs-1, n_outputs)

    ret = np.exp(1j*2*np.pi*np.einsum('n,k->nk', n, k)/n_inputs)
    
    return ret