import cupy as cp

log_encoding_kernel = cp.RawKernel(
    """
//cuda
extern "C" __global__
void log_encoding_kernel(
    float *values,        // (n_frames, n_x, n_y) values to encode
    int n_timesteps,      
    float tau,
    float thresh,
    int *out,           // (n_frames, n_x, n_y)
    bool *spikes)       // (n_frames, n_timesteps, n_x, n_y)
{
    unsigned int x = threadIdx.x;
    unsigned int y = blockIdx.x;
    unsigned int frame = blockIdx.y;
    
    unsigned int n_x = blockDim.x;
    unsigned int n_y = gridDim.x;

    unsigned int neuron_idx = (x * n_y) + (y);
    unsigned int f_idx_o = frame * (n_x * n_y) + neuron_idx; // index offset for out
    unsigned int f_idx_s = frame * (n_timesteps * n_x * n_y); // index offset for spikes

    float value = values[f_idx_o];
    float tau_inv = 1 / tau;

    float u = 0;

    //////////////////////////////////////////////////////////////////////////////////
    // ---------------------------------------------------------------------------- //
    // Run Neuron Dynamics                                                          //
    // ---------------------------------------------------------------------------- //
    //////////////////////////////////////////////////////////////////////////////////

    for (int t = 0; t < n_timesteps; t++)
    {
        // Log-Time-Encodeded Spiking
        u += (value + u) * tau_inv;
        if (u >= thresh - value)
        {
            out[f_idx_o] = t;
            spikes[f_idx_s + t*(n_x*n_y) + neuron_idx] = true;
            return;
        }        
    }
    out[f_idx_o] = n_timesteps; // default value incase neuron does not spike
}
""",
    "log_encoding_kernel",
)
