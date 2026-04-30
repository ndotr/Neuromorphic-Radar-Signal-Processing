import cupy as cp

log_os_cfar_kernel = cp.RawKernel(
    """
//cuda
extern "C" {

__device__ int wrap_index(int i, int dim_size) {
    return ((i %= dim_size) < 0) ? i+dim_size : i;
}

__device__ int reflect_index(int i, int dim_size) {
    int period = 2 * (dim_size - 1);
    int i_mod = abs(i) % period;
    return (i_mod < dim_size) ? i_mod : (period - i_mod);
}

__global__ void log_os_cfar_kernel(
    const bool *spikes,           // (n_frames, n_timesteps, n_x, n_y)
    const int n_timesteps, 
    const bool *cfar_kernel, 
    const int *cfar_kernel_shape,
    const bool wrap_x, 
    const bool wrap_y, 
    const int *t_inhib,
    const int n_t_inhib,          // len of t_inhib
    const int *k,
    const int n_k,                // len of k
    bool *out                     // (n_frames, n_x, n_y, n_t_inhib, n_k)
) {
    const int x = threadIdx.x;
    const int y = blockIdx.x;
    const int frame_idx = blockIdx.y;

    const int n_x = blockDim.x;
    const int n_y = gridDim.x;

    if (x >= n_x || y >= n_y) return;

    const int neuron_idx = x * n_y + y;                                            // neuron at index (x, y)
    const int f_idx_o = (frame_idx * (n_x * n_y) + neuron_idx) * n_t_inhib * n_k;  // index offset for out
    const int f_idx_s = frame_idx * (n_timesteps * n_x * n_y);                     // index offset for spikes

    // alignment anchors for convolution mask
    const int k_height = cfar_kernel_shape[0];
    const int k_width = cfar_kernel_shape[1];
    const int k_center_x = k_height / 2;
    const int k_center_y = k_width / 2;

    int u = 0;  // -1* #(neighbor spikes)
    int t_cut = n_timesteps - 1;  // TODO explaination
    bool spike_flag = false; // flag to indicate if the CUT neuron has spiked

    for (int t = 0; t < n_timesteps; ++t) {
        int idx_offset = t * n_x * n_y;
        int kernel_idx = 0;

        for (int i = 0; i < k_height; ++i) {
            for (int j = 0; j < k_width; ++j) {

                int x_i = x - k_center_x + i;
                int y_j = y - k_center_y + j;

                x_i = wrap_x ? wrap_index(x_i, n_x) : reflect_index(x_i, n_x);
                y_j = wrap_y ? wrap_index(y_j, n_y) : reflect_index(y_j, n_y);

                if (cfar_kernel[kernel_idx]) {
                    // spike at index (t, x_i, y_j)
                    u -= spikes[f_idx_s + idx_offset + x_i * n_y + y_j];
                }
                kernel_idx++;
            }
        }

        // check if presynaptic cut spikes (index (t, x, y))
        if (spikes[f_idx_s + idx_offset + neuron_idx]) {
          t_cut = t;
          spike_flag = true;
        }

        // bulk process t_inhib
        for (int i = 0; i < n_t_inhib; ++i) {
            // spiking condition
            if (t == min(n_timesteps - 1, t_cut + t_inhib[i])) {
                // bulk process k
                for (int j = 0; j < n_k; ++j) {
                    out[f_idx_o + (i * n_k) + j] = u + k[j] * spike_flag > 0;
                }
            }
        }
  }
}
}
""",
    "log_os_cfar_kernel",
)
