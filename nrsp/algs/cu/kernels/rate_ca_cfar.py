import cupy as cp

rate_ca_cfar_kernel = cp.RawKernel(
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

__global__ void rate_ca_cfar_kernel(
    const bool *spikes,    // (frames, timesteps, n_x, n_y)
    const int n_timesteps,
    const bool *cfar_kernel,
    const int *cfar_kernel_shape,
    const bool wrap_x,
    const bool wrap_y,
    const float* alphas,
    const int n_alphas,
    const float *offsets,
    const int n_offsets,
    bool *out              // (frames, n_x, n_y, n_alphas, n_offsets)
) {
    const int x = threadIdx.x;
    const int y = blockIdx.x;
    const int frame_idx = blockIdx.y;

    const int n_x = blockDim.x;
    const int n_y = gridDim.x;

    if (x >= n_x || y >= n_y) return;

    const int neuron_idx = x * n_y + y;                                               // neuron at index (x, y)
    const int f_idx_o = (frame_idx * n_x * n_y + neuron_idx) * n_alphas * n_offsets;  // index offset for out
    const int f_idx_s = frame_idx * n_timesteps * n_x * n_y;                          // index offset for spikes

    // alignment anchors for convolution mask
    const int k_height = cfar_kernel_shape[0];
    const int k_width = cfar_kernel_shape[1];
    const int k_center_x = k_height / 2;
    const int k_center_y = k_width / 2;

    float s_cut = 0;
    float s_neigh_avg = 0;

    for (int t = 0; t < n_timesteps; ++t) {
        float sum_s_neigh = 0;
        float n_neigh = 0;

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
                    sum_s_neigh += spikes[f_idx_s + idx_offset + x_i * n_y + y_j];
                    n_neigh += 1;
                }
                kernel_idx++;
            }
        }

        s_neigh_avg += sum_s_neigh / n_neigh;
        s_cut += spikes[f_idx_s + idx_offset + neuron_idx]; // presynaptic cut spikes
    }

    // spiking condition
    // bulk process alphas and offsets
    for (int i = 0; i < n_alphas; ++i) {
        for (int j = 0; j < n_offsets; ++j) {
            out[f_idx_o + i * n_offsets + j] = s_cut - alphas[i] * s_neigh_avg > offsets[j];
        }
    }
}
}
""",
    "rate_ca_cfar_kernel",
)
