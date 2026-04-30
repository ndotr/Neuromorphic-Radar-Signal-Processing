import cupy as cp

# Single-frame OS-CFAR kernel (NO frame dimension).
#
# Suggested launch:
#   grid  = (n_y, 1, 1)
#   block = (n_x, 1, 1)
#
# ABI:
#   spikes:            bool  (n_timesteps, n_x, n_y)   flattened t*(n_x*n_y) + x*n_y + y
#   cfar_kernel:       bool  (k_height, k_width)      flattened i*k_width + j
#   cfar_kernel_shape: int   (2,)   [k_height, k_width]
#   t_inhib:           int   (n_t_inhib,)
#   k:                 int   (n_k,)
#   t_cut_max:         int   CUT must spike at/before this timestep, else return false
#   out:               bool  (n_x, n_y, n_t_inhib, n_k) flattened ((x*n_y + y)*n_t_inhib + i)*n_k + j
#
log_os_cfar_kernel = cp.RawKernel(
"""
//cuda
extern "C" {

__device__ __forceinline__ int wrap_index(int i, int dim_size) {
    i %= dim_size;
    return (i < 0) ? i + dim_size : i;
}

__device__ __forceinline__ int reflect_index(int i, int dim_size) {
    const int period = 2 * (dim_size - 1);
    const int i_mod = abs(i) % period;
    return (i_mod < dim_size) ? i_mod : (period - i_mod);
}

__global__ void log_os_cfar_kernel(
    const bool * __restrict__ spikes,           // (n_timesteps, n_x, n_y)
    const int n_timesteps,
    const bool * __restrict__ cfar_kernel,      // (k_height, k_width)
    const int * __restrict__ cfar_kernel_shape, // (2,) = [k_height, k_width]
    const bool wrap_x,
    const bool wrap_y,
    const int * __restrict__ t_inhib,           // (n_t_inhib,)
    const int n_t_inhib,
    const int * __restrict__ k,                 // (n_k,)
    const int n_k,
    const int t_cut_max,                        // CUT must spike early enough
    bool * __restrict__ out                     // (n_x, n_y, n_t_inhib, n_k)
) {
    const int x = (int)threadIdx.x;  // 0..n_x-1
    const int y = (int)blockIdx.x;   // 0..n_y-1

    const int n_x = (int)blockDim.x;
    const int n_y = (int)gridDim.x;

    if (x >= n_x || y >= n_y) return;

    const int neuron_idx = x * n_y + y;

    // Output base for this neuron: out[(x,y), :, :]
    const int out_base = neuron_idx * n_t_inhib * n_k;

    // Convolution mask anchors
    const int k_height   = cfar_kernel_shape[0];
    const int k_width    = cfar_kernel_shape[1];
    const int k_center_x = k_height / 2;
    const int k_center_y = k_width  / 2;

    int  u = 0;                    // - #(neighbor spikes accumulated over time)
    int  t_cut = n_timesteps - 1;  // last CUT spike time (default end)
    bool spike_flag = false;       // whether CUT spiked at least once

    for (int t = 0; t < n_timesteps; ++t) {
        const int t_base = t * (n_x * n_y);

        // Accumulate neighbor spikes per time step
        int kernel_idx = 0;
        for (int i = 0; i < k_height; ++i) {
            for (int j = 0; j < k_width; ++j, ++kernel_idx) {

                int x_i = x - k_center_x + i;
                int y_j = y - k_center_y + j;

                x_i = wrap_x ? wrap_index(x_i, n_x) : reflect_index(x_i, n_x);
                y_j = wrap_y ? wrap_index(y_j, n_y) : reflect_index(y_j, n_y);

                if (cfar_kernel[kernel_idx]) {
                    u -= (int)spikes[t_base + x_i * n_y + y_j];
                }
            }
        }

        // CUT spike?
        if (spikes[t_base + neuron_idx]) {
            t_cut = t;
            spike_flag = true;
        }

        // Evaluate inhibition offsets
        for (int i = 0; i < n_t_inhib; ++i) {
            const int t_fire = min(n_timesteps - 1, t_cut + t_inhib[i]);
            if (t == t_fire) {
                const int out_i_base = out_base + i * n_k;

                // Gate: CUT must have spiked, and must have spiked early enough
                const bool cut_valid = (t_cut < t_cut_max);

                for (int j = 0; j < n_k; ++j) {
                    out[out_i_base + j] = cut_valid && ((u + k[j] * (int)spike_flag) > 0);
                }
            }
        }
    }
}

} // extern "C"
""",
    name="log_os_cfar_kernel",
)