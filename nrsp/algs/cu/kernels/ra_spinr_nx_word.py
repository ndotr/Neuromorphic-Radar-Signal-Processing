import cupy as cp

ra_spinr_nx_kernel = cp.RawKernel(
    """
//cuda

__device__ int clip_ws(int value) {
    // Clip value to fit in the signed 16-bit integer range (-32768 to 32767)
    return min(max(value, -32768), 32767);
}

__device__ int clip_wu(int value) {
    // Clip value to fit in the unsigned 16-bit integer range (0 to 65535)
    return min(max(value, 0), 65535);
}

__device__ int clip_ls(int value) {
    // Clip value to fit in the signed 24-bit integer range (-8388608 to 8388607)
    return min(max(value, -8388608), 8388607);
}

__device__ int clip_lu(int value) {
    // Clip value to fit in the unsigned 24-bit integer range (0 to 16777215)
    return min(max(value, 0), 16777215);
}


extern "C" __global__
void ra_spinr_nx_kernel(
    const int *x,              // [n_frames, n_samples, n_channels, 2]
    const int *W,              //
    const float *R,            //
    const int n_samples,
    const int n_channels,
    const float *params,       //
    int *grad,                 // [n_frames, n_distances, n_angles]
    int *out,                  // [n_frames, n_distances, n_angles]
    bool *spikes,              // [n_frames, n_samples, n_distances, n_angles]
    const int *distance_bins,
    const int *angle_bins,
    const int weight_shr,
    const int grd_shl
) {
    const int distance_idx = distance_bins[threadIdx.x];
    const int angle_idx    = angle_bins[blockIdx.x];
    const int frame_idx    = blockIdx.y;

    const int n_distances  = blockDim.x;
    const int n_angles     = gridDim.x;

    // Flat indices
    const int neuron_idx   = distance_idx * n_angles + angle_idx;
    const int w_idx        = angle_idx * n_channels * 2;
    const int R_idx        = 2 * distance_idx;

    const int f_idx_x      = frame_idx * n_samples * n_channels * 2;            // index offset for x
    const int f_idx_out    = frame_idx * n_distances * n_angles + neuron_idx;   // index offset for out/grad
    const int f_idx_s      = frame_idx * n_samples * n_distances * n_angles;    // index offset for spikes

    // Parameters
    const float alpha_grd           = params[0];
    const float beta_grd            = 1 - alpha_grd;
    const int t_grd                 = static_cast<int>(params[1]);
    const int spike_func            = static_cast<int>(params[2]);
    const float tau_rate            = params[3];
    const int thresh_rate           = static_cast<int>(params[4]);
    const int rest_rate             = static_cast<int>(params[5]);
    const float tau_log             = params[6];
    const int thresh_log            = static_cast<int>(params[7]);
    const int t_enc                 = static_cast<int>(params[8]);

    // State and gradient variables
    int s[2] = {0, 0};
    int gradient = 0;
    int upper = 0;
    int width = 0;
    int delta_width = 0;
    int delta_upper = 0;

    // Spiking state
    int u_log = 0;
    int u_rate = 0;
    bool spike_block_log = false;

    out[f_idx_out] = n_samples + 1; // default value

    for (int t = 0; t < n_samples; ++t) {
        // State rotation: R * s
        int re_0 = clip_ws(static_cast<int>(R[R_idx + 0] * s[0]));
        int re_1 = clip_ws(static_cast<int>(R[R_idx + 1] * s[1]));
        int im_0 = clip_ws(static_cast<int>(R[R_idx + 1] * s[0]));
        int im_1 = clip_ws(static_cast<int>(R[R_idx + 0] * s[1]));

        s[0] = clip_ws(re_0 - re_1); // not accurate since in ucode "-" could overflow
        s[1] = clip_ws(im_0 + im_1);

        // TODO not sure how synaptic weight mul is handled, what dtype etc
        // Weighted sum: w * x_t
        int inp[2] = {0, 0};
        for (int i = 0; i < n_channels; ++i) {
            const int idx_0 = f_idx_x + 2 * (t * n_channels + i);
            const int idx_1 = idx_0 + 1;
            inp[0] += W[w_idx + 2 * i + 0] * x[idx_0] - W[w_idx + 2 * i + 1] * x[idx_1];
            inp[1] += W[w_idx + 2 * i + 1] * x[idx_0] + W[w_idx + 2 * i + 0] * x[idx_1];
        }
        inp[0] >>= weight_shr;
        inp[1] >>= weight_shr;

        // Add to state
        s[0] = clip_ws(s[0] + inp[0]);
        s[1] = clip_ws(s[1] + inp[1]);


        // Magnitude and delta updates
        const int mag = clip_wu(abs(s[0]) + abs(s[1]));
        delta_upper = (mag - upper) * (mag > upper);
        delta_width = (upper - mag - width) * ((upper - mag) > width);

        // Exponential filtering of gradient estimations with starting point t_grd
        if (t > t_grd) {
            float alpha_grd_ = alpha_grd * powf(2.0f, static_cast<float>(grd_shl));
            gradient = static_cast<int>(beta_grd * gradient) + 
                       clip_ws(static_cast<int>(alpha_grd_ * (delta_upper - delta_width)));
            gradient = clip_lu(gradient);
        }

        upper = clip_wu(upper + delta_upper);
        width = clip_wu(width + delta_width);

        // Log-time spiking
        if (spike_func == 0) {
            if (t > t_enc) {
                u_log += static_cast<int>((gradient + u_log) / tau_log);
            }

            if ((u_log >= (thresh_log - gradient)) && !spike_block_log) {
                spike_block_log = true;
                out[f_idx_out] = t;
                spikes[f_idx_s + t * n_distances * n_angles + neuron_idx] = true;
            }
        } 
        
        // Rate-based spiking
        if (spike_func == 1) {
            if (t > t_enc) {
                u_rate += static_cast<int>((gradient + rest_rate - u_rate) / tau_rate);
            }

            if (u_rate >= thresh_rate) {
                u_rate -= thresh_rate;
                out[f_idx_out] += 1;
                spikes[f_idx_s + t * n_distances * n_angles + neuron_idx] = true;
            }
        }
    }

    grad[f_idx_out] = gradient;
}
""",
    name="ra_spinr_nx_kernel",
)
