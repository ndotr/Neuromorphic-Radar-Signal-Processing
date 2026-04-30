import cupy as cp

ra_spinr_kernel = cp.RawKernel(
    """
//cuda
extern "C" __global__
void ra_spinr_kernel(
    const float *x,            // [n_frames, n_samples, n_channels, 2]
    const float *W,            //
    const float *R,            //
    const int n_samples,
    const int n_channels,
    const float *params,       //
    float *grad,               // [n_frames, n_distances, n_angles]
    int *out,                  // [n_frames, n_distances, n_angles]
    bool *spikes,              // [n_frames, n_samples, n_distances, n_angles]
    const int *distance_bins,
    const int *angle_bins
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
    const int t_grd                 = static_cast<int>(params[1]);
    const int spike_func            = static_cast<int>(params[2]);
    const float tau_rate            = params[3];
    const float thresh_rate         = params[4];
    const float rest_rate           = params[5];
    const float tau_log             = params[6];
    const float thresh_log          = params[7];
    const int t_enc                 = static_cast<int>(params[8]);

    // State and gradient variables
    float s[2] = {0, 0};
    float _s[2] = {0, 0};
    float gradient = 0;
    float width = 0;
    float upper = 0;
    float delta_width = 0;
    float delta_upper = 0;

    // Spiking state
    float u_log = 0;
    float u_rate = 0;
    bool spike_block_log = false;

    // initialize log encoded spike time to n_samples incase neuron does not spike
    if (spike_func == 0) {
        out[f_idx_out] = n_samples;
    } 

    for (int t = 0; t < n_samples; ++t) {
        // State rotation: R * s
        _s[0] = R[R_idx + 0] * s[0] - R[R_idx + 1] * s[1];
        _s[1] = R[R_idx + 1] * s[0] + R[R_idx + 0] * s[1];
        s[0] = _s[0];
        s[1] = _s[1];

        // Weighted sum: w * x_t
        float inp[2] = {0, 0};
        for (int i = 0; i < n_channels; ++i) {
            const int idx_0 = f_idx_x + 2 * (t * n_channels + i);
            const int idx_1 = idx_0 + 1;
            inp[0] += W[w_idx + 2 * i + 0] * x[idx_0] - W[w_idx + 2 * i + 1] * x[idx_1];
            inp[1] += W[w_idx + 2 * i + 1] * x[idx_0] + W[w_idx + 2 * i + 0] * x[idx_1];
        }

        // Add to state
        s[0] += inp[0];
        s[1] += inp[1];

        // Magnitude and delta updates
        const float mag = sqrtf(s[0] * s[0] + s[1] * s[1]);
        delta_upper = (mag - upper) * (upper < mag);
        delta_width = (upper - mag - width) * ((upper - mag) > width);

        // Exponential filtering of gradient estimations with starting point t_grd
        if (t > t_grd) {
            gradient = (1 - alpha_grd) * gradient + alpha_grd * (delta_upper - delta_width);
            gradient = fmaxf(gradient, 0); // enforce non-negativity
        }

        upper += delta_upper;
        width += delta_width;

        // Log-time spiking
        if (spike_func == 0) {
            if (t > t_enc) {
                u_log += (gradient + u_log) / tau_log;
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
                u_rate += (gradient + rest_rate - u_rate) / tau_rate;
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
    name="ra_spinr_kernel",
)
