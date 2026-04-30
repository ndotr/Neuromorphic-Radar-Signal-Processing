import cupy as cp

ra_spinr_debug_kernel = cp.RawKernel(
    """
//cuda
extern "C" __global__
void ra_spinr_debug_kernel(
    const float *x,            // [n_samples, n_channels, 2]
    const float *W,            
    const float *R,            
    const int n_samples,
    const int n_channels,
    const float *params,       
    float *grad_out,           // [n_samples, n_neurons] exponentially filtered gradient per timestep
    float *grad_raw_out,       // [n_samples, n_neurons] raw gradient estimate per timestep
    float *mag_out,            // [n_samples, n_neurons] state magnitude per timestep
    const int *dist_angle_bins // [n_neurons, 2] distance/angle bin indices per neuron
) {

    const int neuron_idx    = threadIdx.x;
    const int distance_idx  = dist_angle_bins[2 * neuron_idx];
    const int angle_idx     = dist_angle_bins[2 * neuron_idx + 1];

    const int n_neurons     = blockDim.x;

    const int w_idx         = angle_idx * n_channels * 2;
    const int R_idx         = 2 * distance_idx;

    // Parameters
    const float alpha_grd   = params[0];
    const int t_grd         = static_cast<int>(params[1]);

    // State and gradient variables
    float s[2] = {0, 0};
    float _s[2] = {0, 0};
    float gradient = 0;
    float width = 0;
    float upper = 0;
    float delta_width = 0;
    float delta_upper = 0;

    for (int t = 0; t < n_samples; ++t) {
        // State rotation: R * s
        _s[0] = R[R_idx + 0] * s[0] - R[R_idx + 1] * s[1];
        _s[1] = R[R_idx + 1] * s[0] + R[R_idx + 0] * s[1];
        s[0] = _s[0];
        s[1] = _s[1];

        // Weighted sum: w * x_t
        float inp[2] = {0, 0};
        for (int i = 0; i < n_channels; ++i) {
            const int idx_0 = 2 * (t * n_channels + i);
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

        // store debug outputs
        int out_idx             = t * n_neurons + neuron_idx;
        grad_out[out_idx]       = gradient;
        grad_raw_out[out_idx]   = delta_upper - delta_width;
        mag_out[out_idx]        = mag;
    }
}
""",
    name="ra_spinr_debug_kernel",
)
