import cupy as cp

rd_spinr_turnoff_v2_nx_kernel = cp.RawKernel(
    """
//cuda

__device__ int clip_ls(int value) {
    // Clip value to fit in the signed 24-bit integer range (-8388608 to 8388607)
    return min(max(value, -8388608), 8388607);
}

__device__ int clip_lu(int value) {
    // Clip value to fit in the unsigned 24-bit integer range (0 to 16777215)
    return min(max(value, 0), 16777215);
}


extern "C" __global__
void rd_spinr_turnoff_v2_nx_kernel(
    const int *x,              // [n_frames, n_chirps, n_samples, 2]
    const float *R,            // [n_distances, 2] (range phasor weights)
    const float *D,            // [n_velocities, 2] (doppler phasor weights)
    const int n_samples,
    const int n_chirps,
    const float *params,       //
    int *grad,                 // [n_frames, n_distances, n_velocities]
    int *out,                  // [n_frames, n_distances, n_velocities]
    bool *spikes,              // [n_frames, n_t, n_distances, n_velocities] with n_t = n_chirps*n_samples - t_enc 
    bool *inactive_neurons,    // [n_frames, n_distances, n_velocities]
    int *inactive_count,       // [n_frames, n_chirps*n_samples] // count of neurons that turned off at each timestep
    const int grd_shl
) {
    const int distance_idx = threadIdx.x;
    const int doppler_idx  = blockIdx.x;
    const int frame_idx    = blockIdx.y;

    const int n_distances  = blockDim.x;
    const int n_velocities     = gridDim.x;

    // Parameters
    const long long int alpha_grd             = params[0];
    const long long int beta_grd              = 32768 - alpha_grd; // 2^15 - alpha_grd
    const int t_grd                 = static_cast<int>(params[1]);
    const int spike_func            = static_cast<int>(params[2]);
    // const float tau_rate            = params[3];
    // const int thresh_rate           = static_cast<int>(params[4]);
    // const int rest_rate             = static_cast<int>(params[5]);
    const float tau_log             = params[6];
    const int thresh_log            = static_cast<int>(params[7]);
    const int t_enc                 = static_cast<int>(params[8]);
    const int thresh_silent         = static_cast<int>(params[9]); // -1 means no silencing
    const int thresh_silent_chirp  = static_cast<int>(params[10]);
    const int monotonicity_thresh   = static_cast<int>(params[11]);
    const int t_monotonicity = static_cast<int>(params[12]);

    // Flat indices
    const int neuron_idx   = distance_idx * n_velocities + doppler_idx;
    const int f_idx_out    = frame_idx * n_distances * n_velocities + neuron_idx;   // index for out/grad
    const int n_t = n_chirps * n_samples - t_enc; 
    const int f_idx_s      = frame_idx * n_t * n_distances * n_velocities;    // index offset for spikes

    // State and gradient variables
    int s[2] = {0, 0};
    int gradient = 0;

    float monotonicity = 0; // float is fine, otherwise just scale up so match leak term (monotonicity - 0.02)
    int upper = 0;
    int lower = 0;
    int delta = 0;

    // turnoff
    bool turnoff = false;
    int silent = 0;
    int silent_chirp = 0;

    // Spiking state
    int u_log = 0;

    // initialize log encoded spike time incase neuron does not spike
    if (spike_func == 0) {
        out[f_idx_out] = n_chirps*n_samples-t_enc;
    }

    int n_timesteps = n_chirps * n_samples;
    
    for (int chirp = 0; chirp < n_chirps; ++chirp) {
        // Chirp rotation
        int re_0 = clip_ls(static_cast<int>(D[2 * doppler_idx + 0] * s[0]));
        int re_1 = clip_ls(static_cast<int>(D[2 * doppler_idx + 1] * s[1]));
        int im_0 = clip_ls(static_cast<int>(D[2 * doppler_idx + 1] * s[0]));
        int im_1 = clip_ls(static_cast<int>(D[2 * doppler_idx + 0] * s[1]));
        s[0] = clip_ls(re_0 - re_1);
        s[1] = clip_ls(im_0 + im_1);

        for (int t = 0; t < n_samples; ++t) {
            int sample_idx = chirp * n_samples + t;
            if (turnoff) {
                inactive_neurons[f_idx_out] = true;
                atomicAdd(&inactive_count[frame_idx * n_timesteps + sample_idx], 1);
                return;
            }        

            // Sample rotation
            int re_0 = clip_ls(static_cast<int>(R[2 * distance_idx + 0] * s[0]));
            int re_1 = clip_ls(static_cast<int>(R[2 * distance_idx + 1] * s[1]));
            int im_0 = clip_ls(static_cast<int>(R[2 * distance_idx + 1] * s[0]));
            int im_1 = clip_ls(static_cast<int>(R[2 * distance_idx + 0] * s[1]));
            s[0] = clip_ls(re_0 - re_1);
            s[1] = clip_ls(im_0 + im_1);

            // Add input to state
            int x_idx = frame_idx * n_chirps * n_samples * 2 + chirp * n_samples * 2 + t * 2;
            s[0] = clip_ls(s[0] + x[x_idx + 0]);
            s[1] = clip_ls(s[1] + x[x_idx + 1]);

            // Magnitude and delta updates
            // const int mag =  clip_lu(static_cast<int>(sqrtf(s[0] * s[0] + s[1] * s[1])));
            const int mag = clip_lu(abs(s[0]) + abs(s[1]));

            delta = 0;
            if (mag > upper) {
                delta = (mag - upper);
                upper = mag;
                lower += delta;
                silent = 0;
                monotonicity += 1;
            } else {
                if (lower > mag) {
                    delta = mag - lower;
                    lower = mag;
                    monotonicity -= 1;
                }
                monotonicity = monotonicity - 0.02;
                silent++;
            }

            // turnoff logic
            if (sample_idx < t_monotonicity){
                monotonicity = 0;
            }
            // Decline
            if ((monotonicity < monotonicity_thresh) && (sample_idx > t_monotonicity)){
                turnoff=true;
            }
            
            if (thresh_silent != -1 && silent >= thresh_silent) {
                silent = 0;
                silent_chirp += 1;
            }
            if (silent_chirp == thresh_silent_chirp) {
                turnoff = true; // neuron will return in next timestep
            }

            // Exponential filtering of gradient estimations with starting point t_grd
            if (sample_idx > t_grd) {
                int grad_decay = (beta_grd * gradient) >> 15;
                int grad_incr = (alpha_grd * delta) >> (15-grd_shl);
                gradient = clip_ls(grad_decay + grad_incr);
            }
            
            // int spike_idx = f_idx_s + sample_idx * n_distances * n_velocities + neuron_idx;
            int spike_idx = f_idx_s + (sample_idx - t_enc) * n_distances * n_velocities + neuron_idx;

            grad[f_idx_out] = gradient; // store gradient in each timestep as neurons might return early

            // Log-time spiking
            if (spike_func == 0) {
                if (sample_idx >= t_enc) {
                    u_log += static_cast<int>((gradient + u_log) / tau_log);
                    if ((u_log >= (thresh_log - gradient)) ) {
                        turnoff = true;
                        out[f_idx_out] = sample_idx - t_enc;
                        spikes[spike_idx] = true;
                    }
                }
            } 
        }
    }
}
""",
    name="rd_spinr_turnoff_v2_nx_kernel",
)
