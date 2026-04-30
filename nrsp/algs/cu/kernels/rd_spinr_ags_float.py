import cupy as cp

# Single-frame kernel: NO frame dimension anywhere.
# Launch:
#   grid  = (n_velocities, 1, 1)
#   block = (n_distances, 1, 1)
#
# ABI:
#   x:              float32, shape (n_chirps, n_samples, 2)   (Re, Im)
#   R:              float32, shape (n_distances, 2)
#   D:              float32, shape (n_velocities, 2)
#   grad:           float32, shape (n_distances, n_velocities)
#   out:            int32,   shape (n_distances, n_velocities)   (spike time encoded here)
#   spikes:         OPTIONAL bool, shape (n_t, n_distances, n_velocities), n_t = n_chirps*n_samples - t_enc
#   inactive_neurons: bool,  shape (n_distances, n_velocities)
#   inactive_count:   int32, shape (n_chirps*n_samples)
#   record_spikes:    int32  (0/1)
#
rd_spinr_ags_float_kernel = cp.RawKernel(
"""
//cuda
extern "C" __global__
void rd_spinr_ags_float_kernel(
    const float * __restrict__ x,            // [n_chirps, n_samples, 2]
    const float * __restrict__ R,            // [n_distances, 2]
    const float * __restrict__ D,            // [n_velocities, 2]
    const int n_samples,
    const int n_chirps,
    const float * __restrict__ params,       // float32 len>=13 (see below)
    float * __restrict__ grad,               // [n_distances, n_velocities]
    int   * __restrict__ out,                // [n_distances, n_velocities]
    bool  * spikes,                          // OPTIONAL: [n_t, n_distances, n_velocities]
    bool  * __restrict__ inactive_neurons,   // [n_distances, n_velocities]
    int   * __restrict__ inactive_count,     // [n_chirps*n_samples]
    const int record_spikes                  // 0/1
) {
    // Mapping:
    //   threadIdx.x  -> distance bin (range)
    //   blockIdx.x   -> doppler bin (velocity)
    const int distance_idx = (int)threadIdx.x;
    const int doppler_idx  = (int)blockIdx.x;

    const int n_distances  = (int)blockDim.x;
    const int n_velocities = (int)gridDim.x;

    // ---- Params ABI (aligned with your packing) ----
    const float alpha_grd = params[0];
    const float beta_grd  = 1.0f - alpha_grd;
    const int   t_grd     = (int)params[1];
    const int   spike_func= (int)params[2];

    const float tau_log   = params[6];
    const float thresh_log= params[7];
    const int   t_enc     = (int)params[8];

    const int   thresh_silent       = (int)params[9];   // -1 disables
    const int   thresh_silent_chirp = (int)params[10];

    const float monotonicity_thresh = params[11];
    const int   t_monotonicity      = (int)params[12];

    // ---- Flat indices (NO frame stride) ----
    const int neuron_idx = distance_idx * n_velocities + doppler_idx;
    const int out_idx    = neuron_idx;

    const int n_timesteps = n_chirps * n_samples;
    const int n_t = n_timesteps - t_enc;

    // Optional spike recording
    const int do_record_spikes = (record_spikes != 0) && (spikes != NULL);

    // ---- Cache phasors ----
    const float R_re = R[2 * distance_idx + 0];
    const float R_im = R[2 * distance_idx + 1];
    const float D_re = D[2 * doppler_idx  + 0];
    const float D_im = D[2 * doppler_idx  + 1];

    // ---- State ----
    float s_re = 0.0f;
    float s_im = 0.0f;

    float gradient = 0.0f;
    float mag = 0.0f;

    float monotonicity = 0.0f;
    float upper = 0.0f;
    float lower = 0.0f;
    float delta = 0.0f;

    bool turnoff = false;
    int silent = 0;
    int silent_chirp = 0;

    float u_log = 0.0f;

    // Default: encode "no spike" as end-of-window time (for spike_func==0)
    if (spike_func == 0) {
        out[out_idx] = n_timesteps - t_enc;
    }

    // Main loops
    for (int chirp = 0; chirp < n_chirps; ++chirp) {

        // Doppler rotation: s <- D * s
        {
            const float re = D_re * s_re - D_im * s_im;
            const float im = D_im * s_re + D_re * s_im;
            s_re = re;
            s_im = im;
        }

        const int chirp_x_base = chirp * (n_samples * 2);

        for (int t = 0; t < n_samples; ++t) {
            const int sample_idx = chirp * n_samples + t;

            //  Turnoff finalized on NEXT timestep (same semantics as your old kernel)
            if (turnoff) {
                inactive_neurons[out_idx] = true;
                grad[out_idx] = gradient;
                out[out_idx] = 1-int(turnoff);
                atomicAdd(&inactive_count[sample_idx], 1);
                return;
            }

            // Range rotation: s <- R * s
            {
                const float re = R_re * s_re - R_im * s_im;
                const float im = R_im * s_re + R_re * s_im;
                s_re = re;
                s_im = im;
            }

            // Add input sample x[chirp, t, :]
            const int x_idx = chirp_x_base + t * 2;
            s_re += x[x_idx + 0];
            s_im += x[x_idx + 1];

            // Magnitude
            mag = sqrtf(s_re * s_re + s_im * s_im);

            // Envelope + monotonicity
            delta = 0.0f;

            if (mag > upper) {
                delta = (mag - upper);
                upper = mag;
                lower += delta;

                silent = 0;
                monotonicity += 2.0f;
            } else {
                if (lower > mag) {
                    delta = mag - lower; // negative
                    lower = mag;
                    monotonicity -= 1.5f;
                }
                monotonicity -= 0.08f;
                silent += 1;
            }

            // Turnoff logic
            if (sample_idx < t_monotonicity) {
                monotonicity = 0.0f;
            }
            if ((sample_idx > t_monotonicity) && (monotonicity < monotonicity_thresh) && (monotonicity_thresh != -1)) {
                turnoff = true;
            }

            if (thresh_silent != -1 && silent >= thresh_silent) {
                silent = 0;
                silent_chirp += 1;
            }
            if (silent_chirp >= thresh_silent_chirp && thresh_silent_chirp != -1) {
                turnoff = true;
            }

            if ((sample_idx == n_timesteps-1) && (gradient < 0.0f)) {
                // If we reach the end of the window and the gradient is negative, we can also turn off (no detection)
                turnoff = true;
            }

            // Gradient EWMA
            if (sample_idx >= t_grd) {
                gradient = beta_grd * gradient + alpha_grd * delta;
            }

            // Log-time spiking
            //if (spike_func == 0 && sample_idx >= t_enc) {
            //    if(sample_idx == t_enc) {
            //        // Initialize u_log at t_enc based on current gradient
            //        u_log = gradient;}
            //    else {
            //        u_log += (u_log - gradient) * (tau_log);
            //    }
            //    

            //    if (u_log >= (thresh_log)) {
            //        const int t_rel = sample_idx - t_enc;

            //        turnoff = true;
            //        out[out_idx] = t_rel;

            //        if (do_record_spikes) {
            //            // spikes[t_rel, distance_idx, doppler_idx] flattened as:
            //            // t_rel*(n_distances*n_velocities) + neuron_idx
            //            const int spike_idx = t_rel * (n_distances * n_velocities) + neuron_idx;
            //            spikes[spike_idx] = true;
            //        }
            //    }
            //}
        }
    }

    // Finished without early return
    grad[out_idx] = gradient;
    out[out_idx] = 1-int(turnoff);
}
""",
    name="rd_spinr_ags_float_kernel",
)