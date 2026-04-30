import cupy as cp

kernel = cp.RawKernel("""//CUDA//
    //cuda
    extern "C" __global__ void float_spinr_kernel(
                            float *x, 
                            float *W, 
                            float *sample_R, 
                            float *chirp_R, 
                            float *states,
                            bool *spikes, 
                            int* n_samples_, 
                            int* n_chirps_, 
                            int* n_channels_,
                            float *params, 
                            bool* debug_)
    {
        unsigned int neuron_idx_0 = threadIdx.x; // angle
        unsigned int neuron_idx_1 = blockIdx.x; // distance
        unsigned int neuron_idx_2 = blockIdx.y; // velocity
        
        unsigned int n_neurons_0 = blockDim.x; 
        unsigned int n_neurons_1 = gridDim.x;
        unsigned int n_neurons_2 = gridDim.y;

        unsigned int n_channels = n_channels_[0];
        unsigned int n_samples = n_samples_[0];
        unsigned int n_chirps = n_chirps_[0];
        unsigned int n_timesteps = n_samples * n_chirps;
                          
        //unsigned int start_idx_v = n_timesteps*n_angles*n_distances*velocity_idx;
        //unsigned int start_idx_d = n_timesteps*n_angles*distance_idx;
        //unsigned int start_idx_a = n_timesteps*angle_idx;
        unsigned int start_idx_2 = n_neurons_0*n_neurons_1*neuron_idx_2;
        unsigned int start_idx_1 = n_neurons_0*neuron_idx_1;
        unsigned int start_idx_0 = neuron_idx_0;

        // input data
        unsigned int input_dim = 2;

        // W
        // start_idx_W + 2*antenna_idx + 0/1 (real/imag)
        unsigned int W_dim = 2;
        unsigned int start_idx_W = 2*n_channels*neuron_idx_0;

        // Spike data
        // Usage:
        // spike_idx + spike_dim*n_samples*chirp_idx + spike_dim*sample_idx + spike_dim_idx
        unsigned int spike_dim = 1;
        unsigned int spike_idx = spike_dim*n_timesteps*(start_idx_2 + start_idx_1 + start_idx_0);

        // State data
        // Usage:
        // out_idx + out_dim*n_samples*chirp_idx + out_dim*sample_idx + out_dim_idx
        unsigned int state_dim = 2;
        unsigned int state_idx = state_dim*(start_idx_2 + start_idx_1 + start_idx_0);


        // Parameters
        // Global params 
        bool debug = debug_[0];

        // Input params
        float alpha_l = params[0]; // weight of negative gradient / lower limit, default: 1
        float alpha_a_smooth = params[1]; // exponential filtering of magnitude, default: 0.001
        float alpha_grad_smooth = params[2]; // exponential filtering of gradient, default: 0.001

        // Start time for gradient estimation
        unsigned int t_start = params[3];

        // Spiking Function Params
        float alpha_u = params[4];     // weight for potential, default: 0
        float u_threshold = params[5]; // threshold
        float u_rest = params[6];      // rest

        bool allow_spike = true;

        // Init dynamic states
        // Input states
        float inp[2] = {0,0};
            
        // RF Neuron states
        float s[2] = {0, 0};
        float s_[2] = {0,0};
        s[0] = states[state_idx + 0];
        s[1] = states[state_idx + 1];

        // Magnitude
        float a = 0;
        
        // Envelopes
        float lower = 0;
        float upper = 0;
        float delta_lower = 0;
        float delta_upper = 0;

        // Gradient
        float grad = 0;

        // Rate and time LIF
        float u = 0;

        for (int t=0; t<n_timesteps; t++) {
            // CHIRP
            // For each chirp, only if more than 1 neuron in 2
            if (n_neurons_2 > 1) {
                if ((t%n_samples==0)){
                    // Reset output dynamics
                    u = 0;

                    // Reset envelope (Why?)
                    lower = 0;
                    upper = 0;

                    // Chirp rotation
                    s[0] = chirp_R[2*neuron_idx_2 + 0] * s_[0] - chirp_R[2*neuron_idx_2 + 1] * s_[1];
                    s[1] = chirp_R[2*neuron_idx_2 + 0] * s_[1] + chirp_R[2*neuron_idx_2 + 1] * s_[0];
                }
            } // else perform resetting if bool true
            else if ((false)) {

                // RF Neuron states
                memset(s, 0, sizeof(s));
                memset(s_, 0, sizeof(s_));

                // Reset envelope
                lower = 0;
                upper = 0;

                // Reset gradient of neuron state
                u = 0;
            }
                       
            // SAMPLE

            // Rotation of states s_t+1 = R s_t
            // with rotation matrix R
            s_[0] = 1*(sample_R[2*neuron_idx_1 + 0] * s[0] - sample_R[2*neuron_idx_1 + 1] * s[1]);
            s_[1] = 1*(sample_R[2*neuron_idx_1 + 1] * s[0] + sample_R[2*neuron_idx_1 + 0] * s[1]);

            // Variables for matrix multiplication inp = W x_t
            inp[0] = 0;
            inp[1] = 0;

            // Matrix multiplication W x_t
            for (int channel_idx=0; channel_idx<n_channels; channel_idx++) {
                
                inp[0] += W[start_idx_W + W_dim*channel_idx + 0] * float(x[input_dim*n_timesteps*channel_idx + input_dim*t + 0])
                              - W[start_idx_W + W_dim*channel_idx + 1] * float(x[input_dim*n_timesteps*channel_idx + input_dim*t + 1]);
                inp[1] += W[start_idx_W + W_dim*channel_idx + 1] * float(x[input_dim*n_timesteps*channel_idx + input_dim*t + 0])
                               + W[start_idx_W + W_dim*channel_idx + 0] * float(x[input_dim*n_timesteps*channel_idx + input_dim*t + 1]);
                          
            }

            // Adding s_t + Wx_t
            s_[0] += inp[0];
            s_[1] += inp[1];
                          
            a = sqrtf(s_[0]*s_[0] + s_[1]*s_[1]);
            //a = (1-alpha_a_smooth)*a + alpha_a_smooth * (abs(s_[0]) + abs(s_[1]));
            
            // Envelopes
            delta_upper =  1*(a-upper)*((upper)<(a));
            delta_lower = ((upper + lower) > a)*(a-lower-upper)*alpha_l;

            // Exponential filtering
            // Estimate gradient with starting point
            if ((t>=t_start)) {
                grad =  (1-alpha_grad_smooth)*grad + alpha_grad_smooth*(delta_upper+delta_lower);
                u += (grad/alpha_grad_smooth + u_rest - u)*alpha_u;
            }

            // Spiking block 
            //spikes[spike_idx + spike_dim*t + 0] = sub_reset_spike(u, u_threshold, u_threshold, true);
            if ((u>u_threshold)){
                u -= u_threshold;
                spikes[spike_idx + spike_dim * t] = true;
            }
            //spikes[spike_idx + spike_dim*t + 1] = sub_reset_spike(u, u_threshold, u_reset, allow_spike)
            //if ((u>u_threshold) && allow_spike){
            //    u -= u_threshold;
            //    spikes[spike_idx + spike_dim*t + 0] = true;
            //}
                          
            // Updating 
            upper += delta_upper;
            lower += delta_lower;
            s[0] = s_[0];
            s[1] = s_[1];

            // Storing 
            //if ((debug==true) && (t == n_timesteps-1)) {
            //    states[state_idx + 0] += spikes[spike_idx + spike_dim*t + 0];
            //    states[state_idx + 1] += 0;
            //}
        }
    }
    //!cuda
    """, 'float_spinr_kernel')
