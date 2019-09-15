// Launched with (image_width, image_height) grid of threads.
// One thread handle one line of events, so threads are indexes by frame pixel indices.
kernel void event_correlator(
    const global int* vol_times_array,
    const global DTYPE* vol_data_array,
    const global uint* offsets,
    const global int* q_mask,
    global int* res_tau,
    global uint* sums,
    int image_height,
    int Nt
) {
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    if ((x >= IMAGE_WIDTH) || (y >= image_height)) return;
    uint pos = y*IMAGE_WIDTH + x;

    int bin_idx = q_mask[pos] - 1;
    if (bin_idx < 0) return;

    uint offset = offsets[pos];
    int n_events = offsets[pos+1] - offset;
    if (n_events == 0) return;

    global int* vol_times = vol_times_array + offset;
    global DTYPE* vol_data = vol_data_array + offset;

    // Fetch vol_XXX[y, x, :] in fast memory
    int times[MAX_EVT_COUNT] = {0};
    DTYPE data[MAX_EVT_COUNT] = {0};
    for (int i = 0; i < n_events; i++) {
        times[i] = vol_times[i];
        data[i] = vol_data[i];
    }

    // Compute the correlatin and store the result in the current bin
    global uint* my_res_tau = res_tau + bin_idx * Nt;
    global uint* my_sums = sums + bin_idx * Nt;
    for (int i_tau = 0; i_tau < n_events; i_tau++) {
        atomic_add(my_sums + times[i_tau], data[i_tau]);
        for (int i_t = i_tau; i_t < n_events; i_t++) {
            int tau = times[i_t] - times[i_t - i_tau];
            atomic_add(my_res_tau + tau, data[i_t] * data[i_t - i_tau]);
        }
    }
}


#ifndef SCALE_FACTOR
  #define SCALE_FACTOR 1.0f
#endif


// Normalize < <I(t, p) * I(t-tau, p)>_p >_t
// by  <I(t, p)>_p * <I(t-tau, p)>_p >_t
// i.e res_int is divided by correlate(sums, sums, "full")
// The kernel is launched with a grid size (n_frames, n_bins)
kernel void normalize_correlation(
    global int* res_int,
    global float* res,
    global uint* sums,
    global float* scale_factors,
    int Nt
) {
    uint tau = get_global_id(0);
    uint bin_idx = get_global_id(1);
    if ((tau >= Nt) || (bin_idx >= NUM_BINS)) return;

    global uint* my_sum = sums + bin_idx * Nt;
    float corr = 0.0f;
    for (int t = tau; t < Nt; t++) {
        corr += (my_sum[t] * my_sum[t - tau]);
    }
    corr /= scale_factors[bin_idx]; // passing 1/scale_factor in preprocessor is not numerically accurate
    res[bin_idx + Nt + tau] = res_int[bin_idx*Nt + tau] / corr;
}

