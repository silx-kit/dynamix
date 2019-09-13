// Launched with (image_width, image_height) grid of threads.
// One thread handle one line of events, so threads are indexes by frame pixel indices.
kernel void event_correlator_oneQ(
    const global int* vol_times_array,
    const global DTYPE* vol_data_array,
    const global uint* offsets,
    global int* res_tau,
    global uint* sums,
    int image_height
) {
    uint x = get_global_id(0);
    uint y = get_global_id(1);

    if ((x >= IMAGE_WIDTH) || (y >= image_height)) return;

    uint pos = y*IMAGE_WIDTH + x;
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

    // Compute r(p, t, t-tau)
    DTYPE sum = 0;
    for (int i_tau = 0; i_tau < n_events; i_tau++) {
        atomic_add(sums + times[i_tau], data[i_tau]);
        for (int i_t = i_tau; i_t < n_events; i_t++) {
            int tau = times[i_t] - times[i_t - i_tau];
            atomic_add(res_tau + tau, data[i_t] * data[i_t - i_tau]);
        }
    }
}






#ifndef SCALE_FACTOR
  #define SCALE_FACTOR 1.0f
#endif


// Normalize < <I(t, p) * I(t-tau, p)>_p >_t
// by  <I(t, p)>_p * <I(t-tau, p)>_p >_t
kernel void normalize_correlation_oneQ(
    global int* res_int,
    global float* res,
    global uint* sums,
    int Nt
) {
    uint tau = get_global_id(0);
    if (tau >= Nt) return;
    float s = 0.0f;
    for (int t = tau; t < Nt; t++) {
        s += (sums[t] * sums[t - tau]);
    }
    s /= SCALE_FACTOR;
    res[tau] = res_int[tau] / s;
}

