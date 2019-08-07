// "Event correlator"
// One thread : r(p, (t, t-tau)) -> atomic_add


kernel void event_correlator_oneQ(
    const global int* vol_times,
    const global DTYPE* vol_data,
    const global int* ctr,
    global int* res_tau,
    //~ global uint* res_t_tau,
    int image_height
) {
    uint x = get_global_id(0);
    uint y = get_global_id(1);

    if ((x >= IMAGE_WIDTH) || (y >= image_height)) return;

    uint pos = y*IMAGE_WIDTH + x;
    int n_events = ctr[pos];
    if (n_events == 0) return;


    // Fetch vol_XXX[y, x, :] in fast memory
    // TODO investigate registers usage
    int times[MAX_EVT_COUNT];
    DTYPE data[MAX_EVT_COUNT];
    for (int i = 0; i < n_events; i++) {
        times[i] = vol_times[pos*MAX_EVT_COUNT+i];
        data[i] = vol_data[pos*MAX_EVT_COUNT+i];
    }

    // Compute r(p, t, t-tau)
    for (int i_tau = 0; i_tau < n_events; i_tau++) {
        for (int i_t = i_tau; i_t < n_events; i_t++) {
            int tau = times[i_t] - times[i_t - i_tau];
            atomic_add(res_tau + tau, data[i_t] * data[i_t - i_tau]);
        }
    }
}
