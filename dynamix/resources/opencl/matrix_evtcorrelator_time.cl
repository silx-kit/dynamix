#include "dtypes.h"
#include "utils.h"

/**
Build (half) the correlation matrix, using time-compacted data.


Each thread reads at most max_nnz_t elements.
This seems to be much faster than space-based compaction, but the data has to be re-compacted.
**/
kernel void build_correlation_matrix_times_representation(
    const global uint* vol_times_array,
    const global DTYPE* vol_data_array,
    const global OFFSET_DTYPE* offsets,
    const global QMASK_DTYPE* q_mask,
    global RES_DTYPE* corr_matrix,
    global RES_DTYPE* sums,
    int image_width,
    int image_height,
    int n_frames,
    int n_times,
    int pre_sort
) {
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    if ((x >= image_width) || (y >= image_height)) return;
    uint pos = y*image_width + x;

    int bin_idx = q_mask[pos] - 1;
    if (bin_idx < 0) return;

    uint offset = offsets[pos];
    int n_events = offsets[pos+1] - offset;
    if (n_events == 0) return;

    global uint* vol_times = vol_times_array + offset;
    global DTYPE* vol_data = vol_data_array + offset;

    uint times[MAX_EVT_COUNT] = {0};
    DTYPE data[MAX_EVT_COUNT] = {0};
    for (int i = 0; i < n_events; i++) {
        times[i] = vol_times[i];
        data[i] = vol_data[i];
    }


    // If time-compacted data is unordered, it has to be sorted here
    if (pre_sort) {
        int i = 1, j;
        while (i < n_events) {
            j = i;
            while (j > 0 && times[j-1] > times[j]) {
                uint tmp = times[j];
                times[j] = times[j-1];
                times[j-1] = tmp;

                DTYPE tmp2 = data[j];
                data[j] = data[j-1];
                data[j-1] = tmp2;
                j--;
            }
            i++;
        }
    }
    // ---

    size_t cor_matrix_flat_size = (n_frames * (n_times + 1)) / 2;

    global RES_DTYPE* my_sums = sums + bin_idx * n_frames;
    for (int i_tau = 0; i_tau < n_events; i_tau++) {
        atomic_add(my_sums + times[i_tau], data[i_tau]);
        for (int i_t = i_tau; i_t < n_events; i_t++) {
            int tau = times[i_t] - times[i_t - i_tau];
            // size_t out_idx = get_index(n_times, times[i_tau], times[i_t - i_tau]);
            size_t out_idx = get_index(n_times, times[i_t], times[i_t - i_tau]);
            out_idx += bin_idx * cor_matrix_flat_size;
            atomic_add(corr_matrix + out_idx, data[i_t] * data[i_t - i_tau]);
        }
    }
}
