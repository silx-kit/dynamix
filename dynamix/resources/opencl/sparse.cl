#include "dtypes.h"

// OK, but results may be un-sorted
kernel void space_compact_to_time_compact(
    const global DTYPE* data,
    const global uint* pixel_idx,
    const global OFFSET_DTYPE* frame_offset,

    global DTYPE* t_data,
    global uint* t_times,
    global uint* counters,

    int n_frames,
    int Nx,
    int Ny
) {

    uint idx = get_global_id(0);

    uint start, stop;
    size_t img_pix_idx, vol_pix_idx;
    uint j = 0;
    for (uint i = 0; i < n_frames; i++) {
        start = frame_offset[i];
        stop = frame_offset[i+1];
        if (start + idx < stop) {
            img_pix_idx = pixel_idx[start + idx];
            j = atomic_inc(counters + img_pix_idx);
            vol_pix_idx = img_pix_idx + j * Ny * Nx;
            t_data[vol_pix_idx] = data[start + idx];
            t_times[vol_pix_idx] = i;
        }
    }
}

// Slower
kernel void space_compact_to_time_compact_alternate(
    const global DTYPE* data,
    const global uint* pixel_idx,
    const global OFFSET_DTYPE* frame_offset,
    const global int* q_mask,

    global DTYPE* t_data,
    global uint* t_times,
    global uint* counters,

    int n_frames,
    int Nx,
    int Ny
) {

    uint x = get_global_id(0);
    uint y = get_global_id(1);

    if (x >= Nx || y >= Ny) return;
    size_t pos = y * Nx + x;
    int qbin = q_mask[pos] - 1;
    if (qbin < 0) return;


    uint t = 0;
    for (uint i = 0; i < n_frames; i++) {
        uint start = frame_offset[i];
        uint stop = frame_offset[i+1];
        uint pos2 = binary_search(pos, pixel_idx + start, stop - start);
        if (pos2 == stop - start) continue;
        size_t pos_in_vol = (t * Ny + y) * Nx + x;
        t_times[pos_in_vol] = i;
        t_data[pos_in_vol] = data[start + pos2];
        t++;
    }
    counters[pos] = t;
}


kernel void space_compact_to_time_compact_stage2(
    const global DTYPE* t_data_tmp,
    const global uint* t_times_tmp,
    const global OFFSET_DTYPE* t_offsets,

    global DTYPE* t_data,
    global uint* t_times,

    int n_pix
) {

    uint pix_idx = get_global_id(0);
    if (pix_idx >= n_pix) return;
    uint start = t_offsets[pix_idx];
    uint stop = t_offsets[pix_idx+1];
    if (start == stop) return;

    for (uint i = start; i < stop; i++) {
        t_data[i] = t_data_tmp[(i - start) * n_pix + pix_idx];
        t_times[i] = t_times_tmp[(i - start) * n_pix + pix_idx];
    }

}

/**
 * Should be launched with a grid (n_pixels_tot, 1).
 * This kernel does the same as "space_compact_to_time_compact_stage2",
 * but it also sorts the elements by time before merging them.
 *
*/
kernel void space_compact_to_time_compact_stage2_sort(
    const global DTYPE* t_data_tmp,
    const global uint* t_times_tmp,
    const global OFFSET_DTYPE* t_offsets,

    global DTYPE* t_data,
    global uint* t_times,

    int n_pix
) {

    uint pix_idx = get_global_id(0);
    if (pix_idx >= n_pix) return;
    uint start = t_offsets[pix_idx];
    uint stop = t_offsets[pix_idx+1];
    if (start == stop) return;


    uint l_times[MAX_EVT_COUNT] = {0};
    DTYPE l_data[MAX_EVT_COUNT] = {0};

    for (uint i = 0; i < stop-start; i++) {
        l_data[i] = t_data_tmp[i * n_pix + pix_idx];
        l_times[i] = t_times_tmp[i * n_pix + pix_idx];
    }

    int i = 1, j;
    while (i < stop-start) {
        j = i;
        while (j > 0 && l_times[j-1] > l_times[j]) {
            uint tmp = l_times[j];
            l_times[j] = l_times[j-1];
            l_times[j-1] = tmp;

            DTYPE tmp2 = l_data[j];
            l_data[j] = l_data[j-1];
            l_data[j-1] = tmp2;
            j--;
        }
        i++;
    }

    for (uint i = start; i < stop; i++) {
        t_data[i] = l_data[i - start];
        t_times[i] = l_times[i - start];
    }
}

