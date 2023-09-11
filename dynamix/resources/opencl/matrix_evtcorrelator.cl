#include "dtypes.h"
#include "utils.h"


/**
Must be launched with grid (max_nonzero_per_frame, n_frames).

Each "line of threads" handles one vector of space-compacted data.


        threads line 0 |  threads line 1  | threads line 2   ...
data = [nnz_frame_0    | nnz_frame_1      | nnz_frame_2      ...]   // space-compacted
    off[0]           off[1]             off[2]                      // offsets


Therefore, a given thread has one pixel_idx.
It will look (through binary search) if this pixel index is found in the other frames vectors.

Each threads reads at most 1 + n_times/2 * log2(max_nnz)   elements,
  where max_nnz is the max length of a space-compacted vector (i.e the maximum number of non-zero items in frames)

Here again, a ~2X speed-up could be obtain by better balancing the work (see notes in first kernel)
**/
kernel void build_correlation_matrix(
    const global DTYPE* data,
    const global uint* pixel_idx,
    const global OFFSET_DTYPE* frame_offset,
    const global QMASK_DTYPE* qmask,
    global RES_DTYPE* corr_matrix,
    global RES_DTYPE* sums,
    int n_frames,
    int n_times
) {
    uint frame_idx = get_global_id(1);
    if (frame_idx  >= n_frames) return;
    uint idx = get_global_id(0);
    uint i_start_0 = frame_offset[frame_idx];
    uint i_stop_0 = frame_offset[frame_idx + 1];
    if (i_start_0 + idx >= i_stop_0) return;

    char qbin = qmask[i_start_0 + idx] - 1;
    if (qbin < 0) return;
    size_t cor_matrix_flat_size = (n_frames * (n_times + 1)) / 2;

    uint my_pix_idx = pixel_idx[i_start_0 + idx];
    RES_DTYPE d = (RES_DTYPE) data[i_start_0 + idx];

    // corr_matrix[i, i] = sum_in_bin(frame[i] * frame[i])
    size_t out_idx = get_index(n_times, frame_idx, frame_idx);
    out_idx += qbin * cor_matrix_flat_size;
    atomic_add(corr_matrix + out_idx, d * d);
    // sums[i] = sum_in_bin(frame[i])
    atomic_add(sums + qbin * n_frames + frame_idx, d);

    // TODO balance workload among threads: each thread handles n_times/2 frames
    for (uint other_frame_idx = frame_idx + 1; other_frame_idx < n_times /* && other_frame_idx - frame_idx < n_times/2 */ ; other_frame_idx++) {
        // data for current frame is in data[i_start:i_stop]
        uint i_start = frame_offset[other_frame_idx];
        uint i_stop = frame_offset[other_frame_idx + 1];
        // is there an index 'i' in pixel_idx[i_start:i_stop] such that i == my_pix_idx ?
        uint i = binary_search(my_pix_idx, pixel_idx + i_start, i_stop - i_start);
        if (i == i_stop - i_start) continue;
        // if so, accumulate the result
        RES_DTYPE d_other = (RES_DTYPE) data[i_start + i];
        out_idx = get_index(n_times, other_frame_idx, frame_idx);
        out_idx += qbin * cor_matrix_flat_size;
        atomic_add(corr_matrix + out_idx, d * d_other);
    }
}

