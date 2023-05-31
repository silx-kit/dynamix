#include "dtypes.h"

/*
  From the index (x, y) in the 2D correlation matrix,
  get the index in the flattened correlation array.

  The output correlation matrix is flattened, and only the upper part is retained.
  So instead of having W*W elements (where W is the matrix width), [W | W | ... | W]
  We have W * (W+1)//2 elements  [W | W - 1 | W - 2 | ... 1 ]
  Element at index (x, y) in the 2D matrix corresponds to index   W * y - y*(y-1)//2 + x - y  in the flattened array

  W: matrix width
  x: column index
  y : row index
*/
static inline size_t get_index(uint W, uint x, uint y) {
    return (size_t) (W * y) - (size_t) (y*(y-1)/2) + x - y;
}

/*
  Binary search in an array of (unsigned) integers.

  Parameters
  -----------
  val: query element
  arr: (sorted) array
  n: array size

  Returns
  -------
  idx: Location of found element. If no element is found, return the array size.
*/
static inline uint binary_search(uint val, uint* arr, uint n) {
    uint L = 0, R = n - 1;
    uint m = 0;
    while (L != R) {
        m = (L + R + 1)/2;
        if (arr[m] > val) R = m - 1;
        else L = m;
    }
    if (arr[L] == val) return L;
    return n;
}





/*
  Build (half) the correlation matrix from the "events" data structure.
  Threads are launched by a 1D grid (n_times, 1).
  Each thread computes corr[i, i] = sum_{current_qbin}(frame[i] * frame[j])
    then corr[i, i+1] = sum_{curr_bin}(frame[i] * frame[i+1])
    and so on

  This means that threads 0 read the entire dataset (!),
    thread 1 reads frame 1, 2, .... N-1,
    thread 2 reads frame 2, 3, .... N-1

  Parameters
  -----------
  data: array of 'total_nnz' elements, where 'total_nnz' is the total number of non-zero pixels in the XPCS data
    (i.e total number of 'events')
  pixel_idx: array of 'total_nnz' elements
  offset: array of n_frames + 1 elements. offset[i+1]-offset[i] gives the number of non-zero elements in frame number 'i'.
  qmask_compacted: array of 'total_nnz' elements. qmask_comacted[idx] gives the q-bin of pixel at (compacted) index 'idx'.

*/

kernel void build_correlation_matrix_v1(
    const global DTYPE* data,
    const global uint* pixel_idx,
    const global OFFSET_DTYPE* offset,
    const global char* qmask,
    global RES_DTYPE* corr_matrix,
    int n_frames,
    int n_bins
) {

    uint frame_idx = get_global_id(0);
    uint current_qbin = get_global_id(1) + 1;
    if (current_qbin > n_bins || frame_idx >= n_frames) return;

    OFFSET_DTYPE i_start = offset[frame_idx], i_stop = offset[frame_idx+1];

    RES_DTYPE res = 0;
    for (int i = i_start; i < i_stop; i++) {
        if (qmask[i] == current_qbin) {
           res += ((RES_DTYPE) data[i]) * ((RES_DTYPE) data[i]);
        }
    }
    corr_matrix[frame_idx * n_frames + frame_idx] = res;

    // TODO same for off-diagonal elements

}

/**
Build (half) the correlation matrix.
Threads are launched with a 2D grid (n_frames, n_times)  where n_times == n_frames most of the time.
This means that each given frame will be read by all threads.
**/
kernel void build_correlation_matrix_v2(
    const global DTYPE* data,
    const global uint* pixel_idx,
    const global OFFSET_DTYPE* offset,
    const global char* qmask,
    global RES_DTYPE* corr_matrix,
    int n_frames,
    int n_times,
    int current_qbin
) {

    uint time_idx = get_global_id(0);
    uint frame_idx = get_global_id(1);
    if (time_idx >= n_times || frame_idx >= n_frames || frame_idx > time_idx) return;

    OFFSET_DTYPE i_start = offset[frame_idx], i_stop = offset[frame_idx+1];
    RES_DTYPE res = 0;


    if (time_idx == frame_idx) {
        // Main diagonal - simple case!
        RES_DTYPE d = 0;
        for (int i = i_start; i < i_stop; i++) {
            if (qmask[i] == current_qbin) {
                d = (RES_DTYPE) data[i];
                res += d * d;
            }
        }
    }
    else {
        // Off-diagonal
        OFFSET_DTYPE i_start_other = offset[time_idx], i_stop_other = offset[time_idx+1];

        uint i_other = 0;
        uint n_pix_other = i_stop_other - i_start_other;
        for (uint i = 0; i < i_stop - i_start; i++) {
            while (i_other < n_pix_other && pixel_idx[i_start_other + i_other] < pixel_idx[i_start + i]) {
                i_other += 1;
            }
            if (i_other == n_pix_other) continue;
            if (pixel_idx[i_start_other + i_other] == pixel_idx[i_start + i]
                && qmask[i_start + i] == current_qbin
                && qmask[i_start_other + i_other] == current_qbin
            ){
                res += data[i_start + i] * data[i_start_other + i_other];
            }
        }
    }
    size_t out_idx = get_index(n_times, time_idx, frame_idx);
    corr_matrix[out_idx] = res;
}



/**
Same as above, but threads are launched by groups, so that each frame is read at most n_frames / group_size times.
**/
kernel void build_correlation_matrix_v2b(
    const global DTYPE* data,
    const global uint* pixel_idx,
    const global OFFSET_DTYPE* frame_offset,
    const global char* qmask,
    global RES_DTYPE* corr_matrix,
    int n_frames,
    int n_times,
    int current_qbin,
    local DTYPE* s_data,
    local char* s_qmask
) {

    uint time_idx = get_global_id(0);
    uint frame_idx = get_global_id(1);
    uint tid = get_local_id(0);
    uint wg_size = get_local_size(0);
    // if (time_idx >= n_times || frame_idx >= n_frames || frame_idx > time_idx) return;
    if (time_idx >= n_times || frame_idx >= n_frames) return;

    OFFSET_DTYPE i_start = frame_offset[frame_idx], i_stop = frame_offset[frame_idx+1];
    RES_DTYPE res = 0;

    uint s_offset = 0, offset = 0;

    for (s_offset = 0; i_start + s_offset < i_stop; s_offset += SHARED_ARRAYS_SIZE) {


      // ========================================================================
      // Fetch XX[i_start:i_stop] into shared memory where XX is 'data' and 'pixel_idx'
      // local DTYPE s_data[SHARED_ARRAYS_SIZE];
      // local char s_qmask[SHARED_ARRAYS_SIZE];
      for (offset = 0; offset + tid < SHARED_ARRAYS_SIZE ; offset += wg_size) {
          if (i_start + s_offset + offset + tid >= i_stop) break;
          s_data[offset + tid] = data[i_start + s_offset + offset + tid];
          s_qmask[offset + tid] = qmask[i_start + s_offset + offset + tid];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      // ========================================================================

      if (time_idx == frame_idx) {
          // Main diagonal - simple case!
          RES_DTYPE d = 0;
          for (uint i = 0; i < SHARED_ARRAYS_SIZE && i_start + s_offset + i < i_stop; i++) {
              if (s_qmask[i] == current_qbin) {
                  d = (RES_DTYPE) s_data[i];
                  res += d * d;
              }
          }
      }

      // ========================================================================  off-diagonal
      // else {
      //     // Off-diagonal
      //     OFFSET_DTYPE i_start_other = frame_offset[time_idx], i_stop_other = frame_offset[time_idx+1];

      //     uint i_other = 0;
      //     uint n_pix_other = i_stop_other - i_start_other;
      //     for (uint i = 0; i < i_stop - i_start; i++) {
      //         while (i_other < n_pix_other && pixel_idx[i_start_other + i_other] < pixel_idx[i_start + i]) {
      //             i_other += 1;
      //         }
      //         if (i_other == n_pix_other) continue;
      //         if (pixel_idx[i_start_other + i_other] == pixel_idx[i_start + i]
      //             && qmask[i_start + i] == current_qbin
      //             && qmask[i_start_other + i_other] == current_qbin
      //         ){
      //             res += data[i_start + i] * data[i_start_other + i_other];
      //         }
      //     }
      // }
      // ========================================================================  /off-diagonal


      barrier(CLK_LOCAL_MEM_FENCE);
    }

    // if (frame_idx > time_idx) return;

    if (frame_idx == time_idx) {
    size_t out_idx = get_index(n_times, time_idx, frame_idx);
    corr_matrix[out_idx] = res;
    }
    // corr_matrix[out_idx] = tid;
}


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

**/
kernel void build_correlation_matrix_v3(
    const global DTYPE* data,
    const global uint* pixel_idx,
    const global OFFSET_DTYPE* frame_offset,
    const global char* qmask,
    global RES_DTYPE* corr_matrix,
    global RES_DTYPE* sums,
    int n_frames,
    int n_times,
    int current_qbin
) {

    uint frame_idx = get_global_id(1);
    if (frame_idx  >= n_frames) return;
    uint idx = get_global_id(0);
    uint i_start_0 = frame_offset[frame_idx];
    uint i_stop_0 = frame_offset[frame_idx + 1];
    if (i_start_0 + idx >= i_stop_0) return;

    char qbin = qmask[i_start_0 + idx] - 1;
    if ((qbin < 0) || (current_qbin > 0 && qbin != current_qbin)) return;
    size_t cor_matrix_flat_size = (n_frames * (n_times + 1)) / 2;

    uint my_pix_idx = pixel_idx[i_start_0 + idx];
    RES_DTYPE d = (RES_DTYPE) data[i_start_0 + idx];


    // corr_matrix[i, i] = sum_in_bin(frame[i] * frame[i])
    size_t out_idx = get_index(n_times, frame_idx, frame_idx);
    if (current_qbin < 0) out_idx += qbin * cor_matrix_flat_size;
    atomic_add(corr_matrix + out_idx, d * d);
    // sums[i] = sum_in_bin(frame[i])
    atomic_add(sums + frame_idx, d);

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
        if (current_qbin < 0) out_idx += qbin * cor_matrix_flat_size;
        atomic_add(corr_matrix + out_idx, d * d_other);
    }
}



/**
Build (half) the correlation matrix, using time-compacted data.


Each thread reads at most max_nnz_t elements.
This seems to be much faster than space-based compaction, but the data has to be re-compacted.
**/
kernel void build_correlation_matrix_times_representation(
    const global uint* vol_times_array,
    const global DTYPE* vol_data_array,
    const global uint* offsets,
    const global int* q_mask,
    global RES_DTYPE* corr_matrix,
    global uint* sums,
    int image_width,
    int image_height,
    int n_frames,
    int n_times
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

    int times[MAX_EVT_COUNT] = {0};
    DTYPE data[MAX_EVT_COUNT] = {0};
    for (int i = 0; i < n_events; i++) {
        times[i] = vol_times[i];
        data[i] = vol_data[i];
    }

    size_t cor_matrix_flat_size = (n_frames * (n_times + 1)) / 2;

    global uint* my_sums = sums + bin_idx * n_frames;
    for (int i_tau = 0; i_tau < n_events; i_tau++) {
        atomic_add(my_sums + times[i_tau], data[i_tau]);
        for (int i_t = i_tau; i_t < n_events; i_t++) {
            int tau = times[i_t] - times[i_t - i_tau];
            // size_t out_idx = get_index(n_times, times[i_tau], times[i_t - i_tau]);
            size_t out_idx = get_index(n_times, times[i_t], times[i_t - i_tau]);
            out_idx += bin_idx * cor_matrix_flat_size;
            atomic_add(corr_matrix + out_idx, data[i_t] * data[i_t - i_tau]);

            // was: atomic_add(my_res_tau + tau, data[i_t] * data[i_t - i_tau]);
        }
    }
}

