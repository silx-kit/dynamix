

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

#ifndef DTYPE
  #define DTYPE uchar
#endif

#ifndef OFFSET_DTYPE
  // must be extended to unsigned long if total_nnz > 2147483647
  #define OFFSET_DTYPE uint
#endif

#ifndef SHARED_ARRAYS_SIZE
  #define SHARED_ARRAYS_SIZE 4096
#endif

#ifndef QMASK_DTYPE
  // must be extended to int if number of q-bins > 127
  #define QMASK_DTYPE char
#endif

#ifndef RES_DTYPE
  // must be extended to unsigned long for large nnz_per_frame and/or events counts
  #define RES_DTYPE uint
#endif


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
  Build (half) the correlation matrix from the "events" data structure.
  data: array of 'total_nnz' elements, where 'total_nnz' is the total number of non-zero pixels in the XPCS data
    (i.e total number of 'events')
  pixel_idx: array of 'total_nnz' elements
  offset: array of n_frames + 1 elements. offset[i+1]-offset[i] gives the number of non-zero elements in frame number 'i'.
  qmask_compacted: array of 'total_nnz' elements. qmask_comacted[idx] gives the q-bin of pixel at (compacted) index 'idx'.


*/

kernel void build_correlation_matrix_diagonal(
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
    corr_matrix[frame_idx * n_frames + frame_idx] = res; // current_qbin; // res; // i_stop - i_start; // res;

}



kernel void build_correlation_matrix_flattened(
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


kernel void build_correlation_matrix_flattened_wg(
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
    if (time_idx >= n_times || frame_idx >= n_frames || frame_idx > time_idx) return;

    OFFSET_DTYPE i_start = frame_offset[frame_idx], i_stop = frame_offset[frame_idx+1];
    RES_DTYPE res = 0;

    RES_DTYPE debug = 0;

    uint s_offset = 0, offset = 0;

    for (s_offset = 0; i_start + s_offset < i_stop; s_offset += SHARED_ARRAYS_SIZE) {


      // ========================================================================
      // Fetch XX[i_start:i_stop] into shared memory where XX is 'data' and 'pixel_idx'
      // local DTYPE s_data[SHARED_ARRAYS_SIZE];
      // local char s_qmask[SHARED_ARRAYS_SIZE];
      // for (uint offset = s_offset; i_start + offset < i_stop; offset += wg_size) {
      for (offset = 0; offset < SHARED_ARRAYS_SIZE/wg_size ; offset += wg_size) {
          debug = s_offset + offset + tid;
          if (s_offset + offset + tid >= i_stop - i_start) break;
          s_data[offset + tid] = data[i_start + s_offset + offset + tid];
          s_qmask[offset + tid] = qmask[i_start + s_offset + offset + tid];

          // if (i_start + offset + tid < i_stop) {
              // if (offset + tid < SHARED_ARRAYS_SIZE) s_data[offset + tid] = data[i_start + offset + tid];
              // if (offset + tid < SHARED_ARRAYS_SIZE) s_qmask[offset + tid] = qmask[i_start + offset + tid];
          // }



      }
      barrier(CLK_LOCAL_MEM_FENCE);
      // ========================================================================




      if (tid == 0 && time_idx == frame_idx) {
          // Main diagonal - simple case!
          RES_DTYPE d = 0;
          for (uint i = 0; i < SHARED_ARRAYS_SIZE && i_start + s_offset + i < i_stop; i++) {
              if (1 || s_qmask[i] == current_qbin) {
                  // d = (RES_DTYPE) s_data[i];
                  // res += d * d;
                res += 1; // ((RES_DTYPE) s_data[i]) * ((RES_DTYPE) s_data[i]);
              }
          }
      }

      // ========================================================================  off-diagonal
      else {
          return;
          // Off-diagonal
          OFFSET_DTYPE i_start_other = frame_offset[time_idx], i_stop_other = frame_offset[time_idx+1];

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
      // ========================================================================  /off-diagonal


  }


    size_t out_idx = get_index(n_times, time_idx, frame_idx);
    if (tid == 0) corr_matrix[out_idx] = res;
}







