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
static inline size_t get_index(uint W0, uint x0, uint y0) {
    // return (size_t) (W * y) - (size_t) (y*(y-1)/2) + x - y;
    size_t W = (size_t) W0;
    size_t x = (size_t) x0;
    size_t y = (size_t) y0;
    return (size_t) (W * y) - (size_t) (y*(y-1)/2) + x - y;
}

/*
    Compute ceil(a/b) where a and b are integers
*/
static inline uint updiv(uint a, uint b) {
    return (a + (b - 1)) / b;
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

// launched with (ntimes, nframes, qbins) grid
kernel void build_flattened_scalar_correlation_matrix(
    const global RES_DTYPE* arr,
    global RES_DTYPE* res,
    int n_frames,
    int n_times,
    int n_bins
) {
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint qbin_idx = get_global_id(2);
    if ((x >= n_times) || (n_times * y > n_frames * x) || (qbin_idx >= n_bins)) return;
    // res[y * n + x] = arr[x] * arr[y];
    size_t idx = get_index(n_times, x, y);
    idx += ((n_frames * (n_times + 1)) / 2) * qbin_idx;
    res[idx] = arr[x] * arr[y];
}








static inline double2 update_variance(double2 avg_m2, int i, double newval) {
    double avg = avg_m2.s0, m2 = avg_m2.s1;
    double dev1 = newval - avg;
    avg = (i*avg + newval)/(1+i);
    double dev2 = newval - avg;
    m2 += dev1 * dev2;
    return (double2) (avg, m2);
}


/**
Launched with (n_frames, n_bins) threads.

Both numerator and denominator are stored as flattened matrix:
  [ W items | W-1 items | W-2 items | ... | 1 item]    W = n_frames

The aim is to compute
  - sum(diag_num)
  - sum(diag_denom)
  - variance(diag_num / diag_denom)


**/
kernel void get_g2_and_std_v1(
    const global RES_DTYPE* numerator,
    const global RES_DTYPE* denominator,
    global double* g2,
    global double* std,
    int n_frames,
    int n_times,
    int n_bins
)
{
    uint tid = get_global_id(0);
    uint qbin_idx = get_global_id(1);
    if (qbin_idx >= n_bins || tid >= n_frames) return;

    uint offset_for_qbin = ((n_frames * (n_times + 1)) / 2) * qbin_idx;

    double sum_diag_num = 0, sum_diag_denom = 0;
    double2 avg_m2 = (0, 0);

    uint offset = 0;
    for (uint i = 0; i < n_frames; i++) {
        if (tid >= n_frames - i) break; // continue if workgroup
        RES_DTYPE n = numerator[offset_for_qbin + offset + tid];
        RES_DTYPE d = denominator[offset_for_qbin + offset + tid];

        sum_diag_num += (double) n;
        sum_diag_denom += (double) d;
        avg_m2 = update_variance(avg_m2, i, n/ ((double) d));

        offset += n_frames - i;
        // barrier(CLK_GLOBAL_MEM_FENCE); // + big workgroup size

    }
    g2[qbin_idx * n_frames + tid] = sum_diag_num / sum_diag_denom;
    std[qbin_idx * n_frames + tid] = sqrt(avg_m2.s1) / (n_frames - tid);
}




/*
    Reduction function used in compute_final_reductions().
    It uses four items:
      - s0 is the accumulator for sum(diag_numerator)
      - s1 is the accumulator for sum(diag_denominator)
      - s2 is the accumulator for sum(diag_numerator/diag_denominator)
      - s3 is the accumulator for sum((diag_numerator/diag_denominator)**2)
*/

/*
static inline double4 reduce_diagonal_elements(double4* a, double4* b) {
    return (double4) (a.s0 + b.s0, a.s1 + b.s1, );

}
*/




/*
    Launched with (WORKGROUP_SIZE, n_times, n_bins) threads.
    Each group of thread handles exactly one diagonal,
    i.e workgroup 0 handles diag(k=0), workgroup 1 handles diag(k=1), and so on.
    This means that workgroup 0 has the most work.

    Matrices "numerator" and "denominator" are in compacted form,
    i.e half the elements are stored. See the function get_index() for more details
    on how the data is stored within.
    This means that they are not accessed with the easy cartesian pattern.

*/

/*
kernel void compute_final_reductions(
    const global RES_DTYPE* numerator,
    const global RES_DTYPE* denominator,
    local double2* diagonal_elements,
    int n_frames,
    int n_times,
    int n_bins
) {
    uint i = get_global_id(0);
    uint k = get_global_id(1);
    uint qbin_idx = get_global_id(2);
    uint workgroup_size = get_local_size(0);
    uint lid = get_local_id(0); // lid == i for this kernel
    if (i >= workgroup_size || k >= n_times || qbin_idx >= n_bins) return;

    // Number of elements in diagonal k = matrix width - k = n_times - k
    uint n = n_times - k;
    uint n_reduction_steps = updiv(n, workgroup_size);




    for (uint r = 0; r < n_reduction_steps; r++) {

        // processing diag(k)[r*workgroup_size:(r+1)*workgroup_size]
        // TODO: fetch diag(k) into diagonal_elements[]

        // Parallel reduction: sum of diag(numerator, k), sum of diag(denominator, k), sum(numerator/denominator, k)
        for (uint block=workgroup_size/2; block > 1; block /= 2) {
            if ((lid < block) && ((lid + block) < workgroup_size)) {
                diagonal_elements[lid] = REDUCE(diagonal_elements[lid], diagonal_elements[lid + block]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0){
            if (workgroup_size > 1) {
                acc = REDUCE(diagonal_elements[0], diagonal_elements[1]);
            }
            else {
                acc = diagonal_elements[0];
            }
            out[get_group_id(0)] = acc;
        }


    }
}

*/
