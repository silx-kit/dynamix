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
    global float* res,
    int n_frames,
    int n_times
) {
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint qbin = get_global_id(2);
    if ((x >= n_times) || (n_times * y > n_frames * x)) return;
    // res[y * n + x] = arr[x] * arr[y];
    size_t idx = get_index(n_times, x, y);
    idx += ((n_frames * (n_times + 1)) / 2) * qbin;
    res[idx] = arr[x] * arr[y];
}