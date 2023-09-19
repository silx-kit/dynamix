#ifndef UTILS_H
#define UTILS_H

static inline size_t get_index(uint W, uint x, uint y);
static inline uint binary_search(uint val, uint* arr, uint n);
kernel void build_flattened_scalar_correlation_matrix(const global RES_DTYPE* arr, global RES_DTYPE* res, int n_frames, int n_times, int n_bins);

#endif
