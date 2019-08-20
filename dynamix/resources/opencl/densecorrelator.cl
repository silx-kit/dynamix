// This is a general-purpose kernel for "dense" data.
// If IMAGE_WIDTH > 1024, the kernel will have to be adapted
// (unlikely since the XPCS data is dense only in a reduced ROI)
// WG threads horizontally (must have WG >= NUM_BINS), ideally WG >= IMAGE_WIDTH
// Nt thread vertically
kernel void correlator_multiQ_dense(
    const global char* frames,
    const global int* q_mask,
    const global float* norm_mask,
    const global float* normalization,
    global float* output,
    int image_height,
    int Nt
) {
    uint tid = get_local_id(0);
    uint tau = get_global_id(1);
    if (tau >= Nt) return;
    if (tid >= IMAGE_WIDTH) return;

    uint my_sum[NUM_BINS];
    local uint sum_p[NUM_BINS];

    // memset private "my_sum"
    for (int q = 0; q < NUM_BINS; q++) my_sum[q] = 0;
    // memset shared "sum_p"
    if (tid < NUM_BINS) sum_p[tid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Invert loops on "t" and "row_idx" to do less q_mask reads
    for (int row_idx = 0; row_idx < image_height; row_idx++) {

        int q = q_mask[row_idx*IMAGE_WIDTH + tid] -1;
        if (q == -1) continue; //
        for (int t = tau; t < Nt; t++) {
            // frames[t, row_idx, :] * frames[t-tau, row_idx, :]
            uint val1 = frames[(t*image_height+row_idx)*IMAGE_WIDTH + tid];
            uint val2 = frames[((t-tau)*image_height+row_idx)*IMAGE_WIDTH + tid];
            my_sum[q] += val1*val2;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    // Gather
    for (int q = 0; q < NUM_BINS; q++) {
        atomic_add(sum_p + q, my_sum[q]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < NUM_BINS)  // tid <-> q
        output[tid*Nt + tau] = sum_p[tid] / normalization[tid*Nt + tau];
}



#define REDUCE_SHARED_ITEMS(stride) for (int q = 0; q < NUM_BINS; q++) s_buf[q*(2*SUM_WG_SIZE) + tid] += s_buf[q*(2*SUM_WG_SIZE) + tid + stride];


// Must be launched with worgroup size SUM_WG_SIZE horizontally,
// and with "Nt" threads vertically
kernel void compute_sums_dense(
    const global DTYPE* frames,
    const global int* q_mask,
    global DTYPE_SUMS* sums,
    int image_height,
    int Nt
) {
    uint tid = get_local_id(0);
    uint frame_id = get_global_id(1);
    if (frame_id >= Nt) return;

    // frame for current group of threads
    size_t numels = IMAGE_WIDTH * image_height;
    const global DTYPE* frame = frames + frame_id*numels;

    // Allocate (+memset) a shared buffer twice bigger than workgroup size
    local DTYPE_SUMS s_buf[2*SUM_WG_SIZE*NUM_BINS];
    for (int q = 0; q < NUM_BINS; q++) {
        s_buf[q*(2*SUM_WG_SIZE) + tid] = 0;
        s_buf[q*(2*SUM_WG_SIZE) + tid + SUM_WG_SIZE] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 1: reduce global data to local data
    for (int offset = 0; tid + offset < numels; offset += SUM_WG_SIZE) {
        int q = q_mask[tid + offset] - 1;
        if (q == -1) continue;
        s_buf[q*(2*SUM_WG_SIZE) + tid] += frame[tid + offset];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 2: parallel reduction within shared memory
    #if SUM_WG_SIZE >= 1024
    if (tid < 1024) REDUCE_SHARED_ITEMS(1024)
    barrier(CLK_LOCAL_MEM_FENCE);
    #endif

    #if SUM_WG_SIZE >= 512
    if (tid < 512) REDUCE_SHARED_ITEMS(512)
    barrier(CLK_LOCAL_MEM_FENCE);
    #endif

    #if SUM_WG_SIZE >= 256
    if (tid < 256) REDUCE_SHARED_ITEMS(256)
    barrier(CLK_LOCAL_MEM_FENCE);
    #endif

    #if SUM_WG_SIZE >= 128
    if (tid < 128) REDUCE_SHARED_ITEMS(128)
    barrier(CLK_LOCAL_MEM_FENCE);
    #endif

    #if SUM_WG_SIZE >= 64
    if (tid < 64) REDUCE_SHARED_ITEMS(64)
    barrier(CLK_LOCAL_MEM_FENCE);
    #endif

    // No synchronization needed on Nvidia hardware beyond this point
    if (tid < 32) {
        REDUCE_SHARED_ITEMS(32)
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid < 16) {
        REDUCE_SHARED_ITEMS(16)
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid < 8) {
        REDUCE_SHARED_ITEMS(8)
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid < 4) {
        REDUCE_SHARED_ITEMS(4)
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid < 2) {
        REDUCE_SHARED_ITEMS(2)
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) {
        REDUCE_SHARED_ITEMS(1)
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid < NUM_BINS) { // tid <-> q
        sums[tid * Nt + frame_id] = s_buf[tid*(2*SUM_WG_SIZE) + 0];
    }
}




// Launched with a grid of size (Nt, n_bins)
kernel void correlate_1D(
    const global DTYPE_SUMS* sums,
    global float* output
) {
    uint tau = get_global_id(0);
    uint q = get_global_id(1);
    if (tau >= N_FRAMES) return;
    if (q >= NUM_BINS) return;

    // Using integers for correlation computation avoids float32 loss of precision,
    // But is error-prone if many pixels have a high value
    #if COR_USE_INT_COMPUTATION == 1
    uint s = 0;
    #else
    float s = 0.0f;
    #endif
    for (int t = tau; t < N_FRAMES; t++) {
        s += sums[q * N_FRAMES + t] * sums[q * N_FRAMES + t - tau];
    }
    output[q*N_FRAMES + tau] = s;
}


