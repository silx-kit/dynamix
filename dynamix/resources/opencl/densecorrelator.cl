// This is a general-purpose kernel for "dense" data.
// If IMAGE_WIDTH > 1024, the kernel will have to be adapted
// (unlikely since the XPCS data is dense only in a reduced ROI)
// WG threads horizontally (must have WG >= NUM_BINS), ideally WG >= IMAGE_WIDTH
// Nt thread vertically
kernel void correlator_multiQ_dense(
    const global char* frames,
    const global int* q_mask,
    const global float* norm_mask,
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
        output[tid*Nt + tau] = sum_p[tid];
}
