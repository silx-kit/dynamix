#ifndef DTYPE
  #error "Please define DTYPE parameter"
#endif

#ifndef IDX_DTYPE
  #error "Please define IDX_DTYPE parameter"
#endif

#define OUT_DTYPE float

// Must be launched with worgroup size SUM_WG_SIZE horizontally,
// and with "Nt" threads vertically
kernel void compute_sums(
    const global DTYPE* data,
    const global IDX_DTYPE* offsets,
    global OUT_DTYPE* output,
    int Nt
) {
    uint tid = get_local_id(0);
    uint frame_id = get_global_id(1);
    if (frame_id > Nt) return;

    // Allocate (+memset) a shared buffer twice bigger than workgroup size
    local OUT_DTYPE s_buf[2*SUM_WG_SIZE];
    s_buf[tid] = 0;
    s_buf[tid + SUM_WG_SIZE] = 0;
    // One thread of horizontal workgroup will fetch "offset" to avoid read conflicts
    // s_offsets[0] = my_offset ; s_offsets[1]-s_offsets[0] = my_len
    local IDX_DTYPE s_offsets[2];
    if (tid <= 1) s_offsets[tid] = offsets[frame_id + tid];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Get data corresponding to current frame
    IDX_DTYPE my_offset = s_offsets[0];
    IDX_DTYPE my_len = s_offsets[1] - my_offset;
    const global DTYPE* my_data = data + my_offset;

    // Step 1: reduce global data to local data
    for (int offset = 0; tid + offset < my_len; offset += SUM_WG_SIZE) {
        s_buf[tid] += my_data[tid + offset];
    }
    barrier(CLK_LOCAL_MEM_FENCE);


    // Step 2: parallel reduction within shared memory
    #if SUM_WG_SIZE >= 1024
    if (tid < 1024) s_buf[tid] += s_buf[tid + 1024];
    barrier(CLK_LOCAL_MEM_FENCE);
    #endif

    #if SUM_WG_SIZE >= 512
    if (tid < 512) s_buf[tid] += s_buf[tid + 512];
    barrier(CLK_LOCAL_MEM_FENCE);
    #endif

    #if SUM_WG_SIZE >= 256
    if (tid < 256) s_buf[tid] += s_buf[tid + 256];
    barrier(CLK_LOCAL_MEM_FENCE);
    #endif

    #if SUM_WG_SIZE >= 128
    if (tid < 128) s_buf[tid] += s_buf[tid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);
    #endif

    #if SUM_WG_SIZE >= 64
    if (tid < 64) s_buf[tid] += s_buf[tid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);
    #endif

    // No synchronization needed on Nvidia hardware beyond this point
    if (tid < 32) {
        s_buf[tid] += s_buf[tid + 32];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid < 16) {
        s_buf[tid] += s_buf[tid + 16];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid < 8) {
        s_buf[tid] += s_buf[tid + 8];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid < 4) {
        s_buf[tid] += s_buf[tid + 4];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid < 2) {
        s_buf[tid] += s_buf[tid + 2];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        output[frame_id] = s_buf[tid] + s_buf[tid + 1];
    }
}
