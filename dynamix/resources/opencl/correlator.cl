#ifndef IMAGE_WIDTH
  #error "Please define IMAGE_WIDTH parameter"
#endif

#ifndef DTYPE
  #error "Please define DTYPE parameter"
#endif

#ifndef IDX_DTYPE
  #error "Please define IDX_DTYPE parameter"
#endif


#define OUT_DTYPE float



// Along dimension Y, one thread handles one "tau".
// Along dimension X, one *group* of threads handles one product.
kernel void correlator_oneQ_Nt(
    const global DTYPE* data,
    const global IDX_DTYPE* indices,
    const global IDX_DTYPE* indptr,
    const global IDX_DTYPE* offsets,
    const global OUT_DTYPE* norm_mask,
    const global OUT_DTYPE* images_sums,
    global OUT_DTYPE* output,
    int image_height,
    int Nt
) {
    uint tid = get_local_id(0);
    uint tau = get_global_id(1);
    if (tau >= Nt) return;

    // Local mem
    local uint image_sum[1];
    local IDX_DTYPE indices1_line[IMAGE_WIDTH];
    local DTYPE data1_line[IMAGE_WIDTH];
    local IDX_DTYPE indices2_line[IMAGE_WIDTH];
    local DTYPE data2_line[IMAGE_WIDTH];
    //~ local OUT_DTYPE norm_mask[IMAGE_WIDTH];

    // Memset sum(images[t]*images[t-tau])
    OUT_DTYPE normalization = 0;
    if (tid == 0) {
        image_sum[0] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // One thread (group) handles one couple (t, t-tau)
    for (int t = tau; t < Nt; t++) {
        // Get images[t-tau]
        IDX_DTYPE offset1 = offsets[t-tau];
        const global DTYPE* data1 = data + offset1;
        const global IDX_DTYPE* indices1 = indices + offset1;
        const global IDX_DTYPE* indptr1 = indptr + (image_height+1)*(t-tau);
        // Get images[t]
        IDX_DTYPE offset2 = offsets[t];
        const global DTYPE* data2 = data + offset2;
        const global IDX_DTYPE* indices2 = indices + offset2;
        const global IDX_DTYPE* indptr2 = indptr + (image_height+1)*t;

        // Loop over lines of (images[tau]*images[t-tau])
        for (int row_idx = 0; row_idx < image_height; row_idx++) {
            // masking is not implemented yet
            //~ OUT_DTYPE weight = norm_mask[row_idx * IMAGE_WIDTH + tid]; //norm_mask_line[ind1]; // TODO check if shared mem can speed-up the access
            // data[start:end] is the current image line
            IDX_DTYPE start1 = indptr1[row_idx], end1 = indptr1[row_idx+1];
            IDX_DTYPE start2 = indptr2[row_idx], end2 = indptr2[row_idx+1];

            // Fetch one line of image into shared
            // Hopefully this alleviates the bad access patterns when doing
            // sparse elementwise product, but this comes at the expense
            // of the use of synchronization barrriers
            if (start1 + tid < end1) {
                data1_line[tid] = data1[start1+tid];
                indices1_line[tid] = indices1[start1+tid];
            }
            if (start2 + tid < end2) {
                data2_line[tid] = data2[start2+tid];
                indices2_line[tid] = indices2[start2+tid];
            }
            barrier(CLK_LOCAL_MEM_FENCE);


            // Do the elementwise multiplication.
            // Each local thread, having image1[i, j], will fetch image2[i, j]
            if (start1 + tid < end1) {
                // column index "j"
                IDX_DTYPE ind1 = indices1_line[tid];
                DTYPE val1 = data1_line[tid]; // always nonzero
                DTYPE val2 = 0; // check if DTYPE if float32

                // In CSR format, column indices are sorted - use bisection
                IDX_DTYPE low = 0, high=end2-start2-1;
                while (low <= high) {
                    IDX_DTYPE mid = ((low+high) >> 1);
                    IDX_DTYPE ind2 = indices2_line[mid];

                    if (ind2 > ind1) high=mid-1;
                    else if (ind2 < ind1) low=mid+1;
                    else {
                        val2 = data2_line[mid];
                        break;
                    }
                }
                DTYPE val = val1*val2; // must be int
                atomic_add(image_sum, val);

            } // start1+tid < end1

            // At this point, the workgroup finished processing one line of image1 * image2
            // Wait for other thread to finish their line
            barrier(CLK_LOCAL_MEM_FENCE);

        } // End loop over row_idx (rows of images[t])

        if (tid == 0) normalization += images_sums[t]*images_sums[t-tau];


    } // End loop over "t" (images[t])

    if (tid == 0) {
        output[tau] = image_sum[0] * 1.0f / normalization;
    }

}
