/*
    Kernel re-ordering the pixel in each slice so that they are contiguous
    ======================================================================

Pixel with qmask=0 are discarded.

Size of the kernel: 3D 
    dim0: Collaborative threads writing
    dim1: qbin  groupsize: 1
    dim2: tau   groupsize: 1

Constrains:
  Collaborative writing
  No shared memory
  No constrains on workgroup size

Parameters:
        frames: 3D array, stack of SAXS frames
        q_mask_ptr: 1D array with indexes of start of qbin in q_mask_pixels
        q_mask_pixels: 1d array with indexes of pixel contributing in qbin
        Nt: number of time-steps recorded
        nbin: number of bins in mask
        image_size: width*height
        output: 3D array, stack of SAXS frames with pixel re-ordered.

*/
kernel void multiQ_reorder(    const global DTYPE* frames,
                               const global int* q_mask_ptr,
                               const global int* q_mask_pixels,
                               const int Nt,
                               const int nbin,
                               const int image_size,
                               global DTYPE* output){
    const uint tid = get_local_id(0);
    const uint ws = get_local_size(0);
    const uint qbin = get_global_id(1);
    const uint tau = get_global_id(2);
    
    // Barrier for thread index: bin0 is discarded
    if (tau >= Nt) return;
    if (qbin == 0) return; 
    if (qbin >= nbin) return;
    
    const int start = q_mask_ptr[qbin];
    const int stop = q_mask_ptr[qbin+1];
    const int npix = stop - start;
    const int offset = q_mask_ptr[1];
    const int nb_pix = q_mask_ptr[nbin] - offset; // number of (valid) pixels in a frame
    
    for (int idx = start+tid; idx<stop; idx+=ws) {
        output[tau*nb_pix + idx - offset] = frames[tau*image_size + q_mask_pixels[idx]];
    }
}

/*

  Collaborative implementation of the dense correlator
  ====================================================

Size of the kernel: 
    dim0: collaborative read/reduction workgoup size: 128 
    dim1: qbin  groupsize: 1
    dim2: tau   groupsize: 1

Constrains:
      Collaboration between treads in dim0, expected WG threads.
      Workgroup size should be at least the size of a memory transaction and limited by the amount of shared memory
      Shared memory: 3x WG * sizeof(uint)
      Requires double-precision floating point unit

Parameters:
        frames: 3D array, stack of SAXS frames
        q_mask_ptr: 1D array with indexes of start of qbin in q_mask_pixels
        output_mean: 2d array (qbin, tau) with XPCS result
        output_stc: 2d array (qbin, tau) with deviation of XPCS result
        Nt: number of time-steps recorded
        nbin: Number of bins in qbin mask

References: 
    Variance propagation implemented according to doi:10.1145/3221269.3223036
*/

#ifndef SUM_WG_SIZE
#define SUM_WG_SIZE 1024
#endif

// Sum-up all elements in 3 arrays, reset themt, and return the sum of each of them
inline DTYPE_SUMS3 summed3(local DTYPE_SUMS* ary1,
                     local DTYPE_SUMS* ary2,
                     local DTYPE_SUMS* ary3){
    uint wg = get_local_size(0);
    const uint tid = get_local_id(0);
    barrier(CLK_LOCAL_MEM_FENCE);
    while (wg>1){
        wg /= 2;
        if (tid<wg){
            ary1[tid] += ary1[tid+wg];
            ary2[tid] += ary2[tid+wg];
            ary3[tid] += ary3[tid+wg];
        }
        if (wg>32)
            barrier(CLK_LOCAL_MEM_FENCE);
    }
    DTYPE_SUMS3 value = (DTYPE_SUMS3)(ary1[0], ary2[0], ary3[0]);
    barrier(CLK_LOCAL_MEM_FENCE);
    ary1[tid] = 0;
    ary2[tid] = 0;
    ary3[tid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    return value;
}


kernel void correlator_multiQ_ordered(
                                    const global DTYPE* frames,
                                    const global int* q_mask_ptr,
                                    global float* output_mean,
                                    global float* output_std,
                                    const int Nt,
                                    const int nbin) {
    const uint qbin = get_global_id(1);
    const uint tau = get_global_id(2);
    const uint tid = get_local_id(0);
    const uint ws = get_local_size(0);
    // Barrier for thread index
    if (tau >= Nt) return;
    if (qbin == 0) return;
    if (qbin >= nbin) return;
    if (ws>WG){
        if (tid==0){
            printf("Actual workgroup size %d is larger than allocated memory %s\n",ws, WG);
        }
        return;
    }
    const int offset = q_mask_ptr[1];
    const int nb_pix = q_mask_ptr[nbin] - offset;
    const int start = q_mask_ptr[qbin] - offset;
    const int stop = q_mask_ptr[qbin+1] -offset;
    const int npix = stop - start;
    
    ulong dia_n_sum = 0;
    double dia_d_sum = 0.0;
    
    //Shared arrays
    local DTYPE_SUMS shared_sum1[WG];
    local DTYPE_SUMS shared_sum2[WG];
    local DTYPE_SUMS shared_dia_n_val[WG];
    shared_sum1[tid] = 0;
    shared_sum2[tid] = 0;
    shared_dia_n_val[tid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    double sumV = 0.0;  // sum of values
    double varV = 0.0;  // sum on variance contribution
    uint cnt = 0;       // sum of weights, i.e counter
    for (uint t = tau; t < Nt; t++) {
        DTYPE_SUMS sum1 = 0;
        DTYPE_SUMS sum2 = 0;
        DTYPE_SUMS dia_n_val = 0;
        for (uint idx = start+tid; idx < stop; idx+=ws) {
            uint val1 = frames[t*nb_pix + idx];
            uint val2 = frames[(t-tau)*nb_pix + idx];
            shared_dia_n_val[tid] += val1 * val2;
            shared_sum1[tid] += val1;
            shared_sum2[tid] += val2;
        }
        DTYPE_SUMS3 partial_sum = summed3(shared_dia_n_val, shared_sum1, shared_sum2);
        dia_n_val = partial_sum.s0;
        sum1 = partial_sum.s1;
        sum2 = partial_sum.s2;
        
        if (tid==0){
            double dia_d_val = 1.0 * sum1 * sum2 / (npix * npix);

            dia_n_sum += dia_n_val;
            dia_d_sum += dia_d_val;

            //Variance propagation
            double val = dia_n_val / (dia_d_val * npix);
            if (cnt>0){
                double num = sumV - cnt*val;
                varV += num*num / (cnt*(cnt+1));
            }
            sumV += val;        
            cnt++;
        }
    }
    if (tid==0){
        uint pos = (qbin-1)*Nt + tau;
        output_mean[pos] = dia_n_sum * 1.0 / (dia_d_sum * npix);
        output_std[pos] = sqrt(varV) / cnt;
    }
}
