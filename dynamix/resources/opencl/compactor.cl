/* Compact some frames into the sparse data-structure.
 * 
 * 
 */ 

/* Launched with (image_width, image_height) grid of threads.
 *  One thread handle one line of events, so threads are indexes by frame pixel indices.
 *
 * Parameters:
 * slab: contains a sub-stack of input images
 * timestamp_first and timestamp_last are the indices of begin and end of the sub-stack.
 * image_width and image_height contain the size of the images
 * nnz is the maximum expected number of events in a pixel
 * times_array is am (y,x,nnz) array with the timestamps of pixels with signal
 * data_array is am (y,x,nnz) array with the values associated with times_array
 * counter is an (y,x) array with the number of active points found in the stack
 * 
 */
kernel void compactor1( const global DTYPE* slab,
                        const int timestamp_first,
                        const int timestamp_last,
                        const int image_width,
                        const int image_height,
                        const int nnz,
                        global DTYPE* data_array,
                        global int* times_array,
                        global uint* counter){
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    if ((x >= image_width) || (y >= image_height)){
        return;
    }
    uint pos = y*image_width + x;
    for (int k=0; k<timestamp_last-timestamp_first; k++){
        DTYPE value = slab[k*image_width*image_height+pos];
        if (value>0){
            uint cnt = counter[pos]++;
            if (cnt<nnz){
                times_array[cnt+nnz*pos] = k +  timestamp_first;
                data_array[cnt+nnz*pos] = value;
            }
        }
    }
}
/*Launched with (image_width, image_height, 32) and (1,1,32) for the block-size
 * One workgroup handles the copy of a complete pixel
 * 
 * 
 * TODO: implement !
 */
kernel void compactor2(                        
        const int image_width,
        const int image_height,
        const int nnz,
        global int* times_array,
        global DTYPE* data_array,
        const global uint* pixel_ptr){
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    if ((x >= image_width) || (y >= image_height)){
        return;
    }
    uint pos = y*image_width + x;
    
}