#ifndef DTYPES_H
#define DTYPES_H

#ifndef DTYPE
  #define DTYPE uchar
#endif

#ifndef OFFSET_DTYPE
  // must be extended to unsigned long if total_nnz > 4294967295
  #define OFFSET_DTYPE uint
#endif

#ifndef QMASK_DTYPE
  // must be extended to int if number of q-bins > 127
  #define QMASK_DTYPE char
#endif

#ifndef RES_DTYPE
  // must be extended to unsigned long for large nnz_per_frame and/or events counts
  #define RES_DTYPE uint
#endif

#endif 