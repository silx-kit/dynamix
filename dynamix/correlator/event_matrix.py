from os import path
from time import perf_counter
import numpy as np
from pyopencl import LocalMemory
import pyopencl.array as parray
from ..utils import get_opencl_srcfile, updiv
from .common import OpenclCorrelator

class MatrixEventCorrelator(OpenclCorrelator):

    kernel_files = ["matrix_evtcorrelator.cl"]

    """
    A class to compute the correlation function for sparse XPCS data.
    """

    def __init__(
        self, shape, nframes, n_times=None,
        qmask=None, dtype="f", weights=None, scale_factor=None,
        extra_options={}, **oclprocessing_kwargs
    ):
        """
        """
        super().__init__(
            shape, nframes, qmask=qmask, dtype=dtype, weights=weights,
            scale_factor=scale_factor, extra_options=extra_options,
            **oclprocessing_kwargs
        )
        self._allocate_events_arrays(n_times)
        self._setup_kernels()


    def _allocate_events_arrays(self, n_times):
        if n_times is None:
            n_times = self.nframes
        self.n_times = n_times
        self.correlation_matrix_full_shape = (self.nframes, self.n_times)
        if (self.nframes * (self.n_times + 1) % 2) != 0:
            print("Warning: incrementing n_times to have a proper-sized matrix")
            self.n_times += 1
        self.correlation_matrix_flat_size = self.nframes * (self.n_times + 1) // 2
        # TODO dtype + q-bin selection if needed (to reduce array size)
        self.d_corr_matrix = parray.zeros(self.queue, (self.n_bins, self.correlation_matrix_flat_size), np.uint32)


    def _setup_kernels(self):
        self._max_nnz = 250 # TODO
        kernel_files = list(map(get_opencl_srcfile, self.kernel_files))
        self.compile_kernels(
            kernel_files=kernel_files,
            compile_options=[
                "-DDTYPE=%s" % self.c_dtype,
                "-DMAX_EVT_COUNT=%d" % self._max_nnz,
                "-I%s" % path.dirname(get_opencl_srcfile("dtypes.h")),
                "-DSHARED_ARRAYS_SIZE=%d" % 11000,
            ]
        )
        self.build_correlation_matrix_kernel_v1 = self.kernels.get_kernel("build_correlation_matrix_v1")
        self.build_correlation_matrix_kernel_v2 = self.kernels.get_kernel("build_correlation_matrix_v2")
        self.build_correlation_matrix_kernel_v2b = self.kernels.get_kernel("build_correlation_matrix_v2b")
        self.build_correlation_matrix_kernel_v3 = self.kernels.get_kernel("build_correlation_matrix_v3")
        self.build_correlation_matrix_kernel_times = self.kernels.get_kernel("build_correlation_matrix_times_representation")

        # wg_size = 16 # Tune ?
        # self.wg = (wg_size, 1) # None
        # self.grid = [int(x) for x in [self.n_times , self.nframes]] # sanitize
        # # self.grid[0] = updiv(updiv(self.n_times, wg_size), wg_size)*wg_size


    def _get_compacted_qmask(self, pixel_indices, offsets):
        # compaction of qmask (duplicates information!)
        # qmask_compacted[pixel_compact_idx] give the associated q-bin
        qmask_compacted = []
        qmask1D = self.qmask.ravel()
        for frame_idx in range(self.nframes):
            i_start = offsets[frame_idx]
            i_stop = offsets[frame_idx+1]
            qmask_compacted.append(qmask1D[pixel_indices[i_start:i_stop]])
        qmask_compacted = np.hstack(qmask_compacted)
        return qmask_compacted
        #

    def build_correlation_matrix_v1(self, data, pixel_indices, offsets):
        qmask_compacted = self._get_compacted_qmask(pixel_indices, offsets)
        d_qmask_compacted = parray.to_device(self.queue, qmask_compacted.astype(np.int8)) # !

        # TODO data setter
        d_data = self.d_data = parray.to_device(self.queue, data.astype(np.uint8))
        d_pixel_indices = parray.to_device(self.queue, pixel_indices.astype(np.uint32))
        d_offsets = parray.to_device(self.queue, offsets.astype(np.uint32))
        self.d_sums = parray.zeros(self.queue, self.nframes, np.uint32) # TODO dtype
        #

        wg = None
        grid = (self.n_times, self.n_bins)

        evt = self.build_correlation_matrix_kernel_v1(
            self.queue,
            grid,
            wg,
            d_data.data,
            d_pixel_indices.data,
            d_offsets.data,
            d_qmask_compacted.data,
            self.d_corr_matrix.data,
            # self.d_sums.data,
            np.int32(self.nframes),
            np.int32(self.n_times),
            np.int32(1),
        )
        evt.wait()
        self.profile_add(evt, "Build matrix correlation (v1)")

        return self.d_corr_matrix.get()


    def build_correlation_matrix_v2(self, data, pixel_indices, offsets):
        qmask_compacted = self._get_compacted_qmask(pixel_indices, offsets)
        d_qmask_compacted = parray.to_device(self.queue, qmask_compacted.astype(np.int8)) # !

        # TODO data setter
        d_data = self.d_data = parray.to_device(self.queue, data.astype(np.uint8))
        d_pixel_indices = parray.to_device(self.queue, pixel_indices.astype(np.uint32))
        d_offsets = parray.to_device(self.queue, offsets.astype(np.uint32))
        self.d_sums = parray.zeros(self.queue, self.nframes, np.uint32) # TODO dtype
        #

        wg = None
        grid = (self.n_times, self.nframes)

        evt = self.build_correlation_matrix_kernel_v2(
            self.queue,
            grid,
            wg,
            d_data.data,
            d_pixel_indices.data,
            d_offsets.data,
            d_qmask_compacted.data,
            self.d_corr_matrix.data,
            # self.d_sums.data,
            np.int32(self.nframes),
            np.int32(self.n_times),
            np.int32(1), # current q-bin
        )
        evt.wait()
        self.profile_add(evt, "Build matrix correlation (v2)")

        return self.d_corr_matrix.get()



    def build_correlation_matrix_v2b(self, data, pixel_indices, offsets):
        qmask_compacted = self._get_compacted_qmask(pixel_indices, offsets)
        d_qmask_compacted = parray.to_device(self.queue, qmask_compacted.astype(np.int8)) # !

        # TODO proper set_data
        d_data = self.d_data = parray.to_device(self.queue, data.astype(np.uint8))
        d_pixel_indices = parray.to_device(self.queue, pixel_indices.astype(np.uint32))
        d_offsets = parray.to_device(self.queue, offsets.astype(np.uint32))
        #

        wg = (1024, 1) # TODO tune
        grid = (updiv(self.n_times, wg[0])*wg[0], self.nframes)

        evt = self.build_correlation_matrix_kernel_v2b(
            self.queue,
            grid,
            wg,
            d_data.data,
            d_pixel_indices.data,
            d_offsets.data,
            d_qmask_compacted.data,
            self.d_corr_matrix.data,
            np.int32(self.nframes),
            np.int32(self.n_times),
            # np.int32(self.n_bins)
            np.int32(1),
            LocalMemory(11000 * 1),
            LocalMemory(11000 * 1),
        )
        evt.wait()
        self.profile_add(evt, "Build correlation matrix (v2b)")

        return self.d_corr_matrix.get()






    def build_correlation_matrix_v3(self, data, pixel_indices, offsets):
        qmask_compacted = self._get_compacted_qmask(pixel_indices, offsets)
        d_qmask_compacted = parray.to_device(self.queue, qmask_compacted.astype(np.int8)) # !

        # TODO data setter
        d_data = self.d_data = parray.to_device(self.queue, data.astype(np.uint8))
        d_pixel_indices = parray.to_device(self.queue, pixel_indices.astype(np.uint32))
        d_offsets = parray.to_device(self.queue, offsets.astype(np.uint32))
        self.d_sums = parray.zeros(self.queue, self.nframes, np.uint32) # TODO dtype
        #

        max_nnz_space = np.diff(offsets).max()
        wg = (512, 1) # TODO tune
        grid = (updiv(max_nnz_space, wg[0])*wg[0], self.nframes)

        evt = self.build_correlation_matrix_kernel_v3(
            self.queue,
            grid,
            wg,
            d_data.data,
            d_pixel_indices.data,
            d_offsets.data,
            d_qmask_compacted.data,
            self.d_corr_matrix.data,
            self.d_sums.data,
            np.int32(self.nframes),
            np.int32(self.n_times),
            np.int32(-1),
        )
        evt.wait()
        self.profile_add(evt, "Build matrix correlation (v3)")

        return self.d_corr_matrix.get()


    def build_correlation_matrix_times(self, data, times, offsets):
        wg = None
        grid = tuple(self.shape[::-1])

        # TODO data setter
        d_data = parray.to_device(self.queue, data)
        d_times = parray.to_device(self.queue, times)
        d_offsets = parray.to_device(self.queue, offsets)

        self.d_sums = parray.zeros(self.queue, self.nframes, np.uint32) # TODO dtype
        #

        self.d_corr_matrix.fill(0)
        self.d_sums.fill(0)

        evt = self.build_correlation_matrix_kernel_times(
            self.queue,
            grid,
            wg,
            d_times.data,
            d_data.data,
            d_offsets.data,
            self.d_qmask.data,
            self.d_corr_matrix.data,
            self.d_sums.data,
            np.int32(self.shape[1]),
            np.int32(self.shape[0]),
            np.int32(self.nframes),
            np.int32(self.n_times),
        )
        evt.wait()
        self.profile_add(evt, "Build matrix correlation (times repr.)")

        return self.d_corr_matrix.get()

