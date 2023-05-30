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
        extra_options={}, ctx=None, devicetype="all", platformid=None,
        deviceid=None, block_size=None, memory=None, profile=False
    ):
        """
        """
        super().__init__(
            shape, nframes, qmask=qmask, dtype=dtype, weights=weights,
            scale_factor=scale_factor, extra_options=extra_options,
            ctx=ctx, devicetype=devicetype, platformid=platformid,
            deviceid=deviceid, block_size=block_size, memory=memory,
            profile=profile
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
            ]
        )
        self.build_correlation_matrix_kernel = self.kernels.get_kernel("build_correlation_matrix_flattened_wg")
        self.build_correlation_matrix_image = self.kernels.get_kernel("build_correlation_matrix_image")
        self.build_correlation_matrix_times_representation = self.kernels.get_kernel("build_correlation_matrix_times_representation")
        self.space_compact_to_time_compact_kernel = self.kernels.get_kernel("space_compact_to_time_compact")
        self.space_compact_to_time_compact_stage2_kernel = self.kernels.get_kernel("space_compact_to_time_compact_stage2_sort")

        wg_size = 16 # Tune ?
        self.wg = (wg_size, 1) # None
        self.grid = [int(x) for x in [self.n_times , self.nframes]] # sanitize
        # self.grid[0] = updiv(updiv(self.n_times, wg_size), wg_size)*wg_size




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


    def build_correlation_matrix(self, data, pixel_indices, offsets):
        # self._check_event_arrays(self, data, pixel_indices, offsets) # TODO
        # self._set_data({
        #     "vol_times": vol_times,
        #     "vol_data": vol_data,
        #     "offsets": offsets
        # })

        # self.d_res.fill(0)
        # self.d_sums.fill(0)

        d_data = self.d_data = parray.to_device(self.queue, data.astype(np.uint8))
        d_pixel_indices = parray.to_device(self.queue, pixel_indices.astype(np.uint32))
        d_offsets = parray.to_device(self.queue, offsets.astype(np.uint32))

        t0 = perf_counter()
        qmask_compacted = self._get_compacted_qmask(pixel_indices, offsets)
        print("_get_compacted_qmask: ", (perf_counter() - t0) * 1e3)
        d_qmask_compacted = parray.to_device(self.queue, qmask_compacted.astype(np.int8)) # !

        t0 = perf_counter()
        evt = self.build_correlation_matrix_kernel(
            self.queue,
            self.grid,
            self.wg,
            d_data.data,
            d_pixel_indices.data,
            d_offsets.data,
            d_qmask_compacted.data,
            self.d_corr_matrix.data,
            np.int32(self.nframes),
            np.int32(self.n_times),
            # np.int32(self.n_bins)
            np.int32(1),
            LocalMemory(4096 * 1),
            LocalMemory(4096 * 1),
        )
        evt.wait()
        self.profile_add(evt, "Event correlator")
        print("build correlation matrix:", (perf_counter() - t0) * 1e3)

        # self._reset_arrays(["vol_times", "vol_data", "offsets"])





    def build_correlation_matrix_v2(self, data, pixel_indices, offsets):
        # wg = None
        # grid = (11*1024, self.nframes, 1)
        wg = (512, 1)
        grid = (updiv(11000, wg[0])*wg[0], self.nframes)
        print(grid)

        d_data = self.d_data = parray.to_device(self.queue, data.astype(np.uint8))
        d_pixel_indices = parray.to_device(self.queue, pixel_indices.astype(np.uint32))
        d_offsets = parray.to_device(self.queue, offsets.astype(np.uint32))
        self.d_sums = parray.zeros(self.queue, self.nframes, np.uint32) # TODO dtype


        t0 = perf_counter()
        qmask_compacted = self._get_compacted_qmask(pixel_indices, offsets)
        print("_get_compacted_qmask: ", (perf_counter() - t0) * 1e3)
        d_qmask_compacted = parray.to_device(self.queue, qmask_compacted.astype(np.int8)) # !

        t0 = perf_counter()
        evt = self.build_correlation_matrix_image(
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
        self.profile_add(evt, "Event correlator")
        print("build correlation matrix [binary_search]:", (perf_counter() - t0) * 1e3)

        # self._reset_arrays(["vol_times", "vol_data", "offsets"])
        return self.d_corr_matrix.get()


    def build_correlation_matrix_times(self, data, times, offsets):
        wg = None
        grid = tuple(self.shape[::-1])

        d_data = parray.to_device(self.queue, data)
        d_times = parray.to_device(self.queue, times)
        d_offsets = parray.to_device(self.queue, offsets)

        self.d_corr_matrix.fill(0)
        self.d_sums.fill(0)

        t0 = perf_counter()
        evt = self.build_correlation_matrix_times_representation(
            self.queue,
            grid,
            wg,
            d_times.data,
            d_data.data,
            d_offsets.data,
            self.d_qmask.data,
            self.d_corr_matrix.data,
            self.d_sums.data,
            np.int32(self.shape[0]),
            np.int32(self.nframes),
            np.int32(self.n_times),
        )
        evt.wait()
        self.profile_add(evt, "Event correlator")
        print("build correlation matrix (times repr.):", (perf_counter() - t0) * 1e3)

        # self._reset_arrays(["vol_times", "vol_data", "offsets"])
        return self.d_corr_matrix.get()



    def space_compact_to_time_compact(self, data, pixel_indices, offsets):
        wg = None
        grid = (11000, 1)

        d_data = self.d_data = parray.to_device(self.queue, data.astype(np.uint8))
        d_pixel_indices = parray.to_device(self.queue, pixel_indices.astype(np.uint32))
        d_offsets = parray.to_device(self.queue, offsets.astype(np.uint32))

        # TODO estimate "max_nnz"

        d_t_data = parray.zeros(self.queue, self._max_nnz * np.prod(self.shape), np.uint8)
        d_t_times = parray.zeros(self.queue, self._max_nnz * np.prod(self.shape), np.uint32)
        d_counter = parray.zeros(self.queue, np.prod(self.shape), np.uint32)


        t0 = perf_counter()
        evt = self.space_compact_to_time_compact_kernel(
            self.queue,
            grid,
            wg,

            d_data.data,
            d_pixel_indices.data,
            d_offsets.data,

            d_t_data.data,
            d_t_times.data,
            d_counter.data,

            np.int32(self.nframes),
            np.int32(self.shape[1]),
            np.int32(self.shape[0]),
        )
        evt.wait()
        self.profile_add(evt, "Event correlator")
        print("re-compact data:", (perf_counter() - t0) * 1e3)

        # self._reset_arrays(["vol_times", "vol_data", "offsets"])
        return d_t_data.get(), d_t_times.get(), d_counter.get()




    def space_compact_to_time_compact_v2(self, data, pixel_indices, offsets):
        wg = None
        grid = self.shape[::-1]

        d_data = self.d_data = parray.to_device(self.queue, data.astype(np.uint8))
        d_pixel_indices = parray.to_device(self.queue, pixel_indices.astype(np.uint32))
        d_offsets = parray.to_device(self.queue, offsets.astype(np.uint32))

        d_t_data = parray.zeros(self.queue, self._max_nnz * np.prod(self.shape), np.uint8)
        d_t_times = parray.zeros(self.queue, self._max_nnz * np.prod(self.shape), np.uint32)
        d_counter = parray.zeros(self.queue, np.prod(self.shape), np.uint32)


        t0 = perf_counter()
        evt = self.space_compact_to_time_compact_kernel_v2(
            self.queue,
            grid,
            wg,

            d_data.data,
            d_pixel_indices.data,
            d_offsets.data,
            self.d_qmask.data,

            d_t_data.data,
            d_t_times.data,
            d_counter.data,

            np.int32(self.nframes),
            np.int32(self.shape[1]),
            np.int32(self.shape[0]),
        )
        evt.wait()
        self.profile_add(evt, "Event correlator")
        print("re-compact data (v2):", (perf_counter() - t0) * 1e3)

        # self._reset_arrays(["vol_times", "vol_data", "offsets"])
        return d_t_data.get(), d_t_times.get(), d_counter.get()



    def space_compact_to_time_compact_stage2(self, t_data_tmp, t_times_tmp, t_counter):

        t0 = perf_counter()
        offsets = np.hstack([np.array([0], dtype=np.uint32), np.cumsum(t_counter, dtype=np.uint32)])
        print("[space->times stage 2] cumsum:", (perf_counter() - t0) * 1e3)

        d_t_data_tmp = parray.to_device(self.queue, t_data_tmp)
        d_t_times_tmp = parray.to_device(self.queue, t_times_tmp)

        d_t_offsets = parray.to_device(self.queue, offsets)
        d_t_data = parray.zeros(self.queue, offsets[-1], t_data_tmp.dtype)
        d_t_times = parray.zeros(self.queue, d_t_data.size, np.uint32)

        wg = None
        grid = (np.prod(self.shape), 1)

        t0 = perf_counter()
        evt = self.space_compact_to_time_compact_stage2_kernel(
            self.queue,
            grid,
            wg,

            d_t_data_tmp.data,
            d_t_times_tmp.data,
            d_t_offsets.data,

            d_t_data.data,
            d_t_times.data,

            np.int32(np.prod(self.shape)),
        )
        evt.wait()
        self.profile_add(evt, "Event correlator")
        print("[space->times stage 2] kernel:", (perf_counter() - t0) * 1e3)

        return d_t_data.get(), d_t_times.get(), offsets


