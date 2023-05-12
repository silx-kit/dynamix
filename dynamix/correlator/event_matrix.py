from time import perf_counter
import numpy as np
from pyopencl import LocalMemory
import pyopencl.array as parray
from ..utils import get_opencl_srcfile
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
        self.d_corr_matrix = parray.zeros(self.queue, self.correlation_matrix_flat_size, np.uint32) # TODO dtype


    def _setup_kernels(self):
        kernel_files = list(map(get_opencl_srcfile, self.kernel_files))
        self.compile_kernels(
            kernel_files=kernel_files,
            compile_options=[
                "-DDTYPE=%s" % self.c_dtype,
            ]
        )
        self.build_correlation_matrix_kernel = self.kernels.get_kernel("build_correlation_matrix_flattened_wg")

        self.grid = (self.nframes, self.n_times)
        self.wg = (32, 1) # None # tune ?




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

        return self.d_corr_matrix.get()
