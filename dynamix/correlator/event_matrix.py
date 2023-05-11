from time import perf_counter
import numpy as np
import pyopencl.array as parray
from ..utils import get_opencl_srcfile
from .common import OpenclCorrelator

class MatrixEventCorrelator(OpenclCorrelator):

    kernel_files = ["matrix_evtcorrelator.cl"]

    """
    A class to compute the correlation function for sparse XPCS data.
    """

    def __init__(
        self, shape, nframes,
        qmask=None, dtype="f", weights=None, scale_factor=None,
        extra_options={}, ctx=None, devicetype="all", platformid=None,
        deviceid=None, block_size=None, memory=None, profile=False
    ):
        """
        Initialize an EventCorrelator instance.

        Parameters
        -----------
        Please refer to the documentation of dynamix.correlator.common.OpenclCorrelator

        Specific parameters
        --------------------
        max_events_count: int
            Expected maximum number of events (non-zero values) in the frames
            along the time axis. If `frames_stack` is a numpy.ndarray of shape
            `(n_frames, n_y, n_x)`, then `total_events_count` can be computed as

            ```python
            (frames_stack > 0).sum(axis=0).max()
            ```
        total_events_count: int, optional
            Expected total number of events (non-zero values) in all the frames.
            If `frames_stack` is a numpy.ndarray of shape
            `(n_frames, n_y, n_x)`, then `total_events_count` can be computed as

            ```python
            (frames_stack > 0).sum()
            ```
        """
        super().__init__(
            shape, nframes, qmask=qmask, dtype=dtype, weights=weights,
            scale_factor=scale_factor, extra_options=extra_options,
            ctx=ctx, devicetype=devicetype, platformid=platformid,
            deviceid=deviceid, block_size=block_size, memory=memory,
            profile=profile
        )
        # self._allocate_events_arrays()
        self._setup_kernels()



    # def _allocate_events_arrays(self, is_reallocating=False):
    #     tot_nnz = self.total_events_count

    #     self.d_vol_times = parray.zeros(self.queue, tot_nnz, dtype=np.int32)
    #     self.d_vol_data = parray.zeros(self.queue, tot_nnz, dtype=self.dtype)
    #     self.d_offsets = parray.zeros(self.queue, np.prod(self.shape)+1, dtype=np.uint32)

    #     self._old_d_vol_times = None
    #     self._old_d_vol_data = None
    #     self._old_d_offsets = None

    #     if not(is_reallocating):
    #         self.d_res_int = parray.zeros(self.queue, self.output_shape, dtype=np.int32)
    #         self.d_sums = parray.zeros(self.queue, self.output_shape, np.uint32)
    #         self.d_res = parray.zeros(self.queue, self.output_shape, np.float32)
    #         self.d_scale_factors = parray.to_device(self.queue, np.array(list(self.scale_factors.values()), dtype=np.float32))



    def _setup_kernels(self):
        kernel_files = list(map(get_opencl_srcfile, self.kernel_files))
        self.compile_kernels(
            kernel_files=kernel_files,
            compile_options=[
                "-DDTYPE=%s" % self.c_dtype,
            ]
        )
        self.build_correlation_matrix_kernel = self.kernels.get_kernel("build_correlation_matrix")

        self.grid = (self.nframes, 1)
        self.wg = None # tune ?

        self.d_corr_matrix = parray.zeros(self.queue, (self.nframes, self.nframes), np.uint32)


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
            # np.int32(self.n_bins)
            np.int32(1)
        )
        evt.wait()
        self.profile_add(evt, "Event correlator")
        print("build correlation matrix:", (perf_counter() - t0) * 1e3)

        # self._reset_arrays(["vol_times", "vol_data", "offsets"])

        return self.d_corr_matrix.get()
