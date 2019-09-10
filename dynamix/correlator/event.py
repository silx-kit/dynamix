import numpy as np
# from silx.opencl.common import pyopencl as cl
import pyopencl.array as parray
from ..utils import get_opencl_srcfile
from .common import OpenclCorrelator

class EventCorrelator(OpenclCorrelator):

    kernel_files = ["evtcorrelator.cl"]


    def __init__(
        self, shape, nframes, total_events_count, allow_reallocate=True,
        dtype="f", bins=0, weights=None, extra_options={},
        ctx=None, devicetype="all", platformid=None, deviceid=None,
        block_size=None, memory=None, profile=False
    ):
        """
        Initialize an EventCorrelator instance.

        Parameters
        -----------
        Please refer to the documentation of dynamix.correlator.common.OpenclCorrelator

        Specific parameters
        --------------------
        max_events_count: int
            Expected number of events (non-zero values) in all the frames.
            axis. If `frames_stack` is a numpy.ndarray of shape
            `(n_frames, n_y, n_x)`, then `total_events_count` can be computed as

            ```python
            (frames_stack > 0).sum()
            ```
        """
        super().__init__(
            shape, nframes, dtype=dtype, bins=bins, weights=weights,
            extra_options=extra_options,
            ctx=ctx, devicetype=devicetype, platformid=platformid,
            deviceid=deviceid, block_size=block_size, memory=memory,
            profile=profile
        )
        self._events_count = total_events_count
        self.allow_reallocate = allow_reallocate
        self._allocate_events_arrays()
        self._setup_kernels()


    def _setup_kernels(self):
        kernel_files = list(map(get_opencl_srcfile, self.kernel_files))
        self.compile_kernels(
            kernel_files=kernel_files,
            compile_options=[
                "-DIMAGE_WIDTH=%d" % self.shape[1],
                "-DDTYPE=%s" % self.c_dtype,
                "-DSUM_WG_SIZE=%d" % 1024, # TODO tune ?
                "-DMAX_EVT_COUNT=%d" % self.max_events_count,
                "-DSCALE_FACTOR=%f" % (1./(np.prod(self.shape))),
            ]
        )
        self.correlation_kernel = self.kernels.get_kernel("event_correlator_oneQ_simple") # TODO tune
        self.normalization_kernel = self.kernels.get_kernel("normalize_correlation_oneQ") # TODO tune

        self.grid = self.shape[::-1]
        self.wg = None # tune ?


    def _allocate_events_arrays(self, is_reallocating=False):
        tot_nnz = self._events_count

        self.d_vol_times = parray.zeros(self.queue, tot_nnz, dtype=np.int32)
        self.d_vol_data = parray.zeros(self.queue, tot_nnz, dtype=self.dtype)
        self.d_offsets = parray.zeros(self.queue, np.prod(self.shape)+1, dtype=np.uint32)

        if not(is_reallocating):
            self.d_res_int = parray.zeros(self.queue, self.nframes, dtype=np.int32)
            self.d_sums = parray.zeros(self.queue, self.nframes, np.uint32)
            self.d_res = parray.zeros(self.queue, self.nframes, np.float32)


    def _check_event_arrays(self, vol_times, vol_data, offsets):
        for arr in [vol_times, vol_data, offsets]:
            assert arr.ndim == 1
        assert vol_times.size == vol_data.size
        assert offsets.size == np.prod(self.shape) + 1
        numels = vol_data.size
        if numels > self._events_count:
            if self.allow_reallocate:
                self._events_count = numels
                self._allocate_events_arrays(is_reallocating=True)
            else:
                raise ValueError("Too many events and allow_reallocate was set to False")


    def correlate(self, vol_times, vol_data, offsets):
        self._check_event_arrays(vol_times, vol_data, offsets)
        self._set_data({
            "vol_times": vol_times,
            "vol_data": vol_data,
            "offsets": offsets
        })

        evt = self.correlation_kernel(
            self.queue,
            self.grid,
            self.wg,
            self.d_vol_times.data,
            self.d_vol_data.data,
            self.d_offsets.data,
            self.d_res_int.data,
            self.d_sums.data,
            np.int32(self.nframes)
        )
        evt.wait()
        self.profile_add(evt, "Event correlator")

        evt = self.normalization_kernel(
            self.queue,
            (self.nframes, 1),
            None, # tune wg ?
            self.d_res_int.data,
            self.d_res.data,
            self.d_sums.data,
            np.int32(self.nframes)
        )
        evt.wait()
        self.profile_add(evt, "Normalization")

        self._reset_arrays(["d_vol_times", "d_vol_data", "d_ctr"])

        return self.d_res.get()




    @staticmethod
    def build_events_structure(frames):
        """
        Helper function to build the events data structure from a stack of frames.
        It returns a tuple (data, times, offsets).
        """
        assert frames.ndim == 3
        framesT = np.moveaxis(frames, 0, -1)
        times = np.arange(frames.shape[0], dtype=np.int32)
        nnz_indices = np.where(framesT > 0)

        res_data = framesT[nnz_indices]
        res_times = times[nnz_indices[-1]]

        offsets = np.unique(np.cumsum((framesT > 0).sum(axis=-1).ravel()))
        res_offsets = np.ascontiguousarray(offsets, dtype=np.uint32)

        return res_data, res_times, res_offsets







"""
Let `frames` be a stack of `Nt` frames, each of them having `Nx * Ny` pixels.

At each location `(x, y)` in the frame space, we can extract a 1D array
of `Nt` elements (i.e `frames[:, y, x]`).
In this array, many elements will be zero, so it is not useful to store them.

The non-zero elements are compacted, along with their "time" index
in the original frames volume (i.e their indices along axis 0 in the volume space).
Schematically:
[0 0 ...     0   p1  0 ...  0   p2 0 ... 0 p3 0 ...] # pixels values
[0 1 ... (i1-1)  i1 .... (i2-1) i2 0 ... 0 i3 0 ...] # indices values
gives once compacted:
[p1 p2 p3 ...] # (nonzero) pixel values
[i1 i2 i3 ...] # corresponding "time" locations

This is done for each pixel location (x, y). Therefore, for each pixel in the
frame space (there are Nx*Ny such pixels), we build two arrays E(x, y) and T(x, y)
containing respectively the compacted non-zero elements, and their time indices.

 -----------
|      []   |  at location (x, y)   --> (E(x, y), T(x, y)) = ([p1, p2, ...], [t1, t2, ...])
|           |
|           |
 -----------

[ E(0, 0) | E(0, 1) | ... | E(x, y) | ... | E(Ny-1, Nx-1) ] # concatenated nonzero data values
[ T(0, 0) | T(0, 1) | ... | T(x, y) | ... | T(Ny-1, Nx-1) ] # concatenated corresponding time indices

In order to access the correct arrays E(x, y) and T(x, y) given a coordinate (x, y),
we need some mapping. This is done by building an "offset" array "L":

[L0 = 0   | L1      | ... | L(x, y) | ... ] # offsets
[ E(0, 0) | E(0, 1) | ... | E(x, y) | ... | E(Ny-1, Nx-1) ] # concatenated nonzero data values

The offset to access E(0, 0) is 0.
The offset to access E(0, 1) is 0 + <number of non-zero pixels in E(0 0)>,
and so on.


"""