import numpy as np
# from silx.opencl.common import pyopencl as cl
import pyopencl.array as parray
from ..utils import get_opencl_srcfile
from .common import OpenclCorrelator

class EventCorrelator(OpenclCorrelator):

    kernel_files = ["evtcorrelator.cl"]

    """
    A class to compute the correlation function for sparse XPCS data.
    """

    def __init__(
        self, shape, nframes,
        max_events_count, total_events_count=None, allow_reallocate=True,
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
        self._set_events_size(max_events_count, total_events_count)
        self.allow_reallocate = allow_reallocate
        self._allocate_events_arrays()
        self._setup_kernels()


    def _set_events_size(self, max_events_count, total_events_count):
        self.max_events_count = max_events_count
        self.total_events_count = total_events_count or max_events_count * np.prod(self.shape)


    def _setup_kernels(self):
        kernel_files = list(map(get_opencl_srcfile, self.kernel_files))
        self.compile_kernels(
            kernel_files=kernel_files,
            compile_options=[
                "-DIMAGE_WIDTH=%d" % self.shape[1],
                "-DDTYPE=%s" % self.c_dtype,
                "-DSUM_WG_SIZE=%d" % 1024, # TODO tune ?
                "-DMAX_EVT_COUNT=%d" % self.max_events_count,
                "-DSCALE_FACTOR=%f" % self.scale_factors[1],
                "-DNUM_BINS=%d" % self.n_bins,
            ]
        )
        self.correlation_kernel = self.kernels.get_kernel("event_correlator")
        self.normalization_kernel = self.kernels.get_kernel("normalize_correlation")

        self.grid = self.shape[::-1]
        self.wg = None # tune ?


    def _allocate_events_arrays(self, is_reallocating=False):
        tot_nnz = self.total_events_count

        self.d_vol_times = parray.zeros(self.queue, tot_nnz, dtype=np.int32)
        self.d_vol_data = parray.zeros(self.queue, tot_nnz, dtype=self.dtype)
        self.d_offsets = parray.zeros(self.queue, np.prod(self.shape)+1, dtype=np.uint32)

        self._old_d_vol_times = None
        self._old_d_vol_data = None
        self._old_d_offsets = None

        if not(is_reallocating):
            self.d_res_int = parray.zeros(self.queue, self.output_shape, dtype=np.int32)
            self.d_sums = parray.zeros(self.queue, self.output_shape, np.uint32)
            self.d_res = parray.zeros(self.queue, self.output_shape, np.float32)
            self.d_scale_factors = parray.to_device(self.queue, np.array(list(self.scale_factors.values()), dtype=np.float32))


    def _check_event_arrays(self, vol_times, vol_data, offsets):
        for arr in [vol_times, vol_data, offsets]:
            assert arr.ndim == 1
        assert vol_times.size == vol_data.size
        assert offsets.size == np.prod(self.shape) + 1
        numels = vol_data.size
        if numels > self.total_events_count:
            if self.allow_reallocate:
                self.total_events_count = numels
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

        self.d_res.fill(0)
        self.d_sums.fill(0)

        evt = self.correlation_kernel(
            self.queue,
            self.grid,
            self.wg,
            self.d_vol_times.data,
            self.d_vol_data.data,
            self.d_offsets.data,
            self.d_qmask.data,
            self.d_res_int.data,
            self.d_sums.data,
            np.int32(self.shape[1]),
            np.int32(self.nframes)
        )
        evt.wait()
        self.profile_add(evt, "Event correlator")

        evt = self.normalization_kernel(
            self.queue,
            (self.nframes, self.n_bins),
            None, # tune wg ?
            self.d_res_int.data,
            self.d_res.data,
            self.d_sums.data,
            self.d_scale_factors.data,
            np.int32(self.nframes)
        )
        evt.wait()
        self.profile_add(evt, "Normalization")

        self._reset_arrays(["vol_times", "vol_data", "offsets"])

        return self.d_res.get()



class FramesCompressor(object):
    """
    A class for compressing frames on-the-fly.

    It builds the data structure used by EventCorrelator, which is explained below.

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

    def __init__(self, shape, nframes, max_nnz, dtype=np.int8):
        self.shape = shape
        self.nframes = nframes
        self.dtype = dtype
        self.max_nnz = max_nnz
        self.npix = np.prod(self.shape)
        self._init_events_datastructure()


    def _init_events_datastructure(self):
        self.events_counter = np.zeros(self.npix, dtype=np.int32)
        self.events = np.zeros((self.npix, self.max_nnz), dtype=self.dtype)
        self.times = np.zeros((self.npix, self.max_nnz), dtype=np.int32)
        self.frames_counter = 0


    @staticmethod
    def compress_all_stack(frames):
        """
        Build the whole event structure assuming that all the frames are given.
        """
        assert frames.ndim == 3
        framesT = np.moveaxis(frames, 0, -1)
        times = np.arange(frames.shape[0], dtype=np.int32)
        nnz_indices = np.where(framesT > 0)

        res_data = framesT[nnz_indices]
        res_times = times[nnz_indices[-1]]

        offsets = np.cumsum((frames > 0).sum(axis=0).ravel())
        res_offsets = np.zeros(np.prod(frames.shape[1:])+1, dtype=np.uint32)
        res_offsets[1:] = offsets[:]

        return res_data, res_times, res_offsets


    def process_frame(self, frame):
        """
        Compress a single frame, and update the events stack state.
        """
        frame1d = frame.ravel()

        mask = frame1d > 0
        self.events[mask, self.events_counter[mask]] = frame1d[mask]
        self.times[mask, self.events_counter[mask]] = self.frames_counter
        self.events_counter[mask] += 1
        self.frames_counter += 1


    def get_compacted_events(self, wait_for_all_frames=True):
        """
        Compact all compressed frames into three 1D structures:
          - events: 1D array containing all the nonzero data points
          - times: time indices corresponding to nonzero data points
          - offsets: events[offsets[k]:offsets[k+1]] corresponds to all events for pixel index k
        """
        if self.frames_counter < self.nframes - 1 and wait_for_all_frames:
            raise RuntimeError(
                "Not all frames were compressed yet (%d/%d)" %
                (self.frames_counter, self.nframes)
            )
        offsets = np.zeros(
            self.events_counter.size + 1, dtype=np.uint32
        )
        offsets[1:] = np.cumsum(self.events_counter)

        m = self.events.ravel() > 0
        events = self.events.ravel()[m]
        times = self.times.ravel()[m]

        return events, times, offsets



