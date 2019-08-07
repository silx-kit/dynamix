import numpy as np
from silx.opencl.common import pyopencl as cl
from silx.opencl.processing import OpenclProcessing, KernelContainer
from pyopencl.tools import dtype_to_ctype
import pyopencl.cltypes as cltypes
import pyopencl.array as parray
from ..utils import get_opencl_srcfile
from .common import OpenclCorrelator

class EventCorrelator(OpenclCorrelator):

    kernel_files = ["evtcorrelator.cl", "sums.cl"]


    def __init__(
        self, shape, nframes, max_events_count=10,
        dtype="f", bins=0, weights=None, extra_options={},
        ctx=None, devicetype="all", platformid=None, deviceid=None,
        block_size=None, memory=None, profile=False
    ):
        """
        TODO docstring
        """
        super().__init__(
            shape, nframes, dtype=dtype, bins=bins, weights=weights, extra_options=extra_options,
            ctx=ctx, devicetype=devicetype, platformid=platformid,
            deviceid=deviceid, block_size=block_size, memory=memory,
            profile=profile
        )
        self.max_events_count = max_events_count
        self._setup_kernels()
        self._allocate_event_arrays()



    def _setup_kernels(self):
        kernel_files = ["evtcorrelator.cl", "sums.cl"]
        kernel_files = list(map(get_opencl_srcfile, self.kernel_files))
        self.compile_kernels(
            kernel_files=kernel_files,
            compile_options=[
                "-DIMAGE_WIDTH=%d" % self.shape[1],
                "-DDTYPE=%s" % self.c_dtype,
                "-DIDX_DTYPE=%s" % self.idx_c_dtype,
                "-DSUM_WG_SIZE=%d" % 1024, # TODO tune ?
                "-DMAX_EVT_COUNT=%d" % self.max_events_count
            ]
        )
        self.correlation_kernel = self.kernels.get_kernel("event_correlator_oneQ") # TODO tune
        self.sum_kernel = self.kernels.get_kernel("compute_sums")

        self.grid = self.shape[::-1]
        self.wg = None # tune ?


    def _allocate_event_arrays(self):
        vol_shape = self.shape + (self.max_events_count, )
        self.d_vol_times = parray.zeros(self.queue, vol_shape, dtype=np.int32) # tune "times" dtype ?
        self.d_vol_data = parray.zeros(self.queue, vol_shape, dtype=self.dtype)
        self.d_ctr = parray.zeros(self.queue, self.shape, dtype=np.int32) # uint ?
        self.d_res_int = parray.zeros(self.queue, self.nframes, dtype=np.int32)
        self._old_d_vol_times = None
        self._old_d_vol_data = None
        self._old_d_ctr = None


    def set_data(self, vol_times, vol_data, ctr):
        arrays = {
            "vol_times": vol_times,
            "vol_data": vol_data,
            "ctr": ctr
        }
        for arr_name, array in arrays.items():
            my_array_name = "d_" + arr_name
            my_array = getattr(self, my_array_name)
            assert my_array.shape == array.shape
            assert my_array.dtype == array.dtype
            if isinstance(array, np.ndarray):
                my_array.set(array)
            elif isinstance(parray.Array):
                setattr(self, "_old_" + my_array_name, my_array)
                setattr(self, my_array_name, array)
            else: # support buffers ?
                raise ValueError("Unknown array type %s" % str(type(array)))

    def _reset_arrays(self):
        for array_name in ["d_vol_times", "d_vol_data", "d_ctr"]:
            old_array_name = "_old_" + array_name
            old_array = getattr(self, old_array_name)
            if old_array is not None:
                setattr(self, array_name, old_array)
                setattr(self, old_array_name, None)


    def correlate(self, vol_times, vol_data, ctr):

        self.set_data(vol_times, vol_data, ctr)

        evt = self.correlation_kernel(
            self.queue,
            self.grid,
            self.wg,
            self.d_vol_times.data,
            self.d_vol_data.data,
            self.d_ctr.data,
            self.d_res_int.data,
            np.int32(self.nframes)
        )
        evt.wait()
        self.profile_add(evt, "Event correlator")

        ###
        self.d_sums *= 1.0 / np.sqrt((np.prod(self.shape))) # !
        sums = self.d_sums.get()
        N = self.nframes
        normalization = np.zeros(N)
        for i in range(N):
            for j in range(i, N):
                normalization[i] += sums[j]*sums[j-i]
        ###
        self._reset_arrays()

        res = self.d_res_int.get()*1.0 / normalization # TODO
        return res


    def build_events_volume(self, frames):
        """
        Helper function to build a
        from a stack of (uncompressed) frames.
        """
        max_events = self.max_events_count
        dtype = self.dtype

        frame_shape = frames.shape[1:]
        n_frames = frames.shape[0]

        vol_shape = frame_shape + (max_events, )
        volume_times = np.zeros(vol_shape, dtype=np.int32)
        volume_data = np.zeros(vol_shape, dtype=dtype)
        ctr = np.zeros(frame_shape, dtype=np.int32)

        rows, cols = np.indices(frame_shape)

        for i in range(n_frames):
            frame = frames[i]
            mask = frame > 0

            R = rows[mask]
            C = cols[mask]
            depth_pos = ctr[R, C]
            volume_times[R, C, depth_pos] = i
            volume_data[R, C, depth_pos] = frame[mask]
            ctr[R, C] += 1

        return volume_times, volume_data, ctr












