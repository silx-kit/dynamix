import numpy as np
import pyopencl.array as parray
from ..utils import get_opencl_srcfile
from .common import OpenclCorrelator

class DenseCorrelator(OpenclCorrelator):

    kernel_files = ["densecorrelator.cl"]


    def __init__(
        self, shape, nframes, 
        qmask=None, dtype="f", weights=None, extra_options={},
        ctx=None, devicetype="all", platformid=None, deviceid=None,
        block_size=None, memory=None, profile=False
    ):
        """
        TODO docstring
        """
        super().__init__(
            shape, nframes, qmask=qmask, dtype=dtype, weights=weights, 
            extra_options=extra_options,
            ctx=ctx, devicetype=devicetype, platformid=platformid,
            deviceid=deviceid, block_size=block_size, memory=memory,
            profile=profile
        )
        self._setup_kernels()
        self._allocate_arrays()


    def _setup_kernels(self):
        kernel_files = list(map(get_opencl_srcfile, self.kernel_files))
        self.compile_kernels(
            kernel_files=kernel_files,
            compile_options=[
                "-DIMAGE_WIDTH=%d" % self.shape[1],
                "-DNUM_BINS=%d" % self.n_bins,
                "-DDTYPE=%s" % self.c_dtype,
                "-DDTYPE_SUMS=%s" % self.c_sums_dtype,
                "-DN_FRAMES=%d" % self.nframes,
                "-DUSE_SHARED=%d" % 0, # <
                "-DSUM_WG_SIZE=%d" % 1024, # tune ?
            ]
        )
        self.correlation_kernel = self.kernels.get_kernel("correlator_multiQ_dense")
        self.wg = (128, 1) # TODO determine as nextpow2(image_width)
        self.grid = (max(self.wg[0], self.shape[1]), self.nframes)
        self.sums_kernel = self.kernels.get_kernel("compute_sums_dense")
        self.corr1D_kernel = self.kernels.get_kernel("correlate_1D")


    def _allocate_arrays(self):
        self.d_frames = parray.zeros(
            self.queue,
            (self.nframes, ) + self.shape,
            self.dtype
        )
        self._old_d_frames = None


    def correlate(self, frames):
        self._set_data({"frames": frames})

        evt = self.correlation_kernel(
            self.queue,
            self.grid,
            self.wg,
            self.d_frames.data,
            self.d_qmask.data,
            self.d_norm_mask.data,
            self.d_output.data,
            np.int32(self.shape[0]),
            np.int32(self.nframes),
        )
        evt.wait()
        self.profile_add(evt, "Dense correlator")


        return self.d_res.get()

