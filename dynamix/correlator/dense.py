import numpy as np
import pyopencl.array as parray
from ..utils import nextpow2, get_opencl_srcfile
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
                "-DSUM_WG_SIZE=%d" % min(1024, nextpow2(self.shape[1])),
            ]
        )
        self.correlation_kernel = self.kernels.get_kernel("correlator_multiQ_dense")
        self.wg = (
            min(1024, nextpow2(self.shape[1])),
            1
            )
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
        self.d_sums = parray.zeros(
            self.queue,
            self.output_shape,
            self.sums_dtype
        )
        self.d_sums_f = parray.zeros(
            self.queue,
            self.output_shape,
            self.output_dtype,
        )


    def _normalize_sums(self):
        if self.n_bins == 0:
            self.d_sums_f[:] *= self.scale_factor
        else:
            for i, factor in enumerate(self.scale_factors):
                self.d_sums_f[i] /= np.array([factor], dtype=self.output_dtype)[0]
        # self.d_sums_f.finish()


    def correlate(self, frames):
        self._set_data({"frames": frames})

        # Denominator
        self._sum_frames()
        self._correlate_1d()
        self._normalize_sums()

        # Numerator
        evt = self.correlation_kernel(
            self.queue,
            self.grid,
            self.wg,
            self.d_frames.data,
            self.d_qmask.data,
            self.d_norm_mask.data,
            self.d_sums_f.data,
            self.d_output.data,
            np.int32(self.shape[0]),
            np.int32(self.nframes),
        )
        evt.wait()
        self.profile_add(evt, "Dense correlator")

        self._reset_arrays(["frames"])

        return self.d_output.get() # get ?


    def _sum_frames(self):
        evt = self.sums_kernel(
            self.queue,
            (self.wg[0], self.nframes),
            (self.wg[0], 1),
            self.d_frames.data,
            self.d_qmask.data,
            self.d_sums.data,
            np.int32(self.shape[0]),
            np.int32(self.nframes)
        )
        evt.wait()
        self.profile_add(evt, "Sum kernel")


    def _correlate_1d(self):
        evt = self.corr1D_kernel(
            self.queue,
            (self.nframes, self.n_bins),
            None,
            self.d_sums.data,
            self.d_sums_f.data,
        )
        evt.wait()
        self.profile_add(evt, "Corr 1D kernel")


def py_dense_correlator(xpcs_data, mask):
    """
    Reference/"naive" implementation of the dense correlator.

    Parameters
    -----------
    xpcs_data: numpy.ndarray
        Stack of XPCS frames with shape (n_frames, n_rows, n_columns)
    mask: numpy.ndarray
        Mask of bins in the format (n_rows, n_columns).
        Zero pixels indicate unused pixels.
    """
    ind = np.where(mask > 0) # unused pixels are 0
    xpcs_data = np.array(xpcs_data[:, ind[0], ind[1]], np.float32) # (n_tau, n_pix)
    meanmatr = np.mean(xpcs_data, axis=1) # xpcs_data.sum(axis=-1).sum(axis=-1)/n_pix
    ltimes, lenmatr = np.shape(xpcs_data) # n_tau, n_pix
    meanmatr.shape = 1, ltimes

    num = np.dot(xpcs_data, xpcs_data.T)
    denom = np.dot(meanmatr.T, meanmatr)

    res = np.zeros(ltimes-1) # was ones()
    for i in range(1, ltimes): # was ltimes-1, so res[-1] was always 1 !
        dia_n = np.diag(num, k=i)
        dia_d = np.diag(denom, k=i)
        res[i-1] = np.sum(dia_n)/np.sum(dia_d) / lenmatr
    return res
