import numpy as np
import pyopencl.array as parray
from os import path
from multiprocessing import cpu_count
from ..utils import nextpow2, updiv, get_opencl_srcfile, get_next_power
from .common import OpenclCorrelator, BaseCorrelator

from silx.math.fft.fftw import FFTW
try:
    from silx.math.fft.cufft import CUFFT
    import pycuda.gpuarray as garray
    from pycuda.compiler import SourceModule
except ImportError:
    CUFFT = None
try:
    import skcuda.linalg as cublas
    import skcuda.misc as skmisc
except ImportError:
    cublas = None
try:
    import pyfftw
except ImportError:
    pyfftw = None

NCPU = cpu_count()


def py_dense_correlator(xpcs_data, mask):
    """
    Reference implementation of the dense correlator.

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

    res = np.zeros(ltimes) # was ones()
    for i in range(ltimes): # was ltimes-1, so res[-1] was always 1 !
        dia_n = np.diag(num, k=i)
        dia_d = np.diag(denom, k=i)
        res[i] = np.sum(dia_n)/np.sum(dia_d) / lenmatr
    return res


class MatMulCorrelator(BaseCorrelator):

    def __init__(self, shape, nframes,
                 qmask=None,
                 scale_factor=None,
                 extra_options={}):

        super().__init__()
        super()._set_parameters(shape, nframes, qmask, scale_factor, extra_options)


    def correlate(self, frames):
        res = np.zeros((self.n_bins, self.nframes), dtype=np.float32)
        for i, bin_value in enumerate(self.bins):
            mask = (self.qmask == bin_value)
            res[i] = py_dense_correlator(frames, mask)
        return res




class DenseCorrelator(OpenclCorrelator):

    kernel_files = ["densecorrelator.cl"]

    def __init__(
        self, shape, nframes,
        qmask=None, dtype="f", weights=None, extra_options={},
        ctx=None, devicetype="all", platformid=None, deviceid=None,
        block_size=None, memory=None, profile=False
    ):
        """
        Class for OpenCL dense correlator.
        This correlator is usually slower than all the other correlators.
        """
        super(DenseCorrelator, self).__init__(
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
        self.d_output = parray.zeros(
            self.queue,
            (self.n_bins, self.nframes),
            np.float32
        )


    def _normalize_sums(self):
        if self.n_bins == 0:
            self.d_sums_f[:] *= self.scale_factors[0]
        else:
            for i, factor in enumerate(self.scale_factors.values()):
                self.d_sums_f[i] /= np.array([factor], dtype=self.output_dtype)[0]
        self.d_sums_f.finish()


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






class FFTCorrelator(BaseCorrelator):
    def __init__(self, shape, nframes,
                 qmask=None,
                 weights=None,
                 scale_factor=None,
                 precompute_fft_plans=False,
                 extra_options={}):
        super().__init__()
        super()._set_parameters(shape, nframes, qmask, scale_factor, extra_options)
        self._init_fft_plans(precompute_fft_plans)

    def _init_fft_plans(self, precompute_fft_plans):
        self.precompute_fft_plans = precompute_fft_plans
        self.Nf = int(get_next_power(2 * self.nframes))
        self.n_pix = {}
        for bin_val in self.bins:
            mask = (self.qmask == bin_val)
            self.n_pix[bin_val] = mask.sum()

        self.fft_plans = {}
        for bin_val, npix in self.n_pix.items():
            if precompute_fft_plans:
                self.fft_plans[bin_val] = self._create_fft_plan(npix)
            else:
                self.fft_plans[bin_val] = None

    def _create_fft_plan(self, npix):
        raise NotImplementedError("This must be implemented by child class")

    def _get_plan(self, bin_val):
        fft_plan = self.fft_plans[bin_val]
        if fft_plan is None:
            fft_plan = self._create_fft_plan(self.n_pix[bin_val])
        return fft_plan

    def _correlate_fft(self, frames_flat, fftw_plan):
        raise NotImplementedError("This must be implemented by child class")

    def correlate(self, frames):
        res = np.zeros((self.n_bins, self.nframes), dtype=np.float32)
        frames_flat = frames.reshape((self.nframes, -1))
        for i, bin_val in enumerate(self.bins):
            mask = (self.qmask == bin_val).ravel()
            frames_flat_currbin = frames_flat[:, mask]
            fft_plan = self._get_plan(bin_val)
            res[i] = self._correlate_fft(frames_flat_currbin, fft_plan)
        return res




class FFTWCorrelator(FFTCorrelator):
    def __init__(self, shape, nframes,
                 qmask=None,
                 weights=None,
                 scale_factor=None,
                 precompute_fft_plans=False,
                 extra_options={}):
        super().__init__(
            shape, nframes, qmask=qmask,
            weights=weights, scale_factor=scale_factor,
            precompute_fft_plans=precompute_fft_plans, extra_options=extra_options
        )
        if pyfftw is None:
            raise ImportError("pyfftw needs to be installed")

    def _create_fft_plan(self, npix):
        return FFTW(
            shape=(npix, self.Nf), dtype=np.float32,
            num_threads=NCPU, axes=(-1,)
        )

    def _correlate_fft(self, frames_flat, fftw_plan):
        npix = frames_flat.shape[1]

        fftw_plan.data_in.fill(0)
        f_out1 = fftw_plan.data_out
        f_out2 = np.zeros_like(fftw_plan.data_out)

        fftw_plan.data_in[:, :self.nframes] = frames_flat.T

        f_out1 = fftw_plan.fft(None, output=f_out1)
        fftw_plan.data_in.fill(0)
        fftw_plan.data_in[:, :self.nframes] = frames_flat.T[:, ::-1]

        f_out2 = fftw_plan.fft(None, output=f_out2)
        f_out1 *= f_out2
        num = fftw_plan.ifft(f_out1)

        num = num.sum(axis=0)[self.nframes-1:self.nframes-1 + self.nframes]
        sums = frames_flat.sum(axis=1)
        denom = np.correlate(sums, sums, "full")[sums.size-1:] / npix

        res = num/denom
        return res



def export_wisdom(basedir):
    w = pyfftw.export_wisdom()
    for i in range(len(w)):
        fname = path.join(basedir, "wis%d.dat" % i)
        with open(fname, "wb") as fid:
            fid.write(w[i])

def import_wisdom(basedir):
    w = []
    for i in range(3): # always 3 ?
        fname = path.join(basedir, "wis%d.dat" % i)
        if not(path.isfile(fname)):
            raise RuntimeError("Could find wisdom file %s" % fname)
        with open(fname, "rb") as fid:
            w.append(fid.read())
    pyfftw.import_wisdom(w)

