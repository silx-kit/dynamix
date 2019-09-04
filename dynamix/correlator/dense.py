import numpy as np
import pyopencl.array as parray
from ..utils import nextpow2, get_opencl_srcfile
from .common import OpenclCorrelator, BaseCorrelator
import multiprocessing
try:
    from silx.math.fft.cufft import CUFFT
    import pycuda.gpuarray as garray
except ImportError:
    CUFFT = None

try:
    import pyfftw
except ImportError:
    pyfftw = None

from collections import namedtuple
FFTwPlan = namedtuple("FFTwPlan", "fft ifft data_direct data_reciprocal")
NCPU = multiprocessing.cpu_count()

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

class DenseFFTwCorrelator(BaseCorrelator):
    """
    Not an OpenCL correlator, as we are not using OpenCL.
    
    Based on FFTw
    """

    def __init__(self, shape, nframes, 
                 qmask=None, 
                 weights=None, 
                 scale_factor=None, 
                 precompute_fft_plans=False,
                 extra_options={}):
        BaseCorrelator.__init__(self)
        if pyfftw is None:
            raise ImportError("pyfftw needs to be installed")
        BaseCorrelator._set_parameters(self, shape, nframes, qmask, scale_factor, extra_options)
        self._init_fft_plans(precompute_fft_plans)

    def _configure_extra_options(self, extra_options):
        BaseCorrelator._configure_extra_options(self, extra_options)
        self.extra_options["save_fft_plans"] =  True

    def _init_fft_plans(self, precompute_fft_plans):
        """
        Create one couple of (FFT, IFFT) plans for each bin value
        """
        self.precompute_fft_plans = precompute_fft_plans
        self.fft_sizes = {}
        self.ffts = {}
        bins = self.bins if self.bins is not None else [0]
        for bin_val in bins:
            if bin_val == 0:
                n_mask_pixels = np.prod(self.shape)
            else:
                n_mask_pixels = (self.qmask == bin_val).sum()
            fft_size = int(nextpow2(2 * self.nframes * int(n_mask_pixels)))
            self.fft_sizes[bin_val] = fft_size
            
            if precompute_fft_plans:
                self.ffts[bin_val] = self.get_plan(bin_val)
            else:
                self.ffts[bin_val] = None

    @staticmethod
    def _compute_denom_means(frames):
        # frames: (n_frames, n_pix), float32
        # Do it on GPU ? Cumbersome, and not sure if the perf gain is worth it
        return frames.mean(axis=1)

    def get_plan(self, bin_val):
        """
        Get the FFT plan associated with a bin value.
        """
        N = self.fft_sizes[bin_val]
        fft = self.ffts.get(bin_val)
        if fft is None: # plan is not precomputed - it is time to compute it
            data_direct =  pyfftw.empty_aligned(N, 'complex64')
            data_rec = pyfftw.empty_aligned(N, 'complex64')
            fft = FFTwPlan(pyfftw.FFTW(data_direct, data_rec, direction='FFTW_FORWARD', threads=NCPU,  flags=["FFTW_ESTIMATE"]),
                           pyfftw.FFTW(data_rec, data_direct, direction='FFTW_BACKWARD', threads=NCPU,  flags=["FFTW_ESTIMATE"]),
                           data_direct, data_rec) 
            
            if self.extra_options.get("save_fft_plans"):
                self.ffts[bin_val] = fft
        return fft

    def flush_plans(self, bin_val=None):
        """
        Clear stored FFT plans in order to free some memory.
        """
        bins = [bin_val] if bin_val is not None else list(self.ffts.keys())
        for binval in bins:
            self.ffts[binval] = None


    def _correlate_1d(self, frames, bin_val=0):
        # frames: (n_frames, n_pix), float32
        fft = self.get_plan(bin_val)

        fft.data_direct[:frames.size] = np.ascontiguousarray(frames.ravel()[:], dtype="complex64")
        fft.fft()
        out1 = np.copy(fft.data_reciprocal)
        fft.data_direct[:frames.size] = np.ascontiguousarray(frames.ravel()[::-1], dtype="complex64")
        fft.fft()
        out2 = np.copy(fft.data_reciprocal)
        fft.data_reciprocal[...] = out1 * out2 
        res = np.ascontiguousarray(fft.ifft().real, dtype="float32")

        numerator = res[:frames.size].reshape((self.nframes, -1))[:, -1][::-1]
        sums = self._compute_denom_means(frames)
        denominator = np.correlate(sums, sums, "full")[sums.size-1:] # with fft and/or gpu ?

        return numerator/denominator/self.scale_factors[bin_val]

    def get_pixels_in_bin(self, frames, bin_val, check=True, convert_to_float=True):
        """
        From a stack of frames, extract the pixels belonging to a given bin.
        The result is a 2D array of size (nframes, npixels) where npixels
        is the number of pixels falling in the given bin.

        Parameters
        -----------
        frames: numpy.ndarray
            Stack of frames in the format (nframes, nrows, ncolumns)
        bin_val: int
            Value of the current bin
        check: bool, optional
            Whether to check if the stack of frames is valid with the current instance.
        convert_to_float: bool, optional
            Whether to convert the result in float32.
        """
        if check:
            if bin_val > 0:
                assert bin_val in self.bins
            assert frames.ndim == 3
            assert frames.shape[0] == self.nframes
            assert frames[0].shape == self.shape
            # assert frames.dtype == self.dtype # should not be relevant here
        if bin_val == 0: # no qmask
            res = frames.reshape((frames.shape[0], -1))
        else:
            mask = (self.qmask == bin_val)
            res = frames.reshape((frames.shape[0], -1))[:, mask.ravel()]
        if convert_to_float:
            res = np.ascontiguousarray(res, dtype=np.float32)
        return res

    def correlate(self, frames):
        bins = [0] if self.bins is None else self.bins
        results = np.zeros((len(bins), self.nframes), dtype="f")
        for i, bin_val in enumerate(bins):
            arr = self.get_pixels_in_bin(frames, bin_val)
            results[i] = self._correlate_1d(arr, bin_val=bin_val)
        return results



class DenseCuFFTCorrelator(BaseCorrelator):
    """
    Not an OpenCL correlator, as we are not using OpenCL.
    CLFFT does not support 1D FFT of too big arrays for some reason.
    """

    def __init__(self, shape, nframes, 
                 qmask=None, 
                 weights=None, 
                 scale_factor=None, 
                 precompute_fft_plans=False,
                 extra_options={}):
        BaseCorrelator.__init__(self)
        if CUFFT is None:
            raise ImportError("pycuda and scikit-cuda need to be installed")
        BaseCorrelator._set_parameters(self, shape, nframes, qmask, scale_factor, extra_options)
        self._init_fft_plans(precompute_fft_plans)

    def _configure_extra_options(self, extra_options):
        BaseCorrelator._configure_extra_options(self, extra_options)
        self.extra_options["save_fft_plans"] =  True

    def _init_fft_plans(self, precompute_fft_plans):
        """
        Create one couple of (FFT, IFFT) plans for each bin value
        """
        self.precompute_fft_plans = precompute_fft_plans
        self.fft_sizes = {} # key: size, value: next power of two 
        self.ffts = {}
        bins = self.bins if self.bins is not None else [0]
        for bin_val in bins:
            if bin_val == 0:
                n_mask_pixels = np.prod(self.shape)
            else:
                n_mask_pixels = (self.qmask == bin_val).sum()
            fft_size = int(nextpow2(2 * self.nframes * int(n_mask_pixels)))
            self.fft_sizes[bin_val] = fft_size
            self.ffts[bin_val] = None
            if precompute_fft_plans:
                self.get_plan(bin_val)  

    @staticmethod
    def _compute_denom_means(frames):
        # frames: (n_frames, n_pix), float32
        # Do it on GPU ? Cumbersome, and not sure if the perf gain is worth it
        return frames.mean(axis=1)

    def get_plan(self, bin_val):
        """
        Get the FFT plan associated with a bin value.
        """
        N = self.fft_sizes[bin_val]
        fft = self.ffts.get(bin_val)
        if fft is None: # plan is not precomputed - it is time to compute it
            fft = CUFFT(template=np.zeros(N, dtype=np.float32))
            if self.extra_options.get("save_fft_plans"):
                self.ffts[bin_val] = fft
        else:
            fft.data_in.fill(0)
            fft.data_out.fill(0)
        return fft

    def flush_plans(self, bin_val=None):
        """
        Clear stored FFT plans in order to free some GPU memory.
        """
        bins = [bin_val] if bin_val is not None else list(self.fft_sizes.keys())
        for binval in bins:
            self.ffts[binval] = None


    def _correlate_1d(self, frames, bin_val=0):
        # frames: (n_frames, n_pix), float32
        fft = self.get_plan(bin_val)

        fft.data_in[:frames.size] = frames.ravel()[:]

        d_out1 = fft.data_out
        d_out2 = garray.zeros_like(fft.data_out) # pre-allocate ?
        fft.fft(fft.data_in, output=d_out1)
        fft.data_in[:frames.size] = np.ascontiguousarray(frames.ravel()[::-1])
        fft.fft(fft.data_in, output=d_out2)

        d_out1 *= d_out2
        fft.ifft(d_out1, output=fft.data_in)
        res = fft.data_in.get()

        numerator = res[:frames.size].reshape((self.nframes, -1))[:, -1][::-1]
        sums = self._compute_denom_means(frames)
        denominator = np.correlate(sums, sums, "full")[sums.size-1:] # with fft and/or gpu ?

        return numerator/denominator/self.scale_factors[bin_val]

    def get_pixels_in_bin(self, frames, bin_val, check=True, convert_to_float=True):
        """
        From a stack of frames, extract the pixels belonging to a given bin.
        The result is a 2D array of size (nframes, npixels) where npixels
        is the number of pixels falling in the given bin.

        Parameters
        -----------
        frames: numpy.ndarray
            Stack of frames in the format (nframes, nrows, ncolumns)
        bin_val: int
            Value of the current bin
        check: bool, optional
            Whether to check if the stack of frames is valid with the current instance.
        convert_to_float: bool, optional
            Whether to convert the result in float32.
        """
        if check:
            if bin_val > 0:
                assert bin_val in self.bins
            assert frames.ndim == 3
            assert frames.shape[0] == self.nframes
            assert frames[0].shape == self.shape
            # assert frames.dtype == self.dtype # should not be relevant here
        if bin_val == 0: # no qmask
            res = frames.reshape((frames.shape[0], -1))
        else:
            mask = (self.qmask == bin_val)
            res = frames.reshape((frames.shape[0], -1))[:, mask.ravel()]
        if convert_to_float:
            res = np.ascontiguousarray(res, dtype=np.float32)
        return res

    def correlate(self, frames):
        bins = [0] if self.bins is None else self.bins
        results = np.zeros((len(bins), self.nframes), dtype="f")
        for i, bin_val in enumerate(bins):
            arr = self.get_pixels_in_bin(frames, bin_val)
            results[i] = self._correlate_1d(arr, bin_val=bin_val)
        return results

DenseFFTCorrelator = DenseFFTwCorrelator