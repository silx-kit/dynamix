import numpy as np
from ..utils import nextpow2, updiv, get_next_power
from .dense import MatMulCorrelator, FFTCorrelator

try:
    from silx.math.fft.cufft import CUFFT
    import pycuda.gpuarray as garray
    from pycuda.compiler import SourceModule
    from pycuda.driver import Memcpy2D, Memcpy3D
except ImportError:
    CUFFT = None
try:
    import skcuda.linalg as cublas
    import skcuda.misc as skmisc
except ImportError:
    cublas = None


class CublasMatMulCorrelator(MatMulCorrelator):

    """
    The CublasMatMulCorrelator is a CUDA-accelerated version of MatMulCorrelator.
    """

    def __init__(self, shape, nframes,
                 qmask=None,
                 scale_factor=None,
                 extra_options={}):

        """
        Initialize a CUBLAS matrix multiplication correlator.
        Please refer to the documentation of BaseCorrelator for the documentation
        of each parameters.

        Extra options
        --------------
        cublas_handle: int
            If provided, use this cublas handle instead of creating a new one.
        """
        if cublas is None:
            raise ImportError("scikit-cuda is needed to use this correlator")

        super().__init__(
            shape, nframes,
            qmask=qmask, scale_factor=scale_factor, extra_options=extra_options
        )
        self._init_cublas()
        self._compile_kernels()


    def _init_cublas(self):
        import pycuda.autoinit
        if "cublas_handle" in self.extra_options:
            handle = self.extra_options["cublas_handle"]
        else:
            handle = skmisc._global_cublas_handle
            if handle is None:
                cublas.init() # cublas handle + allocator
                handle = skmisc._global_cublas_handle
        self.cublas_handle = handle


    def _compile_kernels(self):
        mod = SourceModule(
            """
            // Extract the upper diagonals of a square (N, N) matrix.
            __global__ void extract_upper_diags(float* matrix, float* diags, int N) {
                int x = blockDim.x * blockIdx.x + threadIdx.x;
                int y = blockDim.y * blockIdx.y + threadIdx.y;
                if ((x >= N) || (y >= N) || (y > x)) return;
                int pos = y*N+x;
                int my_diag = x-y;
                diags[my_diag * N + x] = matrix[pos];
            }
            """
        )
        self.extract_diags_kernel = mod.get_function("extract_upper_diags")
        self._blocks = (32, 32, 1)
        self._grid = (
            updiv(self.nframes, self._blocks[0]),
            updiv(self.nframes, self._blocks[1]),
            1
        )
        self.d_diags = garray.zeros((self.nframes, self.nframes), dtype=np.float32)
        self.d_sumdiags1 = garray.zeros(self.nframes, dtype=np.float32)
        self.d_sumdiags2 = garray.zeros_like(self.d_sumdiags1)
        self._kern_args = [
            None,
            self.d_diags,
            np.int32(self.nframes),
        ]


    def sum_diagonals(self, d_arr, d_out):
        self.d_diags.fill(0)
        self._kern_args[0] = d_arr.gpudata
        self.extract_diags_kernel(*self._kern_args, grid=self._grid, block=self._blocks)
        skmisc.sum(self.d_diags, axis=1, out=d_out)


    def _correlate_matmul_cublas(self, frames_flat, mask):
        arr = np.ascontiguousarray(frames_flat[:, mask], dtype=np.float32)
        npix = arr.shape[1]
        # Pre-allocating memory for all bins might save a bit of time,
        # but would take more memory
        d_arr = garray.to_gpu(arr)
        d_outer = cublas.dot(d_arr, d_arr, transb="T", handle=self.cublas_handle)
        d_means = skmisc.mean(d_arr, axis=1, keepdims=True)
        d_denom_mat = cublas.dot(d_means, d_means, transb="T", handle=self.cublas_handle)

        self.sum_diagonals(d_outer, self.d_sumdiags1)
        self.sum_diagonals(d_denom_mat, self.d_sumdiags2)
        self.d_sumdiags1 /= self.d_sumdiags2
        self.d_sumdiags1 /= npix

        return self.d_sumdiags1.get()

    def correlate(self, frames):
        res = np.zeros((self.n_bins, self.nframes), dtype=np.float32)
        frames_flat = frames.reshape((self.nframes, -1))

        for i, bin_val in enumerate(self.bins):
            mask = (self.qmask.ravel() == bin_val)
            res[i] = self._correlate_matmul_cublas(frames_flat, mask)
        return res



class CUFFTCorrelator(FFTCorrelator):
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
        if CUFFT is None:
            raise ImportError("pycuda and scikit-cuda need to be installed")
        self.d_sums = garray.zeros(self.Nf, np.float32)
        if skmisc._global_cublas_handle is None:
            cublas.init()

    def _create_fft_plan(self, npix):
        return CUFFT(shape=(npix, self.Nf), dtype=np.float32,  axes=(-1,))

    def _correlate_fft(self, frames_flat, cufft_plan):
        npix = frames_flat.shape[1]

        cufft_plan.data_in.fill(0)
        d_in = cufft_plan.data_in
        f_out1 = cufft_plan.data_out
        f_out2 = garray.zeros_like(cufft_plan.data_out)

        cufft_plan.data_in[:, :self.nframes] = frames_flat.T.astype("f")

        f_out1 = cufft_plan.fft(d_in, output=f_out1)
        cufft_plan.data_in.fill(0)
        cufft_plan.data_in[:, :self.nframes] = frames_flat.T[:, ::-1].astype("f")

        f_out2 = cufft_plan.fft(d_in, output=f_out2)
        f_out1 *= f_out2
        num = cufft_plan.ifft(f_out1, output=cufft_plan.data_in)
        skmisc.sum(num, axis=0, out=self.d_sums)

        num = self.d_sums.get()[self.nframes-1:self.nframes-1 + self.nframes]
        # gpu ?
        sums = frames_flat.sum(axis=1)
        denom = np.correlate(sums, sums, "full")[sums.size-1:] / npix # TODO GPU, it takes 2/3 of the time
        #

        res = num/denom
        return res

