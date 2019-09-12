import numpy as np
from silx.opencl.common import pyopencl as cl
from .common import OpenclCorrelator
from silx.opencl.processing import OpenclProcessing, KernelContainer
from pyopencl.tools import dtype_to_ctype
import pyopencl.array as parray
from ..utils import get_opencl_srcfile

class SparseCorrelator(OpenclCorrelator):

    def __init__(
        self, shape, nframes, dtype="f", bins=0, weights=None, extra_options={},
        ctx=None, devicetype="all", platformid=None, deviceid=None,
        block_size=None, memory=None, profile=False
    ):
        """
        Compute the Intensity Correlation function on sparse data.

        :param shape: tuple or int
            Shape of each frame, in the form (num_rows, num_colums).
            If the shape is an integer, the frame is assumed to be square.
        :param nframes: int
            Total number of frames for computing the correlation function.
        :param dtype: str or numpy.dtype, optional
            Numeric data type. By default, sparse matrix data will be float32.
        :param bins: int, optional
            Number of bins. Default is 0 (ensemble averaging on all pixels)
        :param weights: numpy.ndarray, optional
            Array which must have the same shape as the frames.
            It contains the weights applied to each frames.
        :param extra_options: dict, optional
            Advanced extra options.
            Not available yet.

        Opencl processing parameters
        -----------------------------
        Please refer to the documentation of silx.opencl.processing.OpenclProcessing
        for information on the other parameters.
        """
        super().__init__(
            shape, nframes, dtype=dtype, bins=bins, weights=weights, extra_options=extra_options,
            ctx=ctx, devicetype=devicetype, platformid=platformid,
            deviceid=deviceid, block_size=block_size, memory=memory,
            profile=profile
        )
        self._setup_kernels()

    def _determine_kernel(self):
        if self.bins == 0:
            kernel_name = "correlator_oneQ_Nt"
        else:
            kernel_name = "correlator"
        return kernel_name

    def _setup_kernels(self):
        self.kernel_name = self._determine_kernel()
        self.compile_kernels(
            get_opencl_srcfile("correlator.cl"),
            compile_options=[
                "-DIMAGE_WIDTH=%d" % self.shape[1],
                "-DDTYPE=%s" % self.c_dtype,
                "-DIDX_DTYPE=%s" % self.idx_c_dtype,
                "-DSUM_WG_SIZE=%d" % 1024, # TODO tune ?
            ]
        )
        self.correlation_kernel = self.kernels.get_kernel(self.kernel_name)
        self.sum_kernel = self.kernels.get_kernel("compute_sums")


    def set_data(self, data, indices, indptr, offsets):
        # TODO handle input on GPU
        assert data.size == indices.size
        assert indptr.size == (self.shape[0]+1)*self.nframes
        assert offsets.size == self.nframes+1
        to_device = lambda x : parray.to_device(self.queue, x)
        for arr_name, arr in {"data": data, "indices": indices, "indptr": indptr, "offsets": offsets}.items():
            setattr(self, "d_" + arr_name, to_device(arr))


    def correlate(self, data, indices, indptr, offsets, output=None):
        # TODO
        #  - handle input data on GPU
        #  - handle output on GPU
        self.set_data(data, indices, indptr, offsets)
        self.d_output.fill(0) # ?



        # Compute sums
        wg = (1024, 1)
        grid = (wg[0], self.nframes)
        evt = self.sum_kernel(
            self.queue, grid, wg,
            self.d_data.data,
            self.d_offsets.data,
            self.d_sums.data,
            np.int32(self.nframes)
        )
        evt.wait()


        # Normalization is done by < I(t, p) >_p * < I(t+tau, p >_p
        # The numerator is normalized by 1/N_pixels
        # The denominator has two 1/N_pixels factors
        # We should normalize by prod(shape) after the correlation
        # but the correlation would accumulate very small numbers, making it
        # numerically inaccurate
        self.d_sums *= 1./np.sqrt((np.prod(self.shape)))

        # Correlate
        wg = (64, 1) # TODO determine best wg as function of max(nnz_line)
        grid = (wg[0], self.nframes)

        evt = self.correlation_kernel(
            self.queue, grid, wg,
            self.d_data.data,
            self.d_indices.data,
            self.d_indptr.data,
            self.d_offsets.data,
            self.d_norm_mask.data,
            self.d_sums.data,
            self.d_output.data,
            np.int32(self.shape[0]), np.int32(self.nframes)
        )
        evt.wait()

        # TODO output on device
        res = self.d_output.get().ravel()
        return res


