import numpy as np
from os import linesep

import pyopencl.array as parray
from pyopencl.tools import dtype_to_ctype
from silx.opencl.common import pyopencl as cl
from silx.opencl.processing import OpenclProcessing, KernelContainer

from ..utils import get_opencl_srcfile


class OpenclCorrelator(OpenclProcessing):

    def __init__(
        self, shape, nframes, qmask=None, dtype=np.int8, weights=None,
        scale_factor=None, extra_options={},
        ctx=None, devicetype="all", platformid=None, deviceid=None,
        block_size=None, memory=None, profile=False
    ):
        """
        Opencl Correlator
        -----------------

        This is the base class for all OpenCL correlators. Currently there are
        three kinds of correlators:
          - Dense correlator
          - CSR correlator
          - Event correlator
        Although using a different method for computing the correlation function,
        all of them should return the same result.

        Parameters
        -----------

        shape: tuple
            Shape of each XPCS frame.
        nframes: int
            Number of frames
        qmask: numpy.ndarray, optional
            Mask indicating the bins location on the frames.
            It must have the same shape as the frames.
            Value zero indicate that the pixel is not taken into account.
            If no qmask is provided, all the pixels are used for the summation.
        dtype: str or numpy.dtype, optional
            Data type of the frames
        weights: numpy.ndarray, optional
            Array of weights used to multiply each frame. Can be used for
            flat-field correction, for example.
            It must have the same shape as the frames.
        scale_factor: float or numpy.ndarray, optional
            Value used for spatial averaging.
            If all the pixels in the frames are used for the correlation
            (i.e qmask is None), the spatial averaging is sum(frame)/frame.size.
            If "qmask" is not None, sum(frame) is divided by the number of pixels
            falling into the current bin (i.e len(qmask == bin)).
            You can speficy a different value: either a float (in this case
            the same normalization is used for all bins), or an array which
            must have the same length as the number of bins.
        extra_options: dict, optional
            Advanced extra options. None available yet.


        Other parameters
        -----------------

        Please see silx.opencl.processing.OpenclProcessing for other arguments.
        """
        OpenclProcessing.__init__(
            self, ctx=ctx, devicetype=devicetype, platformid=platformid,
            deviceid=deviceid, block_size=block_size, memory=memory,
            profile=profile
        )
        self._set_parameters(
            shape, nframes, dtype, qmask, weights, scale_factor, extra_options
        )
        self._allocate_memory()

    def _set_parameters(
        self, shape, nframes, dtype, qmask, weights, scale_factor, extra_options
    ):
        self.nframes = nframes
        self._set_shape(shape)
        self._set_dtype(dtype=dtype)
        self._set_qmask(qmask=qmask)
        self._set_weights(weights=weights)
        self._set_scale_factor(scale_factor=scale_factor)
        self._configure_extra_options(extra_options)
        self.is_cpu = (self.device.type == "CPU") # move to OpenclProcessing ?


    def _set_shape(self, shape):
        if np.isscalar(shape):
            self.shape = (int(shape), int(shape))
        else:
            assert len(shape) == 2
            self.shape = shape

    def _set_dtype(self, dtype="f"):
        # add checks ?
        self.dtype = dtype
        self.output_dtype = np.float32 # TODO custom ?
        self.sums_dtype = np.uint32 # TODO custom ?
        self.c_dtype = dtype_to_ctype(self.dtype)
        self.c_sums_dtype = dtype_to_ctype(self.sums_dtype)
        self.idx_c_dtype = "int" # TODO custom ?

    def _set_qmask(self, qmask=None):
        self.qmask = qmask
        if qmask is None:
            self.bins = None
            self.n_bins = 0
            self.output_shape = (self.nframes, )
        else:
            self.bins = np.unique(qmask)[1:]
            self.n_bins = self.bins.size
            self.output_shape = (self.n_bins, self.nframes)
            self.qmask = np.ascontiguousarray(self.qmask, dtype=np.int32) #

    def _set_weights(self, weights=None):
        if weights is None:
            self.weights = np.ones(self.shape, dtype=self.output_dtype)
            return
        assert weights.shape == self.shape
        self.weights = np.ascontiguousarray(weights, dtype=self.output_dtype)
        raise ValueError("Advanced weighting is not implemented yet")

    def _set_scale_factor(self, scale_factor=None):
        if self.n_bins == 0:
            self.scale_factor = scale_factor or np.prod(self.shape)
            self.scale_factors = None
        else:
            self.scale_factor = None
            if scale_factor is not None:
                assert np.iterable(scale_factor)
                assert len(scale_factor) == self.n_bins
                self.scale_factors = scale_factor
            else:
                self.scale_factors = []
                for bin_val in self.bins:
                    self.scale_factors.append(np.sum(self.qmask == bin_val))



    def _configure_extra_options(self, extra_options):
        self.extra_options = {}
        if extra_options is not None:
            self.extra_options.update(extra_options)

    def _allocate_memory(self):
        self.d_output = parray.zeros(
            self.queue,
            self.output_shape,
            self.output_dtype
        )
        self.d_norm_mask = parray.to_device(self.queue, self.weights)
        if self.qmask is not None:
            self.d_qmask = parray.to_device(self.queue, self.qmask)


    def _set_data(self, arrays):
        """
        General-purpose function for setting the internal arrays (copy for
        numpy arrays, swap for pyopencl arrays).
        The parameter "arrays" must be a mapping array_name -> array.
        """
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


    def _reset_arrays(self, arrays_names):
        for array_name in arrays_names:
            old_array_name = "_old_" + array_name
            old_array = getattr(self, old_array_name)
            if old_array is not None:
                setattr(self, array_name, old_array)
                setattr(self, old_array_name, None)



    # Overwrite OpenclProcessing.compile_kernel, as it does not support
    # kernels outside silx/opencl/resources
    def compile_kernels(self, kernel_files=None, compile_options=None):
        kernel_files = kernel_files or self.kernel_files

        allkernels_src = []
        for kernel_file in kernel_files:
            with open(kernel_file) as fid:
                kernel_src = fid.read()
            allkernels_src.append(kernel_src)
        allkernels_src = linesep.join(allkernels_src)

        compile_options = compile_options or self.get_compiler_options()
        try:
            self.program = cl.Program(self.ctx, allkernels_src).build(options=compile_options)
        except (cl.MemoryError, cl.LogicError) as error:
            raise MemoryError(error)
        else:
            self.kernels = KernelContainer(self.program)
