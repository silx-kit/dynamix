import numpy as np

import pyopencl.array as parray
from pyopencl.tools import dtype_to_ctype
from silx.opencl.common import pyopencl as cl
from silx.opencl.processing import OpenclProcessing, KernelContainer
from ..utils import _compile_kernels


class BaseCorrelator:
    "Abstract base class for all Correlators"

    default_extra_options = {
        # dtype for indices offsets
        # must be extended to unsigned long (np.uint64) if total_nnz > 4294967295
        "offset_dtype": np.uint32,
        # dtype for intermediate result (correlation in integer).
        # must be extended to unsigned long for large nnz_per_frame and/or events counts
        "res_dtype": np.uint32,
        "sums_dtype": np.uint32,
        # dtype for q-mask.
        #  must be extended to int if number of q-bins > 127
        "qmask_dtype": np.int8,
    }


    def __init__(self):
        self.nframes = None
        self.shape = None
        self.bins = None
        self.n_bins = None
        self.output_shape = None
        self.weights = None
        self.scale_factors = None


    def _set_dtype(self, dtype):
        # Configure data types - important for OpenCL kernels,
        # as some operations are better performed on integer types (ex. atomic),
        # but some overflow can occur for large/non-sparse data.

        # data type for data. Usually the data values lie in a small range (less than 255)
        self.dtype = dtype
        # Other data types
        self._offset_dtype = self.extra_options["offset_dtype"]
        self._qmask_dtype = self.extra_options["qmask_dtype"]
        self._res_dtype = self.extra_options["res_dtype"]
        self._sums_dtype = self.extra_options["sums_dtype"]
        self._pix_idx_dtype = np.uint32 # won't change (goes from 0 to N_x*N_y)
        self._output_dtype = np.float32 # won't change


    def _set_parameters(self, shape, nframes, qmask, scale_factor, extra_options, dtype):
        self._configure_extra_options(extra_options)
        self._set_dtype(dtype)
        self.nframes = nframes
        self._set_shape(shape)
        self._set_qmask(qmask=qmask)
        self._set_scale_factor(scale_factor=scale_factor)

    def _set_shape(self, shape):
        if np.isscalar(shape):
            self.shape = (int(shape), int(shape))
        else:
            assert len(shape) == 2
            self.shape = shape

    def _set_qmask(self, qmask=None):
        self.qmask = None
        if qmask is None:
            self.bins = np.array([1], dtype=self._qmask_dtype)
            self.n_bins = 1
            self.qmask = np.ones(self.shape, dtype=self._qmask_dtype)
        else:
            self.qmask = np.ascontiguousarray(qmask, dtype=self._qmask_dtype)
            self.n_bins = self.qmask.max()
            self.bins = np.arange(1, self.n_bins + 1, dtype=self._qmask_dtype)
        self.output_shape = (self.n_bins, self.nframes)

    def _set_weights(self, weights=None):
        if weights is None:
            self.weights = np.ones(self.shape, dtype=self._output_dtype)
            return
        assert weights.shape == self.shape
        self.weights = np.ascontiguousarray(weights, dtype=self._output_dtype)
        raise ValueError("Advanced weighting is not implemented yet")

    def _set_scale_factor(self, scale_factor=None):
        if self.n_bins == 0:
            s = scale_factor or np.prod(self.shape)
            self.scale_factors = {1: s}
            return
        if scale_factor is not None:
            if not(np.iterable(scale_factor)):
                scale_factor = [scale_factor]
            assert len(scale_factor) == self.n_bins
            if isinstance(scale_factor, dict):
                self.scale_factors = scale_factor
            else:
                self.scale_factors = {k: v for k, v in zip(self.bins, scale_factor)}
        else:
            self.scale_factors = {}
            for bin_val in self.bins:
                self.scale_factors[bin_val] = np.sum(self.qmask == bin_val)

    def _configure_extra_options(self, extra_options):
        """
        :param extra_options: dict
        """
        self.extra_options = self.default_extra_options.copy()
        self.extra_options.update(extra_options or {})


class OpenclCorrelator(BaseCorrelator, OpenclProcessing):

    def __init__(
        self, shape, nframes, qmask=None, dtype=np.uint8, weights=None,
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

        shape: tuple or int
            Shape of each XPCS frame. If shape is an integer, the frames are
            assumed to be square.
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
            Advanced extra options. Available options are:
                - "offset_dtype": np.uint32
                - "res_dtype": np.uint32,
                - "sums_dtype": np.uint32,
                - "qmask_dtype": np.int8,

        Other parameters
        -----------------

        Please see silx.opencl.processing.OpenclProcessing for other arguments.
        """
        OpenclProcessing.__init__(
            self, ctx=ctx, devicetype=devicetype, platformid=platformid,
            deviceid=deviceid, block_size=block_size, memory=memory,
            profile=profile
        )
        BaseCorrelator.__init__(self)
        self._set_parameters(shape, nframes, dtype, qmask, weights, scale_factor, extra_options)
        self._allocate_memory()

    def _set_parameters(self, shape, nframes, dtype, qmask, weights, scale_factor, extra_options):
        BaseCorrelator._set_parameters(self, shape, nframes, qmask, scale_factor, extra_options, dtype)
        self._set_cl_dtypes()
        self._set_weights(weights)
        self.is_cpu = (self.device.type == "CPU")
        self.extra_options = self.default_extra_options.copy().update((extra_options or {}))

    def _set_cl_dtypes(self):
        self.c_dtype = dtype_to_ctype(self.dtype)
        self.c_sums_dtype = dtype_to_ctype(self._sums_dtype)
        self.idx_c_dtype = "int"  # TODO custom ?
        self._res_c_dtype = dtype_to_ctype(self._res_dtype)
        self._dtype_compilation_flags = [
            "-DDTYPE=%s" % self.c_dtype,
            "-DOFFSET_DTYPE=%s" % (dtype_to_ctype(self._offset_dtype)),
            "-DQMASK_DTYPE=%s" % (dtype_to_ctype(self._qmask_dtype)),
            "-DRES_DTYPE=%s" % self._res_c_dtype
        ]


    def _allocate_memory(self):
        # self.d_norm_mask = parray.to_device(self.queue, self.weights)
        if self.qmask is not None:
            self.d_qmask = parray.to_device(self.queue, self.qmask)
        self._allocated = {}
        # send scale_factors to device
        self.allocate_array("d_scale_factors", (self.n_bins,), dtype=np.float64)
        for bin_idx, scale_factor in enumerate(self.scale_factors.values()):
            self.d_scale_factors[bin_idx] = scale_factor


    def _set_data(self, arrays):
        """
        General-purpose function for setting the internal arrays (copy for
        numpy arrays, swap for pyopencl arrays).
        The parameter "arrays" must be a mapping array_name -> array.
        """
        for arr_name, array in arrays.items():
            my_array_name = "d_" + arr_name
            my_array = getattr(self, my_array_name)
            assert my_array.shape == array.shape, "%s should have shape %s, got %s" % (my_array_name, str(my_array.shape), str(array.shape))

            assert my_array.dtype == array.dtype
            if isinstance(array, np.ndarray):
                my_array.set(array)
            elif isinstance(array, parray.Array):
                setattr(self, "_old_" + my_array_name, my_array)
                setattr(self, my_array_name, array)
            else:  # support buffers ?
                raise ValueError("Unknown array type %s" % str(type(array)))

    def _reset_arrays(self, arrays_names):
        for array_name in arrays_names:
            old_array_name = "_old_d_" + array_name
            old_array = getattr(self, old_array_name)
            if old_array is not None:
                setattr(self, array_name, old_array)
                setattr(self, old_array_name, None)


    # -----------

    def allocate_array(self, array_name, shape, dtype=np.float32):
        """
        Allocate a GPU array on the current context/stream/device,
        and set 'self.array_name' to this array.

        Parameters
        ----------
        array_name: str
            Name of the array (for book-keeping)
        shape: tuple of int
            Array shape
        dtype: numpy.dtype, optional
            Data type. Default is float32.
        """
        if not self._allocated.get(array_name, False):
            new_device_arr = parray.zeros(self.queue, shape, dtype)
            setattr(self, array_name, new_device_arr)
            self._allocated[array_name] = True
        return getattr(self, array_name)

    def set_array(self, array_name, array_ref):
        """
        Set the content of a device array.

        Parameters
        ----------
        array_name: str
            Array name. This method will look for self.array_name.
        array_ref: array (numpy or GPU array)
            Array containing the data to copy to 'array_name'.
        """
        if isinstance(array_ref, parray.Array):
            current_arr = getattr(self, array_name, None)
            setattr(self, "_old_" + array_name, current_arr)
            setattr(self, array_name, array_ref)
        elif isinstance(array_ref, np.ndarray):
            self.allocate_array(array_name, array_ref.shape, dtype=array_ref.dtype)
            getattr(self, array_name).set(array_ref)
        else:
            raise ValueError("Expected numpy array or pyopencl array")
        return getattr(self, array_name)

    def get_array(self, array_name):
        return getattr(self, array_name, None)


    # -----------


    # Overwrite OpenclProcessing.compile_kernel, as it does not support
    # kernels outside silx/opencl/resources
    def compile_kernels(self, kernel_files=None, compile_options=None):
        _compile_kernels(self, kernel_files=kernel_files, compile_options=compile_options)


    def get_timings(self):
        if not(self.profile):
            raise RuntimeError("Need to instantiate this class with profile=True")
        evd = lambda e: (e.stop - e.start)/1e6
        return {e.name: evd(e) for e in self.events}


