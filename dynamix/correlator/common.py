import numpy as np
from os import linesep

import pyopencl.array as parray
from pyopencl.tools import dtype_to_ctype
from silx.opencl.common import pyopencl as cl
from silx.opencl.processing import OpenclProcessing, KernelContainer

from ..utils import get_opencl_srcfile


class OpenclCorrelator(OpenclProcessing):

    def __init__(
        self, shape, nframes, dtype="f", bins=0, weights=None, extra_options={},
        ctx=None, devicetype="all", platformid=None, deviceid=None,
        block_size=None, memory=None, profile=False
    ):
        """
        TODO docstring
        """
        OpenclProcessing.__init__(
            self, ctx=ctx, devicetype=devicetype, platformid=platformid,
            deviceid=deviceid, block_size=block_size, memory=memory,
            profile=profile
        )
        self._set_parameters(shape, nframes, dtype, bins, weights, extra_options)
        self._allocate_memory()

    def _set_parameters(self, shape, nframes, dtype, bins, weights, extra_options):
        self.nframes = nframes
        self._set_shape(shape)
        self._set_dtype(dtype=dtype)
        self._set_bins(bins=bins)
        self._set_weights(weights=weights)
        self._configure_extra_options(extra_options)

    def _set_shape(self, shape):
        if np.isscalar(shape):
            self.shape = (int(shape), int(shape))
        else:
            assert len(shape) == 2
            self.shape = shape

    def _set_dtype(self, dtype="f"):
        # add checks ?
        self.dtype = dtype
        self.output_dtype = np.float32
        self.c_dtype = dtype_to_ctype(self.dtype)
        self.idx_c_dtype = "int" # TODO custom ?

    def _set_bins(self, bins=0):
        # add checks ?
        self.bins = bins
        if self.bins > 0:
            self.output_shape = (self.nframes, self.bins)
        else:
            self.output_shape = (self.nframes, )

    def _set_weights(self, weights=None):
        if weights is None:
            self.weights = np.ones(self.shape, dtype=self.output_dtype)
            return
        assert weights.shape == self.shape
        self.weights = np.ascontiguousarray(weights, dtype=self.output_dtype)

    def _configure_extra_options(self, extra_options):
        self.extra_options = {}
        if extra_options is not None:
            self.extra_options.update(extra_options)

    def _allocate_memory(self):
        self.is_cpu = (self.device.type == "CPU") # move to OpenclProcessing ?
        self.d_output = parray.zeros(
            self.queue,
            self.output_shape,
            self.output_dtype
        )
        self.d_norm_mask = parray.to_device(self.queue, self.weights)
        self.d_sums = parray.zeros(self.queue, self.nframes, self.output_dtype)


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
        except (pyopencl.MemoryError, pyopencl.LogicError) as error:
            raise MemoryError(error)
        else:
            self.kernels = KernelContainer(self.program)
