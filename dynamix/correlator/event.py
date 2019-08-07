import numpy as np
from silx.opencl.common import pyopencl as cl
from silx.opencl.processing import OpenclProcessing, KernelContainer
from pyopencl.tools import dtype_to_ctype
import pyopencl.array as parray
from ..utils import get_opencl_srcfile

class EventCorrelator(OpenclCorrelator):

    kernel_files = ["evtcorrelator.cl", "sums.cl"]


    def __init__(
        self, shape, nframes, dtype="f", bins=0, weights=None, extra_options={},
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
        self._setup_kernels()



    def _setup_kernels(self):
        kernel_files = ["evtcorrelator.cl", "sums.cl"]
        kernel_files = list(map(get_opencl_srcfile, self.kernel_files))
        self.compile_kernels(
            kernel_files=kernel_files,
            compile_options=[
                "-DIMAGE_WIDTH=%d" % self.shape[1],
                "-DDTYPE=%s" % self.c_dtype,
                "-DMAX_EVT_COUNT=10" # TODO tune
            ]
        )
        self.correlation_kernel = self.kernels.get_kernel("event_correlator_oneQ") # TODO tune
        self.sum_kernel = self.kernels.get_kernel("compute_sums")

        self.grid = self.shape[::-1]
        self.wg = None # tune ?
