# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2019-2019 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/
"""tests for the dense correlator.

"""

__authors__ = ["P. Paleo"]
__license__ = "MIT"
__date__ = "04/09/2019"


from time import time
import logging
import unittest
import numpy as np
from silx.opencl.common import ocl
from dynamix.test.utils import XPCSDataset
from dynamix.correlator.dense import DenseCorrelator, py_dense_correlator, FFTWCorrelator, MatMulCorrelator
from dynamix.correlator.cuda import CublasMatMulCorrelator, CUFFTCorrelator, CUFFT, cublas

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

try:
    import pyfftw
except ImportError:
    pyfftw = None



class TestDense(unittest.TestCase):

    ref = None

    @classmethod
    def load_data(cls):
        logger.debug("Loading data")
        cls.dataset = XPCSDataset("eiger_514_10k")
        cls.shape = cls.dataset.dataset_desc.frame_shape
        cls.nframes = cls.dataset.dataset_desc.nframes

    @classmethod
    def compute_reference_correlation(cls):
        if cls.ref is not None:
            return # dont re-compute ref
        t0 = time()
        ref = np.zeros(
            (cls.dataset.dataset_desc.bins, cls.dataset.dataset_desc.nframes),
            dtype=np.float32
        )
        for bin_val in range(1, cls.dataset.dataset_desc.bins+1):
            mask = (cls.dataset.qmask == bin_val)
            ref[bin_val-1] = py_dense_correlator(cls.dataset.data, mask)
        logger.info("Numpy dense correlator took %.1f ms" % ((time() - t0)*1e3))
        cls.ref = ref

    @classmethod
    def setUpClass(cls):
        cls.load_data()
        cls.compute_reference_correlation()

    def setUp(self):
        self.tol = 5e-3


    def tearDown(self):
        pass

    def compare(self, res, method_name):
        errors = res - self.ref
        errors_max = np.max(np.abs(errors), axis=1)
        for bin_idx in range(errors_max.shape[0]):
            self.assertLess(
                errors_max[bin_idx], self.tol,
                "%s: something wrong in bin index %d" % (method_name, bin_idx)
            )

    def test_dense_correlator(self):
        if ocl is None:
            self.skipTest("Need pyopencl and a working OpenCL device")
        self.correlator = DenseCorrelator(
            self.shape,
            self.nframes,
            qmask=self.dataset.qmask,
            dtype=self.dataset.dataset_desc.dtype,
            profile=True
        )
        t0 = time()
        res = self.correlator.correlate(
            self.dataset.data
        )
        logger.info("OpenCL dense correlator took %.1f ms" % ((time() - t0)*1e3))
        self.compare(res, "OpenCL dense correlator")


    def test_matmul_correlator(self):
        correlator = MatMulCorrelator(
            self.shape, self.nframes, self.dataset.qmask
        )
        t0 = time()
        res = correlator.correlate(self.dataset.data)
        logger.info("Matmul correlator took %.1f ms" % ((time() - t0)*1e3))
        self.compare(res, "Matmul correlator")


    def test_cuda_matmul_correlator(self):
        if cublas is None:
            self.skipTest("Need scikit-cuda for this test")
        correlator = CublasMatMulCorrelator(
            self.shape, self.nframes, self.dataset.qmask
        )
        t0 = time()
        res = correlator.correlate(self.dataset.data)
        logger.info("Cublas Matmul correlator took %.1f ms" % ((time() - t0)*1e3))
        self.compare(res, "Cublas Matmul correlator")


    def test_fftw_dense_correlator(self):
        if pyfftw is None:
            self.skipTest("Need pyfftw")
        self.fftcorrelator = FFTWCorrelator(
            self.shape,
            self.nframes,
            qmask=self.dataset.qmask,
            extra_options={"save_fft_plans": False}
        )
        t0 = time()
        res = self.fftcorrelator.correlate(self.dataset.data)
        logger.info("FFTw dense correlator took %.1f ms" % ((time() - t0)*1e3))
        self.compare(res, "FFTw dense correlator")


    def test_cufft_dense_correlator(self):
        if CUFFT is None:
            self.skipTest("Need pycuda scikit-cuda")
        self.fftcorrelator = CUFFTCorrelator(
            self.shape,
            self.nframes,
            qmask=self.dataset.qmask,
            extra_options={"save_fft_plans": False}
        )
        t0 = time()
        res = self.fftcorrelator.correlate(self.dataset.data)
        logger.info("CUFFT dense correlator took %.1f ms" % ((time() - t0)*1e3))
        self.compare(res, "CUFFT dense correlator")






def suite():
    testsuite = unittest.TestSuite()
    testsuite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestDense)
    )
    return testsuite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")

