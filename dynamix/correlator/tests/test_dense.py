from time import time
import logging
import unittest
import numpy as np
from dynamix.test.utils import XPCSDataset
from dynamix.correlator.dense import DenseCorrelator, py_dense_correlator

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestDense(unittest.TestCase):

    def setUp(self):
        logger.debug("Loading data")
        self.dataset = XPCSDataset("eiger_514_10k")
        logger.setLevel(logging.DEBUG)

    def tearDown(self):
        pass

    def test_dense_correlator(self):
        self.correlator = DenseCorrelator(
            self.dataset.dataset_desc.frame_shape,
            self.dataset.dataset_desc.nframes,
            qmask=self.dataset.qmask,
            dtype=self.dataset.dataset_desc.dtype,
            profile=True
        )
        logger.debug("Running OpenCL dense correlator")
        t0 = time()
        res = self.correlator.correlate(
            self.dataset.data
        )
        logger.debug("OpenCL dense correlator took %.1f ms" % ((time() - t0)*1e3))
        logger.debug("Running numpy dense correlator")
        t0 = time()
        ref = self.compute_reference_correlation()
        logger.debug("Numpy dense correlator took %.1f ms" % ((time() - t0)*1e3))

        self.assertTrue(
            np.allclose(res[:, 1:], ref),
            "Something wrong with Dense correlator"
        )


    def compute_reference_correlation(self):
        ref = np.zeros(
            (self.dataset.dataset_desc.bins, self.dataset.dataset_desc.nframes -1),
            dtype=np.float32
        )
        for bin_val in range(1, self.dataset.dataset_desc.bins+1):
            mask = (self.dataset.qmask == bin_val)
            ref[bin_val-1] = py_dense_correlator(self.dataset.data, mask)
        return ref


def suite():
    testsuite = unittest.TestSuite()
    testsuite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestDense)
    )
    return testsuite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")

