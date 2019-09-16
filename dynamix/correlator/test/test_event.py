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
"""tests for the event correlator.

"""

__authors__ = ["P. Paleo"]
__license__ = "MIT"
__date__ = "04/09/2019"


from time import time
import logging
import unittest
import numpy as np
from dynamix.test.utils import XPCSDataset
from dynamix.correlator.event import EventCorrelator, FramesCompressor

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class TestEventDataStructure(unittest.TestCase):

    def setUp(self):
        logger.debug("Loading data")
        self.dataset = XPCSDataset("andor_1024_3k")
        self.shape = self.dataset.dataset_desc.frame_shape
        self.nframes = self.dataset.dataset_desc.nframes
        self.compressor = FramesCompressor(self.shape, self.nframes, 330, np.int8)

    def tearDown(self):
        pass

    def compute_reference_datastructure(self):
        logger.debug("Computing reference data compaction to events")
        return self.compressor.compress_all_stack(self.dataset.data)


    def test_progressive_compression(self):
        # Simulate progressive acquisition + compaction of frames
        logger.debug("Computing progressive data compaction")
        for frame in self.dataset.data:
            self.compressor.process_frame(frame)
        # Compare with reference implementation (compact all frames in a single pass)
        ref_data, ref_times, ref_offsets = self.compute_reference_datastructure()
        data, times, offsets = self.compressor.get_compacted_events()

        self.assertTrue(np.allclose(data, ref_data))
        self.assertTrue(np.allclose(times, ref_times))
        self.assertTrue(np.allclose(offsets, ref_offsets))


class TestEvent(unittest.TestCase):

    ref = None

    def setUp(self):
        logger.debug("Loading data")
        # self.dataset = XPCSDataset("andor_1024_10k")
        self.dataset = XPCSDataset("andor_1024_3k")
        self.max_nnz = 330 # for andor_1024_10k
        self.scale_factor = 1025171 # number of pixels actually used
        self.shape = self.dataset.dataset_desc.frame_shape
        self.nframes = self.dataset.dataset_desc.nframes
        self.ref = self.dataset.result
        self.compact_frames()
        self.tol = 5e-3

    def tearDown(self):
        pass

    def compact_frames(self):
        logger.debug("Compacting frames")
        self.frames_compressor = FramesCompressor(
                self.shape,
                self.nframes,
                self.max_nnz,
                dtype=self.dataset.dataset_desc.dtype
            )
        self.events_struct = self.frames_compressor.compress_all_stack(self.dataset.data)

    def compare(self, res, method_name):
        errors = res[:, 1:] - self.ref
        errors_max = np.max(np.abs(errors[:, :-1]), axis=1) # last point is wrong in ref...

        for bin_idx in range(errors_max.shape[0]):
            self.assertLess(
                errors_max[bin_idx], self.tol,
                "%s: something wrong in bin index %d" % (method_name, bin_idx)
            )

    def test_event_correlator(self):
        # Get events data structure
        vol_data, vol_times, offsets = self.events_struct
        # Init correlator
        self.correlator = EventCorrelator(
            self.shape,
            self.nframes,
            dtype=self.dataset.dataset_desc.dtype,
            max_events_count=self.max_nnz, # np.diff(offsets).max()
            total_events_count=vol_data.size,
            scale_factor=self.scale_factor,
            profile=True
        )
        # Correlate
        t0 = time()
        res = self.correlator.correlate(
            vol_times,
            vol_data,
            offsets
        )
        logger.info("OpenCL event correlator took %.1f ms" % ((time() - t0)*1e3))
        self.compare(res, "OpenCL event correlator")



def suite():
    testsuite = unittest.TestSuite()
    testsuite.addTest(
        # unittest.defaultTestLoader.loadTestsFromTestCase(TestEventDataStructure)
        unittest.defaultTestLoader.loadTestsFromTestCase(TestEvent)
    )
    return testsuite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")

