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
__date__ = "01/07/2021"


import time
import logging
import unittest
import numpy 
from ...test.utils import XPCSDataset, DatasetDescription
from ..event import EventCorrelator, FramesCompressor
from ...tools.decorators import timeit
np = numpy
logger = logging.getLogger(__name__)

@timeit
def build_random_stack(shape=(10,11), nframes=111, dtype="uint8", nnz=10):
    dtype = numpy.dtype(dtype)
    npix = numpy.prod(shape)
    maxi = numpy.iinfo(dtype).max
    stack = np.zeros((nframes,)+(shape), dtype=dtype)
    nnz2d = numpy.random.randint(1,nnz, size=shape)
    pix_ptr = numpy.cumsum(nnz2d)
    last = pix_ptr[-1]
    pix_ptr = numpy.concatenate(([0],pix_ptr))
    values = numpy.random.randint(0, maxi, last)
    times = numpy.random.randint(0,nframes,last)
    for i in range(npix):
        line = stack[:, i//shape[-1], i%shape[-1]]
        line[times[pix_ptr[i]:pix_ptr[i+1]]] = values[pix_ptr[i]:pix_ptr[i+1]]
    return stack
 
def build_fake_dataset(shape=(233,231), nframes=513, dtype="uint8", nnz=17):
    "Build a completely fake  dataset to test sparsification"
    data = build_random_stack(shape, nframes, dtype, nnz)
    
    self = XPCSDataset()
    self.name = "fake"
    descr = {"name":"fake",
             "data":data, 
             "result":None, "description":f"Fake dataset with nnz={nnz}", 
             "nframes":nframes, 
             "dtype":dtype,
             "frame_shape":shape, 
             "bins":None}
    self.dataset_desc = DatasetDescription(**descr)
    self.data = data
    self.data_file = None
    self.result_file = None
    self.result = None
    return self

class TestEventDataStructure(unittest.TestCase):

    def setUp(self):
        logger.debug("TestEventDataStructure.setUp")
        #self.dataset = XPCSDataset("andor_1024_3k")
        #self.nnz = 330
        self.nnz = 10
        self.dtype = np.dtype("int8") 
        self.shape = (101, 103)
        self.nframes = 257        
        self.dataset = build_fake_dataset(dtype=self.dtype, shape=self.shape, nnz=self.nnz, nframes=self.nframes)

    def tearDown(self):
        self.dataset = self.shape = self.nframes = self.dtype = self.nnz = None
    
    @property
    def compressor(self):
        "Single use compressor"
        return FramesCompressor(self.shape, self.nframes, self.nnz, self.dtype)
    
    @timeit
    def compute_reference_datastructure(self):
        return self.compressor.compress_all_stack(self.dataset.data)

    def test_progressive_compression(self):
        # Simulate progressive acquisition + compaction of frames
        logger.debug("Computing progressive data compaction")
        compressor = self.compressor
        for frame in self.dataset.data:
            compressor.process_frame(frame)
        # Compare with reference implementation (compact all frames in a single pass)
        ref_data, ref_times, ref_offsets = self.compute_reference_datastructure()
        data, times, offsets = compressor.get_compacted_events()

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
        self.events_struct = None
        self.compact_frames()
        self.tol = 5e-3

    def tearDown(self):
        self.dataset = None
        self.events_struct = None

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
        t0 = time.perf_counter()
        res = self.correlator.correlate(
            vol_times,
            vol_data,
            offsets
        )
        logger.info("OpenCL event correlator took %.1f ms", ((time.perf_counter() - t0)*1e3))
        self.compare(res, "OpenCL event correlator")



def suite():
    testsuite = unittest.TestSuite()
    testsuite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestEventDataStructure))
    testsuite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestEvent))
    return testsuite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")

