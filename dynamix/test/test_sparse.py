import unittest
import numpy as np
from silx.io.url import DataUrl
from silx.io import get_data
from dynamix.sparse import (
    dense_to_space,
    dense_to_times,
    space_to_dense,
    times_to_dense,
    space_to_times,
    SpaceToTimeCompaction,
    estimate_max_events_in_times_from_space_compacted_data,
)


tests_config = {
    # TODO
    "xpcs_data_path": DataUrl(
        file_path="/data/id10/inhouse/software/dynamix/datasets/dataset03/scan0002/eiger4m_0000.h5",
        data_path="/entry_0000/measurement/data",
        data_slice=(slice(0, 100), slice(None, None), slice(None, None)),
        scheme="silx",
    ),
    "qmask_path": "/data/id10/inhouse/software/dynamix/datasets/dataset03/analysis/Pt1_10GPa_2_242C/scan0002_0_5000/Pt1_10GPa_2_242C_qmask.npy",
    "data_type": np.uint8,
}


class TestSparseFormats(unittest.TestCase):
    def setUp(self):
        import hdf5plugin

        self.xpcs_frames = get_data(tests_config["xpcs_data_path"])
        self.qmask = np.load(tests_config["qmask_path"])
        self.xpcs_frames = (self.xpcs_frames * (self.qmask > 0)).astype(tests_config["data_type"])
        self.n_frames = self.xpcs_frames.shape[0]
        self.frame_shape = self.xpcs_frames.shape[1:]

    def tearDown(self):
        pass

    @staticmethod
    def compare_int_volumes(arr1, arr2, err_msg=""):
        # integer comparison - don't cast to float32, offsets int might be wrong after conversion
        abs_diff = np.abs(arr1.astype(np.int64) - arr2.astype(np.int64))
        assert abs_diff.max() == 0, err_msg

    def test_space_compaction(self):
        data, pix_idx, offset = dense_to_space(self.xpcs_frames)
        # TODO compare with a reference ?
        dense = space_to_dense(data, pix_idx, offset, self.frame_shape)
        self.compare_int_volumes(dense, self.xpcs_frames, "space compaction")

    def test_times_compaction(self):
        data, times, offsets = dense_to_times(self.xpcs_frames)
        # TODO compare with a reference ?
        dense = times_to_dense(data, times, offsets, self.n_frames, self.frame_shape)
        self.compare_int_volumes(dense, self.xpcs_frames, "times compaction")

    def test_from_space_to_times(self):
        data, pix_idx, offset = dense_to_space(self.xpcs_frames)
        t_data0, t_times0, t_offsets0 = dense_to_times(self.xpcs_frames)
        t_data, t_times, t_offsets = space_to_times(data, pix_idx, offset, self.frame_shape, 20)
        self.compare_int_volumes(t_data, t_data0, "space -> times: data")
        self.compare_int_volumes(t_times, t_times0, "space -> times: times")
        self.compare_int_volumes(t_offsets, t_offsets0, "space -> times: offsets")

    def test_from_space_to_times_opencl(self):
        data, pix_idx, offset = dense_to_space(self.xpcs_frames)
        t_data, t_times, t_offsets = dense_to_times(self.xpcs_frames)

        estimated_time_nnz = estimate_max_events_in_times_from_space_compacted_data(
            pix_idx, offset, estimate_from_n_frames=self.n_frames
        )
        true_time_nnz = (self.xpcs_frames > 0).sum(axis=0).max()
        assert estimated_time_nnz == true_time_nnz

        estimated_time_nnz2 = estimate_max_events_in_times_from_space_compacted_data(
            pix_idx, offset, estimate_from_n_frames=100
        )
        assert estimated_time_nnz2 >= true_time_nnz

        space_to_time_ocl = SpaceToTimeCompaction(
            self.frame_shape, max_time_nnz=estimated_time_nnz2, dtype=self.xpcs_frames.dtype
        )

        d_t_data, d_t_times, d_t_offsets = space_to_time_ocl.space_compact_to_time_compact(data, pix_idx, offset)

        self.compare_int_volumes(d_t_data.get(), t_data)
        self.compare_int_volumes(d_t_times.get(), t_times)
        self.compare_int_volumes(d_t_offsets.get(), t_offsets)






def suite():
    testsuite = unittest.TestSuite()
    testsuite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestSparseFormats))
    return testsuite


if __name__ == "__main__":
    unittest.main(defaultTest="suite")
