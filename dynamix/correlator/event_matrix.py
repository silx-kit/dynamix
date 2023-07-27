from os import path
from time import perf_counter
import numpy as np
from pyopencl import LocalMemory
import pyopencl.array as parray
from ..utils import get_opencl_srcfile, updiv
from .common import OpenclCorrelator


class MatrixEventCorrelator(OpenclCorrelator):

    """
    A class to compute the correlation function for sparse XPCS data.
    """

    def __init__(
        self,
        shape,
        nframes,
        n_times=None,
        qmask=None,
        dtype=np.uint8,
        weights=None,
        scale_factor=None,
        extra_options={},
        **oclprocessing_kwargs
    ):
        super().__init__(
            shape,
            nframes,
            qmask=qmask,
            dtype=dtype,
            weights=weights,
            scale_factor=scale_factor,
            extra_options=extra_options,
            **oclprocessing_kwargs
        )
        self._init_corr_matrix(n_times)

    def _init_corr_matrix(self, n_times):
        self.n_times = n_times or self.nframes
        self.correlation_matrix_full_shape = (self.nframes, self.n_times)
        if (self.nframes * (self.n_times + 1) % 2) != 0:
            print("Warning: incrementing n_times to have a proper-sized matrix")
            self.n_times += 1
        self.correlation_matrix_flat_size = self.nframes * (self.n_times + 1) // 2
        self.d_corr_matrix = self.allocate_array(
            "corr_matrix", (self.n_bins, self.correlation_matrix_flat_size), dtype=self._res_dtype
        )
        self.d_sums_corr_matrix = self.allocate_array(
            "sums_corr_matrix", (self.n_bins, self.correlation_matrix_flat_size), dtype=self._res_dtype
        )
        self.d_sums = self.allocate_array("sums", (self.n_bins, self.nframes), dtype=self._res_dtype)


    def _check_arrays(self, data, pixel_indices, offsets):
        if data.dtype != self.dtype:
            raise ValueError("Expected dtype %s for offsets, but got %s" % (self.dtype, data.dtype))
        if pixel_indices.dtype != self._pix_idx_dtype:
            raise ValueError("Expected dtype %s for offsets, but got %s" % (self._pix_idx_dtype, pixel_indices.dtype))
        if offsets.dtype != self._offset_dtype:
            raise ValueError("Expected dtype %s for offsets, but got %s" % (self._offset_dtype, offsets.dtype))

    def _correlate_sums(self):
        wg = None
        grid = (self.nframes, self.n_times, self.n_bins)
        evt = self.build_scalar_correlation_matrix(
            self.queue,
            grid,
            wg,
            self.d_sums.data,
            self.d_sums_corr_matrix.data,
            np.int32(self.nframes),
            np.int32(self.n_times),
        )
        evt.wait()
        self.profile_add(evt, "Correlate d_sums")


class SMatrixEventCorrelator(MatrixEventCorrelator):
    kernel_files = ["matrix_evtcorrelator.cl", "utils.cl"]

    def __init__(
        self,
        shape,
        nframes,
        n_times=None,
        max_space_nnz=None,
        qmask=None,
        dtype="f",
        weights=None,
        scale_factor=None,
        extra_options={},
        **oclprocessing_kwargs
    ):
        super().__init__(
            shape,
            nframes,
            n_times=n_times,
            qmask=qmask,
            dtype=dtype,
            weights=weights,
            scale_factor=scale_factor,
            extra_options=extra_options,
            **oclprocessing_kwargs
        )
        self._setup_kernels(max_space_nnz)

    def _setup_kernels(self, max_space_nnz):
        self.max_space_nnz = max_space_nnz
        kernel_files = list(map(get_opencl_srcfile, self.kernel_files))
        self.compile_kernels(
            kernel_files=kernel_files,
            compile_options=self._dtype_compilation_flags
            + [
                "-I%s" % path.dirname(get_opencl_srcfile("dtypes.h")),
                "-DSHARED_ARRAYS_SIZE=%d" % 11000,  # for build_correlation_matrix_v2b, not working
            ],
        )
        self.build_correlation_matrix_kernel_v1 = self.kernels.get_kernel("build_correlation_matrix_v1")
        self.build_correlation_matrix_kernel_v2 = self.kernels.get_kernel("build_correlation_matrix_v2")
        self.build_correlation_matrix_kernel_v2b = self.kernels.get_kernel("build_correlation_matrix_v2b")
        self.build_correlation_matrix_kernel_v3 = self.kernels.get_kernel("build_correlation_matrix_v3")
        self.build_scalar_correlation_matrix = self.kernels.get_kernel("build_flattened_scalar_correlation_matrix")

    def _get_max_space_nnz(self, max_space_nnz, offsets):
        if max_space_nnz is not None:
            return max_space_nnz
        elif self.max_space_nnz is not None:
            return self.max_space_nnz
        else:
            if isinstance(offsets, parray.Array):
                h_offsets = offsets.get()
            else:
                h_offsets = offsets
            return np.diff(h_offsets).max()

    def _get_compacted_qmask(self, pixel_indices, offsets):
        # compaction of qmask (duplicates information!)
        # qmask_compacted[pixel_compact_idx] give the associated q-bin
        qmask_compacted = []
        qmask1D = self.qmask.ravel()
        for frame_idx in range(self.nframes):
            i_start = offsets[frame_idx]
            i_stop = offsets[frame_idx + 1]
            qmask_compacted.append(qmask1D[pixel_indices[i_start:i_stop]])
        qmask_compacted = np.hstack(qmask_compacted)
        return qmask_compacted
        #

    def _build_correlation_matrix_v1(self, data, pixel_indices, offsets):
        qmask_compacted = self._get_compacted_qmask(pixel_indices, offsets)
        d_qmask_compacted = parray.to_device(self.queue, qmask_compacted.astype(np.int8))  # !

        # TODO data setter
        d_data = self.d_data = parray.to_device(self.queue, data.astype(np.uint8))
        d_pixel_indices = parray.to_device(self.queue, pixel_indices.astype(np.uint32))
        d_offsets = parray.to_device(self.queue, offsets.astype(np.uint32))
        self.d_sums = parray.zeros(self.queue, self.nframes, np.uint32)  # TODO dtype
        #

        wg = None
        grid = (self.n_times, self.n_bins)

        evt = self.build_correlation_matrix_kernel_v1(
            self.queue,
            grid,
            wg,
            d_data.data,
            d_pixel_indices.data,
            d_offsets.data,
            d_qmask_compacted.data,
            self.d_corr_matrix.data,
            # self.d_sums.data,
            np.int32(self.nframes),
            np.int32(self.n_times),
            np.int32(1),
        )
        evt.wait()
        self.profile_add(evt, "Build matrix correlation (v1)")

        return self.d_corr_matrix

    def _build_correlation_matrix_v2(self, data, pixel_indices, offsets):
        qmask_compacted = self._get_compacted_qmask(pixel_indices, offsets)
        d_qmask_compacted = parray.to_device(self.queue, qmask_compacted.astype(np.int8))  # !

        # TODO data setter
        d_data = self.d_data = parray.to_device(self.queue, data.astype(np.uint8))
        d_pixel_indices = parray.to_device(self.queue, pixel_indices.astype(np.uint32))
        d_offsets = parray.to_device(self.queue, offsets.astype(np.uint32))
        self.d_sums = parray.zeros(self.queue, self.nframes, np.uint32)  # TODO dtype
        #

        wg = None
        grid = (self.n_times, self.nframes)

        evt = self.build_correlation_matrix_kernel_v2(
            self.queue,
            grid,
            wg,
            d_data.data,
            d_pixel_indices.data,
            d_offsets.data,
            d_qmask_compacted.data,
            self.d_corr_matrix.data,
            # self.d_sums.data,
            np.int32(self.nframes),
            np.int32(self.n_times),
            np.int32(1),  # current q-bin
        )
        evt.wait()
        self.profile_add(evt, "Build matrix correlation (v2)")

        return self.d_corr_matrix

    def _build_correlation_matrix_v2b(self, data, pixel_indices, offsets):
        qmask_compacted = self._get_compacted_qmask(pixel_indices, offsets)
        d_qmask_compacted = parray.to_device(self.queue, qmask_compacted.astype(np.int8))  # !

        # TODO proper set_data
        d_data = self.d_data = parray.to_device(self.queue, data.astype(np.uint8))
        d_pixel_indices = parray.to_device(self.queue, pixel_indices.astype(np.uint32))
        d_offsets = parray.to_device(self.queue, offsets.astype(np.uint32))
        #

        wg = (1024, 1)  # TODO tune
        grid = (updiv(self.n_times, wg[0]) * wg[0], self.nframes)

        evt = self.build_correlation_matrix_kernel_v2b(
            self.queue,
            grid,
            wg,
            d_data.data,
            d_pixel_indices.data,
            d_offsets.data,
            d_qmask_compacted.data,
            self.d_corr_matrix.data,
            np.int32(self.nframes),
            np.int32(self.n_times),
            # np.int32(self.n_bins)
            np.int32(1),
            LocalMemory(11000 * 1),
            LocalMemory(11000 * 1),
        )
        evt.wait()
        self.profile_add(evt, "Build correlation matrix (v2b)")

        return self.d_corr_matrix

    def _build_correlation_matrix_v3(self, data, pixel_indices, offsets, max_space_nnz=None, check=True):
        qmask_compacted = self._get_compacted_qmask(pixel_indices, offsets)
        d_qmask_compacted = parray.to_device(self.queue, qmask_compacted.astype(np.int8))  # !

        if check:
            self._check_arrays(data, pixel_indices, offsets)

        max_space_nnz = self._get_max_space_nnz(max_space_nnz, offsets)
        d_data = self.set_array("data", data)
        d_pixel_indices = self.set_array("pixel_indices", pixel_indices)
        d_offsets = self.set_array("offsets", offsets)

        wg = (512, 1)  # TODO tune
        grid = (updiv(max_space_nnz, wg[0]) * wg[0], self.nframes)

        evt = self.build_correlation_matrix_kernel_v3(
            self.queue,
            grid,
            wg,
            d_data.data,
            d_pixel_indices.data,
            d_offsets.data,
            d_qmask_compacted.data,
            self.d_corr_matrix.data,
            self.d_sums.data,
            np.int32(self.nframes),
            np.int32(self.n_times),
        )
        evt.wait()
        self.profile_add(evt, "Build matrix correlation (v3)")

        return self.d_corr_matrix

    build_correlation_matrix = _build_correlation_matrix_v3


class TMatrixEventCorrelator(MatrixEventCorrelator):
    kernel_files = ["matrix_evtcorrelator_time.cl", "utils.cl"]

    def __init__(
        self,
        shape,
        nframes,
        n_times=None,
        max_time_nnz=250,
        qmask=None,
        dtype="f",
        weights=None,
        scale_factor=None,
        extra_options={},
        **oclprocessing_kwargs
    ):
        super().__init__(
            shape,
            nframes,
            n_times=n_times,
            qmask=qmask,
            dtype=dtype,
            weights=weights,
            scale_factor=scale_factor,
            extra_options=extra_options,
            **oclprocessing_kwargs
        )
        self._setup_kernels(max_time_nnz)

    def _setup_kernels(self, max_time_nnz):
        self._max_nnz = max_time_nnz
        kernel_files = list(map(get_opencl_srcfile, self.kernel_files))
        self.compile_kernels(
            kernel_files=kernel_files,
            compile_options=self._dtype_compilation_flags
            + [
                "-DMAX_EVT_COUNT=%d" % self._max_nnz,
                "-I%s" % path.dirname(get_opencl_srcfile("dtypes.h")),
            ],
        )
        self.build_correlation_matrix_kernel_times = self.kernels.get_kernel(
            "build_correlation_matrix_times_representation"
        )

    def build_correlation_matrix(self, data, times, offsets, check=True):
        wg = None
        grid = tuple(self.shape[::-1])

        if check:
            self._check_arrays(data, times, offsets)

        d_data = self.set_array("t_data", data)
        d_times = self.set_array("t_times", times)
        d_offsets = self.set_array("t_offsets", offsets)

        self.d_corr_matrix.fill(0)
        self.d_sums.fill(0)

        evt = self.build_correlation_matrix_kernel_times(
            self.queue,
            grid,
            wg,
            d_times.data,
            d_data.data,
            d_offsets.data,
            self.d_qmask.data,
            self.d_corr_matrix.data,
            self.d_sums.data,
            np.int32(self.shape[1]),
            np.int32(self.shape[0]),
            np.int32(self.nframes),
            np.int32(self.n_times),
            np.int32(1), # pre-sort
        )
        evt.wait()
        self.profile_add(evt, "Build matrix correlation (times repr.)")

        return self.d_corr_matrix


def flat_to_square(arr, shape=None, dtype=np.uint32):
    """
    Convert a flattened correlation "matrix" to a rectangular matrix.

    Paramaters
    ----------
    arr: numpy.ndarray
        1D array, flattened correlation matrix
    shape: tuple of int, optional
        Correlation matrix shape. If not given, resulting matrix is assumed square.
    """
    if shape is None:
        n2 = arr.size
        n = int(((1 + 8 * n2) ** 0.5 - 1) / 2)
        shape = (n, n)
    idx = np.triu_indices(shape[0], m=shape[1])
    res = np.zeros(shape, dtype=dtype)
    res[idx] = arr
    return res
