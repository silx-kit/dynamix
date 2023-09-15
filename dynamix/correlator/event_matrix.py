from os import path
from math import sqrt
from multiprocessing.pool import ThreadPool
import numpy as np
from pyopencl.elementwise import ElementwiseKernel
import pyopencl.array as parray
from ..utils import get_opencl_srcfile, updiv
from .common import OpenclCorrelator


class MatrixEventCorrelatorBase(OpenclCorrelator):

    """
    Base class for MatrixEventCorrelator
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
        self.cl_div = None


    def _check_arrays(self, data, pixel_indices, offsets):
        if data.dtype != self.dtype:
            raise ValueError("Expected dtype %s for data, but got %s" % (self.dtype, data.dtype))
        if pixel_indices.dtype != self._pix_idx_dtype:
            raise ValueError("Expected dtype %s for pixel_indices, but got %s" % (self._pix_idx_dtype, pixel_indices.dtype))
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
            np.int32(self.n_bins)
        )
        evt.wait()
        self.profile_add(evt, "Correlate d_sums")


    def _get_normalized_ttcf_opencl(self, d_num, d_denom):
        if self.cl_div is None:
            self.cl_div = ElementwiseKernel(
                self.ctx,
                "float* res, %s* num, %s* denom" % (self._res_c_dtype, self._res_c_dtype),
                "num[i] /= denom[i]",
                "float_int_divide",
                preamble='#include "dtypes.h',
                options=self._compile_options,
            )
        normalized_ttcf = self.allocate_array("normalized_ttcf", d_num.shape, dtype=np.float32)
        self.cl_div(normalized_ttcf, d_num, d_denom)
        return normalized_ttcf


    def _get_normalized_ttcf_numpy(self, num, denom):
        res = num.astype(np.float32)
        denom = denom.astype(np.float32)
        res /= denom
        return res


    def get_normalized_ttcf(self, calculate_on_device=False):
        """
        From the previously-computed numerator and denominator of TTCF,
        compute the normalized matrix, i.e num/denom (including scaling factors).

        Parameters
        ----------
        calculate_on_device: bool, optional
            Whether to perform computations on GPU. This needs an additional memory allocation,
            so it might be not suitable for large number of frames.

        Returns
        -------
        normalized_ttcf: numpy.ndarray
            A numpy array containing the normalized two-times correlation matrix.
            The resulting array has shape (n_bins, n_frames, n_frames), i.e there are n_bins matrices.
            Only the upper half of each matrix was computed.
        """
        num = self.d_corr_matrix
        denom = self.d_sums_corr_matrix
        if calculate_on_device:
            ttcf_allbins = self._get_normalized_ttcf_opencl(num, denom)
        else:
            num = num.get()
            denom = denom.get()
            ttcf_allbins = self._get_normalized_ttcf_numpy(num, denom)
        normalized_ttcf = np.zeros((self.n_bins, self.n_times, self.n_times), "f")
        for bin_idx in range(self.n_bins):
            normalized_ttcf[bin_idx] = flat_to_square(ttcf_allbins[bin_idx], dtype="f")
        return normalized_ttcf


    def get_correlation_function(self, bin_idx, n_threads=1, calc_std=False, return_num_denom=False, dtype=np.float64):
        """
        Compute the two-time correlation function.
        This function only uses numpy, so it's quite slow for large arrays.

        Parameters
        ----------
        bin_idx: integer
            Index of the bin to compute. Starts with 0!
        n_threads: int, optional
            Number of threads to compute the final step
        calc_std: bool, optional
            Whether to also compute standard deviation. Default is False.
        dtype: numpy.dtype, optional
            Data type for performing the final computation. Default is float64
        """
        num = self.d_corr_matrix[bin_idx].get()
        num = flat_to_square(num, shape=(self.nframes, self.n_times), dtype=dtype)

        denom = self.d_sums_corr_matrix[bin_idx].get()
        denom = flat_to_square(denom, shape=(self.nframes, self.n_times), dtype=dtype)

        res = np.zeros(self.n_times, dtype=dtype)
        if calc_std:
            dev = np.zeros_like(res)

        def _compute_diag(i):
            dia_n = np.diag(num, k=i)
            dia_d = np.diag(denom, k=i)
            res[i] = np.sum(dia_n) / np.sum(dia_d)
            if calc_std:
                dev[i] = np.std(dia_n / dia_d) / sqrt(len(dia_d))

        with ThreadPool(n_threads) as tp:
            tp.map(_compute_diag, range(self.n_times))

        res *= self.scale_factors[bin_idx + 1]

        ret = [res]
        if calc_std:
            dev *= self.scale_factors[bin_idx + 1]
            ret += [dev]
        if return_num_denom:
            ret += [num, denom]
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)


    def cc(self, data, pixel_indices, offsets, bins="all", **kwargs):
        """
        Compute XPCS functions from space-compacted or time-compacted data.

        Parameters
        ----------
        data: pyopencl.Array
            space-comacted data or time-compacted data
        pixel_indices: pyopencl.Array
            For space-compacted data: pixel indices
            For time-compacted data: times indices
        offsets: pyopencl.Array
            Frames offsets.
            For space-compacted data: offsets[i+1] - offsets[i] is the number of non-zero elements in frame at time=i
               i.e offsets[i+1] - offsets[i] = (xpcs_data_dense[i, :] > 0).sum()  # assuming 1D spatial indexing in the second axis
            For time-compacted data: offsets[i+1] - offsets[i] is the number of non-zero elements at pixel location i
               i.e offsets[i+1] - offsets[i] = (xpcs_data_dense[:, i] > 0).sum()  # assuming 1D spatial indexing in the second axis
        return_num_and_denom: bool, optional
            Whether to return the TTCF as (num, denom) instead of num/denom.
        bins: str or list of int, optional
            Bins values to compute the correlation function.
            By default the computations are done on all scattering vectors defined in qmask.


        Returns
        --------
        g2: numpy.ndarray
            One-dimensional array of size 'n_frames': the g2 correlation function.
        std: numpy.ndarray
            One-dimensional array of size 'n_frames': the standard deviation on correlation function.
        num_or_ttcf: numpy.ndarray
            If return_num_and_denom is False, this is the normalized TTCF, i.e "num/denom".
            If return_num_and_denom is True, this is "num".
        denom: numpy.ndarray
            If return_num_and_denom is False, this is None.
            If return_num_and_denom is True, this is "denom".
        """
        # Build numerator (call subclass methods)
        self.build_correlation_matrices(data, pixel_indices, offsets, **kwargs) # build_correlation_matrix() has some extra args

        if bins == "all" or bins is None:
            bins_indices = self.bins - 1
        else:
            bins_indices = bins
        for bin_idx in bins_indices:
            g2, std, _, _ = self._compute_final_ttcf(bin_idx, calc_std=True)




class SMatrixEventCorrelator(MatrixEventCorrelatorBase):
    kernel_files = ["matrix_evtcorrelator.cl", "utils.cl"]

    """
    A class for computing the two-times correlation function (TTCF) from space-compacted data.
    """

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
        self._compile_options = self._dtype_compilation_flags + [
            "-I%s" % path.dirname(get_opencl_srcfile("dtypes.h")),
            "-DSHARED_ARRAYS_SIZE=%d" % 11000,  # for build_correlation_matrix_v2b, not working
        ]
        self.compile_kernels(
            kernel_files=kernel_files,
            compile_options=self._compile_options,
        )
        self.build_correlation_matrix_kernel = self.kernels.get_kernel("build_correlation_matrix")
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

    def build_correlation_matrices(self, data, pixel_indices, offsets, max_space_nnz=None, check=True):
        """
        Build the following correlation matrices:
          - numerator: < I(t1, p) * I(t2, p) >_p
          - denominator: < I(t1, p) >_p * < I(t2, p) >_p
        Where < . >_p denotes summing over a given q-bin.
        """
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

        evt = self.build_correlation_matrix_kernel(
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
        self.profile_add(evt, "Build matrix correlation")

        # Build denominator
        self._correlate_sums()

        return self.d_corr_matrix, self.d_sums_corr_matrix



class TMatrixEventCorrelator(MatrixEventCorrelatorBase):
    kernel_files = ["matrix_evtcorrelator_time.cl", "utils.cl"]

    """
    A class for computing the two-times correlation function (TTCF) from time-compacted data.
    """

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
        self._compile_options = self._dtype_compilation_flags + [
            "-DMAX_EVT_COUNT=%d" % self._max_nnz,
            "-I%s" % path.dirname(get_opencl_srcfile("dtypes.h")),
        ]
        self.compile_kernels(
            kernel_files=kernel_files,
            compile_options=self._compile_options,
        )
        self.build_correlation_matrix_kernel_times = self.kernels.get_kernel(
            "build_correlation_matrix_times_representation"
        )
        self.build_scalar_correlation_matrix = self.kernels.get_kernel("build_flattened_scalar_correlation_matrix")

    def build_correlation_matrices(self, data, times, offsets, check=True):
        """
        Build the following correlation matrices:
          - numerator: < I(t1, p) * I(t2, p) >_p
          - denominator: < I(t1, p) >_p * < I(t2, p) >_p
        Where < . >_p denotes summing over a given q-bin.
        """
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

        # Build denominator
        self._correlate_sums()

        return self.d_corr_matrix, self.d_sums_corr_matrix


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
