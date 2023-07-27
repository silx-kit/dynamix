"""
Module for compacting/decompacting data into various formats.

For now two compact formats are available:
  - "times"
  - "space"
"""
from multiprocessing.pool import ThreadPool
import numpy as np
from math import ceil
from pyopencl.tools import dtype_to_ctype
import pyopencl.array as parray
from silx.opencl.processing import OpenclProcessing
from .utils import get_opencl_srcfile, _compile_kernels


def dense_to_times(frames):
    assert frames.ndim == 3
    framesT = np.moveaxis(frames, 0, -1)
    times = np.arange(frames.shape[0], dtype=np.int32)
    nnz_indices = np.where(framesT > 0)

    res_data = framesT[nnz_indices]
    res_times = times[nnz_indices[-1]]

    offsets = np.cumsum((frames > 0).sum(axis=0).ravel())
    res_offsets = np.zeros(np.prod(frames.shape[1:]) + 1, dtype=np.uint32)
    res_offsets[1:] = offsets[:]

    return res_data, res_times, res_offsets


def times_to_dense(t_data, t_times, t_offset, n_frames, frame_shape):
    res = np.zeros((n_frames,) + frame_shape, dtype=t_data.dtype)
    for i in range(frame_shape[0]):
        for j in range(frame_shape[1]):
            idx = i * frame_shape[1] + j
            start, stop = t_offset[idx], t_offset[idx + 1]
            if stop - start == 0:
                continue
            events = t_data[start:stop]
            times = t_times[start:stop]
            res[times, i, j] = events
    return res


# eg. dense_to_space(xpcs * (qmask > 0))
def dense_to_space(xpcs_frames):
    data = []  # int8
    pixel_idx = []  # uint32
    offset = [0]  # uint64

    for frame in xpcs_frames:
        frame_data = frame.ravel()
        nnz_idx = np.nonzero(frame_data)[0]
        data.append(frame_data[nnz_idx])
        pixel_idx.append(nnz_idx)
        offset.append(len(nnz_idx))
    return np.hstack(data), np.hstack(pixel_idx), np.cumsum(offset)


def concatenate_space_compacted_data(*sparse_data):
    if not (isinstance(sparse_data[0], (list, tuple))) or any([len(s) != 3 for s in sparse_data]):
        raise ValueError(
            "Must be used as: concatenate_space_compacted_data((data1, pix_idx1, offset1), (data2, pix_idx2, offset2), ...)"
        )
    res_data = np.concatenate([s[0] for s in sparse_data])
    res_pix_idx = np.concatenate([s[1] for s in sparse_data])
    offsets = [s[2][1:].copy() for s in sparse_data]
    for i in range(1, len(offsets)):
        offsets[i] += offsets[i - 1][-1]
    res_offsets = np.concatenate(offsets)
    res_offsets = np.hstack([[0], res_offsets])
    return res_data, res_pix_idx, res_offsets


def dense_to_space_multi(xpcs_frames, n_threads=16):
    n_frames = xpcs_frames.shape[0]
    chunk_size = ceil(n_frames / n_threads)
    results = [None] * n_threads

    def compact_frames(i):
        results[i] = dense_to_space(xpcs_frames[i * (chunk_size) : (i + 1) * chunk_size])

    with ThreadPool(n_threads) as tp:
        tp.map(compact_frames, range(n_threads))

    return concatenate_space_compacted_data(*results)


def space_to_dense(data, pix_idx, offset, frame_shape):
    n_frames = len(offset) - 1
    xpcs_data = np.zeros((n_frames,) + frame_shape, dtype=data.dtype)

    def compact_to_frame(frame_idx):
        i_start = offset[frame_idx]
        i_stop = offset[frame_idx + 1]
        numels = np.prod(frame_shape)
        d = np.zeros(numels, dtype=data.dtype)
        pix_idx_ = pix_idx[i_start:i_stop]
        data_ = data[i_start:i_stop]
        d[pix_idx_] = data_
        xpcs_data[frame_idx] = d.reshape(frame_shape)

    with ThreadPool(16) as tp:
        tp.map(compact_to_frame, range(n_frames))

    return xpcs_data


def space_to_times(data, pix_idx, offset, shape, max_nnz):
    t_data_tmp = np.zeros((max_nnz, np.prod(shape)), dtype=data.dtype)
    t_times_tmp = np.zeros(t_data_tmp.shape, dtype=np.uint32)
    t_counter = np.zeros(np.prod(shape), dtype=np.uint32)

    n_times = len(offset) - 1
    for i in range(n_times):
        indices = pix_idx[offset[i] : offset[i + 1]]
        z = t_counter[indices]
        t_data_tmp[z, indices] = data[offset[i] : offset[i + 1]]
        t_times_tmp[z, indices] = i
        t_counter[indices] += 1

    t_offsets = np.hstack([[0], np.cumsum(t_counter, dtype=np.uint32)])
    t_data = np.zeros(t_offsets[-1], dtype=data.dtype)
    t_times = np.zeros(t_data.size, dtype=np.uint32)
    for i in range(len(t_offsets) - 1):
        start, stop = t_offsets[i], t_offsets[i + 1]
        if start == stop:
            continue
        t_data[start:stop] = t_data_tmp[: stop - start, i]
        t_times[start:stop] = t_times_tmp[: stop - start, i]
    return t_data, t_times, t_offsets


def estimate_max_events_in_times_from_space_compacted_data(pix_idx, offsets, estimate_from_n_frames=100):
    """
    From data in space-compacted format (data, pix_idx, offsets),
    return an estimation of the maximum number of events along the time axis,
    i.e (xpcs > 0).sum(axis=0).max().
    """
    num_events_in_subset = np.bincount(pix_idx[: offsets[estimate_from_n_frames]]).max()
    return ceil((num_events_in_subset / estimate_from_n_frames) * (offsets.size - 1))


from os import path


class SpaceToTimeCompaction(OpenclProcessing):
    kernel_files = ["sparse.cl"]

    def __init__(self, shape, max_time_nnz=250, dtype=np.uint8, offset_dtype=np.uint32, **oclprocessing_kwargs):
        super().__init__(**oclprocessing_kwargs)
        self.shape = shape
        self.dtype = dtype
        self.offset_dtype = offset_dtype
        self._setup_kernels(max_time_nnz=max_time_nnz)

    def _setup_kernels(self, max_time_nnz=250):
        self.max_time_nnz = max_time_nnz
        kernel_files = list(map(get_opencl_srcfile, self.kernel_files))
        self.compile_kernels(
            kernel_files=kernel_files,
            compile_options=[
                "-DDTYPE=%s" % dtype_to_ctype(self.dtype),
                "-DOFFSET_DTYPE=%s" % dtype_to_ctype(self.offset_dtype),
                "-DMAX_EVT_COUNT=%d" % self.max_time_nnz,
                "-I%s" % path.dirname(get_opencl_srcfile("dtypes.h")),
            ],
        )
        self.space_compact_to_time_compact_kernel = self.kernels.get_kernel("space_compact_to_time_compact")
        self.space_compact_to_time_compact_stage2_kernel = self.kernels.get_kernel(
            "space_compact_to_time_compact_stage2_sort"
        )
        self._d_t_counter = parray.zeros(self.queue, np.prod(self.shape), np.uint32)
        self._d_t_data_tmp = parray.zeros(self.queue, self.max_time_nnz * np.prod(self.shape), self.dtype)
        self._d_t_times_tmp = parray.zeros(self.queue, self.max_time_nnz * np.prod(self.shape), np.uint32)

    def compile_kernels(self, kernel_files=None, compile_options=None):
        _compile_kernels(self, kernel_files=kernel_files, compile_options=compile_options)

    def _space_compact_to_time_compact_stage1(self, data, pixel_indices, offsets, max_nnz_space, start_frame=0):
        self._d_t_data_tmp.fill(0)
        self._d_t_times_tmp.fill(0)

        n_frames = offsets.size - 1

        wg = None
        grid = (max_nnz_space, 1)  # round to something more friendly ?

        evt = self.space_compact_to_time_compact_kernel(
            self.queue,
            grid,
            wg,
            data.data,
            pixel_indices.data,
            offsets.data,
            self._d_t_data_tmp.data,
            self._d_t_times_tmp.data,
            self._d_t_counter.data,
            np.int32(self.shape[1]),
            np.int32(self.shape[0]),
            np.int32(n_frames),
            np.int32(start_frame),
        )
        evt.wait()
        self.profile_add(evt, "space->time stage 1")

    def _space_compact_to_time_compact_stage2(self):
        # Could be made on device directly. But parray.cumsum() takes some time to compile.
        t_counter = self._d_t_counter.get()
        offsets1 = np.cumsum(t_counter, dtype=t_counter.dtype)
        offsets = np.hstack([np.array([0], dtype=np.uint32), offsets1])

        d_t_offsets = parray.to_device(self.queue, offsets)
        d_t_data = parray.zeros(self.queue, offsets[-1], self._d_t_data_tmp.dtype)
        d_t_times = parray.zeros(self.queue, d_t_data.size, np.uint32)

        wg = None
        grid = (np.prod(self.shape), 1)

        evt = self.space_compact_to_time_compact_stage2_kernel(
            self.queue,
            grid,
            wg,
            self._d_t_data_tmp.data,
            self._d_t_times_tmp.data,
            d_t_offsets.data,
            d_t_data.data,
            d_t_times.data,
            np.int32(np.prod(self.shape)),
        )
        evt.wait()
        self.profile_add(evt, "space->time stage 2")

        return d_t_data, d_t_times, d_t_offsets

    def space_compact_to_time_compact(self, data, pixel_indices, offsets, start_frame=0):
        max_nnz_space = np.diff(offsets).max()

        # Might be good to have a more clever memory management
        d_data = parray.to_device(self.queue, data.astype(self.dtype))
        d_pixel_indices = parray.to_device(self.queue, pixel_indices.astype(np.uint32))
        d_offsets = parray.to_device(self.queue, offsets.astype(self.offset_dtype))
        #

        self._space_compact_to_time_compact_stage1(
            d_data, d_pixel_indices, d_offsets, max_nnz_space, start_frame=start_frame
        )
        d_t_data, d_t_times, d_t_offsets = self._space_compact_to_time_compact_stage2()

        return d_t_data, d_t_times, d_t_offsets


class SpaceToTimeCompactionV2(OpenclProcessing):
    kernel_files = ["sparse.cl"]

    def __init__(self, shape, dtype=np.uint8, offset_dtype=np.uint32, **oclprocessing_kwargs):
        super().__init__(**oclprocessing_kwargs)
        self.shape = shape
        self.dtype = dtype
        self.offset_dtype = offset_dtype
        self._setup_kernels()

    def _setup_kernels(self):
        kernel_files = list(map(get_opencl_srcfile, self.kernel_files))
        self.compile_kernels(
            kernel_files=kernel_files,
            compile_options=[
                "-DDTYPE=%s" % dtype_to_ctype(self.dtype),
                "-DOFFSET_DTYPE=%s" % dtype_to_ctype(self.offset_dtype),
                "-DMAX_EVT_COUNT=%d" % 10,  # TODO
                "-I%s" % path.dirname(get_opencl_srcfile("dtypes.h")),
            ],
        )
        self.space_compact_to_time_compact_kernel = self.kernels.get_kernel("space_compact_to_time_compact_v2_stage1")
        self.space_compact_to_time_compact_stage2_kernel = self.kernels.get_kernel(
            "space_compact_to_time_compact_v2_stage2"
        )
        self._d_t_counter = parray.zeros(self.queue, np.prod(self.shape), np.uint32)
        # self._d_t_data_tmp = parray.zeros(self.queue, self.max_time_nnz * np.prod(self.shape), self.dtype)
        # self._d_t_times_tmp = parray.zeros(self.queue, self.max_time_nnz * np.prod(self.shape), np.uint32)

    def compile_kernels(self, kernel_files=None, compile_options=None):
        _compile_kernels(self, kernel_files=kernel_files, compile_options=compile_options)

    def _space_compact_to_time_compact_stage1(self, pixel_indices, offsets, max_nnz_space, start_frame=0):
        n_frames = offsets.size - 1

        wg = None
        grid = (max_nnz_space, 1)  # round to something more friendly ?
        evt = self.space_compact_to_time_compact_kernel(
            self.queue,
            grid,
            wg,
            pixel_indices.data,
            offsets.data,
            self._d_t_counter.data,
            np.int32(self.shape[1]),
            np.int32(self.shape[0]),
            np.int32(n_frames),
        )
        evt.wait()
        self.profile_add(evt, "space->time stage 1")

    def _space_compact_to_time_compact_stage2(self, data, pixel_indices, offsets, n_frames, max_nnz_space):
        # Could be made on device directly. But parray.cumsum() takes some time to compile.
        t_counter = self._d_t_counter.get()
        offsets1 = np.cumsum(t_counter, dtype=t_counter.dtype)
        offsets1 = np.hstack([np.array([0], dtype=np.uint32), offsets1])
        self._d_t_counter.fill(0)

        d_t_offsets = parray.to_device(self.queue, offsets1)
        d_t_offsets2 = d_t_offsets.copy()
        d_t_data = parray.zeros(self.queue, offsets1[-1], self.dtype)
        d_t_times = parray.zeros(self.queue, d_t_data.size, np.uint32)

        wg = None
        grid = (max_nnz_space, 1)

        evt = self.space_compact_to_time_compact_stage2_kernel(
            self.queue,
            grid,
            wg,
            data.data,
            pixel_indices.data,
            offsets.data,
            d_t_data.data,
            d_t_times.data,
            d_t_offsets2.data,
            np.int32(n_frames),
        )
        evt.wait()
        self.profile_add(evt, "space->time stage 2")

        return d_t_data, d_t_times, d_t_offsets

    def space_compact_to_time_compact(self, data, pixel_indices, offsets, start_frame=0):
        max_nnz_space = np.diff(offsets).max()
        n_frames = offsets.size - 1

        # Might be good to have a more clever memory management
        d_data = parray.to_device(self.queue, data.astype(self.dtype))
        d_pixel_indices = parray.to_device(self.queue, pixel_indices.astype(np.uint32))
        d_offsets = parray.to_device(self.queue, offsets.astype(self.offset_dtype))
        #

        self._space_compact_to_time_compact_stage1(d_pixel_indices, d_offsets, max_nnz_space, start_frame=start_frame)
        d_t_data, d_t_times, d_t_offsets = self._space_compact_to_time_compact_stage2(
            d_data, d_pixel_indices, d_offsets, n_frames, max_nnz_space
        )

        return d_t_data, d_t_times, d_t_offsets
