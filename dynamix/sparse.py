"""
Module for compacting/decompacting data into various formats.

For now two compact formats are available:
  - "times"
  - "space"
"""
from multiprocessing.pool import ThreadPool
import numpy as np



def dense_to_times(frames):
    assert frames.ndim == 3
    framesT = np.moveaxis(frames, 0, -1)
    times = np.arange(frames.shape[0], dtype=np.int32)
    nnz_indices = np.where(framesT > 0)

    res_data = framesT[nnz_indices]
    res_times = times[nnz_indices[-1]]

    offsets = np.cumsum((frames > 0).sum(axis=0).ravel())
    res_offsets = np.zeros(np.prod(frames.shape[1:])+1, dtype=np.uint32)
    res_offsets[1:] = offsets[:]

    return res_data, res_times, res_offsets



def times_to_dense(t_data, t_times, t_offset, n_frames, frame_shape):
    res = np.zeros((n_frames, ) + frame_shape, dtype=t_data.dtype)
    for i in range(frame_shape[0]):
        for j in range(frame_shape[1]):
            idx = i * frame_shape[1] + j
            start, stop = t_offset[idx], t_offset[idx+1]
            if stop - start == 0:
                continue
            events = t_data[start:stop]
            times = t_times[start:stop]
            res[times, i, j] = events
    return res


# eg. dense_to_space(xpcs * (qmask > 0))
def dense_to_space(xpcs_frames):
    data = [] # int8
    pixel_idx = [] # uint32
    offset = [0] # uint64

    for frame in xpcs_frames:
        frame_data = frame.ravel()
        nnz_idx = np.nonzero(frame_data)[0]
        data.append(frame_data[nnz_idx])
        pixel_idx.append(nnz_idx)
        offset.append(len(nnz_idx))
    return np.hstack(data), np.hstack(pixel_idx), np.cumsum(offset)



def space_to_dense(data, pix_idx, offset, frame_shape):
    n_frames = len(offset) - 1
    xpcs_data = np.zeros((n_frames, ) + frame_shape, dtype=data.dtype)

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

    n_times = len(offset)-1
    for i in range(n_times):
        indices = pix_idx[offset[i]:offset[i+1]]
        z = t_counter[indices]
        t_data_tmp[z, indices] = data[offset[i]:offset[i+1]]
        t_times_tmp[z, indices] = i
        t_counter[indices] += 1

    t_offsets = np.hstack([[0], np.cumsum(t_counter, dtype=np.uint32)])
    t_data = np.zeros(t_offsets[-1], dtype=data.dtype)
    t_times = np.zeros(t_data.size, dtype=np.uint32)
    for i in range(len(t_offsets) - 1):
        start, stop = t_offsets[i], t_offsets[i+1]
        if start == stop:
            continue
        t_data[start:stop] = t_data_tmp[:stop-start, i]
        t_times[start:stop] = t_times_tmp[:stop-start, i]
    return t_data, t_times, t_offsets


