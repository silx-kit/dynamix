"""
This is a complete test suite of the two-times correlation function (TTCF) matrix builders.

A number of reference datasets are used.
The reference results were computed using the "py_dense_correlator()" function.
"""

from os import path
import numpy as np
from dynamix.correlator.event_matrix import SMatrixEventCorrelator, TMatrixEventCorrelator
from dynamix.sparse import SpaceToTimeCompaction, estimate_max_events_in_times_from_space_compacted_data

# -----------------------------------------------------------------------------
# ---------------------------- Datasets definitions ---------------------------
# -----------------------------------------------------------------------------

datasets = {
    "dataset02": {
        # dataset02: 10k frames. Not really sparse in the ROI
        "data_fname": "/data/id10/inhouse/software/dynamix/datasets/dataset02/scan0005/eiger1_0000.h5",
        "qmask_fname":  "/data/id10/inhouse/software/dynamix/datasets/dataset02/analysis/scan0005_0_10000/DUKE_qmask.npy",
    },
    "dataset03": {
        # dataset03: 40k frames
        "data_fname": "/data/id10/inhouse/software/dynamix/datasets/dataset03/scan0002/eiger4m_0000.h5",
        "qmask_fname": "/data/id10/inhouse/software/dynamix/datasets/dataset03/analysis/Pt1_10GPa_2_242C/scan0002_0_5000/Pt1_10GPa_2_242C_qmask.npy",
    },
    "dataset04": {
        # dataset04: 200k frames
        "data_fname": "/scisoft/dynamix/data/dataset04/Vit4_0GPa_Tg_m_30_Monday_25p0_575K_00001_merged.h5",
        "qmask_fname": "/data/id10/inhouse/software/dynamix/datasets/dataset04/analysis/Vit4_0GPa_Tg_m_30_Monday_25p0_575K/00001_0_200000/Vit4_0GPa_Tg_m_30_Monday_25p0_575K_qmask.npy",
    },
    "dataset05": {
        # dataset04: 1.2M frames
        "data_fname": "/scisoft/dynamix/data/dataset05/dataset05_merged.h5",
        "qmask_fname": "/scisoft/dynamix/data/dataset05/qmask_dummy.npy" # hand-crafted, using data < 10...
    },
}

# dataset01
# 20k frames, sparse, WAXS geometry - some pixels have high data value
for i in range(7, 13+1):
    datasets["dataset01_scan%04d" % i] = {
        "data_fname": "/scisoft/dynamix/data/dataset01_scan%04d/scan%04d_merged.h5" % (i, i),
        # "qmask_fname": "/data/id10/inhouse/software/dynamix/datasets/dataset01/analysis/SiO2-21p67keV/scan%04d_0_20000/SiO2-21p67keV_qmask.npy" % i,
        "qmask_fname": "/scisoft/dynamix/data/dataset01_scan0010/SiO2-21p67keV_qmask_pp.npy",
    }

# ---

# -----------------------------------------------------------------------------
# --------------------------- Dataset and options choice ----------------------
# -----------------------------------------------------------------------------

# List of datasets to test
datasets_to_test = ["dataset01_scan0010"]
# Data type for XPCS data. It is always be uint8 in practice, though dynamix code should work with other data types
dtype = np.uint8
# Whether to print information on datasets sparsity
do_sparse_stats = True
# Whether to print information on execution times
print_timings = True
# Whether to also test the space-based correlators. W
use_space_correlator = True


def benchmark_ttcf(dataset_name, n_frames=None):

    data_fname = datasets[dataset_name]["data_fname"]
    qmask_fname = datasets[dataset_name]["qmask_fname"]

    data, pix_idx, offset, qmask, frame_shape = load_xpcs_compacted_data(
        data_fname, qmask_fname, n_frames=n_frames, cast_to_dtype=dtype
    )
    if do_sparse_stats:
        print("Spatial sparsity:")
        print("  - Max %d non-zero samples per frame" % (np.diff(offset).max()))
        print("  - Total space-sparsity factor: %.1f" % (np.prod(frame_shape)*n_frames/data.size))
        print("  - NNZ samples per frame: %.0f +/- %.0f" % (np.mean(np.diff(offset)), np.std(np.diff(offset))))

    ttcf = None
    if use_space_correlator:
        ttcf_space = SMatrixEventCorrelator(frame_shape, n_frames, qmask=qmask, dtype=dtype, profile=True)
        res_s = ttcf_space._build_correlation_matrix_v3(data, pix_idx, offset).get()


    # To use the time-based TTCF, we first have to convert the data from space-compacted to time-compacted
    space2time = SpaceToTimeCompaction(frame_shape, profile=True, dtype=dtype)
    max_time_nnz = estimate_max_events_in_times_from_space_compacted_data(pix_idx, offset, estimate_from_n_frames=n_frames)
    d_t_data, d_t_times, d_t_offsets = space2time.space_compact_to_time_compact(data, pix_idx, offset) # opencl arrays

    ttcf_time = TMatrixEventCorrelator(frame_shape, n_frames, qmask=qmask, max_time_nnz=max_time_nnz, dtype=dtype, profile=True)
    res_t = ttcf_time.build_correlation_matrix(d_t_data, d_t_times, d_t_offsets).get()

    if print_timings:
        if use_space_correlator:
            print(ttcf_space.get_timings())
        print(ttcf_time.get_timings())
        print(space2time.get_timings())




















def load_xpcs_compacted_data(fname, qmask_fname, n_frames=None, cast_to_dtype=np.uint8):
    if not path.isfile(fname):
        raise FileNotFoundError
    if not path.isfile(qmask_fname):
        raise FileNotFoundError

    f_d = np.load(fname)
    data = f_d["data"]
    pix_idx = f_d["pix_idx"].astype(np.uint32)
    offset = f_d["offset"].astype(np.uint32)
    qmask = f_d["qmask"]
    n_frames_in_file = f_d["n_frames"][()]
    frame_shape = tuple(f_d["frame_shape"].tolist())
    f_d.close()

    if n_frames is not None and n_frames != n_frames_in_file:
        o = offset[n_frames+1]
        data = data[:o]
        pix_idx = pix_idx[:o]
        offset = offset[:n_frames+1]

    qmask = np.load(qmask_fname)
    if cast_to_dtype is not None and data.dtype != cast_to_dtype:
        data = data.astype(cast_to_dtype)
    return data, pix_idx, offset, qmask, frame_shape
