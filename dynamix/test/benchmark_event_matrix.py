"""
This is a complete test suite of the two-times correlation function (TTCF) matrix builders.

A number of reference datasets are used.
The reference results were computed using the "py_dense_correlator()" function.
"""

from os import path
import numpy as np
from math import sqrt
from dynamix.correlator.event_matrix import SMatrixEventCorrelator, TMatrixEventCorrelator
from dynamix.sparse import SpaceToTimeCompaction, estimate_max_events_in_times_from_space_compacted_data
from dynamix.correlator.event_matrix import flat_to_square

# -----------------------------------------------------------------------------
# ---------------------------- Datasets definitions ---------------------------
# -----------------------------------------------------------------------------

datasets = {
    "dataset02": {
        # dataset02: 10k frames. Not really sparse in the ROI
        "raw_data_fname": "/data/id10/inhouse/software/dynamix/datasets/dataset02/scan0005/eiger1_0000.h5",
        "data_fname": "/scisoft/dynamix/data/dataset02/xpcs_010000.npz",
        "qmask_fname":  "/data/id10/inhouse/software/dynamix/datasets/dataset02/analysis/scan0005_0_10000/DUKE_qmask.npy",
        "reference_fname": "/scisoft/dynamix/data/dataset02/reference.npz",
    },
    "dataset03": {
        # dataset03: 40k frames
        "raw_data_fname": "/data/id10/inhouse/software/dynamix/datasets/dataset03/scan0002/eiger4m_0000.h5",
        "data_fname": "/scisoft/dynamix/data/dataset03/xpcs_040000.npz",
        "qmask_fname": "/data/id10/inhouse/software/dynamix/datasets/dataset03/analysis/Pt1_10GPa_2_242C/scan0002_0_5000/Pt1_10GPa_2_242C_qmask.npy",
        "reference_fname": "/scisoft/dynamix/data/dataset03/reference.npz",
    },
    "dataset04": {
        # dataset04: 200k frames
        "raw_data_fname": "/scisoft/dynamix/data/dataset04/Vit4_0GPa_Tg_m_30_Monday_25p0_575K_00001_merged.h5",
        "data_fname": "/scisoft/dynamix/data/dataset04/xpcs_200000.npz",
        "qmask_fname": "/data/id10/inhouse/software/dynamix/datasets/dataset04/analysis/Vit4_0GPa_Tg_m_30_Monday_25p0_575K/00001_0_200000/Vit4_0GPa_Tg_m_30_Monday_25p0_575K_qmask.npy",
        "reference_fname": "/scisoft/dynamix/data/dataset04/reference.npz", # GPU can do only 100k frames for now (with 40GB mem)
    },
    "dataset05": {
        # dataset05: 1.2M frames
        "raw_data_fname": "/scisoft/dynamix/data/dataset05/dataset05_merged.h5",
        "data_fname": "/scisoft/dynamix/data/dataset05/xpcs_1200000.npz",
        "qmask_fname": "/scisoft/dynamix/data/dataset05/qmask_dummy.npy", # hand-crafted, using data < 10...
        "reference_fname": "/scisoft/dynamix/data/dataset05/reference.npz", # GPU can do only 100k frames for now (with 40GB mem)
    },
}

# dataset01
# 20k frames, sparse, WAXS geometry - some pixels have high data value
for i in range(7, 13+1):
    datasets["dataset01_scan%04d" % i] = {
        "raw_data_fname": "/scisoft/dynamix/data/dataset01_scan%04d/scan%04d_merged.h5" % (i, i),
        "data_fname": "/scisoft/dynamix/data/dataset01_scan%04d/xpcs_020000.npz" % i,
        # "qmask_fname": "/data/id10/inhouse/software/dynamix/datasets/dataset01/analysis/SiO2-21p67keV/scan%04d_0_20000/SiO2-21p67keV_qmask.npy" % i,
        "qmask_fname": "/scisoft/dynamix/data/dataset01_scan0010/SiO2-21p67keV_qmask_pp.npy",
        "reference_fname": "/scisoft/dynamix/data/dataset01_scan%04d/reference.npz" % i,
    }

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def benchmark_ttcf(dataset_name, n_frames=None):
    print("-"*80 + "\n" + "Processing %s" % dataset_name + "\n" + "-" * 80)

    data_fname = datasets[dataset_name]["data_fname"]
    qmask_fname = datasets[dataset_name]["qmask_fname"]

    data, pix_idx, offset, qmask, frame_shape = load_xpcs_compacted_data(
        data_fname, qmask_fname, n_frames=n_frames, cast_to_dtype=dtype
    )
    n_frames = offset.size - 1
    if do_sparse_stats:
        print("Spatial sparsity:")
        print("  - Max %d non-zero samples per frame" % (np.diff(offset).max()))
        print("  - Total space-sparsity factor: %.1f" % (np.prod(frame_shape)*n_frames/data.size))
        print("  - NNZ samples per frame: %.0f +/- %.0f" % (np.mean(np.diff(offset)), np.std(np.diff(offset))))


    # Compute TTCF (space-based)
    ttcf_space = None
    if use_space_correlator:
        ttcf_space = SMatrixEventCorrelator(frame_shape, n_frames, qmask=qmask, dtype=dtype, profile=True)
        ttcf_space.build_correlation_matrices(data, pix_idx, offset)

    # To use the time-based TTCF, we first have to convert the data from space-compacted to time-compacted
    space2time = SpaceToTimeCompaction(frame_shape, profile=True, dtype=dtype)
    max_time_nnz = estimate_max_events_in_times_from_space_compacted_data(pix_idx, offset, estimate_from_n_frames=n_frames)
    d_t_data, d_t_times, d_t_offsets = space2time.space_compact_to_time_compact(data, pix_idx, offset) # opencl arrays

    # Compute TTCF (time-based)
    ttcf_time = TMatrixEventCorrelator(frame_shape, n_frames, qmask=qmask, max_time_nnz=max_time_nnz, dtype=dtype, profile=True)
    ttcf_time.build_correlation_matrices(d_t_data, d_t_times, d_t_offsets)
    g2_t, std_t, num_t, denom_t = ttcf_time.get_correlation_function(0, calc_std=True, n_threads=16, return_num_denom=True)

    if print_timings:
        if use_space_correlator:
            print(ttcf_space.get_timings())
        print(ttcf_time.get_timings())
        print(space2time.get_timings())

    try:
        g2_ref, std_ref, num_ref, denom_ref = load_reference_result(dataset_name)
        compare_results(n_frames, g2_t, std_t, num_t, denom_t, g2_ref, std_ref, num_ref, denom_ref, ttcf_time.scale_factors[1])
    except FileNotFoundError:
        print("Can't compare with reference implementation - no reference file")

    return ttcf_time # For Debug




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



def py_dense_correlator(xpcs_data, mask, calc_std=False):
    ind = np.where(mask > 0)  # unused pixels are 0
    xpcs_data = np.array(xpcs_data[:, ind[0], ind[1]], np.float32)  # (n_tau, n_pix)
    meanmatr = np.mean(xpcs_data, axis=1)  # xpcs_data.sum(axis=-1).sum(axis=-1)/n_pix
    ltimes, lenmatr = np.shape(xpcs_data)  # n_tau, n_pix
    meanmatr.shape = 1, ltimes

    num = np.dot(xpcs_data, xpcs_data.T)
    denom = np.dot(meanmatr.T, meanmatr)

    res = np.zeros(ltimes)
    if calc_std:
        dev = np.zeros_like(res)

    for i in range(ltimes):
        dia_n = np.diag(num, k=i) / lenmatr
        dia_d = np.diag(denom, k=i)
        res[i] = np.sum(dia_n) / np.sum(dia_d)
        if calc_std:
            dev[i] = np.std(dia_n / dia_d) / sqrt(len(dia_d))
    if calc_std:
        return (res, dev, num, denom)
    else:
        return res, num, denom


def load_reference_result(dataset_name):
    ref_fname = datasets[dataset_name]["reference_fname"]
    if not path.isfile(ref_fname):
        raise FileNotFoundError(ref_fname)
    f_d = np.load(ref_fname)
    num = f_d["num"][()]
    denom = f_d["denom"][()]
    std = f_d["std"][()]
    ttcf = f_d["ttcf"][()]
    f_d.close()
    return ttcf, std, num, denom



def compare_results(n_frames, g2, std, num, denom, g2_ref, std_ref, num_ref, denom_ref, scale_factor):
    """
    Compare the GPU correlator results with the naive-python-numpy reference implementation.
    The latter was computed only for qbin==1 (hence the [0] in the code below).
    Also, the GPU correlator uses a flat data structure, so we have to use flat_to_square.
    """
    if num_ref.shape[0] != n_frames:
        print(
            "Cannot compare with reference results: reference is computed for n_frames=%d but I currently have n_frames=%d"
            % (num_ref.shape[0], n_frames)
        )
        return

    ma = lambda x: np.max(np.abs(x))
    num_square = num
    denom_square = denom
    denom_square /= scale_factor ** 2
    print("Max error for numerator: %.3e" % (ma(num_square - np.triu(num_ref))))
    print("Max error for denominator: %.3e" % (ma(denom_square - np.triu(denom_ref))))
    print("Max error for TTCF: %.3e (min ref val = %.3e)" % (ma(g2 - g2_ref), np.min(g2_ref)))
    print("Max error for STD: %.3e (min ref val = %.3e)" % (ma(std - std_ref), np.min(std_ref)))





# -----------------------------------------------------------------------------
# --------------------------- Dataset and options choice ----------------------
# -----------------------------------------------------------------------------


# List of datasets to test
# datasets_to_test = ["dataset01_scan0007", "dataset01_scan0008", "dataset01_scan0009",  "dataset01_scan0010", "dataset01_scan0011", "dataset01_scan0012", "dataset01_scan0013"]
# datasets_to_test = ["dataset02", "dataset03"]
datasets_to_test = ["dataset03"]
# Data type for XPCS data. It is always be uint8 in practice, though dynamix code should work with other data types
dtype = np.uint8
# Whether to print information on datasets sparsity
do_sparse_stats = True
# Whether to print information on execution times
print_timings = True
# Whether to also test the space-based correlators.
use_space_correlator = False


if __name__ == "__main__":
    # ttcf_time = benchmark_ttcf("dataset01_scan0009", n_frames=20000)
    ttcf_time = benchmark_ttcf("dataset05", n_frames=20000)
    # for dataset_name in datasets_to_test:
        # benchmark_ttcf(dataset_name)
