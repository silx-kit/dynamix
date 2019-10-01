from collections import namedtuple
from os.path import join as path_join
import numpy as np
from silx.resources import ExternalResources


utilstest = ExternalResources(project="dynamix",
                              url_base="http://www.silx.org/pub/dynamix/",
                              env_key="DYNAMIX_DATA",
                              timeout=60)

DatasetDescription = namedtuple(
    "DatasetDescription", [
        "name", "data", "result", "description", "nframes", "dtype",
        "frame_shape", "bins"
    ]
)

dataset_andor_10k = DatasetDescription(
    name="andor_1024_10k",
    data="Fe79_Andor_9599.npz",
    result="wcf_Fe79_ramp1_240C_down_1_9600.npz",
    description="""
        XPCS data with Andor detector of 1024*1024 pixels, 9599 frames.
        Data sparsity factor is approx. 40.
    """,
    nframes=9599,
    frame_shape=(1024, 1024),
    dtype=np.int8,
    bins=None,
)

dataset_andor_3k = DatasetDescription(
    name="andor_1024_3k",
    data="Fe79_Andor_3000.npz",
    result="wcf_Fe79_ramp1_240C_down_1_3000_raw.npz",
    description="""
        Same dataset as andor_1024_10k, but cropped to 3000 frames.
    """,
    nframes=3000,
    frame_shape=(1024, 1024),
    dtype=np.int8,
    bins=None,
)

dataset_eiger_10k = DatasetDescription(
    name="eiger_514_10k",
    data="eiger_new_duke_100us_ATT0.npz",
    result="wcf_eiger_new_duke_100us_ATT0_results.npz",
    description="""
        XPCS data acquired with Eiger detector of 1030*514 pixels, 9999 frames.
        Data is zero outside of the ROI [314:429, 582:698].
        Inside this ROI, data is "dense".
        Correlation function has to be computed on 10 bins.
    """,
    nframes=9999,
    frame_shape=(115, 116),
    dtype=np.int8,
    bins=10,
)

dataset_eiger_20k = DatasetDescription(
    name="eiger_514_20k",
    data="S6_Eiger_zaptime_2_20000.npz",
    result="wcf_S6_zaptime_002_e_optimized.npz",
    description="""
        XPCS data acquired with Eiger detector of 1030*514 pixels, 20000 frames.
        Data is extremely sparse (average sparsity factor: 21000)
    """,
    nframes=20000,
    frame_shape=(514, 1030),
    dtype=np.int8,
    bins=None,
)

dataset_al_600 = DatasetDescription(
    name="Al_512_600",
    data="Al_600.npz",
    result="unknown.npz",
    description="""
        XPCS simulated data of Al crystal melting with a detector of 512x512 pixels, 600 frames.
        Dense dataset !
    """,
    nframes=600,
    frame_shape=(512, 512),
    dtype=np.uint32,
    bins=None,
)

def get_datasets():
    datasets = [dataset_andor_10k, dataset_andor_3k, dataset_eiger_10k, dataset_eiger_20k, dataset_al_600]
    res = {}
    for dataset in datasets:
        res[dataset.name] = dataset
    return res


class XPCSDataset(object):
    """
    Class for loading XPCS datasets.
    """

    avail_datasets = get_datasets()

    def __init__(self, name):
        self.check_dataset(name)
        self.load_dataset(name)

    def check_dataset(self, name):
        if name not in self.avail_datasets:
            raise ValueError(
                "Unknown dataset %s. Available datasets are: %s" %
                (name, list(self.avail_datasets.keys()))
            )

    def load_dataset(self, name):
        self.name = name
        dataset = self.avail_datasets[name]
        self.dataset_desc = dataset
        # Load data
        data_relpath = path_join(dataset.name, dataset.data)
        self.data_file = utilstest.getfile(data_relpath)
        self.data = np.load(self.data_file)["data"]
        if self.dataset_desc.bins is not None:
            self.qmask = np.load(self.data_file)["qmask"]
        self.check_data()
        # Load results (only the correlation function)
        res_relpath = path_join(dataset.name, dataset.result)
        self.result_file = utilstest.getfile(res_relpath)
        print(self.result_file)
        fd = np.load(self.result_file, allow_pickle=True)
        if self.dataset_desc.bins is not None:
            # Multi-bins dataset
            res = {}
            for bin_name, bin_res in fd.items():
                res[bin_name] = bin_res.item() # dict inside npz
        else:
            # Single-bin dataset
            res = fd["correlation"]
        self.result = res

    def check_data(self):
        assert self.data.shape[0] == self.dataset_desc.nframes
        assert self.data[0].shape == self.dataset_desc.frame_shape
        assert self.data.dtype == self.dataset_desc.dtype


