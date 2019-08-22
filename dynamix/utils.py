import os
import numpy as np

def get_folder_path(foldername=""):
    _file_dir = os.path.dirname(os.path.realpath(__file__))
    package_dir = _file_dir
    return os.path.join(package_dir, foldername)

def get_opencl_srcfile(filename):
    src_relpath =  os.path.join("resources", "opencl")
    opencl_src_folder = get_folder_path(foldername = src_relpath)
    return os.path.join(opencl_src_folder, filename)

def nextpow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def estimate_r2c_memory(fft_shape):
    """
    Estimate the memory (in MB) taken by a R2C (FFT, IFFT) plan couple,
    assuming float32 -> complex64 dtypes.
    """
    real_bytes = np.prod(fft_shape)*4
    cplx_bytes = (np.prod(fft_shape[:-1]) * (fft_shape[-1]//2+1))*8
    return (real_bytes + cplx_bytes)*2/1e6
