from itertools import product
from bisect import bisect
import os
import numpy as np
from collections import namedtuple

CorrelationResult = namedtuple("CorrelationResult", "res dev")
Compacted =  namedtuple("Compacted", "values times pixel_ptr")
""" Compacted form of a XPCS stack containing 3 1D structure:
          - values: pixel values for nonzero data points. dtype: same as reference stack
          - times: time indices corresponding to nonzero data points. dtype: uint32
          - pixel_ptr: Pixel k's information is stored in values[pixel_ptr[k]:pixel_ptr[k+1]] and times[pixel_ptr[k]:pixel_ptr[k+1]]
"""


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

def updiv(a, b):
    """
    return the integer division, plus one if `a` is not a multiple of `b`
    """
    return (a + (b - 1)) // b


def generate_powers():
    """
    Generate a list of powers of [2, 3, 5, 7],
    up to (2**15)*(3**9)*(5**6)*(7**5).
    """
    primes = [2, 3, 5, 7]
    maxpow = {2: 15, 3: 9, 5: 6, 7: 5}
    valuations = []
    for prime in primes:
        # disallow any odd number (for R2C transform), and any number
        # not multiple of 4 (Ram-Lak filter behaves strangely when
        # dwidth_padded/2 is not even)
        minval = 2 if prime == 2 else 0
        valuations.append(range(minval, maxpow[prime]+1))
    powers = product(*valuations)
    res = []
    for pw in powers:
        res.append(np.prod(list(map(lambda x : x[0]**x[1], zip(primes, pw)))))
    return np.unique(res)


def get_next_power(n, powers=None):
    """
    Given a number, get the closest (upper) number p such that
    p is a power of 2, 3, 5 and 7.
    """
    if powers is None:
        powers = generate_powers()
    idx = bisect(powers, n)
    if powers[idx-1] == n:
        return n
    return powers[idx]

