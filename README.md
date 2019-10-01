# dynamix


## About

`dynamix` is a Python module for X-ray photon correlation spectroscopy (XPCS). It primarily aims at offering fast correlation function computation with OpenCL/CUDA implementations.

**Work in progress:** please keep in mind that this is work in progress. The API might change 

`dynamix` provides classes named **correlators** to compute the XPCS correlation function `g2` defined by
![g2](https://latex.codecogs.com/gif.latex?g_2(q,&space;\tau)&space;=&space;\dfrac{&space;\langle&space;\langle&space;I(t,\,&space;p)&space;I(t&space;&plus;&space;\tau,\,&space;p)&space;\rangle_p&space;\rangle_t&space;}&space;{&space;\langle&space;\langle&space;I(t,\,&space;p)&space;\rangle_p&space;\langle&space;I(t&space;&plus;&space;\tau,\,&space;p)&space;\rangle_p&space;\rangle_t&space;})

where `< . >_p` denotes averaging pixels belonging to the current bin (scattering vector), and `< . >_t` denotes time averaging.

All correlators compute the function `g_2`, although through different means.


## Installation

### Python dependencies

dynamix depends on the following python packages:

  - numpy
  - pyopencl
  - [silx](https://github.com/silx-kit/silx)

Optionally:

- pyfftw
- scikit-cuda

These modules should be automatically installed when installing `dynamix`.

### System dependencies

dynamix needs:

- An implementation of OpenCL (>= 1.2)
- Optionally, CUDA and CUFFT
- Optionally, FFTW library and header files

### Releases versions

To install the current release from [pypi](https://pypi.org/project/dynamix/):

```bash
pip install [--user] dynamix
```

### Development versions

To install the current development version:

```bas
pip instal [--user] git+https://github.com/silx-kit/dynamix
```

## Usage

First load a dataset. `dynamix` provides some datasets acquired at ESRF ID10.

```python
from dynamix.test.utils import XPCSDataset
dataset = XPCSDataset("eiger_514_10k")
data = dataset.data
shape = dataset.dataset_desc.frame_shape
nframes = dataset.dataset_desc.nframes
print(dataset.dataset_desc.description)
```

Then use a correlator:

```python
from dynamix.correlator.dense import MatMulCorrelator

correlator = MatMulCorrelator(shape, nframes, mask=dataset.qmask)
result = correlator.correlate(data)
```

This is the basic (and slowest) correlator. Please refer to the documentation to use other backends.

## Question ? Bug ?

Please [open an issue](https://github.com/silx-kit/dynamix/issues) on the project page to report bugs or ask questions.



