import h5py    # HDF5 support
import hdf5plugin
import numpy
import sys
import time
from datetime import datetime

def h5writer(fileName,data):
    '''Writes a NeXus HDF5 file using h5py and numpy'''
    

    print("Write a NeXus HDF5 file")

    #if sys.version_info < (3,):
    #    fileName = "nxdata2d.py2.h5"
    #else:
    #    fileName = "nxdata2d.py3.h5"
    timestamp = str(datetime.now())

    # create the HDF5 NeXus file
    f = h5py.File(fileName, "w")
    # point to the default data to be plotted
    f.attrs['default']          = u'entry'
    # give the HDF5 root some more attributes
    f.attrs['file_name']        = fileName
    f.attrs['file_time']        = timestamp
    f.attrs['creator']          = u'NXdataImage.py'
    f.attrs['HDF5_Version']     = h5py.version.hdf5_version
    f.attrs['h5py_version']     = h5py.version.version

    # create the NXentry group
    nxentry = f.create_group('entry_0000')
    nxentry.attrs['NX_class'] = 'NXentry'
    nxentry.attrs['default'] = u'image_plot'
    nxentry.create_dataset('title', data=u'Lima 2D detector acquisition')

    # create the NXdata group
    nxdata = nxentry.create_group('measurement')
    nxdata.attrs['NX_class'] = u'NXdata'
    nxdata.attrs['signal'] = u'3D data'              # Y axis of default plot
    if sys.version_info < (3,):
        string_dtype = h5py.special_dtype(vlen=unicode)
    else:
        string_dtype = h5py.special_dtype(vlen=str)
    nxdata.attrs['axes'] = numpy.array(['frame_name', 'row_name', 'col_name'], dtype=string_dtype) # X axis of default plot

    # signal data
    ds = nxdata.create_dataset('data', data=data, **hdf5plugin.Bitshuffle(nelems=0, lz4=True))
    ds.attrs['interpretation'] = u'images'

    # time axis data
    ds = nxdata.create_dataset('frame_name', data=numpy.arange(data.shape[0]))
    ds.attrs['units'] = u'number'
    ds.attrs['long_name'] = u'Frame number (number)'    # suggested Y axis plot label 

    # X axis data
    ds = nxdata.create_dataset(u'col_name', data=numpy.arange(data.shape[2]))
    ds.attrs['units'] = u'pixels'
    ds.attrs['long_name'] = u'Pixel Size X (pixels)'    # suggested X axis **hdf5plugin.Bitshuffle(nelems=0, lz4=True)plot label

    # Y axis data
    ds = nxdata.create_dataset('row_name', data=numpy.arange(data.shape[1]))
    ds.attrs['units'] = u'pixels'
    ds.attrs['long_name'] = u'Pixel Size Y (pixels)'    # suggested Y axis plot label

    f.close()   # be CERTAIN to close the file

    print("wrote file:", fileName)


def myreader(fileName):
    '''Read a NeXus HDF5 file using h5py and numpy'''
    
    print("Read a NeXus HDF5 file")
    t0 = time.time()
    f = h5py.File(fileName, "r")

    data = f.get('/entry_0000/measurement/data')
    #data = numpy.array(data,numpy.uint8)
    data = numpy.array(data)

    f.close()
    print("Reading time %3.3f sec" % (time.time()-t0))
    return data
