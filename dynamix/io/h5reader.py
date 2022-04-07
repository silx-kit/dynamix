import h5py    # HDF5 support
import hdf5plugin
import numpy
import numba as nb
import sys
import time
import copy
from datetime import datetime
from dynamix.tools import tools

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

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


def myreader(fileName,detector,nf1,nf2,scan="none"):
    '''Read a NeXus HDF5 file using h5py and numpy'''
    
    t0 = time.time()
    #f = h5py.File(fileName, "r")
    zn = 0
    while True:
        try:
            f = h5py.File(fileName, "r")
            break # Success!
        except OSError :    
            print("Waiting for the file trial %d" % zn)
            time.sleep(5)
            zn +=1 
            if zn==3: 
                print("File %s cannot be read" % fileName) 
                exit() 
    print("Read a NeXus HDF5 file")
    if scan=="none":
        #data = f.get('/entry_0000/measurement/data')
        fdata = f['/entry_0000/measurement/data']
    else:
        fdata = f['/'+scan+'.1/measurement/'+detector]
    fshape = fdata.shape       
    data = numpy.zeros((nf2-nf1,fshape[1],fshape[2]),fdata.dtype)
    print("Data shape", data.shape)
    n = 0
    for i in range(nf1,nf2,1):
        data[n,:,:] = numpy.array(fdata[i,:,:],fdata.dtype)
        n += 1
    #data = numpy.array(data[nf1:nf2,:,:])#,numpy.int8)
    #data = numpy.array(data)

    f.close()
    print("Reading time %3.3f sec" % (time.time()-t0))
    return data
    
def p10_eiger_event_data(fileName,nf1,nf2,mask):
    '''Read a P10 HDF5 master file using h5py and numpy'''
    
    t0 = time.time()
    try:
        f = h5py.File(fileName, "r")
    except OSError :    
        print("File %s cannot be read" % fileName) 
        exit() 
    print("Read a P10 HDF5 file")
    datas = f['/entry/data']
    pixels = []
    s = []
    nframes = 0
    for i in dict(datas):
        #tt = time.time()
        data = numpy.array(datas[i][()],numpy.uint8)
        #print("Reading time %f" % (time.time()-tt)) 
        nframes += numpy.shape(data[:,0,0])[0]
        #print("Number of frames %d" % nframes)
        #tt = time.time()
        try:
            img += numpy.sum(data,0)
        except:
            img = numpy.array(numpy.sum(data,0),numpy.uint16)
        #print("Summing time %f" % (time.time()-tt))    
        #tt = time.time()
        data[:,mask>0] = 0
        #ind = numpy.where(data>20)
        #mask[ind[1],ind[2]] = 1
        #data[:,mask>0] = 0
        trace0 = numpy.sum(numpy.sum(data,1),1)
        #print("Masking time %f" % (time.time()-tt))
        #print("First stage processing %f" % (time.time()-tt))
        mNp = trace0.max()+10
        print("Diagnostics maximum photons per frame %d" % (mNp-10))
        #tt = time.time()
        try:
            pixels0, s0 = tools.events(data, mNp) 
            pixels.extend(pixels0)
            s.extend(s0)
        except:
            pixels = copy.copy(pixels0)
            s = copy.copy(s0)
        #print("Second stage processing %f" % (time.time()-tt))
        if len(s) >= nf2:
            break 
    f.close()
    img = img/len(s)#nframes
    print("Reading time %3.3f sec" % (time.time()-t0))
    return pixels,s,img,nframes,mask
    
def p10_eiger_event_dataf(fileName,nf1,nf2,mask,mNp):
    '''Read a P10 HDF5 master file using h5py and numpy'''
    from dynamix.correlator.WXPCS import eigerpix
    
    t0 = time.time()
    try:
        f = h5py.File(fileName, "r")
    except OSError :    
        print("File %s cannot be read" % fileName) 
        exit() 
    print("Read a P10 HDF5 file")
    datas = f['/entry/data']
    pixels = []
    s = []
    nframes = 0  
    for i in dict(datas):
        #tt = time.time()
        n_frames, nx, ny = datas[i].shape 
        nx = nx*ny  
        for nn in range(n_frames):
             img0 = datas[i][nn,:,:]
             try:
                 img += img0
             except:
                 img = img0 + 0 
             img0[mask>0] = 0    
             matr = numpy.ravel(img0)
             msumpix,mpix = eigerpix(matr,mNp,nx) 
             mpix = mpix[:msumpix]
             pixels.append(mpix)
             s.append(msumpix)
             nframes += 1
             if nframes >= nf2:
                 break 
        #nframes += n_frames 
        if nframes >= nf2:
            break         
    f.close()        
    img = img/nframes
    print("Reading time %3.3f sec" % (time.time()-t0))
    return pixels,s,img,nframes,mask
     

def id10_eiger4m_event_dataf(fileName,detector,nf1,nf2,mask,mNp,scan):
    '''Read a ID10 HDF5 master file using h5py and numpy'''
    
    t0 = time.time()
    try:
        f = h5py.File(fileName, "r")
    except OSError :    
        print("File %s cannot be read" % fileName) 
        exit() 
    print("Read a ID10 HDF5 file")
    datas = f['/'+scan+'.1/measurement/'+detector]
    n_frames, nx, ny = datas.shape
    pixels = []
    s = []
    for i in range(nf1,nf2,1):
        data = numpy.array(datas[i,:,:],numpy.uint8)
        try:
            img += data
        except:
            img = numpy.array(data,numpy.float32)
        mask[data>20] = 1
        data[mask>0] = 0
        trace0 = data.sum()
        mNp = trace0.max()+10
        pixels0, s0 = tools.events(data, mNp) 
        try:
            pixels.extend(pixels0)
            s.extend(s0)
        except:
            pixels = copy.copy(pixels0)
            s = copy.copy(s0)
        if len(s) >= nf2:
            break 
    f.close()
    nframes = nf2-nf1
    print("Number of frames %d" % nframes)
    #print("Number of frames %d" % len(s))
    img = img/len(s)#nframes
    print("Reading time %3.3f sec" % (time.time()-t0))
    return pixels,s,img,nframes,mask

def id10_eiger4m_event_GPU_dataf(fileName,detector,nf1,nf2,mask,scan,thr=20,frc=0.15):
    """ Read a ID10 HDF5 master file using h5py, numpy and eigercompress 

    :param fileName: string name of the h5 file
    :param nf1: iteger first frame number
    :param nf2: integet last frame number
    :param mask: 2D array mask (1 is masked pixel)
    :param scan: string scan number 
    :param thr: integer uper threshold to cut hot pixels
    :param frc: float fraction of frames with envent between 0-1 

    :return: evs,tms,cnt,afr,n_frames,mask,trace (events, times, counters, average image, maks, trace)
    """
    import numpy as np
    import sys
    sys.path.append('/users/opid10/.venv/dynamix/lib64/python3.8/site-packages/dynamix/io')
    from wxpcs import eigercompress
    t0 = time.time()
    try:
        f = h5py.File(fileName, "r")
    except OSError :    
        print("File %s cannot be read" % fileName) 
        exit() 
    print("Read a ID10 HDF5 file")
    data = f['/'+scan+'.1/measurement/'+detector]
    n_frames, nx, ny = data.shape
    n_frames = nf2-nf1
    print("Number of frames %d" % n_frames)
    print("Data size in MB %d" % (n_frames*nx*ny*4/1024**2))
    ll = nx*ny # total number of pixels
    lp = int(n_frames*frc) # total number of frames with events 15%
    mask = np.array(np.ravel(mask),np.int32)
    evs = np.zeros((ll,lp),np.int32)
    tms = np.zeros((ll,lp),np.int32)
    cnt = np.ravel(np.zeros((ll,),np.int32))
    afr = np.ravel(np.zeros((ll,),np.int32))
    tr = 0
    trace = np.zeros((n_frames,),np.int32)
    it = 0    
    for i in range(nf1,nf2,1):
        fr = np.ravel(data[i,:,:])
        evs,tms,cnt,afr,mask,tr = eigercompress(evs,tms,cnt,afr,mask,tr,fr,thr,it,ll,lp)
        trace[i] = tr
        it += 1 
    f.close()
    afr = afr/n_frames
    afr = np.reshape(afr,(nx,ny))
    mask = np.reshape(mask,(nx,ny))
    print("Reading time %3.3f sec" % (time.time()-t0))
    return evs,tms,cnt,afr,n_frames,mask,trace


@nb.jit(nopython=True, parallel=True)
def neigercompress(evs,tms,cnt,afr,m,fr,thr,i,ll):
    """
    Numba implementation 
    Compact one frame:

    :param evs: 2D array (Number_of_pixels,Number_of_frames_with_events) of events containing all the nonzero data values
    :param tms: 2D array (Number_of_pixels,Number_of_frames_with_events) of time indices corresponding to nonzero data values
    :param cnt: 1D array (Number_of_pixels) of counts of number of frames with events in a pixel
    :param afr: 1D array (Number_of_pixels) of total number of events per pixel
    :param m:   1D array (Number_of_pixels) of masked pixels
    :param fr: 1D array (Number_of_pixels) of detector frame to compact
    :param thr: int upper threshold for photon count in a pixel (20)
    :param i:  int frame number
    :param ll: int number of pixels in a detector frame
   
    :return: evs, tms, cnt, afr, m, tr 
    : evs: updated 2D array (Number_of_pixels,Number_of_frames_with_events) of events containing all the nonzero data values
    : tms: updated 2D array (Number_of_pixels,Number_of_frames_with_events) of time indices corresponding to nonzero data values
    : cnt: updated 1D array (Number_of_pixels) of counts of number of frames with events in a pixel
    : afr: updated 1D array (Number_of_pixels) of total number of events per pixel
    : m:   updated 1D array (Number_of_pixels) of masked pixels
    : tr:  int number of photons per frame for trace 
    """
    tr = 0
    for p in nb.prange(ll):
        if fr[p]>0:
            afr[p] += fr[p]
            if fr[p]>thr:
                m[p] = 1
            elif m[p]<1:
               cnt[p] += 1  
               c = cnt[p]
               #c = cnt[p] + 1
               evs[p,c] = fr[p]
               tms[p,c] = i
               #cnt[p] = c          
               tr += fr[p]
    return evs,tms,cnt,afr,m,tr   


@nb.jit(nopython=True)
def nprepare(evs,tms):
    """
    Numba implementation 
    Eliminate 0 velues:

    :param evs: 1D array (Number_of_pixels*Number_of_frames_with_events) of events containing all the nonzero data values
    :param tms: 1D array (Number_of_pixels*Number_of_frames_with_events) of time indices corresponding to nonzero data values
    
   
    :return: evs, tms
    : evs: updated 1D array (Number_of_pixels*Number_of_frames_with_events) of events containing all the nonzero data values
    : tms: updated 1D array (Number_of_pixels,Number_of_frames_with_events) of time indices corresponding to nonzero data values
    """
    ll = evs.size
    i = 0 
    for p in range(ll):
        if evs[p]>0:
            evs[i] = evs[p]
            tms[i] = tms[p]
            i += 1
    return evs[:i],tms[:i]

def id10_eiger4m_event_GPU_datan(fileName,detector,nf1,nf2,mask,scan,thr=20,frc=0.15):
    """ Read a ID10 HDF5 master file using h5py, numpy and eigercompress 

    :param fileName: string name of the h5 file
    :param nf1: iteger first frame number
    :param nf2: integet last frame number
    :param mask: 2D array mask (1 is masked pixel)
    :param scan: string scan number 
    :param thr: integer uper threshold to cut hot pixels
    :param frc: float fraction of frames with envent between 0-1 

    :return: evs,tms,cnt,afr,n_frames,mask,trace (events, times, counters, average image, maks, trace)
    """
    import numpy as np
    import sys
    t0 = time.time()
    try:
        f = h5py.File(fileName, "r")
    except OSError :    
        print("File %s cannot be read" % fileName) 
        exit() 
    print("Read a ID10 HDF5 file")
    data = f['/'+scan+'.1/measurement/'+detector]
    n_frames, nx, ny = data.shape
    n_frames = nf2-nf1
    print("Number of frames %d" % n_frames)
    print("Data size in MB %d" % (n_frames*nx*ny*4/1024**2))
    ll = nx*ny # total number of pixels
    max_e = 1000#int(n_frames*frc) # total number of frames with events 15%
    mask = np.array(np.ravel(mask),np.uint8)
    evs = np.zeros((ll,max_e),np.uint8)
    tms = np.zeros((ll,max_e),np.uint16)
    cnt = np.ravel(np.zeros((ll,),np.uint16))
    afr = np.ravel(np.zeros((ll,),np.uint32))
    trace = np.zeros((n_frames,),np.uint32)
    it = 0    
    for i in range(nf1,nf2,1):
        fr = np.ravel(data[i,:,:])
        evs,tms,cnt,afr,mask,tr = neigercompress(evs,tms,cnt,afr,mask,fr,thr,it,ll)
        trace[it] = tr
        if it >= max_e:
            cntm = cnt.max()
            if cntm == max_e-1:
                nmax_e = int(1.1*n_frames*cnt.max()/it)
                evs = np.concatenate((evs,np.zeros((ll,nmax_e-max_e),np.uint8)),axis=1)
                tms = np.concatenate((tms,np.zeros((ll,nmax_e-max_e),np.uint16)),axis=1)
                max_e = nmax_e+0
                print("Extend array size to %d, file number %d" % (max_e,it))
        it += 1
    f.close()
    afr = afr/n_frames
    afr = np.reshape(afr,(nx,ny))
    mask = np.reshape(mask,(nx,ny))
    evs,tms = nprepare(np.ravel(evs),np.ravel(tms))
    evs = np.array(evs,np.int8)
    print("Reading time %3.3f sec" % (time.time()-t0))
    return evs,tms,cnt,afr,n_frames,mask,trace
