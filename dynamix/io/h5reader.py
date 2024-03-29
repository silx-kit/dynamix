import h5py    # HDF5 support
import hdf5plugin
import numpy
import numba as nb
import time
import copy
from datetime import datetime
from dynamix.tools import tools

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def h5writer(fileName,data):
    '''Writes a NeXus HDF5 file using h5py and numpy'''
    print("Write a NeXus HDF5 file")

    timestamp = str(datetime.now())

    # create the HDF5 NeXus file
    with  h5py.File(fileName, "w") as f:
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

    print("wrote file:", fileName)


def myreader(fileName,nf1,nf2,scan="none"):
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
        data = f['/entry_0000/measurement/data']
    else:
        data = f['/'+scan+'.1/measurement/eiger4m']
    data = numpy.array(data[nf1:nf2,:,:],numpy.int8)
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
        data = datas[i][()].astype(numpy.uint8)
        if nframes == 0:
            img = data.sum(axis=0, dtype=numpy.uint16)
        else:
            img += data.sum(axis=0)
        nframes += data.shape[0]
        data[:, mask>0] = 0
        trace0 = data.sum(axis=(1,2))

        mNp = trace0.max() + 10
        print("Diagnostics maximum photons per frame %d" % (mNp-10))

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
    print("Read a P10 HDF5 file")
    with  h5py.File(fileName, "r") as f:
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
                if nframes==0:
                    img = img0
                else:
                    img += img0
                img0[mask>0] = 0    
                matr = numpy.ravel(img0)
                msumpix, mpix = eigerpix(matr,mNp,nx) 
                mpix = mpix[:msumpix]
                pixels.append(mpix)
                s.append(msumpix)
                nframes += 1
                if nframes >= nf2:
                    break 
            if nframes >= nf2:
                break         
    img = img/nframes
    print("Reading time %3.3f sec" % (time.time()-t0))
    return pixels,s,img,nframes,mask
     

def id10_eiger4m_event_dataf(fileName, nf1, nf2, mask, mNp, scan):
    '''Read a ID10 HDF5 master file using h5py and numpy'''
    
    t0 = time.time()
    print("Read a ID10 HDF5 file")
    with h5py.File(fileName, "r") as f:
        datas = f['/'+scan+'.1/measurement/eiger4m']
        n_frames, nx, ny = datas.shape
        pixels = []
        s = []
        img = None
        for i in range(nf1,nf2,1):
            data = datas[i,:,:].astype(numpy.uint8)
            if img is None:
                img = data.astype(numpy.float32)
            else:
                img += data
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

    nframes = nf2-nf1
    print("Number of frames %d" % nframes)
    #print("Number of frames %d" % len(s))
    img = img/len(s)#nframes
    print("Reading time %3.3f sec" % (time.time()-t0))
    return pixels,s,img,nframes,mask


def id10_eiger4m_event_GPU_dataf(fileName,nf1,nf2,mask,scan,thr=20,frc=0.15):
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
    from ..correlator.WXPCS import eigercompress
    t0 = time.time()
    try:
        f = h5py.File(fileName, "r")
    except OSError :    
        print("File %s cannot be read" % fileName) 
        exit() 
    print("Read a ID10 HDF5 file")
    data = f['/'+scan+'.1/measurement/eiger4m']
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


@nb.jit(nopython=True, parallel=True, fastmath=True)
def neigercompressold(evs,tms,cnt,afr,m,tr,fr,thr,i,ll,max_e):
    tr = 0
    for p in nb.prange(ll):
        afr[p] += fr[p]
        if fr[p]>thr:
            m[p] = 1
        if m[p]>0:
            fr[p] = 0
        if fr[p]>0:
            c = cnt[p] + 1
            evs[p,c] = fr[p]
            tms[p,c] = i
            cnt[p] = c          
        tr += fr[p]
    return evs,tms,cnt,afr,m,tr       


@nb.jit(nopython=True, parallel=True, fastmath=True)
def neigercompress(evs,tms,cnt,afr,m,tr,fr,thr,i,ll,max_e):
    tr = 0
    for p in nb.prange(ll):
        if fr[p]>0:
            afr[p] += fr[p]
            if fr[p]>thr:
                m[p] = 1
            if m[p]<1:
                c = cnt[p] + 1
                evs[p,c] = fr[p]
                tms[p,c] = i
                cnt[p] = c          
                tr += fr[p]
    return evs, tms, cnt, afr, m, tr 


@nb.jit(nopython=True, fastmath=True)
def nprepare(evs,tms):
    ll = evs.size
    i = 0 
    for p in range(ll):
        if evs[p]>0:
            evs[i] = evs[p]
            tms[i] = tms[p]
            i += 1
    return evs, tms, i


def id10_eiger4m_event_GPU_datan(fileName, nf1, nf2, mask, scan, thr=20, frc=0.15):
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
    t0 = time.time()
    print("Read a ID10 HDF5 file")
    with h5py.File(fileName, "r") as f:

        data = f['/'+scan+'.1/measurement/eiger4m']
        _, nx, ny = data.shape
        n_frames = nf2-nf1
        print("Number of frames %d" % n_frames)
        print("Data size in MB %d" % (n_frames*nx*ny*4/1024**2))
        ll = nx*ny # total number of pixels
        lp = int(n_frames*frc) # total number of frames with events 15%
        mask = mask.ravel().astype(numpy.uint8)
        evs = numpy.zeros((ll,lp),numpy.uint8)
        tms = numpy.zeros((ll,lp),numpy.uint16)
        cnt = numpy.ravel(numpy.zeros((ll,),numpy.uint16))
        afr = numpy.ravel(numpy.zeros((ll,),numpy.uint32))
        tr = 0
        trace = numpy.zeros((n_frames,),numpy.uint32)
        it = 0    
        for i in range(nf1,nf2,1):
            fr = numpy.ravel(data[i,:,:])
            evs,tms,cnt,afr,mask,tr = neigercompress(evs,tms,cnt,afr,mask,tr,fr,thr,it,ll,lp)
            trace[i] = tr
            it += 1 
    
    afr = afr/n_frames
    afr = numpy.reshape(afr, (nx,ny))
    mask = numpy.reshape(mask, (nx,ny))
    evs,tms,c = nprepare(numpy.ravel(evs), numpy.ravel(tms))
    evs = numpy.array(evs[:c], numpy.int8)
    tms = tms[:c]
    print(f"Reading time {time.time()-t0:3.3f} sec")
    return evs,tms,cnt,afr,n_frames,mask,trace


