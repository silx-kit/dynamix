#! /usr/bin/env python3
#tools
import sys

import  numpy as np
import time
import os
import pylab as plt
from dynamix.io import readdata, EdfMethods, h5reader
from dynamix.correlator.WXPCS import eigerpix

#####radial averaging ###########

def radi(saxs,mask,cx,cy):
    """ Radial averaging of 2D pattern histogram method

    :param saxs: 2D array of saxs pattern
    :param mask: 2D array mask (1 is masked pixel)
    :param cx: x position of the direct beam pixel (column)
    :param cy: y position of the direct beam pixel (raw)

    :return: 2D array first column pixel, second column average intensity
             2D array of radius in pixels
             2D array radial average pattern
    """
    Y,X = np.indices(saxs.shape)
    X = X - cx
    Y = Y - cy
    q = np.float32(np.sqrt(X**2+Y**2))
    qh = np.int32(q+0.5)#better match with data
    #qh = np.int16(q)#better match with pyfai
    q[mask>0] = 0
    saxs = saxs[mask<1]
    qh = qh[mask<1]
    qmax = np.arange(int(qh.min()),int(qh.max())+1,1)#this is correct
    ring_brightness, radius = np.histogram(qh, weights=saxs, bins=qmax)
    rings, radius = np.histogram(qh, bins=qmax)
    radi = np.zeros((len(radius)-1,2))
    radi[:,0] = radius[:-1]#(radius[:-1]+radius[1:])/2.0
    radi[:,1] = ring_brightness/rings
    new_saxs = q*0
    f1 = q-np.array(q,np.uint32)
    ind = np.array(q,np.uint32)-int(radius[0])
    ind[mask>0] = 0
    val = radi[:,1]
    val = np.append(val,val[-2:])
    ind[ind>radius[-1]]=0
    #print(len(val),ind.max())
    new_saxs[mask<1] = val[ind[mask<1]+1]*f1[mask<1] + val[ind[mask<1]]*(1-f1[mask<1])
    return radi, q, new_saxs

######### Beam center #######
def beam_center(data,mask,cx,cy,lc=60,lw=10):
    """ Finding the beam center 

    :param data: 2D array of saxs pattern
    :param mask: 2D array mask (1 is masked pixel)
    :param cx: x estimate of the direct beam pixel (column)
    :param cy: y estimate of the direct beam pixel (raw)
    :param lc: offest from the beam center
    :param lw: width of the stripe

    :return: cx, cy refined beam center
    """
    data = np.ma.array(data,mask=mask)
    ind = np.where((data>0)&(mask<1))

    rad, r_q, new_saxs = radi(data,mask,cx,cy)#radial averaging
    err = np.abs(data-new_saxs)[ind].mean()/np.mean(data[ind])#error
    print("Initial center cx=%.2f, cy=%.2f, err=%1.5f" % (cx,cy,err))

    ### show the stripes ####
    sdata = np.zeros(data.shape,np.uint32)
    sdata[int(cy-lc-lw/2):int(cy-lc+lw/2+1),:] += np.uint32(1)
    sdata[int(cy+lc-lw/2):int(cy+lc+lw/2+1),:] += np.uint32(1)
    sdata[:,int(cx-lc-lw/2):int(cx-lc+lw/2+1)] += np.uint32(1)
    sdata[:,int(cx+lc-lw/2):int(cx+lc+lw/2+1)] += np.uint32(1)

    plt.figure()
    with np.errstate(divide="ignore", invalid="ignore"):
        plt.imshow(np.log10(data),cmap='jet')
    plt.imshow(sdata,cmap='gray_r',alpha=0.3)
    plt.plot(cx,cy,'r+')
    ##### Find horizontal center x ##########################
    vl1 = np.sum(data[:,int(cx-lc-lw/2):int(cx-lc+lw/2+1)],1)/(lw+1)#vertical line 1
    verr0 = 1e+6
    ### pixel precision ####
    for llc in range(lc-10,lc+10,1):
        vl2 = np.sum(data[:,int(cx+llc-lw/2):int(cx+llc+lw/2+1)],1)/(lw+1)#vertical line
        verr = np.mean(np.abs(vl1-vl2))
        if verr<verr0:
            verr0 = verr+0
            nlc = llc

    vl20 = np.sum(data[:,int(cx+nlc-lw/2):int(cx+nlc+lw/2+1)],1)#vertical line
    verr0 = 1e+6
    nf = 0
    ### subpixel precision ####
    #for f in np.arange(-0.99,1.0,0.01):
    for f in np.arange(-0.51,0.52,0.01):
        if f>=0:
            vl2 = (vl20-data[:,int(cx+nlc-lw/2)]*f+data[:,int(cx+nlc+lw/2+1)+1]*f)/(lw+1)
        else:
            vl2 = (vl20-data[:,int(cx+nlc-lw/2-1)]*f+data[:,int(cx+nlc+lw/2+1)]*f)/(lw+1)
        verr = np.mean(np.abs(vl1-vl2))
        if verr<verr0:
            verr0 = verr+0
            nf = f

    cx = cx+(nlc+nf-lc)/2.0 #new horizontal beam center

    ##### Vertical center y ##########################
    vl1 = np.sum(data[int(cy-lc-lw/2):int(cy-lc+lw/2+1),:],0)/(lw+1)#horizontal line 1
    verr0 = 1e+6
    ### pixel precision ####
    for llc in range(lc-10,lc+10,1):
        vl2 = np.sum(data[int(cy+llc-lw/2):int(cy+llc+lw/2+1),:],0)/(lw+1)#horizontal line
        verr = np.mean(np.abs(vl1-vl2))
        if verr<verr0:
            verr0 = verr+0
            nlc = llc

    vl20 = np.sum(data[int(cy+nlc-lw/2):int(cy+nlc+lw/2+1),:],0)#horizontal line
    verr0 = 1e+6
    nf = 0
    ### subpixel precision ####
    #for f in np.arange(-0.99,1.0,0.01):
    for f in np.arange(-0.51,0.52,0.01):

        if f>=0:
            vl2 = (vl20-data[int(cy+nlc-lw/2),:]*f+data[int(cy+nlc+lw/2+1)+1,:]*f)/(lw+1)
        else:
            vl2 = (vl20-data[int(cy+nlc-lw/2-1),:]*f+data[int(cy+nlc+lw/2+1),:]*f)/(lw+1)
        verr = np.mean(np.abs(vl1-vl2))
        if verr<verr0:
            verr0 = verr+0
            nf = f

    cy = cy+(nlc+nf-lc)/2.0

    rad, r_q, new_saxs = radi(data,mask,cx,cy)#radial averaging
    err = np.abs(data-new_saxs)[ind].mean()/np.mean(data[ind])
    print("Final center cx=%.2f, cy=%.2f, err=%1.5f" % (cx,cy,err))
    return cx,cy

####### multitau correlation function #######
def cftomt(d, par=16):
    """ Transformation of a correlation function 
        with linear spacing into multitau spacing 

    :param data: d array of correlation functions
    :param par: integer 8,16,32,64,128 channel legth

    :return: x correlation function with multitau spacing
    """
    tmp = d[par:,:3]
    nt = []
    nd = []
    nse = []
    for i in range(par):
        nt.append(d[i,0])
        nd.append(d[i,1])
        nse.append(d[i,2])
    while len(tmp[:,0])>=par:
        ntmp = (tmp[:-1,:]+tmp[1:,:])/2
        for i in range(0,par,2):
            nt.append(ntmp[i,0])
            nd.append(ntmp[i,1])
            nse.append(ntmp[i,2])
        tmp = ntmp[par:-1:2,:3]
    x = np.array([nt,nd,nse]).T
    return x


def make_q(config):
    #### Sample description ######################################################################################

    sname = config["sample_description"]["name"]


    #### Data location ######################################################################################

    datdir = config["data_location"]["data_dir"]
    sample_dir = config["data_location"]["sample_dir"]
    prefd = config["data_location"]["data_prefix"]
    sufd = config["data_location"]["data_sufix"]
    nf1 = int(config["data_location"]["first_file"])
    nf2 = int(config["data_location"]["last_file"])
    darkdir = config["data_location"]["dark_dir"]
    df1 = int(config["data_location"]["first_dark"])
    df2 = int(config["data_location"]["last_dark"])
    savdir = config["data_location"]["result_dir"]


    #### Experimental setup ######################################################################################

    geometry = config["exp_setup"]["geometry"]
    cx = float(config["exp_setup"]["dbx"])
    cy = float(config["exp_setup"]["dby"])
    dt = float(config["exp_setup"]["lagtime"])
    lambdaw = float(config["exp_setup"]["wavelength"])
    distance = float(config["exp_setup"]["detector_distance"])
    first_q = float(config["exp_setup"]["firstq"])
    width_q = float(config["exp_setup"]["widthq"])
    step_q = float(config["exp_setup"]["stepq"])
    number_q = int(config["exp_setup"]["numberq"])
    beamstop_mask = config["exp_setup"]["beamstop_mask"]

    #### Correlator info  ######################################################################################


    correlator = config["correlator"]["method"]
    lth = int(config["correlator"]["low_threshold"])
    bADU = int(config["correlator"]["bottom_ADU"])
    tADU = int(config["correlator"]["top_ADU"])
    mNp = float(config["correlator"]["max_number"])
    aduph = int(config["correlator"]["photon_ADU"])
    ttcf_par = int(config["correlator"]["ttcf"])

    #### Detector description  ######################################################################################
    detector = config["detector"]["det_name"]
    pix_size = float(config["detector"]["pixels"])
    mask_file = config["detector"]["mask"]
    flatfield_file = config["detector"]["flatfield"]
    ###################################################################################################################
    print("Making qs")
     
    try:
        data = readdata.readnpz(savdir+sname+"_2D.npz")
    except:
        print("Cannot read "+savdir+sname+"_2D.npz")
        sys.exit()

    #if  beamstop_mask != 'none':
    #    try:
    #        bmask = np.abs(EdfMethods.loadedf(beamstop_mask))#reads edf and npy
    #        bmask[bmask>1] = 1
    #    except:
    #        print("Cannot read beamstop mask %s, skip" % beamstop_mask)

    ### read beamstop mask ####
    bmask = read_beamstop_mask(beamstop_mask)
    
    ### read detector mask ###
    mask = read_det_mask(mask_file,detector)
    
    mask[bmask>0] = 1
    
    data[mask>0] = 0


    ind = np.where((data>0)&(mask<1))

    data = np.ma.array(data,mask=mask)
    qmask = mask*0


    #t0 = time.time()
    rad, r_q, new_saxs = radi(data,mask,cx,cy)#radial averaging
    #print("Calculation time %3.4f sec" % (time.time()-t0))
    new_saxs[np.isnan(new_saxs)] = 0
    np.save(savdir+sname+"_gaus.npy",np.array(new_saxs,np.float32))


    rad[:,0] = 4*np.pi/lambdaw*np.sin(np.arctan(pix_size*rad[:,0]/distance)/2.0)#q vector calculation
    r_q = 4*np.pi/lambdaw*np.sin(np.arctan(pix_size*r_q/distance)/2.0)#q vector calculation
    width_p = np.tan(np.arcsin(width_q*lambdaw/4/np.pi)*2)*distance/pix_size
    qmask = np.array((r_q-first_q+width_q/2)/width_q+1,np.uint16)

    print("Number of Qs %d" % number_q)
    #print("Width of ROI is %1.1f pixels" % width_p)
    np.savetxt(savdir+sname+"_1D.dat",rad)

    #qmask = np.array((r_q-first_q+width_q/2)/width_q+1,np.uint16)
    #qmask[mask>0] = 0
    #np.save(savdir+sname+"_qmask.npy",np.array(qmask,np.uint16))
    #qmask[qmask>number_q] = 0


    #qp = np.linspace(first_q,first_q+(number_q-1)*width_q,number_q)
    qp = np.linspace(first_q,first_q+(number_q-1)*step_q,number_q)
    qmask = mask*0
    i_qp = qp*0
    i_bands = []
    n = 0
    for i in qp:
      i_qp[n] = rad[(np.abs(rad[:,0]-i)==(np.abs(rad[:,0]-i).min())),1]
      ind1 = np.where((rad[:,0]>=i-width_q/2)&(rad[:,0]<=i+width_q/2))[0]
      i_bands.append(ind1)
      n += 1
      indq = np.where(np.abs(r_q-i)<=width_q/2)
      qmask[indq] = n

    qmask[mask>0] = 0
    np.save(savdir+sname+"_qmask.npy",np.array(qmask,np.uint16))
    return


def test_dir(dir_name):
    """ test the existance of the directory 
        if it does not exist then create 

    :param dir_name: directory name

    """
    test_savdir = os.path.dirname(dir_name)
    if not os.path.exists(test_savdir):
        os.makedirs(test_savdir)
        print("Create",test_savdir)
    return


def format_result(CorrelationResult,qqmask,flatfield,cdata,dt,ttcf_par):
    """ format the obained result to usable form

    :param CorrelationResult: nametuple containing the results
    :param qqmask: q mask matrix
    :param flatfield: flatfield matrix
    :param cdata: smoothed image matrix
    :param dt: lat time float
    :param ttcf_par: number of q for trc integer

    :return: res, save_cf, trc formated cf for ploting, formated matrix for saving and ttime resolved cf
    """
    res = []
    n = 0
    nframes = CorrelationResult.res[0].size
    cf = np.zeros((nframes,3),np.float32)
    cf[:,0] = np.arange(1,nframes+1,1)*dt
    number_q = qqmask.max()
    for q in range(1,number_q+1,1):
        fcorrection = (flatfield[qqmask==q]**2).mean()/flatfield[qqmask==q].mean()**2
        correction = (cdata[qqmask==q]**2).mean()/cdata[qqmask==q].mean()**2 * fcorrection
        cf[:,1] = CorrelationResult.res[n,:]/correction # make correct baseline#CPU
        cf[:,2] = CorrelationResult.dev[n,:]/correction # make correct baseline#CPU
        if ttcf_par == q:
            trc = CorrelationResult.trc/correction  
        cfm = cftomt(cf)
        res.append(cfm)
        if q == 1:
            save_cf = cfm
        else:
            save_cf = np.append(save_cf,cfm[:,1:],axis=1)
        n += 1
    if ttcf_par == 0:    
        trc = CorrelationResult.trc
    return res, save_cf, trc
      

def save_cf(file_name,save_cf,qp):
    q_title='#q values 1/A:'
    for q in qp:
        q_title = q_title+" %.5f " % q
    q_title=q_title+'\n'
    f=open(file_name,'w')
    f.write(q_title)
    np.savetxt(f, save_cf)
    f.close()
    return

def events(data, mNp):
    """ convert 3D data matrix to events list

    :param data: 3D data matrix, first axis frame number, second and third axes are image
    :param mNp: maximum number of photons per frame 

    :return: pixels, s list of pixels and intgrated intensity per frame
    """
    #t0 = time.time()
    s = []
    pixels = []
    sshape = np.shape(data)
    if len(sshape) == 3 :
       nframes, nx, ny = np.shape(data)
       nx = nx*ny

       for i in range(nframes):
            matr = np.ravel(data[i,:,:])
            msumpix,mpix = eigerpix(matr,mNp,nx) 
            mpix = mpix[:msumpix]
            pixels.append(mpix)
            s.append(msumpix)

    if len(sshape) == 2:   
        nx, ny = np.shape(data)
        nx = nx*ny
        matr = np.ravel(data)
        msumpix,mpix = eigerpix(matr,mNp,nx) 
        mpix = mpix[:msumpix]
        pixels.append(mpix)
        s.append(msumpix)

    #print("Compaction time %f" % (time.time()-t0))

    return pixels, s 


def make_cdata(file_name,config):
    """ make smooth data

    :param file_name: filename that can contain cdata

    :return: cdata, smooth 2D image for correction
    """
    if os.path.isfile(file_name):
        cdata = np.array(np.load(file_name),np.float32)
    else:
        make_q(config)
        cdata = np.array(np.load(file_name),np.float32)
    cdata[np.isnan(cdata)] = 0

    return cdata

def read_det_mask(det_mask,detector):
    """ read detector mask file

    :param det_mask: filename that contains detector mask

    :return: dmask, 2D array of beamstop mask, 1 is masked values
    """
    if  det_mask != 'none':
        try:
            dmask = np.abs(EdfMethods.loadedf(det_mask))#reads edf and npy
            dmask[dmask>1] = 1  
        except:
            print("Cannot read detector mask %s, skip" % det_mask)
    else:
        if detector == 'andor':
            dshape=(1024,1024)
        elif detector == 'maxipix':
            dshape=(516,516)
        elif detector== 'mpx_si_22':
            dshape=(516,516)
        elif detector == 'eiger500k':
            dshape=(1024,514)    
        elif detector == 'eiger4m':
            dshape=(2162,2068)    
        else:    
            print("Needs detector mask file")
            sys.exit() 
        dmask = np.zeros(dshape,dtype='uint8')
    return dmask

def read_qmask(qmask_file,mask, number_q):
    """ read qmask file 

    :param qmask_file: filename that contains qmask
    :param mask: detector mask 2D array
    :param number_q: number of qs in q mask

    :return: qqmask, 2D array of q mask each reqion is 1,2,3...
    """
    if qmask_file != 'none':
        try:
            qqmask = EdfMethods.loadedf(qmask_file)
            qqmask[mask>0] = 0
            qqmask[qqmask>number_q] = 0
        except:
            print("Cannot read qmask %s, skip" % qmask_file)
            qqmask = mask*0
            qqmask[mask<1] = 1           
    else:
        qqmask = mask*0
        qqmask[mask<1] = 1
    
    return qqmask

def read_beamstop_mask(beamstop_mask):
    """ read beamstop mask file

    :param beamstop_mask: filename that contains beamstop mask

    :return: bmask, 2D array of beamstop mask, 1 is masked values
    """
    if  beamstop_mask != 'none':
        try:
            bmask = np.abs(EdfMethods.loadedf(beamstop_mask))#reads edf and npy
            bmask[bmask>1] = 1  
        except:
            print("Cannot read beamstop mask %s, skip" % beamstop_mask)
    else:
        bmask = 0
    return bmask
     
def reduce_matrix(data,qqmask,cdata,flatfield):
    """ read beamstop mask file

    :param data: 3D array of frames 
    :param qqmaks: 2D array of q mask
    :param cdata: 2D array of smooth data
    :param flatfield: 2D array of flatfield

    :return: data,qqmask,cdata,flatfield reduced array of original data
    """
       
    ind = np.where(qqmask>0)
    sindy = ind[0].min()
    eindy = ind[0].max()
    sindx = ind[1].min()
    eindx = ind[1].max()
    data = data[:,sindy:eindy,sindx:eindx]
    qqmask = qqmask[sindy:eindy,sindx:eindx]
    cdata = cdata[sindy:eindy,sindx:eindx]
    flatfield = flatfield[sindy:eindy,sindx:eindx]
    print("Reduced data size is %2.2f Gigabytes" % (data.size*data.itemsize/1024**3))

    return data,qqmask,cdata,flatfield

def data_compaction(data):
    """ compact data for event correlation 

    :param data: 3D array of frames 

    :return: events, times, offsets 
    """
    t0 = time.time()
    events = []
    times = [] 
    offsets = [0] 
    t,x,y = data.shape
    y =  x*y
    data = np.reshape(data,(t,y))

    for pix in range(y):
        ind = np.where(data[:,pix]>0)
        events.append(data[ind[0],pix])
        times.append(ind[0])
        offsets.append(ind[0].size)

    events = np.array(np.concatenate(events).ravel(),np.int8)
    times = np.array(np.concatenate(times).ravel(),np.int32)
    offsets = np.array(np.cumsum(offsets),np.uint32)
    print("Compaction time %f" % (time.time()-t0))

    return events, times, offsets

