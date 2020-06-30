#! /usr/bin/env python3
#xpcs script to start analysis of the data
#reads the ini configuration file
import sys
import os
#sys.path.append("/users/chushkin/.local/lib/python3.7/site-packages/dynamix")
#sys.path.append("/data/id10/inhouse/Programs/PyXPCS_project/wxpcs")
#sys.path.append("/users/chushkin/Documents/Analysis/Glass_school_2019/wxpcs")
#sys.path.append("/users/chushkin/Documents/Programs/PyXPCS_project/wxpcs")
import _pickle as cPickle
import numpy as np
import time
import fabio
from dynamix.io import readdata, EdfMethods, h5reader
from dynamix.tools import tools
import configparser
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())


def y_dense_correlator(xpcs_data, mask):
    """
    Reference implementation of the dense correlator.

    Parameters
    -----------
    xpcs_data: numpy.ndarray
        Stack of XPCS frames with shape (n_frames, n_rows, n_columns)
    mask: numpy.ndarray
        Mask of bins in the format (n_rows, n_columns).
        Zero pixels indicate unused pixels.
    """
    ind = np.where(mask > 0) # unused pixels are 0
    xpcs_data = xpcs_data[:, ind[0], ind[1]] # (n_tau, n_pix)
    del ind
    ltimes, lenmatr = np.shape(xpcs_data) # n_tau, n_pix
    meanmatr = np.array(np.mean(xpcs_data, axis=1),np.float32) # xpcs_data.sum(axis=-1).sum(axis=-1)/n_pix
    meanmatr.shape = 1, ltimes

 
    if ltimes*lenmatr>1000*512*512:
        nn = 16
        newlen = lenmatr//nn
        num = np.dot(np.array(xpcs_data[:,:newlen],np.float32), np.array(xpcs_data[:,:newlen],np.float32).T) 
        xpcs_data =  xpcs_data[:,newlen:] + 0
        tt0 = time.time()   
        for i in range(1,nn-1,1):
            num += np.dot(np.array(xpcs_data[:,:newlen],np.float32),np.array(xpcs_data[:,:newlen],np.float32).T)     
            xpcs_data = xpcs_data[:,newlen:] + 0 
            print("Progres %d %%, %3.1f seconds to finish" % (int(i/nn*100), (nn/i-1)*(time.time()-tt0))) 
        num += np.dot(np.array(xpcs_data,np.float32), np.array(xpcs_data,np.float32).T) 
    else:
        num = np.dot(np.array(xpcs_data,np.float32), np.array(xpcs_data,np.float32).T)  
    
    num /= lenmatr
    denom = np.dot(meanmatr.T, meanmatr)
    del meanmatr
    res = np.zeros((ltimes-1,3)) # was ones()
    for i in range(1,ltimes,1): # was ltimes-1, so res[-1] was always 1 !
        dia_n = np.diag(num, k=i)
        sdia_d = np.diag(denom, k=i)
        res[i-1,0] = i
        res[i-1,1] = np.sum(dia_n)/np.sum(sdia_d) 
        res[i-1,2] = np.std(dia_n/sdia_d) / len(sdia_d)**0.5
    return res, num/denom


def cftomt_testing(d):
   par = 16
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


################# READING CONFIG FILE ##########################################################################

try:
    config.read(sys.argv[1])
except:
    print("Cannot read "+sys.argv[1]+" input file")
    exit()

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
pix_size = float(config["detector"]["pixels"])
distance = float(config["exp_setup"]["detector_distance"])
first_q = float(config["exp_setup"]["firstq"])
width_q = float(config["exp_setup"]["widthq"])
number_q = int(config["exp_setup"]["numberq"])
qmask_file = config["exp_setup"]["q_mask"]
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
mask_file = config["detector"]["mask"]
flatfield_file = config["detector"]["flatfield"]
###################################################################################################################
from dynamix.plot.draw_result import plot_cf, show_trc
import pylab as plt

#correlator = "event"
#correlator = "intensity"

if correlator == "event":
    ###### event ######################
    from dynamix.correlator.event_y import ecorrts

    #### check if the saving direftory exist and create one #####
    test_savdir = os.path.dirname(savdir)
    if not os.path.exists(test_savdir):
        os.makedirs(test_savdir)
        print("Create",test_savdir)

    #### read and dropletize data ####
    if detector == "Andor":
        #pixels - list of pixels that see a photon
        #s - sum of all photons in a frame
        #for_norm - number of pixels used 
        #img - summed 2D scattering pattern
        pixels, s, for_norm, img =      readdata.get_ccd_event_data(datdir,prefd,sufd,nf1,nf2,darkdir,df1,df2,sname,lth,bADU,tADU,mNp,aduph,savdir,mask_file)
    else:
        pixels, s, for_norm, img = readdata.get_eiger_event_data(sample_dir,prefd,sufd,nf1,nf2,sname,mNp,savdir,mask_file)
    print(len(pixels),len(s))

    #### save summed image #####
    np.savez_compressed(savdir+sname+"_2D_raw.npz",data=np.array(img,np.float32))

    ### read and apply mask ### 
    mask = EdfMethods.loadedf(mask_file)
    img[mask>0] = 0
   
    ### read and apply flat field ###
    if flatfield_file != 'none':
        flatfield = EdfMethods.loadedf(flatfield_file)
        fcorrection = (flatfield[mask<1]**2).mean()/flatfield[mask<1].mean()**2  
        print("Flatfield correction %2.4f" % fcorrection)
        img /= flatfield

    np.savez_compressed(savdir+sname+"_2D.npz",data=np.array(img,np.float32))

    correction = (img[mask<1]**2).mean()/img[mask<1].mean()**2# * fcorrection
    print("Correction %2.4f" % correction)

    #### calculate the correlation function #####
    #cf - intensity correlation funtion
    #mcf - multitau  intensity correlation funtion
    #trc - two-time correlation fuction
    cf,mcf,trc = ecorrts(pixels,s,for_norm)

    #### save results ########
    cf[:,0] *= dt
    cf[:,1] /= correction # make correct baseline
    cf[:,2] /= correction # make correct baseline
    np.savetxt(savdir+sname+"_event_cf.dat",cf)
    mcf[:,0] *= dt
    mcf[:,1] /= correction # make correct baseline
    mcf[:,2] /= correction # make correct baseline
    np.savetxt(savdir+sname+"_event_mcf.dat",mcf)
    np.savetxt(savdir+sname+"_trace.dat",s)

    readdata.savenpz(savdir+sname+"_event_trc.npz",np.array(trc/correction,np.float32))

    #### plot results #######
    plot_cf(cf,sname)
    #show_trc(t,sname)
elif correlator == "intensity":
    ##### intensity #################################
    from dynamix.correlator.dense import MatMulCorrelator
    from dynamix.correlator.cuda import CublasMatMulCorrelator 
    from dynamix.correlator.dense import FFTWCorrelator
    from dynamix.correlator.dense import export_wisdom, import_wisdom
    #### check if the saving direftory exist and create one #####
    test_savdir = os.path.dirname(savdir)
    if not os.path.exists(test_savdir):
        os.makedirs(test_savdir)
        print("Create",test_savdir)
    #cx = int(cx+0.5)
    #cy = int(cy+0.5)  
    ##### read data ##########################################
    if sufd.find("edf") > -1:#== ".edf":
        data = readdata.get_data(sample_dir,prefd,sufd,nf1,nf2)#[:3000,:,:]
    elif sufd == ".h5":
        data = h5reader.myreader(sample_dir+sname+"_raw.h5")[nf1:nf2,:,:]#,cy-64:cy+64,cx-64:cx+64]        
    else:  
        exit()
   
    if os.path.isfile(savdir+sname+"_2D_raw.npz"):
        pass
    else:
        np.savez_compressed(savdir+sname+"_2D_raw.npz",data=np.array(np.mean(data,0),np.float32))


    if mask_file != 'none':
        mask = EdfMethods.loadedf(mask_file)#[cy-64:cy+64,cx-64:cx+64]#reads edf and npy
        data[:,mask>0] = 0    
    if flatfield_file != 'none':
        flatfield = EdfMethods.loadedf(flatfield_file)
        fcorrection = (flatfield[mask<1]**2).mean()/flatfield[mask<1].mean()**2  
        print("Flatfield correction %2.4f" % fcorrection)
        #data = data/flatfield 
    if os.path.isfile(savdir+sname+"_2D.npz"):
        pass
    else:
        np.savez_compressed(savdir+sname+"_2D.npz",data=np.array(np.mean(data,0),np.float32))
        #np.savetxt(savdir+sname+"_trace.dat",np.array(np.mean(np.mean(data,1),1),np.float32))
        #exit()
    data = np.array(data,np.uint8)
    print("Data size is %2.2f Gigabytes" % (data.size*data.itemsize/1024**3))
    print("Data type", data.dtype,data.max())
       
    cdata = np.array(np.load(savdir+sname+"_gaus.npy"),np.float32)#[cy-64:cy+64,cx-64:cx+64]
    #cdata = readdata.readnpz(savdir+sname+"_2D.npz")
    #from scipy.ndimage import gaussian_filter
    #cdata = gaussian_filter(cdata,0.5) 
    #correction = (cdata[mask<1]**2).mean()/cdata[mask<1].mean()**2 * fcorrection
    #print("Correction %2.4f" % correction)


    if qmask_file != 'none':
        qqmask = EdfMethods.loadedf(qmask_file)#[cy-64:cy+64,cx-64:cx+64]
        qqmask[mask>0] = 0 
        qqmask[qqmask>number_q] = 0
    else:
        qqmask = mask*0
        qqmask[mask<1] = 1
    
    #### reduce the matrix size to speed up calculations ####
    
    ind = np.where(qqmask>0)
    sindy = ind[0].min()
    eindy = ind[0].max()
    sindx = ind[1].min()
    eindx = ind[1].max()
    data = data[:,sindy:eindy,sindx:eindx]
    qqmask = qqmask[sindy:eindy,sindx:eindx]
    cdata = cdata[sindy:eindy,sindx:eindx]
    print("Reduced data size is %2.2f Gigabytes" % (data.size*data.itemsize/1024**3))
    
    shape = np.shape(data[0,:,:])
    nframes = np.shape(data)[0]
    print("Number of frames %d" % nframes)
    print("Number of qs %d" % number_q)
    ### correlator ####### 
    t0 = time.time()
    
    #### MatMul Correlator #####################################
    #correlator = FFTWCorrelator(shape,nframes,qmask=qqmask)
     #print("Using FFTW")
    correlator = MatMulCorrelator(shape,nframes,qmask=qqmask)
    print("Using CPU")
    #correlator = CublasMatMulCorrelator(shape, nframes, qmask=qqmask)
    #print("Using GPU") 
    t1 = time.time()
    #import_wisdom("./")
    result = correlator.correlate(data)
    #export_wisdom("./")
    print("Correlator time %3.2f seconds" % (time.time()-t1))
    trace = np.zeros((nframes,number_q),np.float32)
    res = []
    n = 0
    cf = np.zeros((nframes-1,3),np.float32)
    cf[:,0] = np.arange(1,nframes,1)*dt
    for q in range(1,number_q+1,1):
        trace[:,n] =  np.sum(data[:,qqmask==q],1)
        correction = (cdata[qqmask==q]**2).mean()/cdata[qqmask==q].mean()**2# * fcorrection
        #cf[:,1] = result[n][1:]/correction
        #cf[:,2] = result[n][1:]/correction*3e-3
        cf = result[n]
        cf[:,0] *= dt
        cf[:,1] /= correction # make correct baseline
        cf[:,2] /= correction # make correct baseline
        cfm = tools.cftomt(cf)
        res.append(cfm)
        if q == 1:
            save_cf = cfm
        else:
            save_cf = np.append(save_cf,cfm[:,1:],axis=1)
        n += 1
    ''' 
    #### y_dense correlator ########################
    res = []
    trace = np.zeros((nframes,number_q),np.float32)
    n = 0
    t0 = time.time()
    for q in range(1,number_q+1,1):
        trace[:,n] =  np.sum(data[:,qqmask==q],1)
        correction = (cdata[qqmask==q]**2).mean()/cdata[qqmask==q].mean()**2# * fcorrection
        #print("Correction %2.4f, %d" % (correction, q))
        cf, trc = y_dense_correlator(data,(qqmask==q))
        cf[:,0] *= dt
        cf[:,1] /= correction # make correct baseline
        cf[:,2] /= correction # make correct baseline
        #cf = cftomt_testing(cf)
        cf = tools.cftomt(cf)
        res.append(cf)
        if q == 1:
            save_cf = cf
        else:
            save_cf = np.append(save_cf,cf[:,1:],axis=1)
        n += 1
    ''' 
    print("Correlation time %3.2f seconds" % (time.time()-t0))
    q_title='#q values 1/A:'
    qp = np.linspace(first_q,first_q+(number_q-1)*width_q,number_q)
    for q in qp:
        q_title = q_title+" %.5f " % q 
    q_title=q_title+'\n'
    f=open(savdir+sname+"_trace.dat",'w')
    f.write(q_title)
    np.savetxt(f, trace)
    f.close()
    f=open(savdir+sname+"_cf.dat",'w')
    f.write(q_title)
    np.savetxt(f, save_cf)
    f.close()
    #np.savetxt(savdir+sname+"_trace.dat",trace)
    #np.savetxt(savdir+sname+"_cf.dat",save_cf)
    #readdata.savenpz(savdir+sname+"_trc.npz",np.array(trc/correction,np.float32))
    #plot_cf(cf,sname)
    plot_cf(res,sname)


else: pass
exit()


############ plotting ############################
from dynamix.plot.draw_result import plot_cf, show_trc

plot_cf(w,sname)
show_trc(t,sname)

