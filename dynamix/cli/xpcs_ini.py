#! /usr/bin/env python3
#Script for XPCS data analysis 
#Reads the ini configuration file

import shutil
import sys
import os
import _pickle as cPickle
import numpy as np
import time
import fabio
from dynamix.io import readdata, EdfMethods, h5reader
from dynamix.tools import tools

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from collections import namedtuple
CorrelationResult = namedtuple("CorrelationResult", "res dev trc")

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

def main():

    ################# READING CONFIG FILE #########################################################

    try:
        config.read(sys.argv[1])
    except Exception as e:
        print("Cannot read "+sys.argv[1]+" input file, error ",str(e))
        sys.exit()

    #### Sample description #######################################################################

    sname = config["sample_description"]["name"]
    scan = config["sample_description"]["scan"]
    try:
        scan = str(int(scan.split("scan")[1]))
    except:
        pass 
    #### Data location ############################################################################

    datdir = config["data_location"]["data_dir"]
    sample_dir = config["data_location"]["sample_dir"]
    prefd = config["data_location"]["data_prefix"]
    sufd = config["data_location"]["data_sufix"]
    nf1 = int(config["data_location"]["first_file"])
    nf2 = int(config["data_location"]["last_file"])
    try:
        skip = int(config["data_location"]["skip"])
        print("Use every %d image"% skip)
    except:
        skip = 1
    darkdir = config["data_location"]["dark_dir"]
    df1 = int(config["data_location"]["first_dark"])
    df2 = int(config["data_location"]["last_dark"])
    savdir = config["data_location"]["result_dir"]
    toplot = config["data_location"]["toplot"]

    #### Experimental setup #######################################################################

    geometry = config["exp_setup"]["geometry"]
    cx = float(config["exp_setup"]["dbx"])
    cy = float(config["exp_setup"]["dby"])
    dt = config["exp_setup"]["lagtime"]
    lambdaw = float(config["exp_setup"]["wavelength"])
    distance = float(config["exp_setup"]["detector_distance"])
    first_q = float(config["exp_setup"]["firstq"])
    width_q = float(config["exp_setup"]["widthq"])
    step_q = float(config["exp_setup"]["stepq"])
    number_q = int(config["exp_setup"]["numberq"])
    qmask_file = config["exp_setup"]["q_mask"]
    beamstop_mask = config["exp_setup"]["beamstop_mask"]

    #### Correlator info  #########################################################################


    correlator = config["correlator"]["method"]
    engine = config["correlator"]["engine"]
    lth = int(config["correlator"]["low_threshold"])
    bADU = int(config["correlator"]["bottom_ADU"])
    tADU = int(config["correlator"]["top_ADU"])
    mNp = float(config["correlator"]["max_number"])
    aduph = int(config["correlator"]["photon_ADU"])
    ttcf_par = int(config["correlator"]["ttcf"])

    #### Detector description  ####################################################################

    detector = config["detector"]["det_name"]
    pix_size = float(config["detector"]["pixels"])
    mask_file = config["detector"]["mask"]
    flatfield_file = config["detector"]["flatfield"]
    
    ### compaction paramaters ######################################
    fr = 0.4 # fraction of frames with events 
    #small number of events fr should be small
    phthr = 40 # max photon cutoff threshold 
    
    #### check if the saving direftory exist and create one #####
    tools.test_dir(savdir)
    
    
    ### read time period from h5 file #########
    if dt == "auto":
        try:
            dt = readdata.get_dt(sample_dir,prefd,sufd,scan)[0]            
            #config["exp_setup"]["lagtime"] = str(dt)
            #with open(sys.argv[1], 'w') as configfile:
            #    config.write(configfile)    
        except Exception as err:
            print("Auto reading of the time is not good. error was",str(err))
            sys.exit()
    dt = float(dt)*skip
    print("Lag time with the skip factor %s" % str(dt))
    #### Copy config file to the result directory #####
    
    shutil.copy(sys.argv[1], savdir+"input_xpcs_"+sname+".txt")
    
    ###############################################################################################
    from dynamix.plot.draw_result import plot_cf, show_trc
    import pylab as plt
    print("Start analysis of the sample %s Scan number=%s" % (sname,str(scan)))
    print("Running %s correlator using %s processor." % (correlator, engine))

    if correlator == "event":
        ###### event ######################
        from dynamix.correlator.event_y import ecorrts, nbecorrts_q, ecorrts_q, gpu_ecorr, gpu_ecorr_q

        ### read detector mask ###
        mask = tools.read_det_mask(mask_file,detector)

        ### read beamstop mask ####
        bmask = tools.read_beamstop_mask(beamstop_mask)
   
        mask[bmask>0] = 1

        ### make q ### 
        #tools.make_q(config)
                        
        ### read qmask ####
        qqmask = tools.read_qmask(qmask_file,mask, number_q)

        qqmask[mask>0] = 0

        #### read and dropletize data ####
        if detector == "andor":
            #pixels - list of pixels that see a photon
            #s - sum of all photons in a frame
            #for_norm - number of pixels used
            #img - summed 2D scattering pattern
            print("Detector andor")
            #get the delta value from the tenth image to adjust cx position and first_q
            delta = readdata.get_delta(sample_dir,prefd,sufd,nf1,nf2)
            cx = int(cx-distance*np.tan(np.deg2rad(delta))/pix_size)
            first_q = 4*np.pi/lambdaw*np.sin(np.deg2rad(delta)/2)
            print("Automatic adjusting cx=%d and first_q=%3.3f" % (cx,first_q)) 
            if engine == "CPUold": 
                pixels, s, for_norm, img =      readdata.get_ccd_event_data(sample_dir,prefd,sufd,nf1,nf2,darkdir,df1,df2,sname,lth,bADU,tADU,mNp,aduph,savdir,mask_file)
            else:
                events, times, counter, img, nframes, mask, trace = readdata.get_ccd_event_datan(sample_dir,prefd,sufd,nf1,nf2,darkdir,df1,df2,sname,lth,bADU,tADU,mNp,aduph,savdir,mask_file,20,fr)
                if engine == "GPU":
                    times = np.array(times,np.int32)
                    offsets = np.cumsum(counter, dtype=np.uint32)
                    offsets = np.insert(offsets,0,0)
                    max_nnz = counter.max()+1
                    from dynamix.correlator.event import FramesCompressor
                    shape = np.shape(img)
                    dtype = np.int8
                    F = FramesCompressor(shape, nframes, max_nnz)
        elif prefd.find("master") >= 0 :
            events, times, counter, img, nframes, mask, trace = h5reader.p10_eiger4m_event_GPU_datan(sample_dir+prefd+sufd,detector,nf1,nf2,mask,scan,phthr,fr,skip)
            np.savetxt(savdir+sname+"_trace.dat", trace)
            t0 = time.time()
            qqmask[mask>0] = 0
            print("Diagnostics max value %d" % events.max())
            print("Counter max value %d" % counter.max())
            events = np.array(events,np.int8)
            print("Events", events.shape,events.dtype,events.nbytes//1024**2)
            print("Times", times.shape,times.dtype,times.nbytes//1024**2)
            print("Counter", counter.shape,counter.dtype,counter.nbytes//1024**2)
            print("Preprocessing time %f" % (time.time()-t0))
            delta = 37.5#there is no automatic reading of delta for P10 files
            if detector=="eiger4m":
                ccx = int(1024+distance*np.tan(np.deg2rad(delta))/pix_size)
            sq = 4*np.pi/lambdaw*np.sin(np.deg2rad(delta/2))
            print("Delta=%2.2f, suggested cx=%d, central q=%1.3f 1/A" % (delta,ccx, sq))
            sufd = ".5"
            print("Number of frames %d" % nframes)        
            if engine == "GPU":
                times = np.array(times,np.int32)
                offsets = np.cumsum(counter, dtype=np.uint32)
                offsets = np.insert(offsets,0,0)
                max_nnz = counter.max()+1
                from dynamix.correlator.event import FramesCompressor
                shape = np.shape(img)
                dtype = np.int8
                F = FramesCompressor(shape, nframes, max_nnz)
        elif sufd == ".h5":
            delta = readdata.get_delta(sample_dir,prefd,sufd,nf1,nf2,scan)
            if detector=="mpx_si_22":
                 ccx = int(258+distance*np.tan(np.deg2rad(delta))/pix_size)
            if detector=="eiger4m":
                ccx = int(1024+distance*np.tan(np.deg2rad(delta))/pix_size)
            sq = 4*np.pi/lambdaw*np.sin(np.deg2rad(delta/2))
            print("Delta=%2.2f, suggested cx=%d, central q=%1.3f 1/A" % (delta,ccx, sq))
            if engine == "CPUold":
                pixels,s,img,nframes,mask = h5reader.id10_eiger4m_event_dataf(sample_dir+prefd+sufd,detector,nf1,nf2,mask,mNp,scan)
                trace = np.array(s,np.uint16)
                np.savetxt(savdir+sname+"_trace.dat", trace)
            if engine == "CPU":
                events, times, counter, img, nframes, mask, trace = h5reader.id10_eiger4m_event_GPU_datan(sample_dir+prefd+sufd,detector,nf1,nf2,mask,scan,phthr,fr,skip)
                np.savetxt(savdir+sname+"_trace.dat", trace)
                t0 = time.time()
                qqmask[mask>0] = 0
                print("Diagnostics max value %d" % events.max())
                print("Counter max value %d" % counter.max())
                events = np.array(events,np.int8)
                print("Events", events.shape,events.dtype,events.nbytes//1024**2)
                print("Times", times.shape,times.dtype,times.nbytes//1024**2)
                print("Counter", counter.shape,counter.dtype,counter.nbytes//1024**2)
                print("Preprocessing time %f" % (time.time()-t0))
            if engine == "GPU":
                events, times, counter, img, nframes, mask, trace = h5reader.id10_eiger4m_event_GPU_datan(sample_dir+prefd+sufd,detector,nf1,nf2,mask,scan,phthr,fr,skip)
                np.savetxt(savdir+sname+"_trace.dat", trace)
                t0 = time.time()
                qqmask[mask>0] = 0
                print("Diagnostics max value %d" % events.max())
                print("Counter max value %d" % counter.max())
                events = np.array(events,np.int8)
                print("Events", events.shape,events.dtype,events.nbytes//1024**2)
                print("Times", times.shape,times.dtype,times.nbytes//1024**2)
                print("Counter", counter.shape,counter.dtype,counter.nbytes//1024**2)
                print("Preprocessing time %f" % (time.time()-t0))
                times = np.array(times,np.int32)
                offsets = np.cumsum(counter, dtype=np.uint32)
                offsets = np.insert(offsets,0,0)
                max_nnz = counter.max()+1
                from dynamix.correlator.event import FramesCompressor
                shape = np.shape(img)
                dtype = np.int8
                F = FramesCompressor(shape, nframes, max_nnz)
            try:    
                del data
            except: pass    
        else:
            if engine == "CPUold":
                pixels, s, for_norm, img = readdata.get_eiger_event_data(sample_dir,prefd,sufd,nf1,nf2,sname,mNp,savdir,mask_file)
            else:
                events, times, counter, img, nframes, mask, trace = readdata.get_eiger_event_datan(sample_dir,prefd,sufd,nf1,nf2,sname,mNp,savdir,mask_file,phthr,fr)
                if engine == "GPU":
                    times = np.array(times,np.int32)
                    offsets = np.cumsum(counter, dtype=np.uint32)
                    offsets = np.insert(offsets,0,0)
                    max_nnz = counter.max()+1
                    from dynamix.correlator.event import FramesCompressor
                    shape = np.shape(img)
                    dtype = np.int8
                    F = FramesCompressor(shape, nframes, max_nnz)

        #### save summed image #####
        if os.path.isfile(savdir+sname+"_2D_raw.npz"):
            pass
        else:
            np.savez_compressed(savdir+sname+"_2D_raw.npz",data=np.array(img,np.float32))

        ### apply mask ###
        img[mask>0] = 0

        ### read and apply flat field ###
        if flatfield_file != 'none':
            try: 
                flatfield = EdfMethods.loadedf(flatfield_file)
                flatfield[flatfield<0] = 1
            except:
                print("Cannot read flat field %s skip" % flatfield_file)
                flatfield = np.ones(mask.shape,np.float32)
        else: flatfield = np.ones(mask.shape,np.float32)


        if os.path.isfile(savdir+sname+"_2D.npz"):
            pass
        else:
            np.savez_compressed(savdir+sname+"_2D.npz",data=np.array(img/flatfield,np.float32))

        ### qmask #############
        #qqmask = mask*0+1
        #qqmask[mask>0] = 0  
       
        ### make q ### 
        tools.make_q(config)

        
        ### read qmask ####
        qqmask = tools.read_qmask(qmask_file,mask, number_q)

        cdata = tools.make_cdata(savdir+sname+"_gaus.npy",config)

        ### correlate ##########
        if engine == "CPUold":
            CorrelationResult = ecorrts_q(pixels,s,qqmask,True,ttcf_par)
        if engine == "CPU":
            CorrelationResult = nbecorrts_q(events,times,counter,qqmask,nframes,True,ttcf_par) 
        if engine == "GPU":
            #CorrelationResult = gpu_ecorr(events, times, offsets, shape, nframes, dtype, qqmask, F)
            CorrelationResult = gpu_ecorr_q(events, times, offsets, shape, nframes, dtype, qqmask, max_nnz)
        ### format result #######
        res, save_cf, trc = tools.format_result(CorrelationResult,qqmask,flatfield,cdata,dt,ttcf_par)

        ### save cf ##############       
        qp = np.linspace(first_q,first_q+(number_q-1)*step_q,number_q) 
        tools.save_cf(savdir+sname+"_event_cf.dat",save_cf,qp)


        #### plot results #######
        try:
            plot_cf(res,sname+" Scan "+str(scan),savdir,toplot)
        except: pass

        print("Saving trc")
        try:
            if trc.size>1:
                readdata.savenpz(savdir+sname+"_event_trc.npz",np.array(trc,np.float32))
        except: pass

        try:
           print("Plotting trc, please wait")
           show_trc(trc,sname+" Scan "+str(scan),savdir,toplot)
        except: pass

    elif correlator == "intensity":
        ##### intensity #################################
        if engine == 'CPU':
            from dynamix.correlator.dense import MatMulCorrelator
        if engine == 'GPU':
            from dynamix.correlator.cuda import CublasMatMulCorrelator

        ##### read data ##########################################
        if sufd.find("edf") > -1:#== ".edf":
            data = readdata.get_data(sample_dir,prefd,sufd,nf1,nf2)
        elif sufd == ".h5":
            #data = h5reader.myreader(sample_dir+prefd+sufd,nf1,nf2)
            data = h5reader.myreader(sample_dir+prefd+sufd,detector,nf1,nf2,scan,skip)
            #data = np.array(data,np.uint16)
            
        else:
            exit()

        img = np.array(np.mean(data,0),np.float32)
        if os.path.isfile(savdir+sname+"_2D_raw.npz"):
            pass
        else:
            np.savez_compressed(savdir+sname+"_2D_raw.npz",data=img)


        ### read detector mask ###
        mask = EdfMethods.loadedf(mask_file)

        ### read beamstop mask ####
        bmask = tools.read_beamstop_mask(beamstop_mask)

        mask[bmask>0] = 1
        
        img[mask>0] = 0
                
        if flatfield_file != 'none':
            try: 
                flatfield = EdfMethods.loadedf(flatfield_file)
                flatfield[flatfield<0] = 1
            except:
                print("Cannot read flat field %s skip" % flatfield_file)
                flatfield = np.ones(mask.shape,np.float32)
        else: flatfield = np.ones(mask.shape,np.float32)
       
        if os.path.isfile(savdir+sname+"_2D.npz"):
            pass
        else:
            np.savez_compressed(savdir+sname+"_2D.npz",data=np.array(img/flatfield,np.float32))


        print("Original data size is %2.2f Gigabytes" % (data.size*data.itemsize/1024**3))
        print("Original data type", data.dtype,data.max())

        ### make smooth data ####
        cdata = tools.make_cdata(savdir+sname+"_gaus.npy",config)

        ### read qmask ####
        qqmask = tools.read_qmask(qmask_file,mask, number_q)


        #### reduce the matrix size to speed up calculations ####
        data,qqmask,cdata,flatfield = tools.reduce_matrix(data,qqmask,cdata,flatfield)
     
        shape = np.shape(data[0,:,:])
        nframes = np.shape(data)[0]
        print("Number of frames %d" % nframes)
        print("Number of qs %d" % number_q)

        #### calculate trace ####
        trace = np.zeros((nframes,number_q),np.float32)
        n = 0
        for q in range(1,number_q+1,1):
            trace[:,n] =  np.sum(data[:,qqmask==q],1)
            n +=1 
            
        ### save trace  ####
        qp = np.linspace(first_q,first_q+(number_q-1)*step_q,number_q) 
        q_title='#q values 1/A:'
        for q in qp:
            q_title = q_title+" %.5f " % q
        q_title=q_title+'\n'
        f=open(savdir+sname+"_trace.dat",'w')
        f.write(q_title)
        np.savetxt(f, trace)
        f.close()

        ### correlator #######
        t0 = time.time()

        #### MatMul Correlator #####################################
        if engine == "CPU":
            correlator = MatMulCorrelator(shape,nframes,qmask=qqmask)
        if engine == "GPU":
            correlator = CublasMatMulCorrelator(shape, nframes, qmask=qqmask)

        t1 = time.time()
        #result = correlator.correlate(data,calc_std=True)
        CorrelationResult = correlator.correlate(data,True,ttcf_par)
        print("Correlator time %3.2f seconds" % (time.time()-t1))
 
        ### format result #######
        res, save_cf, trc = tools.format_result(CorrelationResult,qqmask,flatfield,cdata,dt,ttcf_par)

        ### save cf ##############   
        qp = np.linspace(first_q,first_q+(number_q-1)*step_q,number_q)     
        tools.save_cf(savdir+sname+"_cf.dat",save_cf,qp)


        #### plot results #######
        try:
            plot_cf(res,sname+" Scan "+str(scan),savdir,toplot)
        except: pass

        print("Saving trc")
        try:
            if trc.size>1:
                readdata.savenpz(savdir+sname+"_trc.npz",np.array(trc,np.float32))
        except: pass

        try:
            print("Plotting trc, please wait")
            show_trc(trc,sname+" Scan "+str(scan),savdir,toplot)
        except: pass
    else: pass
    sys.exit()

if __name__ == "__main__":
 
    main()
