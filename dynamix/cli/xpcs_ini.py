#! /usr/bin/env python3
#xpcs script to start analysis of the data
#reads the ini configuration file
import shutil
import sys
import os
import numpy as np
import time
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
    except:
        print("Cannot read "+sys.argv[1]+" input file")
        exit()

    #### Sample description #######################################################################

    sname = config["sample_description"]["name"]
    scan = config["sample_description"]["scan"]
    scan = str(int(scan.split("scan")[1]))
    #### Data location ############################################################################

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
    toplot = config["data_location"]["toplot"]

    #### Experimental setup #######################################################################

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
    
    
    #### check if the saving direftory exist and create one #####
    tools.test_dir(savdir)
    
    #### Copy config file to the result directory #####
    try:
        shutil.copy(sys.argv[1], savdir+"input_xpcs_"+sname+".txt")
    except:pass
    ###############################################################################################
    from dynamix.plot.draw_result import plot_cf, show_trc
    import pylab as plt
    print("Start analysis of the sample %s" % sname)
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
        if detector == "Andor":
            #pixels - list of pixels that see a photon
            #s - sum of all photons in a frame
            #for_norm - number of pixels used
            #img - summed 2D scattering pattern
            print("Detector Andor")
            #get the delta value from the tenth image to adjust cx position and first_q
            delta = readdata.get_delta(sample_dir,prefd,sufd,nf1,nf2)
            cx = int(cx-distance*np.tan(np.deg2rad(delta))/pix_size)
            first_q = 4*np.pi/lambdaw*np.sin(np.deg2rad(delta)/2)
            print("Automatic ajusting cx=%d and first_q=%3.3f" % (cx,first_q)) 
            if engine == "CPUold": 
                pixels, s, for_norm, img =      readdata.get_ccd_event_data(sample_dir,prefd,sufd,nf1,nf2,darkdir,df1,df2,sname,lth,bADU,tADU,mNp,aduph,savdir,mask_file)
            else:
                events, times, counter, img, nframes, mask, trace = readdata.get_ccd_event_datan(sample_dir,prefd,sufd,nf1,nf2,darkdir,df1,df2,sname,lth,bADU,tADU,mNp,aduph,savdir,mask_file,20,0.33)
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
            #pixels,s,img,nframes,mask = h5reader.p10_eiger_event_data(sample_dir+prefd+sufd,nf1,nf2,mask)
            pixels,s,img,nframes,mask = h5reader.p10_eiger_event_dataf(sample_dir+prefd+sufd,nf1,nf2,mask,mNp)
            trace = np.array(s,np.uint16)
            np.savetxt(savdir+sname+"_trace.dat", trace)
            sufd = ".5"
            print("Number of frames %d" % nframes)    
        elif sufd == ".h5":
            delta = readdata.get_delta(sample_dir,prefd,sufd,nf1,nf2,scan)
            ccx = int(1024+distance*np.tan(np.deg2rad(delta))/pix_size)
            sq = 4*np.pi/lambdaw*np.sin(np.deg2rad(delta/2))
            print("Delta=%2.2f, suggested cx=%d, central q=%1.3f 1/A" % (delta,ccx, sq))
            if engine == "CPUold":
                pixels,s,img,nframes,mask = h5reader.id10_eiger4m_event_dataf(sample_dir+prefd+sufd,nf1,nf2,mask,mNp,scan)
                trace = np.array(s,np.uint16)
                np.savetxt(savdir+sname+"_trace.dat", trace)
            #if engine == "CCPU":
            #    mNp = trace.max()+10
            #    print("Diagnostics maximum photons per frame %d" % (mNp-10))
            #    pixels, s = tools.events(data, mNp) 
            if engine == "CPU":
                events, times, counter, img, nframes, mask, trace = h5reader.id10_eiger4m_event_GPU_datan(sample_dir+prefd+sufd,nf1,nf2,mask,scan,20,0.33)
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
                events, times, counter, img, nframes, mask, trace = h5reader.id10_eiger4m_event_GPU_datan(sample_dir+prefd+sufd,nf1,nf2,mask,scan,20,0.33)
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
                events, times, counter, img, nframes, mask, trace = readdata.get_eiger_event_datan(sample_dir,prefd,sufd,nf1,nf2,sname,mNp,savdir,mask_file,20,0.33)
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
        if toplot =="yes":
            try:
                plot_cf(res,sname)
            except: pass

        print("Saving trc")
        try:
            if trc.size>1:
                readdata.savenpz(savdir+sname+"_event_trc.npz",np.array(trc,np.float32))
        except: pass

        if toplot =="yes":
            try:
                print("Ploting trc, please wait")
                show_trc(trc,sname,savdir)
            except: pass

    elif correlator == "intensity":
        ##### intensity #################################
        from dynamix.correlator.dense import MatMulCorrelator
        from dynamix.correlator.cuda import CublasMatMulCorrelator

        ##### read data ##########################################
        if sufd.find("edf") > -1:#== ".edf":
            data = readdata.get_data(sample_dir,prefd,sufd,nf1,nf2)#[:3000,:,:]
        elif sufd == ".h5":
            data = h5reader.myreader(sample_dir+prefd+sufd)[nf1:nf2,:,:]
            data = np.array(data,np.uint8)
            
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
        ### correlator #######
        t0 = time.time()

        #### MatMul Correlator #####################################
        if engine == "CPU":
            correlator = MatMulCorrelator(shape,nframes,qmask=qqmask)
        if engine == "GPU":
            correlator = CublasMatMulCorrelator(shape, nframes, qmask=qqmask)

        t1 = time.time()
        result = correlator.correlate(data)
        print("Correlator time %3.2f seconds" % (time.time()-t1))

        trace = np.zeros((nframes,number_q),np.float32)
        res = []
        n = 0
        cf = np.zeros((nframes-1,3),np.float32)
        cf[:,0] = np.arange(1,nframes,1)*dt
        for q in range(1,number_q+1,1):
            trace[:,n] =  np.sum(data[:,qqmask==q],1)
            fcorrection = (flatfield[qqmask==q]**2).mean()/flatfield[qqmask==q].mean()**2
            correction = (cdata[qqmask==q]**2).mean()/cdata[qqmask==q].mean()**2 * fcorrection
            if engine == "GPU":
                cf[:,1] = result[n][1:]/correction#GPU
                cf[:,2] = result[n][1:]/correction*1e-4#GPU
            if engine == "CPU":
                cf = result[n]#CPU
                cf[:,0] = cf[:,0]*dt#CPU
                cf[:,1] /= correction # make correct baseline#CPU
                cf[:,2] /= correction # make correct baseline#CPU
            cfm = tools.cftomt(cf)
            res.append(cfm)
            if q == 1:
                save_cf = cfm
            else:
                save_cf = np.append(save_cf,cfm[:,1:],axis=1)
            n += 1

        print("Correlation time %3.2f seconds" % (time.time()-t0))

        ### save cf ##############       
        qp = np.linspace(first_q,first_q+(number_q-1)*step_q,number_q) 
        tools.save_cf(savdir+sname+"_event_cf.dat",save_cf,qp)
        
        ### save trace  ####
        q_title='#q values 1/A:'
        for q in qp:
            q_title = q_title+" %.5f " % q
        q_title=q_title+'\n'
        f=open(savdir+sname+"_trace.dat",'w')
        f.write(q_title)
        np.savetxt(f, trace)
        f.close()

        #np.savetxt(savdir+sname+"_trace.dat",trace)
        #np.savetxt(savdir+sname+"_cf.dat",save_cf)
        #readdata.savenpz(savdir+sname+"_trc.npz",np.array(trc/correction,np.float32))

        #### plot results #######
        if toplot =="yes":
            try:
                plot_cf(res,sname)
            except: pass
            try:
                show_trc(trc,sname,savdir)
            except: pass


    else: pass
    exit()


    # ############ plotting ############################
    # from dynamix.plot.draw_result import plot_cf, show_trc
    #
    # plot_cf(w,sname)
    # show_trc(t,sname)



if __name__ == "__main__":
    main()
