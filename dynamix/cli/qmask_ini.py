#! /usr/bin/env python3
#make q mask
#reads the ini configuration file
import sys
import os
import numpy as np
import time
import fabio
from dynamix.io import readdata, EdfMethods, h5reader
from dynamix.tools import tools
import configparser
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())

def main():

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
    import pylab as plt
    print("Making qs")
    try:
        data = readdata.readnpz(savdir+sname+"_2D.npz")
    except:
        print("Cannot read "+savdir+sname+"_2D.npz")
        exit()

    if  beamstop_mask != 'none':
        try:
            bmask = np.abs(EdfMethods.loadedf(beamstop_mask))#reads edf and npy
            bmask[bmask>1] = 1
        except:
            print("Cannot read beamstop mask %s, skip" % beamstop_mask)

    if mask_file != 'none':
        try:
            mask = np.abs(EdfMethods.loadedf(mask_file))#reads edf and npy
            mask[mask>1] = 1
            try: 
                mask[bmask>0] = 1
            except: pass 
            data[mask>0] = 0
        except:
            print("Cannot read mask %s, exit" % mask_file)
            exit() 


    ind = np.where((data>0)&(mask<1))

    data = np.ma.array(data,mask=mask)
    qmask = mask*0
    ny,nx = data.shape

    t0 = time.time()
    rad, r_q, new_saxs = tools.radi(data,mask,cx,cy)#radial averaging
    print("Calculation time %3.4f sec" % (time.time()-t0))

    np.save(savdir+sname+"_gaus.npy",np.array(new_saxs,np.float32))


    rad[:,0] = 4*np.pi/lambdaw*np.sin(np.arctan(pix_size*rad[:,0]/distance)/2.0)#q vector calculation
    r_q = 4*np.pi/lambdaw*np.sin(np.arctan(pix_size*r_q/distance)/2.0)#q vector calculation
    width_p = np.tan(np.arcsin(width_q*lambdaw/4/np.pi)*2)*distance/pix_size
    qmask = np.array((r_q-first_q+width_q/2)/width_q+1,np.uint16)

    print("Number of Qs %d" % number_q)
    print("Width of ROI is %1.1f pixels" % width_p)
    np.savetxt(savdir+sname+"_1D.dat",rad)

    #qmask = np.array((r_q-first_q+width_q/2)/width_q+1,np.uint16)
    #qmask[mask>0] = 0
    #np.save(savdir+sname+"_qmask.npy",np.array(qmask,np.uint16))
    #qmask[qmask>number_q] = 0


    #qp = np.linspace(first_q,first_q+(number_q-1)*width_q,number_q)
    qp = np.linspace(first_q,first_q+(number_q-1)*step_q,number_q)
    print("Q values = ", qp)
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
      print("Number of pixels per q %d %d" % (n,qmask[indq].size))
    qmask[mask>0] = 0
    np.save(savdir+sname+"_qmask.npy",np.array(qmask,np.uint16))
    qmask[qmask>number_q] = 0


    qmask = np.ma.array(qmask)
    qmask[qmask==0]=np.ma.masked

    new_saxs = np.ma.array(new_saxs,mask=mask)
    err = np.abs(data-new_saxs)[ind].mean()/np.mean(data[ind])
    print("SAXS and radi modulus error %f" % err)
    print("Total calculation time %3.4f sec" % (time.time()-t0))

    plt.figure()
    plt.imshow(np.log10(new_saxs),cmap='jet')
    plt.title("Radi")

    plt.figure()
    plt.imshow(qmask.data,cmap='gray_r')
    if (np.abs(cx)<=nx) and (np.abs(cy)<=ny):
        plt.plot(cx,cy,'r+')
    plt.grid()
    plt.xlabel("Pixel x")
    plt.ylabel("Pixel y")
    plt.title("Q mask")

    plt.figure()
    plt.imshow(np.log10(data),cmap='jet')
    plt.imshow(qmask,cmap='gray_r',alpha=0.5)
    if (np.abs(cx)<=nx) and (np.abs(cy)<=ny):
        plt.plot(cx,cy,'r+')
    plt.grid()
    plt.xlabel("Pixel x")
    plt.ylabel("Pixel y")
    plt.title("Q mask")

    plt.figure()
    plt.loglog(rad[:,0],rad[:,1],'-b')
    plt.loglog(qp,i_qp,'or',mfc='none')
    for n in range(number_q):
        if (n+1)//2 == (n+1)/2:
            plt.fill_between(rad[i_bands[n],0],rad[i_bands[n],1],facecolor='lime',alpha=0.5)
        else:
            plt.fill_between(rad[i_bands[n],0],rad[i_bands[n],1],facecolor='tomato',alpha=0.5)
    plt.xlabel(r"Q ($\AA^{-1}$)")
    plt.ylabel("I (counts)")
    plt.title(sname)
    plt.show()
    exit()




if __name__ == "__main__":
    main()
