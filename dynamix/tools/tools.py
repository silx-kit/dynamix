#! /usr/bin/env python3
#tools
import sys

import  numpy as np
import time
import os
import pylab as plt
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
    qh = np.int16(q+0.5)#better match with data
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
    f1 = q-np.array(q,np.uint16)
    ind = np.array(q,np.uint16)-int(radius[0])
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

