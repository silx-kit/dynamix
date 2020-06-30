#! /usr/bin/env python3
#wxpcs code that works with ini file
import sys
#sys.path.append("/data/id10/inhouse/Programs/PyXPCS_project/wxpcs")
#sys.path.append("/users/chushkin/Documents/Analysis/Glass_school_2019/wxpcs")
#sys.path.append("/users/chushkin/Documents/Programs/PyXPCS_project/wxpcs")

import  numpy as np
from .WXPCS import fecorrt
from time import time

########### Event_correlator standard ################
def ecorrts(pixels,s,for_norm):
    print("start calculation of the correlation functions")
    timee = time()
    lpixels = len(pixels)
    rpixels = range(lpixels)
    t=[]
    for t1 in rpixels:
      t += [t1]*s[t1]
    print("time for pre loop "+str(time()-timee))
    pix = np.concatenate(pixels).ravel()
    print('start sorting')
    times = time()
    pix = np.array(pix,dtype=np.int32) 
    t = np.array(t)
    indpi = np.lexsort((t,pix))
    t = t[indpi]
    pix = pix[indpi] 
    print('sorting took '+ str(time()-times))
    print('start main loop')
    timem = time()
    lenpi = len(pix)
    cor = np.zeros((lpixels,lpixels),dtype=np.uint32)
    timef = time()
    cor = fecorrt(pix,t,cor,lenpi,lpixels)#fortran module    
    # to split for multicores
    #ind = np.where(pix<133128)[0][-1] 
    #cor = fecorrt(pix[:ind],t[:ind],cor,len(pix[:ind]),lpixels)#fortran module    
    #cor1 = np.zeros((lpixels,lpixels),dtype=np.uint32)
    #cor1 = fecorrt(pix[ind:],t[ind:],cor1,len(pix[ind:]),lpixels)#fortran module    
    #cor = cor+cor1
    print("time for correlating "+str(time()-timem))
    print("average photons per frame "+str(np.mean(s)))
    #print(cor.min(),cor.max())
    lens = len(s)
    s = np.array(s,dtype=np.float32)
    cor = np.array(cor,dtype=np.float32)
    s.shape = lens,1
    #norm = dot(s,flipud(s.T))/lpixels
    norm = np.dot(s,s.T)/lpixels
    cor = cor*for_norm/lpixels
    x = np.ones((lpixels-1,3))
    x[:,0] = np.arange(1,lpixels)
    for i in range(1,lpixels):
      dia = np.diag(cor,k=i)
      sdia = np.diag(norm,k=i)
      ind = np.where(np.isfinite(dia))
      x[i-1,1] = np.mean(dia[ind])/np.mean(sdia[ind])
      x[i-1,2] = np.std(dia[ind]/sdia[ind])/len(sdia[ind])**0.5
    wcf = x+0
    mtcf = cftomt_testing(x)
    cor /= norm
    print("Total time for correlating "+str(time()-timee))
    return wcf,mtcf,cor
##############################################################################

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
