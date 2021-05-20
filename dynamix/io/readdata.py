#! /usr/bin/env python3
#readdata
import sys

import  numpy as np
import numba as nb
from dynamix.io import nfiles
import _pickle as cPickle
from dynamix.correlator.WXPCS import dropimgood, eigerpix
from multiprocessing import Process, Queue
from dynamix.io import EdfMethods
from sys import argv,exit,stdout
import time
import h5py    # HDF5 support
import hdf5plugin

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

####### Save pixels and intensities ######
def events_load(fname):
    edata = open(fname,'rb')
    pixels,s = cPickle.load(edata)
    edata.close()
    return pixels,s

def events_save(fname,pixels,s):
    edata = open(fname,'wb')
    cPickle.dump([pixels,s],edata,-1)
    edata.close()
    return

######## Read npz #####
def readnpz(FileName):
    f = np.load(FileName)
    #data = f['data']
    data = f[f.files[0]]
    f.close()
    return data

######## save npz #####
def savenpz(FileName,data):
    np.savez_compressed(FileName,data=data)
    return 

######### processing the file for Eiger,Maxipix etc ##############
def mread_eiger(mfilename,mdataout,ind_g,mNp,nx):
    for mfile in mfilename:
        matr = EdfMethods.loadedf(mfile)
        msumpix,mpix = eigerpix(matr[ind_g],mNp,nx) 
        mpix = mpix[:msumpix]
        mdataout.put([mpix,msumpix,matr])
    mdataout.close()
###########################################


######### processing the file for CCD ##############
def mread_ccd(mfilename,mdataout,darkimg,lth,bADU,tADU,mNp,aduph,nx,ny):
    for mfile in mfilename:
        matr = np.asfarray(EdfMethods.loadedf(mfile),dtype=np.float32)
        try:
            matr[ind]=0
        except: 
            pass
        msumpix,mpix,tmp = dropimgood(matr,darkimg,lth,bADU,tADU,mNp,aduph,nx,ny)#dropletize CCD frames
        mpix = mpix[:msumpix]
        mdataout.put([mpix,msumpix,tmp])
    mdataout.close()
###########################################

def get_data(datdir,prefd,sufd,nf1,nf2):
    # read Maxipix, Eiger edf images ####
    t0 = time.time()

    swrite = stdout.write
    sflush = stdout.flush
    print("start reading the files") 
    #creating filenames
    filenames = nfiles.filename(datdir+prefd,sufd,nf1,nf2)
    lfilenames = len(filenames)
    #reading first image to get dimenstions of the matrix
    headers = EdfMethods.headeredf(filenames[0])
    dim1 = np.intc(headers['Dim_1'])
    dim2 = np.intc(headers['Dim_2'])
    nx = dim2
    ny = dim1
    data = np.zeros((lfilenames,nx,ny),np.uint16)
    i = 0
    for mfile in filenames:
        data[i,:,:] = EdfMethods.loadedf(mfile)
        i += 1
        swrite(4*'\x08')
        swrite(str(int(i*100./lfilenames))+'%')
        sflush()
    print("\n")
    print("Reading time %3.3f sec" % (time.time()-t0)) 
    return data

def get_eiger_event_data(datdir,prefd,sufd,nf1,nf2,sname,mNp,savdir,mask_file):
    ### read maxipix, Eiger edf images and convert for event correlator ####
    time0 = time.time()

    swrite = stdout.write
    sflush = stdout.flush
    print("start reading the files")      
    #creating filenames
    filenames = nfiles.filename(datdir+prefd,sufd,nf1,nf2)
    lfilenames = len(filenames)#-1
    #reading first image to get dimenstions of the matrix
    headers = EdfMethods.headeredf(filenames[0])
    dim1 = np.intc(headers['Dim_1'])
    dim2 = np.intc(headers['Dim_2'])
    nx = dim2
    ny = dim1

    ############reading mask##########
    try:
        mask_data = EdfMethods.loadedf(mask_file)
        print("use mask file "+mask_file)
        ind = np.where(mask_data>0)
        ind_g = np.where(mask_data<1)
        for_norm = dim1*dim2 - len(mask_data[ind])
    except:
        print("no mask applied")
        for_norm = dim1*dim2#1024**2
        pass
    print("Numebr of pixels used "+str(for_norm))

    ########creating image matrix of zeros############  
    img = np.zeros((dim2,dim1))
    pixels = []
    s = []
    data = []
    pread = []
    nx = len(img[ind_g])
    for i in range(2):
        data.append(Queue(2))
        #pread.append(Process(target=mread_eiger, args=(filenames[i:-1:2],data[i],ind_g,mNp,nx)))
    pread.append(Process(target=mread_eiger, args=(filenames[:-1:2],data[0],ind_g,mNp,nx)))
    pread.append(Process(target=mread_eiger, args=(filenames[1::2],data[1],ind_g,mNp,nx)))
    for i in range(2):
        pread[i].start()

    ii=0
    ###########reading and summing files ###########
    for i in range(lfilenames):
        swrite(4*'\x08')
        swrite(str(int(i*100./lfilenames))+'%')
        sflush()
        fromproc = data[ii].get()
        pixels.append(fromproc[0])
        s.append(fromproc[1])
        img += fromproc[2]
        ii+=1
        if ii==2: 
            ii = 0 
 
    for i in range(2):
        pread[i].join()
        data[i].close()

    dtime = time.time()- time0
    print('reading of %d files took %5.2f sec' %(lfilenames,dtime))

    if not os.path.exists(savdir):
        answ = raw_input("create a director (y)/n")
        if answ == "n":
           print("exit")
           exit()
        else:
           os.makedirs(savdir)
           print("directory "+savdir+" has been created")
    return pixels,s,for_norm, img



def get_ccd_event_data(datdir,prefd,sufd,nf1,nf2,darkdir,df1,df2,sname,lth,bADU,tADU,mNp,aduph,savdir,mask_file):
    ### read ccd edf images and convert for event correlator ####
    time0 = time.time()

    swrite = stdout.write
    sflush = stdout.flush
    print("start reading the files")      
    #creating filenames
    filenames = nfiles.filename(datdir+prefd,sufd,nf1,nf2)
    lfilenames = len(filenames)#-1
    #reading first image to get dimenstions of the matrix
    headers = EdfMethods.headeredf(filenames[0])
    dim1 = np.intc(headers['Dim_1'])
    dim2 = np.intc(headers['Dim_2'])
    nx = dim2
    ny = dim1

    ############reading mask##########
    try:
        mask_data = EdfMethods.loadedf(mask_file)
        print("use mask file "+mask_file)
        ind = np.where(mask_data>0)
        for_norm = dim1*dim2 - len(mask_data[ind])
    except:
        print("no mask applied")
        for_norm = dim1*dim2#1024**2
        pass
    print("Numebr of pixels used "+str(for_norm))

    ########creating image matrix of zeros############  
    img = np.zeros((dim2,dim1))
    
    ########reading dark##########
    darkfilenames = nfiles.filename(darkdir+prefd,sufd,df1,df2)
    ndarks = 0  
    for dfile in darkfilenames:
        try:
            darkimg += np.asfarray(EdfMethods.loadedf(dfile),dtype=np.float32)
            ndarks += 1 
        except: 
            darkimg = np.asfarray(EdfMethods.loadedf(dfile),dtype=np.float32)
            ndarks += 1 
            
    darkimg = darkimg/ndarks
    
    #try :
    #    darkimg = np.asfarray(EdfMethods.loadedf(savdir+'dark_'+sname+'.edf'),dtype=np.float32)#*0
    #except:
        #print "make dark"
        #os.system('/data/id10/inhouse/Programs/wxpcs/darkedft.py '+argv[1])
        #darkimg = asfarray(loadedf(savdir+'dark_'+sname+'.edf'),dtype=float32)#*0
        #print("read default dark")
    #    darkimg = np.asfarray(EdfMethods.loadedf(savdir+'dark_Pd_glass.edf'),dtype=np.float32)#*0

    pixels = []
    s = []
    data = []
    pread = []
    for i in range(2):
        data.append(Queue(2))
        #pread.append(Process(target=mread_ccd, args=(filenames[i:-1:2],data[i],darkimg,lth,bADU,tADU,mNp,aduph,nx,ny)))
    pread.append(Process(target=mread_ccd, args=(filenames[:-1:2],data[0],darkimg,lth,bADU,tADU,mNp,aduph,nx,ny)))
    pread.append(Process(target=mread_ccd, args=(filenames[1::2],data[1],darkimg,lth,bADU,tADU,mNp,aduph,nx,ny)))
    for i in range(2):
        pread[i].start()

    ii=0
    ###########reading and summing files###########
    for i in range(lfilenames):
        swrite(4*'\x08')
        swrite(str(int(i*100./lfilenames))+'%')
        sflush()
        fromproc = data[ii].get()
        pixels.append(fromproc[0])
        s.append(fromproc[1])
        img += fromproc[2]
        ii+=1
        if ii==2: 
            ii = 0 
     
    for i in range(2):
        pread[i].join()
        data[i].close()

    dtime = time.time()- time0
    print('reading of %d files took %5.2f sec' %(lfilenames,dtime))

    if not os.path.exists(savdir):
        answ = raw_input("create a director (y)/n")
        if answ == "n":
           print("exit")
           exit()
        else:
           os.makedirs(savdir)
           print("directory "+savdir+" has been created")
    return pixels,s,for_norm, img

def get_delta(datdir,prefd,sufd,nf1,nf2,scan="1"):    
    if sufd ==".edf":
        filenames = nfiles.filename(datdir+prefd,sufd,nf1,nf2)
        h = EdfMethods.headeredf(filenames[10])
        delta = float(h['motor_pos'].split(" ")[h['motor_mne'].split(" ").index('del')])
    if sufd == ".h5":
        filename = datdir+prefd+sufd
        with h5py.File(filename, mode="r") as h5:
            delta = h5['/'+scan+'.1/instrument/positioners/delta'][()]
    return delta
    
def get_eiger_event_datan(datdir,prefd,sufd,nf1,nf2,sname,mNp,savdir,mask_file,thr=20,frc=0.15):
    ### read ccd edf images and convert for Pierre's event correlator using numba ####
    t0 = time.time()

    swrite = stdout.write
    sflush = stdout.flush
    print("start reading the files")      
    #creating filenames
    filenames = nfiles.filename(datdir+prefd,sufd,nf1,nf2)
    lfilenames = len(filenames)#-1
    #reading first image to get dimenstions of the matrix
    headers = EdfMethods.headeredf(filenames[0])
    dim1 = np.intc(headers['Dim_1'])
    dim2 = np.intc(headers['Dim_2'])
    nx = dim2
    ny = dim1

    ############reading mask##########
    try:
        mask_data = EdfMethods.loadedf(mask_file)
        print("use mask file "+mask_file)
        ind = np.where(mask_data>0)
        for_norm = dim1*dim2 - len(mask_data[ind])
    except:
        mask_data = np.zeros((dim2,dim1),np.uint8) 
        print("no mask applied")
        for_norm = dim1*dim2#1024**2
        pass
    print("Numebr of pixels used "+str(for_norm))

       
    n_frames = len(filenames)
    ll = nx*ny # total number of pixels
    lp = int(n_frames*frc) # total number of frames with events 15%
    mask = np.array(np.ravel(mask_data),np.uint8)
    evs = np.zeros((ll,lp),np.uint8)
    tms = np.zeros((ll,lp),np.uint16)
    cnt = np.ravel(np.zeros((ll,),np.uint16))
    afr = np.ravel(np.zeros((ll,),np.uint32))
    tr = 0
    trace = np.zeros((n_frames,),np.uint32)
    it = 0 
    print("Number of frames %d" % n_frames)
    ###########reading and summing files###########
    for i in range(lfilenames):
        swrite(4*'\x08')
        swrite(str(int(i*100./lfilenames))+'%')
        sflush()
        matr = EdfMethods.loadedf(filenames[i])
        try:
            matr[ind]=0
        except: 
            pass
        fr = np.ravel(matr)
        evs,tms,cnt,afr,mask,tr = neigercompress(evs,tms,cnt,afr,mask,tr,fr,thr,it,ll,lp)
        trace[i] = tr 
        it += 1
 
    if not os.path.exists(savdir):
        answ = raw_input("create a director (y)/n")
        if answ == "n":
           print("exit")
           exit()
        else:
           os.makedirs(savdir)
           print("directory "+savdir+" has been created")
    afr = afr/n_frames
    afr = np.reshape(afr,(nx,ny))
    mask = np.reshape(mask,(nx,ny))
    evs,tms,c = nprepare(np.ravel(evs),np.ravel(tms))
    evs = np.array(evs[:c],np.int8)
    tms = tms[:c]
    print("Reading time %3.3f sec" % (time.time()-t0))
    return evs,tms,cnt,afr,n_frames,mask,trace

def get_ccd_event_datan(datdir,prefd,sufd,nf1,nf2,darkdir,df1,df2,sname,lth,bADU,tADU,mNp,aduph,savdir,mask_file,thr=20,frc=0.15):
    ### read ccd edf images and convert for Pierre's event correlator using numba ####
    t0 = time.time()

    swrite = stdout.write
    sflush = stdout.flush
    print("start reading the files")      
    #creating filenames
    filenames = nfiles.filename(datdir+prefd,sufd,nf1,nf2)
    lfilenames = len(filenames)#-1
    #reading first image to get dimenstions of the matrix
    headers = EdfMethods.headeredf(filenames[0])
    dim1 = np.intc(headers['Dim_1'])
    dim2 = np.intc(headers['Dim_2'])
    nx = dim2
    ny = dim1

    ############reading mask##########
    try:
        mask_data = EdfMethods.loadedf(mask_file)
        print("use mask file "+mask_file)
        ind = np.where(mask_data>0)
        for_norm = dim1*dim2 - len(mask_data[ind])
    except:
        mask_data = np.zeros((dim2,dim1),np.uint8) 
        print("no mask applied")
        for_norm = dim1*dim2#1024**2
        pass
    print("Numebr of pixels used "+str(for_norm))

   
    ########reading dark##########
    darkfilenames = nfiles.filename(darkdir+prefd,sufd,df1,df2)
    ndarks = 0  
    for dfile in darkfilenames:
        try:
            darkimg += np.asfarray(EdfMethods.loadedf(dfile),dtype=np.float32)
            ndarks += 1 
        except: 
            darkimg = np.asfarray(EdfMethods.loadedf(dfile),dtype=np.float32)
            ndarks += 1 
            
    darkimg = darkimg/ndarks
    
    n_frames = len(filenames)
    ll = nx*ny # total number of pixels
    lp = int(n_frames*frc) # total number of frames with events 15%
    mask = np.array(np.ravel(mask_data),np.uint8)
    evs = np.zeros((ll,lp),np.uint8)
    tms = np.zeros((ll,lp),np.uint16)
    cnt = np.ravel(np.zeros((ll,),np.uint16))
    afr = np.ravel(np.zeros((ll,),np.uint32))
    tr = 0
    trace = np.zeros((n_frames,),np.uint32)
    it = 0 
    print("Number of frames %d" % n_frames)
    ###########reading and summing files###########
    for i in range(lfilenames):
        swrite(4*'\x08')
        swrite(str(int(i*100./lfilenames))+'%')
        sflush()
        matr = np.asfarray(EdfMethods.loadedf(filenames[i]),dtype=np.float32)
        try:
            matr[ind]=0
        except: 
            pass
        msumpix,mpix,fr = dropimgood(matr,darkimg,lth,bADU,tADU,mNp,aduph,nx,ny)#dropletize CCD frames
        fr = np.ravel(fr)
        evs,tms,cnt,afr,mask,tr = neigercompress(evs,tms,cnt,afr,mask,tr,fr,thr,it,ll,lp)
        trace[i] = tr 
        it += 1
 
    if not os.path.exists(savdir):
        answ = raw_input("create a director (y)/n")
        if answ == "n":
           print("exit")
           exit()
        else:
           os.makedirs(savdir)
           print("directory "+savdir+" has been created")
    afr = afr/n_frames
    afr = np.reshape(afr,(nx,ny))
    mask = np.reshape(mask,(nx,ny))
    evs,tms,c = nprepare(np.ravel(evs),np.ravel(tms))
    evs = np.array(evs[:c],np.int8)
    tms = tms[:c]
    print("Reading time %3.3f sec" % (time.time()-t0))
    return evs,tms,cnt,afr,n_frames,mask,trace


@nb.jit(nopython=True, parallel=True, fastmath=True)
def neigercompress(evs,tms,cnt,afr,m,tr,fr,thr,i,ll,max_e):
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

@nb.jit(nopython=True, fastmath=True)
def nprepare(evs,tms):
    ll = evs.size
    i = 0 
    for p in range(ll):
        if evs[p]>0:
            evs[i] = evs[p]
            tms[i] = tms[p]
            i += 1
    return evs,tms,i
