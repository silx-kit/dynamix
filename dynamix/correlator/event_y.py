#! /usr/bin/env python3
# wxpcs code that works with ini file
import sys
# sys.path.append("/data/id10/inhouse/Programs/PyXPCS_project/wxpcs")
# sys.path.append("/users/chushkin/Documents/Analysis/Glass_school_2019/wxpcs")
# sys.path.append("/users/chushkin/Documents/Programs/PyXPCS_project/wxpcs")

import  numpy as np
import numba as nb
from .WXPCS import fecorrt
import time
from collections import namedtuple

import os
import psutil
nproc = psutil.cpu_count(logical=True)  # len(os.sched_getaffinity(0)) * psutil.cpu_count(logical=False) // psutil.cpu_count(logical=True)

CorrelationResult = namedtuple("CorrelationResult", "res dev trc")


########## Numba implementation ##################################
@nb.jit(nopython=True, parallel=True, fastmath=True)
def ncorrelate(evs, tms, cnt, q, n_frames):
    qm = q.max()
    cc = np.zeros((qm, n_frames, n_frames), np.float32)
    mint = np.zeros((qm, n_frames), np.float32)
    ll = cnt.size
    k = 0
    for i in range(ll):
        qp = q[i] - 1
        if qp >= 0:
            for j in range(cnt[i]):
                t0 = tms[k]
                n = k + 1
                mint[qp, t0] += evs[k]
                for f in range(j + 1, cnt[i], 1):
                    cc[qp, t0, tms[n]] += evs[k] * evs[n]
                    n += 1
                k += 1
        else:
            k += cnt[i]
    return cc, mint


@nb.jit(nopython=True, parallel=True, fastmath=True)
def ncorrelatep(evs, tms, cnt, q, n_frames, nproc):
    qm = q.max()
    cc = np.zeros((nproc, qm, n_frames, n_frames), np.float32)
    mint = np.zeros((nproc, qm, n_frames), np.float32)
    vk = np.zeros((cnt.size + 1), np.uint32)
    for i in range(cnt.size):
        vk[i + 1] = vk[i] + cnt[i]
    for proc in nb.prange(nproc):
        for i in range(proc, cnt.size, nproc):
            k = vk[i]
            qp = q[i] - 1
            if qp >= 0:
                for j in range(cnt[i]):
                    t0 = tms[k]
                    n = k + 1
                    mint[proc, qp, t0] += evs[k]
                    for f in range(j + 1, cnt[i], 1):
                        cc[proc, qp, t0, tms[n]] += evs[k] * evs[n]
                        n += 1
                    k += 1
            else:
                k += cnt[i]
    for i in range(1, nproc, 1):
        cc[0,:,:,:] += cc[i,:,:,:]
        mint[0,:,:] += mint[i,:,:]
    return cc[0,:,:,:], mint[0,:,:]


#### The fastes numba implementation and does not use extra memory ###
@nb.jit(nopython=True, parallel=True, fastmath=True)
def ncorrelatepm(evs, tms, cnt, q, n_frames, nproc):
    qm = q.max()
    cc = np.zeros((qm, n_frames, n_frames), np.float32)
    mint = np.zeros((qm, n_frames), np.float32)
    vk = np.zeros((cnt.size + 1), np.uint32)
    for i in range(cnt.size):
        vk[i + 1] = vk[i] + cnt[i]
    tstep = int(np.ceil(tms.max() / nproc))
    print(tstep)
    for proc in nb.prange(nproc):
        for i in range(cnt.size):
            k = vk[i]
            qp = q[i] - 1
            if qp >= 0:
                for j in range(cnt[i]):
                    t0 = tms[k]
                    if t0 >= tstep * proc and t0 < tstep * (proc + 1):
                    # if t0 % nproc == proc:
                        n = k + 1
                        mint[qp, t0] += evs[k]
                        for f in range(j + 1, cnt[i], 1):
                            cc[qp, t0, tms[n]] += evs[k] * evs[n]
                            n += 1
                    k += 1
            else:
                k += cnt[i]
    return cc, mint


##### Event_correlator standard several qs using Numba #########
def nbecorrts_q(events, times, cnt, qqmask, n_frames, calc_std=False, ttcf_par=0):
    """ Calculation of the event correlation function using Numba 

    :param events: 1D array of events values
    :param times: 1D array of times 
    :param cnt: 1D array of number of events in a pixel
    :param qqmask: 2D array q mask
    :param n_frames: int number of frames


    :return: CorrelationResult structure with correlation functions
    """
    print("start calculation of the correlation functions")
    t0 = time.time()

    qqmask = np.ravel(qqmask)[cnt > 0]
    cnt = cnt[cnt > 0]

    # num, mint = ncorrelate(events,times,cnt,qqmask,n_frames)
    print("Number of used processors: ", nproc)
    num, mint = ncorrelatepm(events, times, cnt, qqmask, n_frames, nproc)  #  parallel
    # num, mint = ncorrelate(events,times,cnt,qqmask,n_frames)#  paralle
    res = []
    qm = qqmask.max()
    res = np.zeros((qm, n_frames - 1), dtype=np.float32)
    if calc_std:
        dev = np.zeros_like(res)
    for q in range(1, qm + 1, 1):
        for_norm = qqmask[qqmask == q].size
        cor = num[q - 1,:,:]  # /q[q==2].size
        s = np.reshape(mint[q - 1,:], (n_frames, 1))  # /for_norm
        print("Number of pixels for q %d is %d" % (q, for_norm))
        print("Average photons per frame for q %d is %f" % (q, np.mean(s)))
        print("Average photons per frame per pixel for q %d is %f" % (q, np.mean(s) / for_norm))

        norm = np.dot(s, s.T) / n_frames
        cor = cor * for_norm / n_frames
        for i in range(1, n_frames):
            dia = np.diag(cor, k=i)
            sdia = np.diag(norm, k=i)
            ind = np.where(np.isfinite(dia))
            res[q - 1, i - 1] = np.mean(dia[ind]) / np.mean(sdia[ind])
            if calc_std:
                dev[q - 1, i - 1] = np.std(dia[ind] / sdia[ind]) / len(sdia[ind]) ** 0.5
        if ttcf_par == q:
            trc = cor / norm
            tmp = np.diag(trc, k=-5)
            if np.sum(tmp) < 1:
                trc += np.rot90(np.fliplr(trc))
            tmp = np.diag(trc, k=1)
            tmp = np.mean(tmp[tmp > 0])
            for j in range(n_frames - 1):
                trc[j, j] = tmp
            del tmp

    print("Total correlation time %2.2f sec" % (time.time() - t0))
    # return wcf,mtcf,cor
    if ttcf_par == 0:
        trc = 0
    if not(calc_std):
        dev = 0
    return CorrelationResult(res, dev, trc)


########### Event_correlator standard several qs ################
def ecorrts_q(pixels, s, qqmask, calc_std=False, ttcf_par=0):
    print("start calculation of the correlation functions")
    t0 = time.time()
    lpixels = len(pixels)  # number of frames
    rpixels = range(lpixels)
    t = []
    for t1 in rpixels:
        t += [t1] * s[t1]
    pix = np.concatenate(pixels).ravel() - 1  # to fit with in1d function result
    pix = np.array(pix, dtype=np.uint32)
    t = np.array(t, dtype=np.uint32)
    indpi = np.lexsort((t, pix))
    t = t[indpi]
    pix = pix[indpi]
    qqmask = np.ravel(qqmask)
    qm = qqmask.max()
    res = np.zeros((qm, lpixels - 1), dtype=np.float32)
    if calc_std:
        dev = np.zeros_like(res)
    for q in range(1, qm + 1, 1):
        for_norm = len(qqmask[qqmask == q])
        indx = np.where(qqmask == q)[0]  # here I can make +1 or pix-1
        indpi = np.in1d(pix, indx)
        tq = t[indpi]
        pixq = pix[indpi]
        s, tt = np.histogram(tq, bins=lpixels)
        print("Number of pixels for q %d is %d" % (q, for_norm))
        print("Average photons per frame for q %d is %f" % (q, np.mean(s)))
        print("Average photons per frame per pixel for q %d is %f" % (q, np.mean(s) / for_norm))
        lenpi = len(pixq)
        cor = np.zeros((lpixels, lpixels), dtype=np.uint32)
        cor = fecorrt(pixq, tq, cor, lenpi, lpixels)  # fortran module
        lens = len(s)
        s = np.array(s, dtype=np.float32)
        cor = np.array(cor, dtype=np.float32)
        s.shape = lens, 1
        # norm = dot(s,flipud(s.T))/lpixels
        norm = np.dot(s, s.T) / lpixels
        cor = cor * for_norm / lpixels
        for i in range(1, lpixels):
            dia = np.diag(cor, k=i)
            sdia = np.diag(norm, k=i)
            ind = np.where(np.isfinite(dia))
            res[q - 1, i - 1] = np.mean(dia[ind]) / np.mean(sdia[ind])
            if calc_std:
                dev[q - 1, i - 1] = np.std(dia[ind] / sdia[ind]) / len(sdia[ind]) ** 0.5
        # wcf = x+0
        # wcf[np.isnan(wcf)] = 0.01
        # mtcf = cftomt_testing(x)
        # mtcf = tools.cftomt(x)
        # res.append(wcf)
        if ttcf_par == q:
            trc = cor / norm
            tmp = np.diag(trc, k=1)
            tmp = np.mean(tmp[tmp > 0])
            for j in range(lpixels - 1):
                trc[j, j] = tmp
            del tmp
    print("Total correlation time %2.2f sec" % (time.time() - t0))
    # return wcf,mtcf,cor
    if ttcf_par == 0:
        trc = 0
    if not(calc_std):
        dev = 0
    return CorrelationResult(res, dev, trc)


########### Event_correlator standard ################
def ecorrts(pixels, s, for_norm):
    print("start calculation of the correlation functions")
    t0 = time.time()
    lpixels = len(pixels)
    rpixels = range(lpixels)
    t = []
    for t1 in rpixels:
       t += [t1] * s[t1]
    # print("time for pre loop "+str(time()-timee))
    pix = np.concatenate(pixels).ravel()
    # print('start sorting')
    times = time()
    pix = np.array(pix, dtype=np.int32)
    t = np.array(t)
    indpi = np.lexsort((t, pix))
    t = t[indpi]
    pix = pix[indpi]
    # print('sorting took '+ str(time()-times))
    # print('start main loop')
    timem = time.time()
    lenpi = len(pix)
    cor = np.zeros((lpixels, lpixels), dtype=np.uint32)
    timef = time.time()
    cor = fecorrt(pix, t, cor, lenpi, lpixels)  # fortran module
    # to split for multicores
    # ind = np.where(pix<133128)[0][-1]
    # cor = fecorrt(pix[:ind],t[:ind],cor,len(pix[:ind]),lpixels)#fortran module
    # cor1 = np.zeros((lpixels,lpixels),dtype=np.uint32)
    # cor1 = fecorrt(pix[ind:],t[ind:],cor1,len(pix[ind:]),lpixels)#fortran module
    # cor = cor+cor1
    # print("time for correlating "+str(time()-timem))
    print("average photons per frame " + str(np.mean(s)))
    # print(cor.min(),cor.max())
    lens = len(s)
    s = np.array(s, dtype=np.float32)
    cor = np.array(cor, dtype=np.float32)
    s.shape = lens, 1
    # norm = dot(s,flipud(s.T))/lpixels
    norm = np.dot(s, s.T) / lpixels
    cor = cor * for_norm / lpixels
    x = np.ones((lpixels - 1, 3))
    x[:, 0] = np.arange(1, lpixels)
    for i in range(1, lpixels):
        dia = np.diag(cor, k=i)
        sdia = np.diag(norm, k=i)
        ind = np.where(np.isfinite(dia))
        x[i - 1, 1] = np.mean(dia[ind]) / np.mean(sdia[ind])
        x[i - 1, 2] = np.std(dia[ind] / sdia[ind]) / len(sdia[ind]) ** 0.5
    wcf = x + 0
    mtcf = cftomt_testing(x)
    cor /= norm
    print("Total time for correlating %2.2f sec" % (time.time() - t0))
    return wcf, mtcf, cor
##############################################################################


def cftomt_testing(d):
    par = 16
    tmp = d[par:,:3]
    nt = []
    nd = []
    nse = []
    for i in range(par):
        nt.append(d[i, 0])
        nd.append(d[i, 1])
        nse.append(d[i, 2])
    while len(tmp[:, 0]) >= par:
    ntmp = (tmp[:-1,:] + tmp[1:,:]) / 2
    for i in range(0, par, 2):
        nt.append(ntmp[i, 0])
        nd.append(ntmp[i, 1])
        nse.append(ntmp[i, 2])
    tmp = ntmp[par:-1:2,:3]
    x = np.array([nt, nd, nse]).T
    return x


def gpu_ecorr(events, times, offsets, shape, nframes, dtype, qqmask, F):

    print("start calculation of the correlation functions")
    from dynamix.correlator.event import EventCorrelator
    t0 = time.time()
    E = EventCorrelator(shape, nframes, F.max_nnz, dtype=dtype, total_events_count=events.size, scale_factor=qqmask[qqmask == 1].size)
    result = E.correlate(times, events, offsets)
    print("Total correlation time %2.2f sec" % (time.time() - t0))
    res = np.zeros((1, nframes - 1), dtype=np.float32)
    dev = np.zeros((1, nframes - 1), dtype=np.float32)
    res[0,:] = result[0][1:]
    dev[0,:] = result[0][1:] * 1e-3
    trc = 0
    return CorrelationResult(res, dev, trc)


def gpu_ecorr_q(events, times, offsets, shape, nframes, dtype, qqmask, max_nnz):
    """ Calculation of the event correlation function using GPU

    :param events: 1D array of events values
    :param times: 1D array of times
    :param offsets: 1D array of offsets
    :param shape: list shape of the 2D frame
    :param nframes: int number of frame
    :param dtype: type of the events array
    :param qqmask: 2D array q mask
    :param max_nnz: maximum number frames with event

    :return: CorrelationResult structure with correlation functions
    """

    print("start calculation of the correlation functions")
    from dynamix.correlator.event import EventCorrelator
    t0 = time.time()
    qm = qqmask.max()
    res = np.zeros((qm, nframes - 1), np.float32)
    dev = np.zeros_like(res)
    E = EventCorrelator(shape, nframes, max_nnz, qmask=qqmask, dtype=dtype, total_events_count=events.size)
    result = E.correlate(times, events, offsets)
    for q in range(1, qm + 1, 1):
        print("Number of pixels for q %d is %d" % (q, E.scale_factors[q]))
        res[q - 1,:] = result[q - 1][1:]  # 1 time correlation function
        dev[q - 1,:] = result[q - 1][1:] * 1e-3  # error of the 1 time correlation function
    print("Total correlation time %2.2f sec" % (time.time() - t0))
    trc = 0
    return CorrelationResult(res, dev, trc)

