#! /usr/bin/env python3
#find the direct beam center
import sys
import os
import numpy as np
import time
import fabio
from dynamix.io import readdata, EdfMethods, h5reader
from dynamix.tools import tools 
from scipy.ndimage import center_of_mass, gaussian_filter
import pylab as plt

import configparser
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())


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

try: 
    data = readdata.readnpz(savdir+sname+"_2D.npz")
except:
    print("Cannot read "+savdir+sname+"_2D.npz") 
    exit()
try:
    mask = EdfMethods.loadedf(mask_file)
except:
    print("Cannot read "+mask_file) 
    exit()


################################################

print("Scipy center of mass of raw data x=%3.2f, y=%3.2f" % center_of_mass(data)[::-1])

t0 = time.time()
cx,cy = tools.beam_center(data,mask,cx,cy,lc=30,lw=10)#change lc and lw according to the needs lc>lw
print("Calculation time t=%2.3f sec." % (time.time()-t0))

plt.show()
exit()
