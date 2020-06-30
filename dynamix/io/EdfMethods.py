from os.path import isfile
from .EdfFile3 import EdfFile, EdfGzipFile
import fabio
####################################
#--- Standard EDF w/r functions ---#
####################################


def loadedf(filename):
    if isfile(filename):
        f=fabio.open(filename)
        data = f.data
        f.close() 
        return data
    else:
        print("file "+filename+" does not exist!")
        raise IOError


def saveedf(filename, data, imgn=0):
    try:
        newf = EdfFile(filename)
        newf.WriteImage({}, data, imgn)
        print("file is saved to ", filename)
        return
    except:
        print("file is not saved!")
        return

def headeredf(filename):
    if isfile(filename):
        f=fabio.open(filename)
        header = f.header
        f.close()
        return header
    else:
        print("file "+filename+" does not exist!")
        raise IOError

