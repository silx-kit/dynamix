#draw results of the caclulations
import pylab as plt
import numpy as np
from scipy.ndimage import gaussian_filter

plt.rc('savefig', directory='') # leave directory empty to use current working directory
plt.rcParams['image.cmap'] = 'jet'
plt.rcParams['font.size'] = 14

def plot_cf(xx,sname=''):
    max_y = 0
    n = 1
    plt.figure()
    try:
        for x in xx:
            plt.fill_between(x[:,0],x[:,1]-x[:,2],x[:,1]+x[:,2],alpha=0.5)
            plt.semilogx(x[:,0],x[:,1],'-',label='q '+str(n))
            #plt.errorbar(x[:,0],x[:,1],yerr=x[:,2])
            if max_y < x[1,1]:
                max_y = x[1,1]*1    
            n+=1
    except:
        plt.fill_between(xx[:,0],xx[:,1]-xx[:,2],xx[:,1]+xx[:,2],alpha=0.5)
        plt.semilogx(xx[:,0],xx[:,1],'-',label='q '+str(0))  
        if max_y < xx[1,1]:
            max_y = xx[1,1]*1  
    plt.xlabel(r"lag time $\tau$ (s)")
    plt.ylabel(r"g$^{(2)}$(q,$\tau$)")
    plt.ylim((1,1+(max_y-1)*1.3))
    plt.legend(loc=1,fontsize=10,ncol=n//6)
    plt.title(sname)
    plt.show()

def show_trc(cor,sname=''):
    plt.figure()
    vvmax = np.diag(cor,k=4)
    vvmax = np.mean(vvmax[np.where((vvmax>0.5)&(vvmax<1.5))])
    plt.imshow(gaussian_filter(cor,1.5),vmin=1,vmax=max(vvmax,1.01),origin="lower",interpolation='nearest')
    plt.colorbar()
    plt.title(sname)
    plt.xlabel('frame')
    plt.ylabel('frame')
    plt.show() 
