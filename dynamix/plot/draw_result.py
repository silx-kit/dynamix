#draw results of the caclulations
import pylab as plt
import numpy as np
from scipy.ndimage import gaussian_filter

plt.rc('savefig', directory='') # leave directory empty to use current working directory
plt.rcParams['image.cmap'] = 'jet'
plt.rcParams['font.size'] = 14

def plot_cf(xx,sname='',savdir="./",toplot="no"):
    """ Plots, saves figure of the g2 functions 

    :param xx: 2D array of the g2
    :param sname: str  sample name for the title and figure file name 
    :param savdir: str directory name where to save the figure
    :param toplot: str yes/no flag to visualise the figure

    """
    max_y = 0
    n = 1
    plt.figure()
    try:
        for x in xx:
            plt.fill_between(x[:,0],x[:,1]-x[:,2],x[:,1]+x[:,2],alpha=0.5)
            plt.semilogx(x[:,0],x[:,1],'-',label='q '+str(n))
            #plt.semilogx(x[:,0],x[:,1],'o',label='q '+str(n))
            #plt.errorbar(x[:,0],x[:,1],yerr=x[:,2])
            if max_y < x[1,1]:
                max_y = x[1,1]*1    
            n+=1
    except:
        plt.fill_between(xx[:,0],xx[:,1]-xx[:,2],xx[:,1]+xx[:,2],alpha=0.5)
        plt.semilogx(xx[:,0],xx[:,1],'-',label='q '+str(0))  
        #plt.semilogx(x[:,0],x[:,1],'o',label='q '+str(n))
        if max_y < xx[1,1]:
            max_y = xx[1,1]*1  
    plt.xlabel(r"lag time $\tau$ (s)")
    plt.ylabel(r"g$^{(2)}$(q,$\tau$)")
    #plt.ylim((1,1+(max_y-1)*1.3))
    try:
        plt.legend(loc=1,fontsize=10,ncol=n//6)
    except:
        plt.legend(loc=1,fontsize=10)
    plt.title(sname)
    plt.tight_layout()
    plt.savefig(savdir+sname.replace(" ","_")+"_cf.png",dpi=300)
    if toplot == "yes":
        plt.show()

def show_trc(cor,sname='',savdir="./",toplot="no"):
    """ Plots, saves figure of the TTCF function

    :param cor: 2D array of the TTCF
    :param sname: str  sample name for the title and figure file name 
    :param savdir: str directory name where to save the figure
    :param toplot: str yes/no flag to visualise the figure

    """
    plt.figure()
    nx,ny = cor.shape
    #For William GPU result
    if np.isnan(cor).any():
        nd= np.zeros(cor.shape,np.float32)
        for i in range(1,nx,1):
            raw = np.arange(0,nx-i,1)
            dd = cor[np.isfinite(cor[:,i]),i]
            col = np.arange(i,i+dd.size,1)
            nd[raw,col] = dd
            nd[col,raw] = dd
  
        nd[np.arange(0,nx,1), np.arange(0,nx,1)] = np.diag(nd,k=1).mean()
        cor[:] = nd
        
    #smooth the data to reduce the noise
    if nx>20000:
        cor = gaussian_filter(cor,9)
    elif nx>10000:
        cor = gaussian_filter(cor,6)
    else:
        cor = gaussian_filter(cor,3)
    vvmax = np.diag(cor,k=4)
    vvmax = np.mean(vvmax[np.where((vvmax>0.5)&(vvmax<1.5))])
    plt.imshow(cor,vmin=1,vmax=max(vvmax,1.01),origin="lower",interpolation='nearest')
    #plt.imshow(cor,vmax=max(vvmax,1.01),origin="lower",interpolation='nearest')
    plt.colorbar()
    plt.title(sname)
    plt.xlabel('frame')
    plt.ylabel('frame')
    plt.tight_layout()
    plt.savefig(savdir+sname.replace(" ","_")+"_trc.png",dpi=300)
    if toplot == "yes":
        plt.show()
