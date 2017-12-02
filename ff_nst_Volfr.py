import numpy as np
#from pylab import *
#import matplotlib
#matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or$
#from scipy.ndimage import measurements
np.set_printoptions(precision = 1)
from matplotlib import colors
import matplotlib.pylab as plt
import SetPub
SetPub.set_pub()


def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None):
    
    #https://tonysyu.github.io/plotting-error-bars.html
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color = color, lw = 1.5, label = strLabel)
    #ax.plot(x, y, color+'o', label = strLabel)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)



def ParticleLabel(regionLabels, x1d, y1d, z1d):  
# Finds particles in each labeled region, tags each particle with ID
    x0 = np.mod(np.ceil(x1d*size_fact/L), size_fact).astype(int)
    x1 = np.mod(np.ceil(y1d*size_fact/L), size_fact).astype(int)
    x2 = np.mod(np.ceil(z1d*size_fact/L), size_fact).astype(int)

    xlabel_ceil = regionLabels[x0,x1,x2]

    x0 = np.mod(np.floor(x1d*size_fact/L), size_fact).astype(int)
    x1 = np.mod(np.floor(y1d*size_fact/L), size_fact).astype(int)
    x2 = np.mod(np.floor(z1d*size_fact/L), size_fact).astype(int)

    xlabel_floor= regionLabels[x0,x1,x2]

    x0 = np.mod(np.round(x1d*size_fact/L), size_fact).astype(int)
    x1 = np.mod(np.round(y1d*size_fact/L), size_fact).astype(int)
    x2 = np.mod(np.round(z1d*size_fact/L), size_fact).astype(int)

    xlabel_round = regionLabels[x0,x1,x2]

    xlabel_all = np.vstack([xlabel_ceil, xlabel_floor , xlabel_round])

    xlabel = np.min(xlabel_all, axis = 0)
    #xlabel = np.max(xlabel_all, axis = 0)    
    #xlabel = xlabel_round   # REMOVE
    
    return xlabel    # Returns 1D labels corresponding to particles


L = 100.
nGr = 128
refFactor = 1
Dn = 60      # 60 for all except 100Mpc-512 (160)
size_fact = nGr*refFactor
lBox = str(int(L))+'Mpc'+str(nGr)

UnderLength = 1  # 0.1
# slice_noX = int(UnderLength*size_fact/(2*L))
sliceNo = 40
#--------#--------#--------#--------#--------
sigList = [0.0, 0.5, 1.0, 1.5, 2.0]
sig = sigList[4]
dLoad = './npy/'+lBox+'/'


file_nstr = dLoad+'numField_050_'+str(refFactor)+'.npy'
nstream = np.load(file_nstr).astype(np.float64)
lBox = str(int(L))+'Mpc'
dir1 = lBox+str(nGr)+'/'
flip = np.load('npy/'+dir1+'flip_snap_050.npy')


img3d_n = np.log(nstream)
img3d_f = np.log(flip+1)

#img3d_n = nstream == 1
#img3d_f = flip == 0

x0_3d = np.load(dLoad+'x0'+'_snap_050.npy')
x1_3d = np.load(dLoad+'x1'+'_snap_050.npy')
x2_3d = np.load(dLoad+'x2'+'_snap_050.npy')

# fileOut = 'npy/'+str( int(L) ) +'Mpc'+str(nGr)+'/x0_'+'snap_051.npy'
x0_1d = np.ravel(x0_3d, 'F')
# fileOut = 'npy/'+str( int(L) ) +'Mpc'+str(nGr)+'/x1_'+'snap_051.npy'
x1_1d = np.ravel(x1_3d, 'F')
# fileOut = 'npy/'+str( int(L) ) +'Mpc'+str(nGr)+'/x2_'+'snap_051.npy'
x2_1d = np.ravel(x2_3d, 'F')

flip_1d = np.ravel((flip), 'F')


particlenstr1d = ParticleLabel( nstream, x0_1d, x1_1d, x2_1d)





plt.figure(figsize = (8,6))
#plt.plot(particlenstr1d, flip_1d , 'o')

nbins = np.arange(-0.5, np.max(flip_1d)-0.5, 1)
y,binEdges = np.histogram( flip_1d, bins = nbins, density=False)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
#y = np.cumsum(y[::-1])[::-1]

strLabel = r"$n_{ff}$(on particles)"
#plt.plot(bincenters, 1.*y, color = 'darkred', label = strLabel)
plt.plot(bincenters, 1.*y/y.sum(), color = 'darkred', label = strLabel)

#errorfill(bincenters,y, np.sqrt(y), color = 'darkred')


nbins = np.arange(0.5, np.max(particlenstr1d)-0.5, 2)
y,binEdges = np.histogram( particlenstr1d, bins = nbins, density=False)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])



#y = np.cumsum(y[::-1])[::-1]

strLabel = r"$n_{str}$(on grid)"
#plt.plot(bincenters, 1.*y, color = 'navy', label = strLabel)
plt.plot(bincenters, 1.*y/y.sum(), color = 'navy', label = strLabel)

#errorfill(bincenters,y, np.sqrt(y), color = 'navy')


nbins = np.arange(0.5, np.max(nstream)-0.5, 2)
y,binEdges = np.histogram( nstream, bins = nbins, density=False)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
#y = np.cumsum(y[::-1])[::-1]

strLabel = r"$n_{str}$(on particles)"
#plt.plot(bincenters, 1.*y, color = 'forestgreen', label = strLabel)
plt.plot(bincenters, 1.*y/y.sum(), color = 'forestgreen', label = strLabel)

#errorfill(bincenters,y, np.sqrt(y), color = 'forestgreen')

plt.yscale('log')
plt.xscale('log')
plt.ylim(1e-4, )

plt.ylabel(r'Volume fraction')
#plt.ylabel(r'Volume fraction $VF(n > n_{th})$')

plt.xlabel(r'$n$')

plt.legend(loc = 'upper right')
#plt.legend(loc = 'lower right', frameon=False)
plt.savefig('plots/ff_nstr_VolFr.pdf', bbox_inches='tight')


plt.tight_layout()

plt.show()
