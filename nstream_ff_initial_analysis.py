import numpy as np
import matplotlib.pylab as plt
from matplotlib import ticker
np.set_printoptions(precision=3)
from matplotlib import colors
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or$
from pylab import *

# http://comp-phys.net/2013/03/25/working-with-percolation-clusters-in-python/
import numpy as np
from pylab import *
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or$
from scipy.ndimage import measurements
np.set_printoptions(precision = 1)
from matplotlib import colors
import matplotlib.pylab as plt

import matplotlib as mpl


def set_pub():
    """ Pretty plotting changes in rc for publications
    Might be slower due to usetex=True
    
    
    plt.minorticks_on()  - activate  for minor ticks in each plot

    """
    plt.rc('font', weight='bold')    # bold fonts are easier to see
    #plt.rc('font',family='serif')
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rc('text', usetex=True)   # Slower
    plt.rc('font',size=18)
    
    plt.rc('lines', lw=1, color='k', markeredgewidth=1.5) # thicker black lines
    #plt.rc('grid', c='0.5', ls='-', lw=0.5)  # solid gray grid lines
    plt.rc('savefig', dpi=300)       # higher res outputs
    
    plt.rc('xtick', labelsize='x-large')
    plt.rc('ytick', labelsize='x-large')
    plt.rc('axes',labelsize= 30)
    
    plt.rcParams['xtick.major.size'] = 12
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['xtick.minor.size'] = 8
    plt.rcParams['xtick.minor.width'] = 1
    
    plt.rcParams['ytick.major.size'] = 12
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['ytick.minor.size'] = 8
    plt.rcParams['ytick.minor.width'] = 1
    
    plt.rcParams['axes.color_cycle'] = ['navy', 'forestgreen', 'darkred']

set_pub()


def plotfaces(img3d):
    
    f, ax = plt.subplots(2,3, figsize = (20,12),  sharex=True)
    f.subplots_adjust( wspace = 0.08, hspace = 0.02)

    ticksLoc = np.linspace(0, img3d.shape[0], 7)
    ticksLabel = [ '', '10', '20', '40', '60', '80', '' ]
    
    cmap = colors.ListedColormap(['gray', 'red', 'white'])
    bounds=[0,1,np.max(img3d)/10., np.max(img3d)]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    
    #figure(n, figsize=(16,5))
    #set_cmap('spectral')
    #subplot(2,3,1)
    plt.sca(ax[0,0])
    imshow(img3d[0,:,:], origin='lower', interpolation='nearest', cmap = cmap, norm = norm)
    #colorbar()
    title("X")
    #plt.minorticks_on()
    plt.ylabel(r" $h^{-1} Mpc$")
    plt.yticks(ticksLoc , ticksLabel  )
    plt.xticks([])
    
    #plt.gca().set_yticks(ticksLabel)
    
    #subplot(2,3,4)
    plt.sca(ax[1,0])
    imshow(img3d[L-1,:,:], origin='lower', interpolation='nearest', cmap = cmap, norm = norm)
    #colorbar()
    #title("2")
    #plt.minorticks_on()
    plt.xlabel(r" $h^{-1} Mpc$")
    plt.ylabel(r" $h^{-1} Mpc$")
    plt.yticks( ticksLoc, ticksLabel  )
    plt.xticks( ticksLoc, ticksLabel  )
    
    
    #subplot(2,3,2)
    plt.sca(ax[0,1])
    imshow(img3d[:,0,:], origin='lower', interpolation='nearest', cmap = cmap, norm = norm)
    #colorbar()
    title("Y")
    #plt.minorticks_on()
    plt.xticks([])
    plt.yticks([])
    
    
    #subplot(2,3,5)
    plt.sca(ax[1,1])
    imshow(img3d[:,L-1,:], origin='lower', interpolation='nearest', cmap = cmap, norm = norm)
    #colorbar()
    #title("4")
    plt.xlabel(r" $h^{-1} Mpc$")
    #plt.minorticks_on()
    plt.xticks( ticksLoc, ticksLabel  )
    plt.yticks([])  
    #plt.minorticks_on()  

    #subplot(2,3,3)
    plt.sca(ax[0,2])
    imshow(img3d[:,:,0], origin='lower', interpolation='nearest', cmap = cmap, norm = norm)
    #colorbar()
    title("Z")
    #plt.minorticks_on()
    plt.xticks([])    #plt.gca().set_yticks([])
    plt.yticks([])  
    

    #subplot(2,3,6)
    plt.sca(ax[1,2])
    imshow(img3d[:,:,L-1], origin='lower', interpolation='nearest', cmap = cmap, norm = norm)
    #colorbar()
    #title("6")
    #show()
    plt.xlabel(r" $h^{-1} Mpc$")
    #plt.minorticks_on()
    plt.xticks( ticksLoc, ticksLabel  )
    plt.yticks([])

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

    xlabel = np.max(xlabel_all, axis = 0)
    
    #xlabel = xlabel_round   # REMOVE
    
    return xlabel    # Returns 1D labels corresponding to particles


nGr = 128
refFactor = 1
size_fact = nGr*refFactor
L = 1.




nstream = np.load('npy/'+str( int(L) ) +'Mpc'+str(nGr)+'/numField_051_'+str(refFactor)+'.npy')
flip = np.load('npy/'+str( int(L) ) +'Mpc'+str(nGr)+'/flip_snap_051.npy')

img3d_n = np.log(nstream)
img3d_f = np.log(flip+1) 

#img3d_n = nstream == 1 
#img3d_f = flip == 0 

fileOut = 'npy/'+str( int(L) ) +'Mpc'+str(nGr)+'/x0_'+'snap_051.npy'
x0_1d = np.ravel(np.load(fileOut), 'F')
fileOut = 'npy/'+str( int(L) ) +'Mpc'+str(nGr)+'/x1_'+'snap_051.npy'
x1_1d = np.ravel(np.load(fileOut), 'F')
fileOut = 'npy/'+str( int(L) ) +'Mpc'+str(nGr)+'/x2_'+'snap_051.npy'
x2_1d = np.ravel(np.load(fileOut), 'F')

flip_1d = np.ravel((flip), 'F')






f, ax = plt.subplots(1,3, figsize = (20,6))
f.subplots_adjust(  wspace = 0.05, hspace = 0.02, bottom = 0.15, left = 0.1, right = 0.85)

ticksLoc = np.linspace(0, img3d_n.shape[0], 6)
ticksLabel = [ '', '0.2', '0.4', '0.6', '0.8', '' ]
    
#cmap = colors.ListedColormap(['white', 'gray', 'red'])
#bounds=[0,1, 2,   np.max(nstream)]
#norm = colors.BoundaryNorm(bounds, cmap.N)
cmap= 'nipy_spectral_r'
cmap= 'CMRmap_r'
    

plt.sca(ax[2])
imshow(img3d_n[5,:,:], origin='lower', interpolation='nearest', cmap = cmap)
#colorbar()
title(r"$n_{str}(z = 0)$")
#plt.minorticks_on()
plt.xlabel(r" $h^{-1} Mpc$")
plt.xticks(ticksLoc , ticksLabel  )
plt.yticks([])

    
#plt.gca().set_yticks(ticksLabel)
    
#subplot(2,3,4)
plt.sca(ax[0])
imshow(img3d_f[5,:,:], origin='lower', interpolation='nearest', cmap = cmap)
#colorbar()
title(r"$ff(z_{ini})$")
#plt.minorticks_on()
plt.xlabel(r" $h^{-1} Mpc$")
plt.ylabel(r" $h^{-1} Mpc$")
plt.yticks( ticksLoc, ticksLabel  )
plt.xticks( ticksLoc, ticksLabel  )


non0ff = np.where( (flip_1d > 0)  & (x0_1d < 0.1))
plt.sca(ax[1])
#plt.plot(x2_1d[non0ff], x1_1d[non0ff], 'o', alpha = 0.3, markersize = 1)

#cmap = colors.ListedColormap(['gray', 'red', 'white'])
cmap = 'prism'
bounds=[1,3, 5, 10, np.max(flip_1d[non0ff])]
norm = colors.BoundaryNorm(bounds, cmap)
    
    
scatter(x2_1d[non0ff], x1_1d[non0ff], 
c = np.log(flip_1d[non0ff]+1), alpha = 0.5, s = 2, cmap = 'gist_stern', edgecolors = 'none')
#colorbar()
title(r"$ff(z=0)$")
#plt.minorticks_on()
plt.xlabel(r" $h^{-1} Mpc$")
#plt.ylabel(r" $h^{-1} Mpc$")
plt.xlim(0,1)
plt.ylim(0,1)
ticksLoc = ticksLabel
plt.yticks([] )
#plt.xticks( ticksLoc, ticksLabel  )


plt.show()



import sys
sys.exit()



import toParaview as toPara
toPara.UnstructuredGrid(xCen_fof*size_fact/L,yCen_fof*size_fact/L,zCen_fof*size_fact/L,r_fof*size_fact/L, 'vti/FOFSph_032_'+str(refFactor))
