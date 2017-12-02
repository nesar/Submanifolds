# http://comp-phys.net/2013/03/25/working-with-percolation-clusters-in-python/
import numpy as np
from pylab import *
#import matplotlib
#matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or$
from scipy.ndimage import measurements
np.set_printoptions(precision = 1)
from matplotlib import colors
import matplotlib.pylab as plt
import SetPub
SetPub.set_pub()

import matplotlib as mpl


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
    # xlabel = np.max(xlabel_all, axis = 0)
    # xlabel = xlabel_round   # REMOVE
    
    return xlabel    # Returns 1D labels corresponding to particles

# ------------------------------

L = 100.
nGr = 128
refFactor = 2
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


# nstream = np.load('npy/'+str( int(L) ) +'Mpc'+str(nGr)+'/numField_051_'+str(refFactor)+'.npy')
# flip = np.load('npy/'+str( int(L) ) +'Mpc'+str(nGr)+'/flip_snap_051.npy')

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


# ------------------------------

particlenstr1d = ParticleLabel( nstream, x0_1d, x1_1d, x2_1d)


ff0 = (   (flip_1d == 0) )
ff1 = (   (flip_1d > 0) )
ns0 = (   (particlenstr1d == 1) )
ns1 = (   (particlenstr1d > 1) )
numF0 = (    (nstream == 0)    )
numF1 = (    (nstream >= 0)    )
# sliceWidth = (x0_1d < UnderLength)
sliceWidth = ( np.abs(x0_1d - sliceNo*L/size_fact) < UnderLength )

# ------------------------------

ff0ns0 = np.where(  sliceWidth   &   ( (ff0) & (ns0)) )
# nstream == 1 &  flip-flop = 0  -- most particles within single-stream region

ff0ns1 = np.where(  sliceWidth  &   ( (ff0) & (ns1)) )
# nstream > 1 &  flip-flop = 0  -- few particles within multi-stream region

ff1ns1 = np.where(  sliceWidth   &  ( (ff1) & (ns1)) )
# nstream > 1 &  flip-flop > 1  -- particles within multi-stream region, flipped 

ff1ns0 = np.where(  sliceWidth   &  ( (ff1) & (ns0)) )
# nstream == 1 &  flip-flop > 0 -- expected None  -- finite value due to interpolation issue
print ff1ns0[0].shape


# ------------------------------


nstream_2d = nstream[sliceNo,:,:]

plt.figure(1000, figsize = (8,8))
# plt.contour(nstream_2d.transpose(), levels = [1], colors = ['r'], linewidths = 2.5,
#             alpha = 0.6, label = r'$n_{str} > 1$')
plt.contourf(nstream_2d.transpose(), levels = [3, nstream_2d.max()], colors = ['r'], alpha = 0.6, label = r'$n_{str} > 1$')
plt.scatter( x1_1d[ff0 & sliceWidth]*size_fact/L, x2_1d[ff0 & sliceWidth]*size_fact/L, alpha = 0.8 , s = 3, facecolors='k', edgecolors='none', label = r'$n_{ff} = 0$')
plt.legend()
# plt.xlim(0,70)
# plt.ylim(0,70)
# plt.colorbar()
plt.savefig('plots/ff0_slice.pdf', bbox_inches='tight')


plt.figure(1001, figsize = (8,8))
plt.contourf(nstream_2d.transpose(), levels = [3, nstream_2d.max()], colors = ['r'], alpha = 0.6, label = r'$n_{str} > 1$')
plt.scatter( x1_1d[ff1 & sliceWidth]*size_fact/L, x2_1d[ff1 & sliceWidth]*size_fact/L, alpha = 0.8 , s = 3, facecolors='k', edgecolors='none', label = r'$n_{ff} > 0$')
plt.legend()
# plt.xlim(0,70)
# plt.ylim(0,70)
plt.savefig('plots/ff1_slice.pdf', bbox_inches='tight')



plt.figure(1002)

flipns0 = flip_1d[ns0] # Flip of particles in single-streaming region

nbins = np.arange(-0.5, np.max(flipns0)+0.5, 1)
y,binEdges = np.histogram( flipns0, bins = nbins, density=False)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
#y = np.cumsum(y[::-1])[::-1]

strLabel = r"$n_{ff}$( single stream)"
#plt.plot(bincenters, 1.*y, color = 'forestgreen', label = strLabel)
plt.plot(bincenters, 1.*y/y.sum(), '-o', color = 'forestgreen', label = strLabel)


# ------------------------------

plt.figure(12)
plt.scatter( x1_1d[ff0ns1], x2_1d[ff0ns1], alpha = 0.4 , s = 10, facecolors='darkred', edgecolors='none', label = r'$n_{ff} = 0$ \& $n_{str} > 1$')
plt.legend()

# ------------------------------


plt.figure(22)
plt.scatter( x1_1d[ff1ns1], x2_1d[ff1ns1], alpha = 0.4 , s = 10, facecolors='navy', edgecolors='none',  label = r'$n_{ff} > 0$ \& $n_{str} > 1$')
plt.legend()


# ------------------------------


plt.figure(figsize = (8,8))

plt.scatter( x1_1d[ff0ns0], x2_1d[ff0ns0], alpha = 0.3 , s = 10, facecolors='forestgreen', edgecolors='none', label = r'$n_{ff} = 0$ \& $n_{str} = 1$')
plt.scatter( x1_1d[ff0ns1], x2_1d[ff0ns1], alpha = 0.4 , s = 10, facecolors='darkred', edgecolors='none', label = r'$n_{ff} = 0$ \& $n_{str} > 1$')
plt.scatter( x1_1d[ff1ns1], x2_1d[ff1ns1], alpha = 0.4 , s = 10, facecolors='navy', edgecolors='none',  label = r'$n_{ff} > 0$ \& $n_{str} > 1$')
plt.scatter( x1_1d[ff1ns0], x2_1d[ff1ns0], alpha = 0.4 , s = 10, facecolors='k', edgecolors='none', label = r'$n_{ff} > 0$ \& $n_{str} = 1$')

nstream_2d = nstream[2,:,:]   ### ???
nstream_2d = (nstream_2d == 1)
cmap = colors.ListedColormap(['k', 'gray'])
bounds= np.array([0, 1]) + 0.5
norm = colors.BoundaryNorm(bounds, cmap.N)
#im2 = plt.imshow(nstream_2d,  alpha = 0.5, cmap = cmap , norm = norm)
plt.imshow(nstream_2d,  alpha = 0.7,  cmap = cmap )


plt.xlim(0,L)
plt.ylim(0,L)
plt.xlabel(r" $h^{-1} Mpc$")
plt.ylabel(r" $h^{-1} Mpc$")
plt.legend(bbox_to_anchor=(1.05, 1.05),  markerscale=4., scatterpoints=3)
plt.tight_layout()
plt.savefig('plots/ff_nst_conditions1.pdf', bbox_inches='tight')


# ------------------------------


from matplotlib import colors
ybeg = zbeg = 0
yend = zend = nGr - 1


f, ax = plt.subplots( 2,2, figsize = (12,12))
f.subplots_adjust( hspace = 0.05, wspace = 0.05)
#f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)


yticks = zticks = np.array([10, 20, 30, 40, 50])/50.

plt.sca(ax[0,0]) 
#plt.gca().set_yticks(yticks)
#plt.gca().set_xticks([])
plt.ylabel(r" $h^{-1} Mpc$")




nstream_2d_a = nstream[sliceNo,ybeg:yend,zbeg:zend]

#nstream_2d_a = nstream[slice_noX-2:slice_noX+2,ybeg:yend,zbeg:zend]

nstream_2d_a[nstream_2d_a >= 7] = -1

nstream_2d_a[nstream_2d_a ==  1] = 0 #0 -- Void
#nstream_2d_a[nstream_2d_a >=  7] = 100 #2 -- Filament
nstream_2d_a[nstream_2d_a >=  3] = 1 #1 -- Wall
nstream_2d_a[nstream_2d_a == -1] = 3 #3 -- Halo

#nstream_2d_a = np.sum(nstream_2d_a, axis = 0)

#nstream_2d = np.where(nstream[slice_noX,ybeg:yend,zbeg:zend] > 1, 1, 0)
cmap = colors.ListedColormap(['white', 'darkgreen', 'navy'])
bounds= np.array([0, 1]) + 0.5
norm = colors.BoundaryNorm(bounds, cmap.N)
#im2 = plt.imshow(nstream_2d,  alpha = 0.5, cmap = cmap , norm = norm)

im3 = plt.imshow(nstream_2d_a.T,  origin='lower', alpha =0.3, cmap = cmap, label = r'$n_{str}(x)$')
#im3.set_extent([ ybeg*L/size_fact , yend*L/size_fact, zbeg*L/size_fact, zend*L/size_fact])

#plt.xlim( ybeg*L/size_fact , yend*L/size_fact) 
#plt.ylim( zbeg*L/size_fact, zend*L/size_fact)
plt.xlim( ybeg , yend) 
plt.ylim( zbeg, zend)

#plt.x
plt.legend()

#plt.subplot(2,2,2)
plt.sca(ax[0,1])
plt.gca().set_xticks([])
#plt.gca().set_yticks([])

#l3_slice = l3[slice_noX,ybeg:yend,zbeg:zend]
#l3binary = np.where(  ( (l3_slice > 0) &  (nstream_2d_a > 0) ) , 1, 0)
#cmap = colors.ListedColormap(['white', 'black'])
#img2 = ax[0,1].imshow(l3binary, cmap = cmap)
#img2.set_extent([ ybeg*L/size_fact , yend*L/size_fact, zbeg*L/size_fact, zend*L/size_fact])
#img2 = plt.imshow(l3_slice, alpha = 0.3, cmap = 'prism')
#

img2 = plt.scatter( x1_1d[ff0ns0], x2_1d[ff0ns0], alpha = 0.2 , s = 10, facecolors='forestgreen', edgecolors='none', label = r'$n_{ff} = 0$ \& $n_{str} = 1$')
#img2.set_extent([ ybeg*L/size_fact , yend*L/size_fact, zbeg*L/size_fact, zend*L/size_fact])
#
plt.scatter( x1_1d[ff1ns0], x2_1d[ff1ns0], alpha = 0.5 , s = 10, facecolors='k', edgecolors='none', label = r'$n_{ff} > 0$ \& $n_{str} = 1$')


plt.xlim( ybeg*L/size_fact , yend*L/size_fact) 
plt.ylim( zbeg*L/size_fact, zend*L/size_fact)

plt.legend()


#plt.subplot(2,2,3)
plt.sca(ax[1, 0])
plt.gca().set_xticks(zticks)
plt.gca().set_yticks(yticks)
plt.xlabel(r" $h^{-1} Mpc$")
plt.ylabel(r" $h^{-1} Mpc$")

#nstream_2d = nstream[slice_noX,ybeg:yend,zbeg:zend]
#nstream_2d = np.where(nstream_2d_a > 0,  1 , 0)

#im2 = ax[1, 0].imshow(nstream_2d,  alpha = 0.25, cmap = 'cubehelix_r' )
im2 = plt.scatter( x1_1d[ff0ns1], x2_1d[ff0ns1], alpha = 0.8 , s = 10, facecolors='darkred', edgecolors='none', label = r'$n_{ff} = 0$ \& $n_{str} > 1$')

plt.xlim( ybeg*L/size_fact , yend*L/size_fact) 
plt.ylim( zbeg*L/size_fact, zend*L/size_fact)

plt.legend()
#plt.subplot(2, 2, 4)
plt.sca(ax[1, 1])
plt.gca().set_xticks(zticks)
plt.gca().set_yticks([])
plt.xlabel(r" $h^{-1} Mpc$")

#im3 = ax[1, 1].imshow(nstream_2d,  alpha = 0.15, cmap = 'cubehelix_r' )
plt.scatter( x1_1d[ff1ns1], x2_1d[ff1ns1], alpha = 0.5 , s = 5, c = flip_1d[ff1ns1], edgecolors='none',  label = r'$n_{ff} > 0$ \& $n_{str} > 1$', cmap = plt.get_cmap('Paired'), vmin=1, vmax=15)

plt.xlim( ybeg*L/size_fact , yend*L/size_fact) 
plt.ylim( zbeg*L/size_fact, zend*L/size_fact)

plt.legend()

plt.savefig('plots/ff_nstCond.pdf', bbox_inches='tight')

plt.show()

# ------------------------------


plt.figure(120)
plt.scatter( x1_1d[ff1ns1], x2_1d[ff1ns1], alpha = 0.05 , s = 20, c = flip_1d[ff1ns1], edgecolors='none',  label = r'$n_{ff} > 0$ \& $n_{str} > 1$', cmap = plt.get_cmap('Set1'), vmin=1, vmax=10)

plt.xlim( ybeg*L/size_fact , yend*L/size_fact) 
plt.ylim( zbeg*L/size_fact, zend*L/size_fact)

plt.legend()

plt.savefig('plots/ff1234_nst1.pdf', bbox_inches='tight')


# ------------------------------


plt.figure(figsize = (8,8))

plt.scatter( x1_1d[ff0ns0], x2_1d[ff0ns0], alpha = 0.3 , s = 6, facecolors='k', edgecolors='none', label = r'$n_{ff} = 0$ \& $n_{str} = 1$')
plt.scatter( x1_1d[ff0ns1], x2_1d[ff0ns1], alpha = 0.4 , s = 6, facecolors='k', edgecolors='none', label = r'$n_{ff} = 0$ \& $n_{str} > 1$')
plt.scatter( x1_1d[ff1ns1], x2_1d[ff1ns1], alpha = 0.4 , s = 6, facecolors='k', edgecolors='none',  label = r'$n_{ff} > 0$ \& $n_{str} > 1$')
plt.scatter( x1_1d[ff1ns0], x2_1d[ff1ns0], alpha = 0.4 , s = 6, facecolors='k', edgecolors='none', label = r'$n_{ff} > 0$ \& $n_{str} = 1$')

nstream_2d = nstream[2,:,:]
nstream_2d = (nstream_2d == 1)
cmap = colors.ListedColormap(['k', 'gray'])
bounds= np.array([0, 1]) + 0.5
norm = colors.BoundaryNorm(bounds, cmap.N)
#im2 = plt.imshow(nstream_2d,  alpha = 0.5, cmap = cmap , norm = norm)
plt.imshow(nstream_2d,  alpha = 0.7,  cmap = cmap )


plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel(r" $h^{-1} Mpc$")
plt.ylabel(r" $h^{-1} Mpc$")
#plt.legend(bbox_to_anchor=(1.05, 1.05),  markerscale=4., scatterpoints=3)
plt.tight_layout()
plt.savefig('plots/allPart.png', bbox_inches='tight')


# ------------------------------



f, ax = plt.subplots( 1,1, figsize = (12,12))
f.subplots_adjust( hspace = 0.05, wspace = 0.05)

#plt.sca(ax[0])
#plt.set_xticks(zticks)
#plt.set_yticks(yticks)
zticks = 100*[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
ax.set_xticklabels(zticks)
ax.set_yticklabels(zticks)
plt.xlabel(r" $h^{-1} Mpc$")
plt.ylabel(r" $h^{-1} Mpc$")

#for nth in [1, 5, 15, 25]:
nstream_2dt = nstream[sliceNo,:,:]

from scipy.ndimage.filters import gaussian_filter
nstream_2dt = gaussian_filter(nstream_2dt, sigma= 2.5)

plt.contour(nstream_2dt,  [1, 3, 30], origin='lower', linewidths = 5,  colors=('k', 'r', 'g') )
plt.tight_layout()
plt.show()
plt.savefig('plots/thresholds.png', bbox_inches='tight')

# ------------------------------
