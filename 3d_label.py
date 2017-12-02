# http://comp-phys.net/2013/03/25/working-with-percolation-clusters-in-python/
import numpy as np
from pylab import *
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or$
from scipy.ndimage import measurements
np.set_printoptions(precision = 1)
from matplotlib import colors
import matplotlib.pylab as plt

import SetPub
SetPub.set_pub()



# Bounding box
def slice(area):
    sliced = measurements.find_objects(area == area.max())
    if(len(sliced) > 0):
        sliceX = sliced[0][1]
        sliceY = sliced[0][0]
        plotxlim=im.axes.get_xlim()
        plotylim=im.axes.get_ylim()
        plot([sliceX.start, sliceX.start, sliceX.stop, sliceX.stop, sliceX.start], \
                        [sliceY.start, sliceY.stop, sliceY.stop, sliceY.start, sliceY.start], \
                        color="white")
        xlim(plotxlim)
        ylim(plotylim)
        
def plotfaces(img3d):
    
    f, ax = plt.subplots(2,3, figsize = (20,13.8),  sharey = True, sharex=True)
    f.subplots_adjust( wspace = 0.08, hspace = 0.02, top = 0.95, bottom = 0.08, right = 0.99, left = 0.05)

    ticksLoc = np.linspace(0, img3d.shape[0], 6)
    ticksLabel = [ '', '20', '40', '60', '80', '' ]
    
    cmap = colors.ListedColormap(['red', 'gray', 'white'])
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
    imshow(img3d[size_fact-1,:,:], origin='lower', interpolation='nearest', cmap = cmap, norm = norm)
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
    #plt.xticks([])
    #plt.yticks([])
    
    
    #subplot(2,3,5)
    plt.sca(ax[1,1])
    imshow(img3d[:,size_fact-1,:], origin='lower', interpolation='nearest', cmap = cmap, norm = norm)
    #colorbar()
    #title("4")
    plt.xlabel(r" $h^{-1} Mpc$")
    #plt.minorticks_on()
    plt.xticks( ticksLoc, ticksLabel  )
    #plt.yticks([])  
    #plt.minorticks_on()  

    #subplot(2,3,3)
    plt.sca(ax[0,2])
    imshow(img3d[:,:,0], origin='lower', interpolation='nearest', cmap = cmap, norm = norm)
    #colorbar()
    title("Z")
    #plt.minorticks_on()
    plt.xticks([])    #plt.gca().set_yticks([])
    #plt.yticks([])  
    

    #subplot(2,3,6)
    plt.sca(ax[1,2])
    imshow(img3d[:,:,size_fact-1], origin='lower', interpolation='nearest', cmap = cmap, norm = norm)
    #colorbar()
    #title("6")
    #show()
    plt.xlabel(r" $h^{-1} Mpc$")
    #plt.minorticks_on()
    plt.xticks( ticksLoc, ticksLabel  )
    #plt.yticks([])
    


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

# field3d = flip # nstream
# minCut = 0  # 1  # field3d.min()
# maxCut = 0 # 51  # field3d.max()
# step = -1 # -2
# OutfileName = 'npy/ff_frESfr1LargestLabel_032'


field3d = nstream # nstream
minCut = 1   # field3d.min()
maxCut = 1   # field3d.max()
step = -2 #
OutfileName = 'npy/nstrfrESfr1LargestLabel_032'



# fileIn = 'npy/numField_032_'+str(refFactor)+'.npy'
# nstream = np.load(fileIn)



# L = np.shape(nstream)[0]
r = field3d
#r = rand(L,L,L)
cutoff = maxCut
z = r == cutoff

#
#figure(1, figsize=(16,5))
#title("Original")
#plotfaces(z)

 


struct= struct = np.reshape(np.ones(27), (3,3,3))
#struct[0,0,0] = struct[0, 0 ,-1] = struct[0,-1,0] = struct[0,-1,-1] = 0
#struct[-1,0,0] = struct[-1, 0 , -1] = struct[-1,-1,0] = struct[-1,-1,-1] = 0

lw, num = measurements.label(z, structure = struct)
#lw, num = measurements.label(z)


def Replace1D(BigArray, oldID, newID):    # Faster
    # Replace original particle labels ( with gaps) by continuous numbers.
#http://stackoverflow.com/questions/29407945/find-and-replace-multiple-values-in-python   
     
    arr = np.empty(BigArray.max()+1, dtype = newID.dtype)
    arr[oldID] = newID
    return arr[BigArray]  

#import sys
#sys.exit()

# Calculate areas
gridOnes = np.ones_like(z)
area = measurements.sum(gridOnes, lw, index= np.unique(lw)[1:])


#argSortArea = np.unique(lw)[1:][np.argsort(area)]
#replacelw = np.append(0, argSortArea)
replacelw = np.append(0, np.sort(area)[::-1])

areaImg = replacelw[lw]

#figure(2, figsize=(16,5))
plotfaces(areaImg)


print
print '-----------------------------'
print
print 'Resolution of multi-stream: %d'%refFactor
print
print '#grids in voids: %d'%(np.sum(z)) , 'FF_V: %.3f %%'%( 100.*np.sum(z)/ (nGr*refFactor)**3. )
print
print '#isolated segments: %d '%num
print
print '#grids in largest voids: %d', np.sort(area)[::-1][0:5]
print
print 'FFx/FF_V: ', (100.*np.sort(area)[::-1][0:5]/np.sum(z) )

plt.savefig('plots/faces_'+str(refFactor)+'.pdf')




plt.show()


## Commented below to avoid sys.exit()

# import sys
# sys.exit()
#
# plt.show()
#
# import toParaview as toPara
# toPara.StructuredScalar(areaImg, '../VTI_numpy_files/Voids_'+str(refFactor), 0, nGr*refFactor)
#
#
#
#
#
#
# figure(3)
# #title("Clusters by area")
# subplot(2,3,1)
# im = imshow(areaImg[0,:,:], origin='lower', interpolation='nearest') # show image clusters as labeled by a shuffled lw
# slice(areaImg[0,:,:])
# autoscale(False)
# colorbar()
# #title("1")
# subplot(2,3,4)
# im = imshow(areaImg[L-1,:,:], origin='lower', interpolation='nearest')
# slice(areaImg[L-1,:,:])
# colorbar()
# #title("2")
# subplot(2,3,2)
# im = imshow(areaImg[:,0,:], origin='lower', interpolation='nearest')
# slice(areaImg[:,0,:])
# colorbar()
# #title("3")
# subplot(2,3,5)
# im = imshow(areaImg[:,L-1,:], origin='lower', interpolation='nearest')
# slice(areaImg[:,L-1,:])
# colorbar()
# #title("4")
# subplot(2,3,3)
# im = imshow(areaImg[:,:,0], origin='lower', interpolation='nearest')
# slice(areaImg[:,:,0])
# colorbar()
# #title("5")
# subplot(2,3,6)
# im = imshow(areaImg[:,:,L-1], origin='lower', interpolation='nearest')
# slice(areaImg[:,:,L-1])
# colorbar()
# #title("6")
#
#
# #figure(4, figsize=(16,5))
# #title("tr")
# #plotfaces(areaImg)
# #slice(areaImg[0,:,:])
#
# show()
# tight_layout()
