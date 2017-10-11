# -*- coding: utf-8 -*-
import numpy as np
import time
import sys
import scipy.ndimage as ndi
import numpy.ma as ma
import toParaview as toPara
import matplotlib.pylab as plt

"""

VOID:      λth > λ1 > λ2 > λ3         All three eigenvectors of the shear tensor are expanding
SHEET:     λ1 > λth ; λ2 ,λ3 < λth    Collapse along one axis, expansion along the two
FILAMENT:  λ1, λ2 > λth ; λ3 < λth    Collapse along two axes, expansion along the other 
KNOT:      λ1 > λ2 > λ3 > λth         All three eigenvectors of the shear tensor are collapsing

λth = 0.0
Can find λth by using void fraction from nstream = 1 regions. 

"""
#------------------------------------------------------------------------------
def RemoveDuplicateRow(a):
    #http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array/
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _,idx = np.unique(b, return_index=True)
    unique_a = np.unique(b).view(a.dtype).reshape(-1, a.shape[1])
    return unique_a    

#------------------------------------------------------------------------------
#def Replace1D(BigArray, oldID, newID):    #Expensive  
#    # Replace original particle labels ( with gaps) by continuous numbers. 
#    return np.array([newID[oldID.index(x)] if any(x==oldID) else x for x in BigArray])

def Replace1D(BigArray, oldID, newID):    # Faster
    # Replace original particle labels ( with gaps) by continuous numbers.
#http://stackoverflow.com/questions/29407945/find-and-replace-multiple-values-in-python   
     
    arr = np.empty(BigArray.max()+1, dtype = newID.dtype)
    arr[oldID] = newID
    return arr[BigArray]  

#------------------------------------------------------------------------------        
def ParticleLabel(regionLabels, x1d, y1d, z1d):  
# Finds particles in each labeled region, tags each particle with ID
    x0 = np.mod(np.ceil(x1d*size_fact/L), size_fact).astype(int)
    x1 = np.mod(np.ceil(y1d*size_fact/L), size_fact).astype(int)
    x2 = np.mod(np.ceil(z1d*size_fact/L), size_fact).astype(int)

    xlabel_ceil = labels[x0,x1,x2]

    x0 = np.mod(np.floor(x1d*size_fact/L), size_fact).astype(int)
    x1 = np.mod(np.floor(y1d*size_fact/L), size_fact).astype(int)
    x2 = np.mod(np.floor(z1d*size_fact/L), size_fact).astype(int)

    xlabel_floor= labels[x0,x1,x2]

    x0 = np.mod(np.round(x1d*size_fact/L), size_fact).astype(int)
    x1 = np.mod(np.round(y1d*size_fact/L), size_fact).astype(int)
    x2 = np.mod(np.round(z1d*size_fact/L), size_fact).astype(int)

    xlabel_round = labels[x0,x1,x2]

    xlabel_all = np.vstack([xlabel_ceil, xlabel_floor , xlabel_round])

    xlabel = np.max(xlabel_all, axis = 0)
    
    #xlabel = xlabel_round   # REMOVE
    
    return xlabel    # Returns 1D labels corresponding to particles

#------------------------------------------------------------------------------
def MaskedKeep(Array3D, toKeep1D):
    #In Array3D, Replaces elements not in toKeep1D by 0
    maskedRandomArray = ma.MaskedArray(Array3D, np.in1d(Array3D, toKeep1D, invert = True), fill_value = 0)
    return maskedRandomArray.filled()

#------------------------------------------------------------------------------    
def MaskedRemove(Array3D, toRemove1D):
    #In Array3D, Replaces elements in toRemove1D by 0
    maskedRandomArray = ma.MaskedArray(Array3D, np.in1d(Array3D, toRemove1D), fill_value = 0)
    return maskedRandomArray.filled()
    
#------------------------------------------------------------------------------ 
def LabelCutStr(labels3d, nstrCutMax, nstrCutMin):
    
    labels1d = np.unique(labels3d)[1:]
    
    print '---------------------------------' 
    print 'Max nstream threshold', nstrCutMax
    print 'Min nstream threshold', nstrCutMin
    print '#Haloes before nstreams-cut:', labels1d.shape

    
    maxnstrEachBlob = np.array(ndi.maximum(nstream, labels=labels3d, index= labels1d))
    c1 = (maxnstrEachBlob < nstrCutMax)   #max(nstream) = 1  ( entire region in void)

    minnstrEachBlob = np.array(ndi.minimum(nstream, labels=labels3d, index= labels1d))
    c2 = (minnstrEachBlob < nstrCutMin)   #max(nstream) = 1  ( entire region in void)

    MaskOutCondition0 = np.where( c1 | c2 )
    maskingValues = labels1d[MaskOutCondition0]
    labels3d_out = MaskedRemove(labels3d, maskingValues)
    
    print '#Haloes after nstreams-cut nstreams:', (np.unique(labels3d_out)[1:]).shape
    


#    maxnstrEachBlob = np.array(ndi.maximum(nstream, labels=labels3d_out, index= (np.unique(labels3d_out)[1:])))
#    minnstrEachBlob = np.array(ndi.minimum(nstream, labels=labels3d_out, index= (np.unique(labels3d_out)[1:])))
#
#    print 'Streams min(min) ', minnstrEachBlob.min()
#    print 'Streams min(max) ', maxnstrEachBlob.min()
                      
                                
    return labels3d_out

#------------------------------------------------------------------------------     
def LabelCutVol(labels3d, volCut):
    
    labels1d = np.unique(labels3d)[1:]
    
    print '---------------------------------' 
    print 'volume threshold', volCut
    print '#Haloes before Volume-cut:', labels1d.shape


    gridOnes = np.ones_like(nstream)
    gridEachBlob = ndi.measurements.sum(gridOnes, labels=labels3d ,index= labels1d)
    c2 = (gridEachBlob < volCut )        #total volume < 8 grids points 




    MaskOutCondition0 = np.where( c2 )
    maskingValues = labels1d[MaskOutCondition0]
    labels3d_out = MaskedRemove(labels3d, maskingValues)
    
    print '#Haloes after Volume-cut:', (np.unique(labels3d_out)[1:]).shape
    
    #print '#Haloes   ',  np.array(np.unique((labels3d_out), return_counts=True))[:,1:].shape
    #print 'Grids (min) ', np.array(np.unique((labels3d_out), return_counts=True))[1,1:].min()
    #    
    #gridEachBlob = ndi.measurements.sum(gridOnes, labels=labels3d_out ,index= (np.unique(labels3d_out)[1:]))
    #print 'GridsEachBlob (min) ', gridEachBlob.min()
       
    return labels3d_out
    
#------------------------------------------------------------------------------ 
def LabelCutDens(labels3d, densCut):
    
    labels1d = np.unique(labels3d)[1:]
    
    print '---------------------------------' 
    print 'density threshold', densCut
    print '#Haloes before density-cut:', labels1d.shape

    
    gridOnes = np.ones_like(nstream)
    gridEachBlob = ndi.measurements.sum(gridOnes, labels=labels3d ,index= labels1d)

    massEachBlob = ndi.measurements.sum(macro, labels=labels3d, index= labels1d) #nParticles
    densEachBlob = massEachBlob/(gridEachBlob*(L/size_fact)**3.)    #nParticles/Mpc^3
    
    
    denBackground = (nGr)**3/(L**3)          # nParticles/Mpc^3
    H = 0.7 * 100 * (10**3) / 3.086e+22  #s−1,  (km/Mpc) is converted
    G = 6.674e-11   # N⋅m2/kg2 
    denCriticalSI = 3.*(H**2.)/(8.*np.pi*G)   #kg/m^3
    denCriticalMsolMpc = denCriticalSI*((0.7*3.086e+22)**3) / 1.988e30  # in Msol/(h^-1 Mpc**3)
    denCritical = denCriticalMsolMpc/m_particle     # nParticles/Mpc^3
    
    dentoVirial = densEachBlob/denBackground
    #dentoVirial = densEachBlob/denCritical
    
    c3 = dentoVirial < densCut
#----------------- 
    MaskOutCondition0 = np.where( c3 )
    maskingValues = labels1d[MaskOutCondition0]
    labels3d_out = MaskedRemove(labels3d, maskingValues)
    
    print '#Haloes after density-cut:', (np.unique(labels3d_out)[1:]).shape
        
    return labels3d_out  
    
    

def LabelCut(labels3d, nstrCut, massCut, volCut, densCut):
    
    labels1d = np.unique(labels3d)[1:]
    
    print '#Haloes before label-cut:', labels1d.shape

    
    maxnstrEachBlob = np.array(ndi.maximum(nstream, labels=labels3d, index= labels1d))
    c1 = (maxnstrEachBlob < nstrCut)   #max(nstream) = 1  ( entire region in void)

    gridOnes = np.ones_like(nstream)
    gridEachBlob = ndi.measurements.sum(gridOnes, labels=labels3d ,index= labels1d)
    c2 = (gridEachBlob < volCut )        #total volume < 8 grids points 

    massEachBlob = ndi.measurements.sum(macro, labels=labels3d, index= labels1d)*m_particle
    densEachBlob = massEachBlob/(gridEachBlob*(L/size_fact)**3)
    denBackground = m_particle*(nGr)**3/(L**3)
    dentoVirial = densEachBlob/denBackground
    c3 = dentoVirial < densCut


    MaskOutCondition0 = np.where( c1 | c2 | c3 )
    maskingValues = labels1d[MaskOutCondition0]
    labels3d_out = MaskedRemove(labels3d, maskingValues)
    
    print '#Haloes after label-cut:', (np.unique(labels3d_out)[1:]).shape
       
    return labels3d_out

#------------------------------------------------------------------------------
def massParticleCut(labels3d, massCut):
    particleLabel1d = ParticleLabel(labels3d, x0_1d, x1_1d, x2_1d)
    
    print 'mass threshold', massCut
    print '#Haloes before mass-cut:', (np.unique(labels3d)[1:]).shape
    
    freqlabel = np.bincount(particleLabel1d)
    non0mass = np.nonzero(freqlabel)[0]
    massLabel = np.vstack( (non0mass, freqlabel[non0mass]) ).T
    massCutlabel = massLabel[massLabel[:,1] >= massCut][:,0]
    labels3d_out = MaskedKeep(labels3d, massCutlabel[1:])
    
    print '#Haloes after mass-cut:', (np.unique(labels3d_out)[1:]).shape
    
    return labels3d_out


#------------------------------------------------------------------------------
    
def VoidParticleUnbind(regionLabels, x3d, y3d, z3d):  
### #remove particles if they belong to nstream = 1 
    x0 = np.mod(np.ceil(x3d*size_fact/L), size_fact).astype(int)
    x1 = np.mod(np.ceil(y3d*size_fact/L), size_fact).astype(int)
    x2 = np.mod(np.ceil(z3d*size_fact/L), size_fact).astype(int)

    regionLabels[nstream[x0, x1, x2] == 1] = 0

    x0 = np.mod(np.floor(x3d*size_fact/L), size_fact).astype(int)
    x1 = np.mod(np.floor(y3d*size_fact/L), size_fact).astype(int)
    x2 = np.mod(np.floor(z3d*size_fact/L), size_fact).astype(int)
#
    regionLabels[nstream[x0, x1, x2] == 1] = 0

    x0 = np.mod(np.round(x3d*size_fact/L), size_fact).astype(int)
    x1 = np.mod(np.round(y3d*size_fact/L), size_fact).astype(int)
    x2 = np.mod(np.round(z3d*size_fact/L), size_fact).astype(int)

    regionLabels[nstream[x0, x1, x2] == 1] = 0
    
    regionLabels[nstream == 1] = 0

        
    return regionLabels
    

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

startT = time.time()


L = 1.
nGr = 128
refFactor = 1
Dn = 60      # 60 for all except 100Mpc-512 (160)
size_fact = nGr*refFactor

lBox = str(int(L))+'Mpc'
#--------#--------#--------#--------#--------
sig = 1.0


l3Cut = 0.0
densCut = 0.0
massCut = 8.0

nstrCutMax = 7
nstrCutMin = 3

volCut = 0

#m_particle = 0.4065E+11
m_particle = 9.94e+09
#--------#--------#--------#--------#--------

print
print 'Smoothening sigma: ', sig
print 'lambda_3 threshold ', l3Cut
#--------x0, x1, x2 -------------------

dLoad = './npy/'

x0_3d = np.load(dLoad+'x0'+'_snap_051.npy') 
x1_3d = np.load(dLoad+'x1'+'_snap_051.npy')
x2_3d = np.load(dLoad+'x2'+'_snap_051.npy')


x0_1d = np.ravel(x0_3d)
x1_1d = np.ravel(x1_3d)
x2_1d = np.ravel(x2_3d)



file_nstr = './npy/numField_051_'+str(refFactor)+'.npy'
nstream = np.load(file_nstr).astype(np.float64)

file_eval = './npy/Evals3_051.npy'
l = np.load(file_eval)

#l3 = np.sum(l, axis = 3)
l3 = l[:,:,:, 2]



#macro = np.load('npy/DensityCIC_051.npy')


struct = np.reshape(np.ones(27), (3,3,3))


z = (  (l3 > l3Cut) & (nstream > 1) )
#z = (  (l3 > l3Cut)  )

labels, num = ndi.label(z, structure = struct)   #struct = array structure to include diagonal
#labels, num = ndi.label(z)    # What about Edges ??  Periodic BC?



# ------------Order ??? cutoff matters!!!  ----------------
"""
labels -> input
labels1 -> intermediate
labels0 -> output
"""

labels1 = LabelCutStr(labels, nstrCutMax, nstrCutMin)      #nstream cut 

#labels1a = labels1
#labels1a = VoidParticleUnbind(labels1, x0_3d, x1_3d, x2_3d)   # Unbind particles in nstr = 1 

labels2 = LabelCutVol(labels1, volCut)       #Volume cut
#labels3 = LabelCutDens(labels2, densCut)     #Density cut

labels3 = labels2
labels0 = massParticleCut(labels3, massCut)  #Mass cut 

#labels0 = LabelCut(labels1, nstrCutMax, massCut, volCut, densCut)  #nstr, density, volume cut


    

# ---------------------Relabeling particles in haloes starts----------
#------------------------------ By Halo mass -------------------------
    
particleLabel1d = ParticleLabel(labels0, x0_1d, x1_1d, x2_1d)
labelidx = np.where(np.in1d( particleLabel1d , np.unique(labels0)[1:] ))[0]
x0_labeled = x0_1d[labelidx]
x1_labeled = x1_1d[labelidx]
x2_labeled = x2_1d[labelidx]
regionLabel = particleLabel1d[labelidx]

freqlabel = np.bincount(regionLabel)
non0mass = np.nonzero(freqlabel)[0]
massLabel = np.vstack( (non0mass, freqlabel[non0mass]) ).T




sortnPart = np.argsort(massLabel[:,1])[::-1]
sortednPart = non0mass[sortnPart]
          
haloID = Replace1D(regionLabel, list(sortednPart), (np.arange(np.size(sortednPart))+1) )


import sys
sys.exit()

#----------------------------------SAVING FILES-------------------------------

np.save('npy/particlesL3Halo', np.vstack((x0_labeled, x1_labeled, x2_labeled, haloID)))
np.save('npy/labels0', labels0)
    
#toPara.Points1d(x0_labeled*size_fact/L , x1_labeled*size_fact/L , x2_labeled*size_fact/L , haloID, 'vti/xLabeledL3_051')
toPara.StructuredScalar(labels0, 'vti/l3_051_'+str(refFactor), 0, size_fact)
toPara.StructuredScalar(nstream, 'vti/nstr_051_'+str(refFactor), 0, size_fact)

import sys
sys.exit()

#----------------------------------PLOTTING & STATISTICS-------------------------------

#---------------------- Make use of Class ---------------

def StatsLabels(labels3d): 
    
    #---------------- Sorting by Volume ---------------
    #gridOnes = np.ones_like(nstream)
    #gridEachBlob = ndi.measurements.sum(gridOnes, labels=labels0, index= np.unique(labels0)[1:])
    #labelsSort = np.argsort(gridEachBlob)[::-1]   # Sorting by size large -> small  ( 0 included )
    #label1d= np.unique(labels0)[1:]#[labelsSort]
    
    label1d = np.unique(labels3d)[1:]
    
    #maxl3InLabel = np.array(ndi.maximum( l3, labels= labels3d, index= label1d ))
    #PDFplot(maxl3InLabel, r'max($\lambda_3$) in Halo')
    
    
    maxPosInLabel = np.array(ndi.maximum_position( l3, labels= labels3d,  index= label1d ))
    comLabel = ndi.measurements.center_of_mass(macro, labels= labels3d, index= label1d )
    
    massEachBlob = ndi.measurements.sum(macro, labels= labels3d, index= label1d )*m_particle  # might not be exact
    
    #strEachBlob = ndi.measurements.sum(nstream, labels= labels3d, index= label1d )
    
    maxnstrEachBlob = np.array(ndi.maximum(nstream, labels=labels3d, index= label1d))
    minnstrEachBlob = np.array(ndi.minimum(nstream, labels=labels3d, index= label1d))
    
    PDFplot(maxnstrEachBlob,  r'max($n_{str}$) in Halo')
    PDFplot(minnstrEachBlob,  r'min($n_{str}$) in Halo')
    
    gridOnes = np.ones_like(nstream)       #  RefinedGrid
    gridEachBlob = ndi.measurements.sum(gridOnes, labels= labels3d , index= label1d )
    
    PDFplot(gridEachBlob,  'Grid-points in each Halo')
    
    
def PDFplot(array1d, strLabel):
    
    print '--------------------------------------------'
    
    print strLabel+' Statistics'
    print '#Haloes:     ', np.shape(array1d)
    print '        Min:  %4.2e' %(array1d.min())
    print '        Max:  %4.2e' %(array1d.max())
    print '        Mean: %4.2e' %(array1d.mean())
    print '        std  : %4.2e' %(array1d.std())
    print '        Median %4.2e' %np.median(array1d)
    print '--------------------------------------------'
    
    plt.figure(figsize = (8,6))
    
    xlim1 = np.min(array1d)
    xlim2 = np.max(array1d)
    #
    
    if (strLabel == 'Grid-points in each Halo'):
        nbins = 90
        y,binEdges = np.histogram( array1d
        , bins = np.linspace((xlim1), (xlim2), nbins), density= False)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    
    else:
        nbins = 100
        y,binEdges = np.histogram( array1d
        , bins = np.logspace(np.log10(xlim1), np.log10(xlim2), nbins), density= False)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    #
    
    #
    #y,binEdges = np.histogram( array1d
    #, bins = np.linspace((xlim1), (xlim2), nbins), density= False)
    #bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    #
    #
    ###plt.plot(bincenters, y, 'ko', lw = 2, label = strLabel)
    plt.errorbar(bincenters, y, 
    yerr = np.sqrt(y),  fmt='ko',  elinewidth=2 , lw = 2, alpha =  0.6,
    label = strLabel)
    
    plt.plot(bincenters, y, 'ko', lw = 2)
    plt.minorticks_on()

    #errorfill(bincenters, y, np.sqrt(y), color='b', alpha_fill=0.3, ax=None)

    plt.xlim( xlim1 ,xlim2)
    #plt.xlim(plt.xlim()[0], xlim2)
    #plt.gca().set_xlim(right=xlim2)
    plt.ylim(1, )

    plt.yscale('log')
    if (strLabel != 'Grid-points in each Halo'): plt.xscale('log')
    plt.xlabel(strLabel)
    plt.ylabel("PDF") 
    #plt.legend(loc = "upper right")
    
    plt.savefig('plots/'+strLabel+'.pdf', bbox_inches='tight')
def errorfill(x, y, yerr, color=None, alpha_fill=0.2, ax=None):
    
    #https://tonysyu.github.io/plotting-error-bars.html
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color = color, lw = 2, label = strLabel)
    ax.plot(x, y, 'bo', label = strLabel)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)
    
        
    

StatsLabels(labels0)
PDFplot(massLabel[:,1], 'Particles in each halo')
plt.show()

#-------------------------------------------------------------

from matplotlib import colors

slice_noX = 20
ybeg = zbeg =  10
yend = zend = 68

"""
labels        l3 > 0 $ nstr > 1
labels1       nstream cut 
labels2       Volume cut            - meh
labels3       Density cut           - meh
labels0       Mass cut  - Final
"""
f, ax = plt.subplots( 2,2, figsize = (8,8))
f.subplots_adjust( hspace = 0.02, wspace = 0.05)
#f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)


yticks = zticks = [0.1,0.2,0.3,0.4, 0.5]


#plt.subplot(2,2,1)
plt.sca(ax[0,0]) 


l3_slice = l3[slice_noX,ybeg:yend,zbeg:zend]
l3binary = np.where(l3_slice > 0, 1, 0)

cmap = colors.ListedColormap(['white', 'black'])
img2 = ax[0,0].imshow(l3binary, cmap = cmap)
img2.set_extent([ ybeg*L/size_fact , yend*L/size_fact, zbeg*L/size_fact, zend*L/size_fact])

img2 = plt.imshow(l3_slice, alpha = 0.3, cmap = 'prism')

plt.gca().set_yticks(yticks)
plt.gca().set_xticks([])
#plt.minorticks_on()


plt.ylabel(r" $h^{-1} Mpc$")

img2.set_extent([ ybeg*L/size_fact , yend*L/size_fact, zbeg*L/size_fact, zend*L/size_fact])

#plt.subplot(2,2,2)
plt.sca(ax[0,1])
labels_2d = np.where( labels[slice_noX,ybeg:yend,zbeg:zend] > 0, 1, 0) 
labels1_2d = np.where( labels1[slice_noX,ybeg:yend,zbeg:zend] > 0, 2, 0)
#labels2_2d = np.where( labels2[slice_noX,ybeg:yend,zbeg:zend] > 0, 4, 0)
#labels3_2d = np.where( labels3[slice_noX,ybeg:yend,zbeg:zend] > 0, 8, 0)
#labels0_2d = np.where( labels0[slice_noX,ybeg:yend,zbeg:zend] > 0, 16, 0)
#labelEffect = labels_2d + labels1_2d + labels2_2d + labels3_2d + labels0_2d
#cmap = colors.ListedColormap(['white', 'green', 'blue', 'red'])
labelEffect = labels_2d + labels1_2d 

cmap = colors.ListedColormap(['white', 'red', 'black'])
bounds= np.unique(labelEffect)  
norm = colors.BoundaryNorm(bounds, cmap.N)
im1 = ax[0, 1].imshow( labelEffect, cmap = cmap)
im1.set_extent([ ybeg*L/size_fact , yend*L/size_fact, zbeg*L/size_fact, zend*L/size_fact])
print np.unique(labelEffect)


nstream_2d = nstream[slice_noX,ybeg:yend,zbeg:zend]

nstream_2d = np.where(nstream[slice_noX,ybeg:yend,zbeg:zend] > 1,  1 , 0)

#nstream_2d = np.where(nstream[slice_noX,ybeg:yend,zbeg:zend] > 1, 1, 0)
#cmap = colors.ListedColormap(['white', 'grey', 'red'])
#bounds= np.array([0, nstrCutMin, nstrCutMax]) + 0.5
#norm = colors.BoundaryNorm(bounds, cmap.N)
#im2 = plt.imshow(nstream_2d,  alpha = 0.5, cmap = cmap , norm = norm)

im2 = ax[0, 1].imshow(nstream_2d,  alpha = 0.3, cmap = 'cubehelix_r' )

im2.set_extent([ ybeg*L/size_fact , yend*L/size_fact, zbeg*L/size_fact, zend*L/size_fact])
#plt.xlabel(r" $h^{-1} Mpc$")
#plt.ylabel(r" $h^{-1} Mpc$")

#plt.minorticks_on()
plt.gca().set_xticks([])
plt.gca().set_yticks([])



#plt.subplot(2,2,3)
plt.sca(ax[1, 0])


labels2_binary = np.where( labels2[slice_noX,ybeg:yend,zbeg:zend] > 0, 1, 0)
cmap = colors.ListedColormap(['white', 'black'])
img2 = ax[1,0].imshow(labels2_binary, cmap = cmap)
img2.set_extent([ ybeg*L/size_fact , yend*L/size_fact, zbeg*L/size_fact, zend*L/size_fact])
plt.gca().set_xticks(zticks)
plt.gca().set_yticks(yticks)

plt.xlabel(r" $h^{-1} Mpc$")
plt.ylabel(r" $h^{-1} Mpc$")

#plt.subplot(2, 2, 4)
plt.sca(ax[1, 1])
#plt.minorticks_on()

labels3_binary = np.where( labels0[slice_noX,ybeg:yend,zbeg:zend] > 0, 1, 0)
cmap = colors.ListedColormap(['white', 'black'])
img3 = ax[1,1].imshow( labels3_binary, cmap = cmap)
img3.set_extent([ ybeg*L/size_fact , yend*L/size_fact, zbeg*L/size_fact, zend*L/size_fact])
plt.gca().set_xticks(zticks)
plt.gca().set_yticks([])
plt.xlabel(r" $h^{-1} Mpc$")


plt.savefig('plots/labels123.pdf', bbox_inches='tight')


#--------------------------------------------------------------------------------

plt.figure(12)

labels_2d = np.where( labels[slice_noX,ybeg:yend,zbeg:zend] > 0, 1, 0) 
labels1_2d = np.where( labels1[slice_noX,ybeg:yend,zbeg:zend] > 0, 2, 0)
labels2_2d = np.where( labels2[slice_noX,ybeg:yend,zbeg:zend] > 0, 4, 0)
labels3_2d = np.where( labels3[slice_noX,ybeg:yend,zbeg:zend] > 0, 8, 0)
labels0_2d = np.where( labels0[slice_noX,ybeg:yend,zbeg:zend] > 0, 16, 0)
labelEffect = labels_2d + labels1_2d + labels2_2d + labels3_2d + labels0_2d
cmap = colors.ListedColormap(['white', 'green', 'blue', 'red'])
bounds= np.unique(labelEffect) 
norm = colors.BoundaryNorm(bounds, cmap.N)
im1 = plt.imshow( labelEffect, cmap = cmap, norm = norm)
im1.set_extent([ ybeg*L/size_fact , yend*L/size_fact, zbeg*L/size_fact, zend*L/size_fact])
print np.unique(labelEffect)





nstream_2d = nstream[slice_noX,ybeg:yend,zbeg:zend]

nstream_2d = np.where(nstream[slice_noX,ybeg:yend,zbeg:zend] > 1,  1 , 0)

#nstream_2d = np.where(nstream[slice_noX,ybeg:yend,zbeg:zend] > 1, 1, 0)
#cmap = colors.ListedColormap(['white', 'grey', 'red'])
#bounds= np.array([0, nstrCutMin, nstrCutMax]) + 0.5
#norm = colors.BoundaryNorm(bounds, cmap.N)
#im2 = plt.imshow(nstream_2d,  alpha = 0.5, cmap = cmap , norm = norm)

im2 = plt.imshow(nstream_2d,  alpha = 0.3, cmap = 'cubehelix_r' )

im2.set_extent([ ybeg*L/size_fact , yend*L/size_fact, zbeg*L/size_fact, zend*L/size_fact])
plt.xlabel(r" $h^{-1} Mpc$")
plt.ylabel(r" $h^{-1} Mpc$")



plt.savefig('plots/labels.pdf', bbox_inches='tight')


#------------------------------------------------------------------------


from matplotlib import colors
slice_noX = 60
ybeg = zbeg = 10
yend = zend = 68
#fig, axes = plt.subplots(nrows=3, ncols=1)

fig, axes = plt.subplots( 1,3, figsize = (20,20))
fig.subplots_adjust( hspace = 0.02, wspace = 0.1)
#fig.subplots_adjust(bottom=0.01, right=0.01, top=0.01)

yticks = np.arange(ybeg,yend, 4)


#yticks = np.arange(ybeg*L/size_fact,yend*L/size_fact, 20*L/size_fact)
yticks = zticks = [0.1, 0.2, 0.3, 0.4]
#zticks = np.arange(ybeg*L/size_fact,yend*L/size_fact, 20*L/size_fact)
#zticks = [10,20,30,40,50]

cmap = 'bwr'
cmap = 'RdGy'
#plt.subplot(1,3,1)
plt.sca(axes[0])

img1 = plt.imshow(l[:,:,:,0][slice_noX,ybeg:yend,zbeg:zend], vmin = -6, vmax = 6, cmap = cmap)
img1.set_extent([ ybeg*L/size_fact , yend*L/size_fact, zbeg*L/size_fact, zend*L/size_fact])
plt.title(r"$\lambda_1$")
plt.gca().set_xticks(yticks)
plt.gca().set_yticks(zticks)
#plt.colorbar(img1)
plt.xlabel(r" $h^{-1} Mpc$")
cbar = plt.colorbar(orientation='horizontal',  pad = 0.1, ticks=[ -5, 0, 5])


plt.ylabel(r" $h^{-1} Mpc$")

#plt.subplot(1,3,2)
plt.sca(axes[1])

plt.title(r"$\lambda_2$")
img2 = plt.imshow(l[:,:,:,1][slice_noX,ybeg:yend,zbeg:zend],  vmin = -6, vmax = 6, cmap = cmap)
img2.set_extent([ ybeg*L/size_fact , yend*L/size_fact, zbeg*L/size_fact, zend*L/size_fact])
plt.gca().set_xticks(yticks)
plt.gca().set_yticks([])
#plt.colorbar(img2)
plt.xlabel(r" $h^{-1} Mpc$")
cbar = plt.colorbar(orientation='horizontal',  pad = 0.1, ticks=[-5, 0, 5])

#plt.subplot(1,3,3)
plt.sca(axes[2])

plt.title(r"$\lambda_3$")
img3 = plt.imshow(l[:,:,:,2][slice_noX,ybeg:yend,zbeg:zend],  vmin = -6, vmax = 6, cmap = cmap)
img3.set_extent([ ybeg*L/size_fact , yend*L/size_fact, zbeg*L/size_fact, zend*L/size_fact])
plt.gca().set_xticks(yticks)
plt.gca().set_yticks([])
#plt.colorbar(img3)
plt.xlabel(r" $h^{-1} Mpc$")
cbar = plt.colorbar(orientation='horizontal',  pad = 0.1, ticks=[-5, 0, 5])
#cbar.ax.set_xticklabels(['-30', '-15', '0', '15', '30'])


#plt.tight_layout()


#plt.colorbar(orientation='horizontal')
#fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.65, 0.35, 0.1, 0.5])
#fig.colorbar(img3, cax=cbar_ax)




plt.savefig('plots/Evals123.pdf', bbox_inches='tight')











# ----------------- min-max str in halo - combined ------------------

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
    ax.plot(x, y, color = color, lw = 2, label = strLabel)
    #ax.plot(x, y, color+'o', label = strLabel)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)
    

nbins = 15


labels3d = labels0

label1d = np.unique(labels3d)[1:]


maxnstrEachBlob = np.array(ndi.maximum(nstream, labels=labels3d, index= label1d))

array1d = maxnstrEachBlob
strLabel = r'max($n_{str}$)'

plt.figure(1233, figsize = (8,6))

xlim1 = np.min(array1d)
xlim2 = np.max(array1d)
bins = np.logspace(np.log10(xlim1), np.log10(xlim2), nbins)
#bins = np.linspace((xlim1), (xlim2), nbins)


y,binEdges = np.histogram( array1d
, bins = bins, density= False)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])



#plt.errorbar(bincenters, y, 
#yerr = np.sqrt(y),  fmt='r.', markersize='5', alpha = 0.7,  elinewidth= 2  )
#plt.plot(bincenters[y!=0], y[y!=0], 'r', alpha = 0.8, lw = 2, label = strLabel)

errorfill(bincenters[y!=0], y[y!=0], np.sqrt(y[y!=0]), color = 'navy')

#plt.xlim(binEdges[1:][0], xlim2)
#plt.ylim(1, )

#plt.yscale('log')
#if (strLabel != '#Grids in Halo'): plt.xscale('log')
#plt.xlabel(strLabel)
#plt.ylabel("pdf") 
#plt.legend(loc = "upper right")


minnstrEachBlob = np.array(ndi.median(nstream, labels=labels3d, index= label1d))
array1d = minnstrEachBlob
strLabel = r'median($n_{str}$)'




xlim1 = np.min(array1d)
xlim2 = np.max(array1d)
bins = np.logspace(np.log10(xlim1), np.log10(xlim2), nbins)


y,binEdges = np.histogram( array1d
, bins = bins , density= False)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])

#plt.errorbar(bincenters, y, 
#yerr = np.sqrt(y),  fmt='b.', markersize='5', alpha = 0.7, elinewidth= 2 )
#
#plt.plot(bincenters[y!=0], y[y!=0], 'b', alpha = 0.8, lw = 2, label = strLabel)

errorfill(bincenters[y!=0], y[y!=0], np.sqrt(y[y!=0]), color = 'forestgreen')


minnstrEachBlob = np.array(ndi.minimum(nstream, labels=labels3d, index= label1d))
array1d = minnstrEachBlob
strLabel = r'min($n_{str}$)'




xlim1 = np.min(array1d)
xlim2 = np.max(array1d)
bins = np.logspace(np.log10(xlim1), np.log10(xlim2), nbins)


y,binEdges = np.histogram( array1d
, bins = bins , density= False)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])

#plt.errorbar(bincenters, y, 
#yerr = np.sqrt(y),  fmt='b.', markersize='5', alpha = 0.7, elinewidth= 2 )
#
#plt.plot(bincenters[y!=0], y[y!=0], 'b', alpha = 0.8, lw = 2, label = strLabel)

errorfill(bincenters[y!=0], y[y!=0], np.sqrt(y[y!=0]), color = 'darkred')



#plt.bar(y[y!=0], bincenters[y!=0], alpha = 0.5, yerr = np.sqrt(y[y!=0]))
#
#plt.hist( array1d
#, bins = bins)
#bincenters = 0.5*(binEdges[1:]+binEdges[:-1] )

plt.xlim(2.5,1500)
plt.ylim(3, 4e3 )

plt.yscale('log')
if (strLabel != '#Grids in Halo'): plt.xscale('log')
plt.xlabel(r'$n_{str}$')
plt.ylabel("PDF") 
plt.legend(loc = "upper right")

plt.savefig('plots/minmaxStrInHalo.pdf', bbox_inches='tight')
plt.show()

