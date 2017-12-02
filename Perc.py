# -*- coding: utf-8 -*-
import numpy as np
import time
import sys
import scipy.ndimage as ndi
import numpy.ma as ma
import toParaview as toPara
import matplotlib.pylab as plt
np.set_printoptions(precision=2)

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

    massEachBlob = ndi.measurements.sum(macro, labels=labels3d, index= labels1d)*m_particle
    densEachBlob = massEachBlob/(gridEachBlob*(L/size_fact)**3)

    denBackground = m_particle*(nGr)**3/(L**3)          # Check if it's alright
    H = 0.7 * 100 * (10**3) / 3.086e+22  #s−1,  (km/Mpc) is converted
    G = 6.674e-11   # N⋅m2/kg2 
    denBackgroundSI = 3.*(H**2.)/(8.*np.pi*G)   #kg/m^3
    denBackground1 = denBackgroundSI*((3.086e+22)**3) / 1.988e30  # in Msol/(Mpc**3)
    
    dentoVirial = densEachBlob/denBackground1

    c3 = dentoVirial < densCut


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

startT = time.time()


# L = 100.
# nGr = 256
# refFactor = 1
# Dn = 60      # 60 for all except 100Mpc-512 (160)
# size_fact = nGr*refFactor
#
# lBox = str(int(L))+'Mpc'
# --------#--------#--------#--------#--------
# sig = 1.0


# l3Cut = 0.0
# densCut = 0.0
# massCut = 1.0

# nstrCutMin = 3
#
#
# nstrCutMax = 3    # Do not change for filaments

# volCut = 1

#m_particle = 0.4065E+11
m_particle = 0.4565e10
#--------#--------#--------#--------#--------

#print
#print 'Smoothening sigma: ', sig
#print 'lambda_3 threshold ', l3Cut
#--------x0, x1, x2 -------------------

#dLoad = './npy/'
#x0_3d = np.load(dLoad+'x0'+'_snap_032.npy') 
#x1_3d = np.load(dLoad+'x1'+'_snap_032.npy')
#x2_3d = np.load(dLoad+'x2'+'_snap_032.npy')
#
#
#x0_1d = np.ravel(x0_3d)
#x1_1d = np.ravel(x1_3d)
#x2_1d = np.ravel(x2_3d)


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
# maxCut = 27 # 51  # field3d.max()
# step = -1 # -2
# OutfileName = 'npy/ff_frESfr1LargestLabel_032'

field3d = nstream # nstream
minCut = 1   # field3d.min()
maxCut = 61   # field3d.max()
step = -2 #
OutfileName = 'npy/nstrfrESfr1LargestLabel_032'


# --------------------------------------------------------------------------------------------


# file_nstr = './npy/numField_032_'+str(refFactor)+'.npy'
# nstream = np.load(file_nstr).astype(np.float64)

#
#file_eval = './npy/Evals3_032.npy'
#l = np.load(file_eval)


#file_nstr = './npy/Half/numFieldHalf_032_'+ str(refFactor)+'.npy'
#nstream = np.load(file_nstr).astype(np.float64)

#file_eval = './npy/Half/Evals3_032_'+ str(refFactor)+'.npy'
#l = np.load(file_eval)

#macro = np.load('npy/DensityCIC_032.npy')
#macro = np.load('npy/Half/DensityCIC_032_'+str(refFactor)+'.npy')


#classified_byEval = np.load('npy/classified_byEval_032.npy')
#classified_bynstream = np.load('npy/classified_bynstream_032.npy')
#
#classified_byEval = np.load('npy/Half/classified_byEvalHalf_'+str(refFactor)+'.npy')
#classified_bynstream = np.load('npy/Half/classified_bynstreamHalf_'+str(refFactor)+'.npy')
#
#FilamentIndex = 2

#z = (classified_byEval == FilamentIndex)
#z = (nstream >= nstrCutMin)
Outfile = np.empty((0,4), float)



struct=np.ones((3,3, 3), dtype="bool8")
struct[0,0,0] = struct[0, 0 ,-1] = struct[0,-1,0] = struct[0,-1,-1] = False
struct[-1,0,0] = struct[-1, 0 , -1] = struct[-1,-1,0] = struct[-1,-1,-1] = False



for nstrCutMin in range(maxCut, minCut, step):


    z = np.where(field3d >= nstrCutMin, 1, 0)


    labels0, num = ndi.label(z, structure = struct)    # What about Edges ??  Periodic BC?
    #labels0, num = ndi.label(z) 
    """
    labels -> input
    labels1 -> intermediate
    labels0 -> output
    """
    
    #labels0 = LabelCutStr(labels, nstrCutMax, nstrCutMin)      #nstream cut 
    #labels0 = LabelCutVol(labels1, volCut)       #Volume cut
    #labels3 = LabelCutDens(labels2, densCut)     #Density cut
    #labels0 = massParticleCut(labels3, massCut)  #Mass cut 


    freqlabel = np.bincount(np.ravel(labels0))
    non0= np.nonzero(freqlabel)[0]
    labelFr = np.vstack( (non0, freqlabel[non0]) ).T
    sortlabel = np.argsort(labelFr[:,1])[::-1]
    sortedLabelFr = labelFr[sortlabel]



#label1d = np.unique(labels0)[1:]
    frES = 1.*np.sum(sortedLabelFr[:,1][1:])/np.sum(sortedLabelFr[:,1][:])
    fr1 = 1.*np.sum(sortedLabelFr[:,1][1])/np.sum(sortedLabelFr[:,1][:])

    print 'nstCut: %d'%nstrCutMin
    print 'fES: %4.1e '%frES
    print 'f1: %4.1e '%fr1
    print 'f1 / fES: %4.1f%%'%(100.*fr1/frES)
    print 'largest clusters'
    #print (sortedLabelFr[:,])#/(256.**3)

    

    Out = np.hstack( [nstrCutMin, frES, fr1, sortedLabelFr[:,0][1]] )
    
    Outfile = np.vstack( [Out, Outfile] )
    
    labels1 = np.where(labels0 == sortedLabelFr[1,0], 1, 2)
    labels1[labels0 == 0] = 0



    np.save('npy/Labels/filamentLabels'+str(nstrCutMin), labels1)
    toPara.StructuredScalar(labels1, 'vti/Labels/filamentsStr_'+str(nstrCutMin), 0, size_fact)


np.save(OutfileName, Outfile)



#np.save('npy/Half/nstrfrESfr1LargestLabel_'+str(refFactor), Outfile)
#np.save('npy/filamentLabels', labels1)
#toPara.StructuredScalar(nstream, 'vti/Half/numField_032_'+str(refFactor), 0, size_fact)

#toPara.StructuredScalar(labels0, 'vti/filaments_032', 0, size_fact)


# plt.figure(1)
plt.figure( 10021,  figsize = (7,7))
sliceNo = 10
# plt.contourf(labels1[sliceNo], levels = [-0.5,0.5,1.5, 2.5], colors = ['w', 'g', 'b', 'r'] )
plt.imshow(labels1[sliceNo] )
plt.colorbar()
plt.show()
