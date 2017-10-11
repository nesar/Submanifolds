import numpy as np
import matplotlib.pylab as plt
import sys

#refFactor = int(sys.argv[1])
#macro = np.load("npy/DensityCIC_032_"+str(refFactor)+".npy")   #mass particles in each cell in Msol

nGr = 128
refFactor = 2
macro = np.load("npy/DensityCIC_032.npy")   #mass particles in each cell in Msol
nstream = np.load("npy/Full/numField_032_"+str(refFactor)+".npy")
#macro1d = np.around(np.sort(np.ravel(macro)), decimals = 3)

#print np.sort(np.unique(macro))[0:5]

volCell = (100./refFactor*nGr)**3.   # Vol of each cell in Mpc^3

#rho = macro/volCell   # Msol/ Mpc^3

rho_b = (128.**3)/(100.**3)   #Msol / Mpc^3


macro_nst = macro[nstream ==1] 

print '-----------------------------------------------------'
print 
print 'refFactor:', refFactor
print
print 'total mass', macro.sum(), 'or', 128**3
print 'total mass in Void', macro_nst.sum()
print 'pc of mass in void', 100*macro_nst.sum()/macro.sum()

print
print 'total volume (grids)',  (nGr*refFactor)**3
print 'total volume in Void',  macro_nst.size
print 'pc of volume in Void', 100.*macro_nst.size/ (nGr*refFactor)**3

print
print 'Avg. density rho_b (in grid units)', macro.sum()/(nGr*refFactor)**3
print 'Avg. density rho_b (in Mpc)', macro.sum()/(100)**3
print 


print 'Mean density rho_V (1): (Mf/Vf) in void',(100*macro_nst.sum()/macro.sum())/(100.*macro_nst.size/ (nGr*refFactor)**3)
print '***** rho_V/rho_b: ', macro_nst.mean() 
print 'rho_V (2):', macro_nst.mean()*(macro.sum()/(100)**3)


print '------------------------------------------------------------'

#print 'rho_V (2):', macro_nst.mean()/ (macro_nst.size*(volCell))


#delta = rho/rho_b


#print np.sort(np.unique(rho))[0:5]




#
#underdense = macro[np.where(macro < rho_b)]
#
#for densCut in np.arange(1e-3, 100, 0.001)[::-1]:
#    print densCut


#dense6 = np.around(  (macro/ (macro.sum()/1e6) ) , decimals = 3)


#np.logspace( np.log10(1e-15), np.log10(2e-4), 1250 )[::-1]


#dense6 = np.around(  (macro/ (macro.sum()/1e6) ) , decimals = 3)


