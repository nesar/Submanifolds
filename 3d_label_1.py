# http://comp-phys.net/2013/03/25/working-with-percolation-clusters-in-python/
import numpy as np
from scipy.ndimage import measurements
   
L = 100.
nGr = 256
refFactorArr = [1]
Dn = 60

lBox = str(int(L))+'Mpc'
dir1 = lBox+'/'+str(nGr)+'/'



for refFactor in refFactorArr:

	fileIn = 'npy/numField_032_'+str(refFactor)+'.npy'

	nstream = np.load(fileIn)


	r = nstream
	#r = rand(L,L,L)
	cutoff = 1
	z = r == cutoff



	struct= struct = np.reshape(np.ones(27), (3,3,3))
	#struct[0,0,0] = struct[0, 0 ,-1] = struct[0,-1,0] = struct[0,-1,-1] = 0
	#struct[-1,0,0] = struct[-1, 0 , -1] = struct[-1,-1,0] = struct[-1,-1,-1] = 0

	lw, num = measurements.label(z, structure = struct)
	

# Calculate areas
	#gridOnes = np.ones_like(z)
	#area = measurements.sum(gridOnes, lw, index= np.unique(lw)[1:])
	


#argSortArea = np.unique(lw)[1:][np.argsort(area)]
#replacelw = np.append(0, argSortArea)

#	replacelw = np.append(0, np.sort(area)[::-1])
#
#	areaImg = replacelw[lw]


	print
	print '-----------------------------'
	print
	print 'Resolution of multi-stream: %d'%refFactor
	print
	print '#grids in voids: %d'%(np.sum(z)) , 'FF_V: %.3f %%'%( 100.*np.sum(z)/ (nGr*refFactor)**3. )
	print
	print '#isolated segments: %d '%num
	print
	#print '#grids in largest voids: %d', np.sort(area)[::-1][0:5]
	print
	#print 'FFx/FF_V: ', (100.*np.sort(area)[::-1][0:5]/np.sum(z) )
	print 'FF1/FF_V: ', (100.*(np.sum(lw==1))/np.sum(z) )

	
