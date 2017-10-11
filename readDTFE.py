""" DTFE read to numpy """

import matplotlib.pylab as plt
import numpy as np
import SetPub
SetPub.set_pub()

plt.rcParams['image.cmap'] = 'CMRmap_r'

refFactor = 1
nGr = 128

size_fact = refFactor*nGr
#fileIn = 'npy/DensityDTFE_051_128.den'
fileIn = 'npy/DensityDTFE_051_128.a_den'



a = np.fromfile(fileIn, dtype=np.float32)
#a = a.astype(float)/np.min(a)

DTFEden = np.reshape(a, (size_fact, size_fact, size_fact), order = 'C')

np.save("npy/DensityDTFE_051_"+str(refFactor), DTFEden)


CICden = np.load("npy/DensityCIC_051_"+str(refFactor)+'.npy')

nstream = np.load("npy/numField_051_"+str(refFactor)+'.npy')


f, ax = plt.subplots(4,3, figsize = (14,20), sharex = True, sharey = True)
f.subplots_adjust( right= 0.95, top= 0.95, hspace = 0.2, wspace = 0.02)
#f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
#plt.minorticks_on()

ticksLoc = np.linspace(0, nGr, 6)
ticksLabel = [ '', '0.2', '0.4', '0.6', '0.8', '' ]

DTFEcut = [ 0.01, 0.05, 0.8, 1.5]
CICcut = [ 0.01,  0.05, 0.8, 1.5]
nstrCut = [1, 3, 5, 25]
from matplotlib import colors

cmap = colors.ListedColormap(['white', 'gray'])


for i in range(4):
    plt.sca(ax[i,0]) 
    a = (DTFEden > DTFEcut[i])
    ax[i,0].imshow(a[5,:,:], label = DTFEcut[i], cmap = cmap)
    plt.title(r'$\rho_{DTFE} > %0.2f$'%DTFEcut[i])
    #ax[i,0].legend(loc = 'bottom right', title = str(DTFEcut[i]))
    #frame = legend.get_frame()
    #frame.set_facecolor('0.90')
    plt.yticks( ticksLoc, ticksLabel  )
    plt.xticks( ticksLoc, ticksLabel  )
    
    plt.sca(ax[i,1])
    c = (CICden > CICcut[i])
    ax[i,1].imshow(c[5,:,:], label = CICcut[i], cmap = cmap)
    plt.title(r'$\rho_{CIC}/\rho_{b} > %0.2f$'%CICcut[i])
    #plt.legend(loc = 'bottom right')
    plt.yticks( ticksLoc, ticksLabel  )
    plt.xticks( ticksLoc, ticksLabel  )

    #
    plt.sca(ax[i,2])
    d = (nstream > nstrCut[i])
    ax[i,2].imshow(d[5,:,:], label = nstrCut[i], cmap = cmap)
    plt.title(r'$n_{str} > %d$'%nstrCut[i])
    #ax[i,2].legend(loc = 'bottom right', title = 'ABCD')
    
    plt.yticks( ticksLoc, ticksLabel  )
    plt.xticks( ticksLoc, ticksLabel  )



plt.savefig('plots/CICnstrDTFE.pdf')

plt.show()




