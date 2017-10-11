# -*- coding: utf-8 -*-
import numpy as np
from numpy import loadtxt
import sys

"""
x    y    z    [h^-1 Mpc]         
velocity       [km/s]      
lkl   
mass           [h^-1 M_s]  
sigma  
sigma_v  
r_sph  
delta  
spin p.

"""

ngr =                  128#  256#   512# 
L =   1# 100#
dir2 = str(L)+'Mpc'
dir3 = str(ngr)

#ID(1)	hostHalo(2)	numSubStruct(3)	Mvir(4)	npart(5)	Xc(6)	Yc(7)	Zc(8)	VXc(9)	VYc(10)	VZc(11)	Rvir(12)	Rmax(13)	r2(14)	mbp_offset(15)	
#com_offset(16)	Vmax(17)	v_esc(18)	sigV(19)	lambda(20)	lambdaE(21)	Lx(22)	Ly(23)	Lz(24)	
#b(25)	c(26)	Eax(27)	Eay(28)	Eaz(29)	Ebx(30)	Eby(31)	Ebz(32)	Ecx(33)	Ecy(34)	Ecz(35)	ovdens(36)	nbins(37)	
#fMhires(38)	Ekin(39)	Epot(40)	SurfP(41)	Phi0(42)	cNFW(43)	
#
#dirOut = '/home/nesar/Dropbox/ImageProc/VTI_numpy_files/'
#fileIn = '/home/nesar/Desktop/032/AHF100Mpc128Unbound.z0.000.AHF_particles'

dirOut = '/home/nesar/Desktop/051/npy'
fileIn = '/home/nesar/Desktop/051/AHF/AHF.z0.000.AHF_particles'

#fileIn = dirOut + 'ahf_haloes_100_128.txt'
#fileIn = '/home/nesar/Desktop/032/c-snapshot_050.AHF_particles'



ParticleList = list(open(fileIn, 'r'))
totHalo = int(ParticleList[0])
b = ParticleList[1:]
#b = b[346000:len(b)]

aGroupLen = []
partID = []  # id’s start at zero (C convention)


for line in b:
    for word in line.split('\s'):
        eachLine = word.split('\t')
        if (len(eachLine) ==1 ):
            #print '-------', (word)
            v = word
            aGroupLen = np.append(aGroupLen, int(str(v).split()[0]))

        else:
            partID = np.append(partID, int(eachLine[0]))

partID = partID-1            


fileOut = 'npy/x0_'+'snap_051.npy'
x0_3dF_1dC = np.ravel(np.load(fileOut), order = 'F')

fileOut = 'npy/x1_'+'snap_051.npy'
x1_3dF_1dC = np.ravel(np.load(fileOut), order = 'F')

fileOut = 'npy/x2_'+'snap_051.npy'
x2_3dF_1dC = np.ravel(np.load(fileOut), order = 'F')



x = x0_3dF_1dC[partID.astype(int)]
y = x1_3dF_1dC[partID.astype(int)]
z = x2_3dF_1dC[partID.astype(int)]

aPos = np.vstack((x,y,z))

print aPos.shape
print
print aGroupLen.shape
print aGroupLen.sum()


idx = [] 

for i in range(1, aGroupLen.size+1):
    
    idx = np.append(idx, i*np.ones(aGroupLen[i-1]).astype(int))
  
  
x, y, z = (aPos)[0], (aPos)[1], (aPos)[2]


#import matplotlib.pylab as plt
#plt.figure(figsize = (5,5))
#idx_e = np.where(idx == 12)
#plt.scatter(x[idx_e], z[idx_e])
#plt.show()

np.save('npy/particlesAHF', np.vstack((x, y, z, idx)))
