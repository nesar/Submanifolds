import numpy as np
from numpy import loadtxt

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
L =   100# 100#
dir2 = str(L)+'Mpc'
dir3 = str(ngr)

#ID(1)	hostHalo(2)	numSubStruct(3)	Mvir(4)	npart(5)	Xc(6)	Yc(7)	Zc(8)	VXc(9)	VYc(10)	VZc(11)	Rvir(12)	Rmax(13)	r2(14)	mbp_offset(15)	
#com_offset(16)	Vmax(17)	v_esc(18)	sigV(19)	lambda(20)	lambdaE(21)	Lx(22)	Ly(23)	Lz(24)	
#b(25)	c(26)	Eax(27)	Eay(28)	Eaz(29)	Ebx(30)	Eby(31)	Ebz(32)	Ecx(33)	Ecy(34)	Ecz(35)	ovdens(36)	nbins(37)	
#fMhires(38)	Ekin(39)	Epot(40)	SurfP(41)	Phi0(42)	cNFW(43)	

dirOut = '/home/nesar/Desktop/051/npy'
fileIn = '/home/nesar/Desktop/051/AHF/AHF.z0.000.AHF_halos'
#fileIn = dirOut + 'ahf_haloes_100_128.txt'


lines = loadtxt(fileIn, comments="#", delimiter="\t", unpack=False)
 
x = lines[:,5]/1000
y = lines[:,6]/1000
z = lines[:,7]/1000
vx = lines[:,8]
vy = lines[:,9]
vz = lines[:,10]
npart = lines[:,4]
mVir = lines[:,3]
numSubStruct = lines[:,2]
hostHalo = lines[:,1]
rVir = lines[:,11]/1000

All = np.vstack([x ,y, z, vx, vy, vz, npart ,mVir, numSubStruct, hostHalo,rVir])


#np.save(dirOut+'AHF_'+dir2+'_'+dir3+'.npy',All)

np.save('npy/AHF_051.npy', All)

