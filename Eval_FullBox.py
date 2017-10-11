import numpy as np
#import matplotlib.pylab as plt
from numpy import linalg as LA
#from mpi4py import MPI

''' MAKE IT PARALLEL cuda????'''



class Box:
    def __init__(self, box3d, pad):
        self.box3d = box3d
        self.pad = pad
        self.N = np.shape(self.box3d)[0]
    
    def extendBox(self): 
        ''' Add slices on all sides
        using periodic BC'''
        
        for i in range(self.pad):
            
            self.box3d = np.dstack((self.box3d[:,:,-(1+2*i)], self.box3d, self.box3d[:,:,2*i])) # dstack - along 3rd axis  --2
            self.box3d = np.swapaxes(self.box3d, 1, 2)
            self.box3d = np.dstack((self.box3d[:,:,-(1+2*i)], self.box3d, self.box3d[:,:,2*i])) # dstack - along 3rd axis  --1
            self.box3d = np.swapaxes(self.box3d, 0, 2)
            self.box3d = np.dstack((self.box3d[:,:,-(1+2*i)], self.box3d, self.box3d[:,:,2*i])) # dstack - along 3rd axis  --0
            self.box3d = np.swapaxes(self.box3d, 0, 2)
            self.box3d = np.swapaxes(self.box3d, 1, 2)
    
        return self.box3d
    
        
    def sliceBox(self):
        '''Remove slices on all sides'''
        for i in range(self.pad):
            self.box3d = self.box3d[1:-1, 1:-1, 1:-1]
            
        return self.box3d
        
        
        

L = 1.
nGr = 128
refFactor = 1
Dn = 60      # 60 for all except 100Mpc-512 (160)  # 0 for small box
size_fact = nGr*refFactor

sig = 1.0

lBox = str(int(L))+'Mpc'
dir1 = lBox+'/'+str(nGr)+'/'
#

fileIn = 'npy/Half/numFieldHalf_032_'+str(refFactor)+'.npy'   #HalfBox
fileIn = 'npy/numField_051_1.npy'            # FullBox

nstream = np.load(fileIn).astype(np.float64)

#nstream = nstream[0:10,0:10,0:10]

nstream = Box(nstream,2).extendBox()

from scipy import ndimage
nstream = ndimage.gaussian_filter(nstream, sigma= sig)

dx = dy = dz = L/size_fact

#dx = dy = dz = 1

Nx, Ny, Nz = np.gradient(-nstream, dx,dy,dz)


Nxx, Nxy, Nxz = np.gradient(Nx, dx,dy,dz)
Nyx, Nyy, Nyz = np.gradient(Ny, dx,dy,dz)
Nzx, Nzy, Nzz = np.gradient(Nz, dx,dy,dz)

Nxy = (Nxy + Nyx)/2.0
Nxz = (Nxz + Nzx)/2.0
Nyz = (Nyz + Nzy)/2.0

extendedSize = np.shape(nstream)[0]

l = np.empty([extendedSize,extendedSize,extendedSize, 3])
v1 = np.empty([extendedSize,extendedSize,extendedSize, 3])
v2 = np.empty([extendedSize,extendedSize,extendedSize, 3])
v3 = np.empty([extendedSize,extendedSize,extendedSize, 3])

#
#comm = MPI.COMM_WORLD
##print type(comm)
#rank = comm.Get_rank()
#size = comm.Get_size()
##print "hello world from process ", rank, "size:", size
#
#if (np.size(hidx)%size) != 0: 
#    print "No. of jobs is:", np.size(hidx)
#    print "Change the no. of processes"
#    comm.Abort()
#
#
#
#
#


#
#niter = np.size(hidx)/size   # no. iteration for each processor(rank)
#for ite in range(niter):

for i in range(extendedSize):
    for j in range(extendedSize):
        for k in range(extendedSize):


            A = np.array([[Nxx[i,j,k], Nxy[i,j,k], Nxz[i,j,k]], [Nxy[i,j,k], Nyy[i,j,k], Nyz[i,j,k]], [Nxz[i,j,k], Nyz[i,j,k], Nzz[i,j,k]]])
            
            #print "======================================================="
            la,v = LA.linalg.eig(A)
            
#==========================================================
#http://stackoverflow.com/questions/10083772/python-numpy-sort-eigenvalues
              
            sort_perm = la.argsort()

            la.sort()     # <-- This sorts the list in place.
            v = v[sort_perm]
#============================================================


            l[i,j,k, :] = la[::-1]
            #print "l1, l2, l3:  ", l[i,j,k, :] #eigenvalues

            v3[i,j,k, :] = v[0]
            v2[i,j,k, :] = v[1]
            v1[i,j,k, :] = v[2]
            
            #print i,j,k

#            print v1[i,j,k, :]
#            print v2[i,j,k, :]
#            print v3[i,j,k, :]
#
#            print 
#
#            print v[:,0]  #first eigenvector
#            print v[:,1]  #second eigenvector
#            print v[:,2]  #third eigenvector
#            print "======================"


#print np.sum(abs(v**2),axis=0) #eigenvectors are unitary
#[ 1.  1. ]
#v1 = np.array(v[:,0]).T
#print LA.norm(A.dot(v1)-l1[i,j,k]*v1) #check the computation
#3.23682852457e-16

#size_fact = 10

l_new = np.empty([size_fact,size_fact,size_fact, 3])
v1_new = np.empty([size_fact,size_fact,size_fact, 3])
v2_new = np.empty([size_fact,size_fact,size_fact, 3])
v3_new = np.empty([size_fact,size_fact,size_fact, 3])

for i in range(3):
    l_new[:,:,:,i] = Box(l[:,:,:,i],2).sliceBox()
    #del l
    v1_new[:,:,:,i] = Box(v1[:,:,:,i],2).sliceBox()
    #del v1
    v2_new[:,:,:,i] = Box(v2[:,:,:,i],2).sliceBox()
    #del v2
    v3_new[:,:,:,i] = Box(v3[:,:,:,i],2).sliceBox()
    #del v3
    
#v0*iHat + v1* jHat + v2* kHat
    

#np.save("npy/Half/Evals3_051_"+str(refFactor), l_new)
np.save("npy/Evals3_051", l_new)
np.save("npy/Evec1_051", v1_new)
np.save("npy/Evec2_051", v2_new)
np.save("npy/Evec3_051", v3_new)
