import numpy as np
import time

#===================================================
def make_corrected_coord(q,x,L):
    dx = x - q
    dx = np.where(dx >  L/2, dx-L, dx)
    dx = np.where(dx < -L/2, dx+L, dx)
    return q + dx

#--------number of streams-----------------\

#==============================================================
# 4 point Cloud-in-cell method to determine density
def rho(x,y,z):   #  0 <  x, y, z < nGr*refFactor in 1D 


    #x = (x-x_l)*Nx/(x_h - x_l)
    #y = (y-y_l)*Ny/(y_h - y_l) 
    
    Np = np.size(x)
    macro = np.zeros([size_fact, size_fact, size_fact])

    for particle in range(Np):
        i = int(x[particle]) 
        j = int(y[particle]) 
        k = int(z[particle])

	#print 'max i', np.max(i)

        dx = 1
        dy = 1
        dz = 1
        
        a1 = np.around(x[particle], decimals = 4) - i*dx
        b1 = np.around(y[particle], decimals = 4) - j*dy
        c1 = np.around(z[particle], decimals = 4) - k*dz
        
        a2 = dx - a1
        b2 = dy - b1
        c2 = dz - c1
        
        wx1 = a1/dx
        wx2 = a2/dx
        wy1 = b1/dy
        wy2 = b2/dy
        wz1 = c1/dz
        wz2 = c2/dz
        
        #print a1, a2, wx1, wz2
        
        
        macro[i, j, k] += (wx1 * wy1 * wz1)
        macro[np.mod(i+1,size_fact), j, k] += (wx2 * wy1 * wz1)
        macro[i, np.mod(j+1,size_fact), k] += (wx1 * wy2 * wz1)
        macro[np.mod(i+1,size_fact), np.mod(j+1,size_fact), k] += (wx2 * wy2 * wz1)
        
        macro[i, j, np.mod(k+1,size_fact)] += (wx1 * wy1 * wz2)
        macro[np.mod(i+1,size_fact), j, np.mod(k+1,size_fact)] += (wx2 * wy1 * wz2)
        macro[i, np.mod(j+1,size_fact), np.mod(k+1,size_fact)] += (wx1 * wy2 * wz2)
        macro[np.mod(i+1,size_fact), np.mod(j+1,size_fact), np.mod(k+1,size_fact)] += (wx2 * wy2 * wz2)

    return macro

#==============================================================



start = time.time()
L = 1.
nGr = 128
refFactor = 1
Dn = 60      # 60 for all except 100Mpc-512 (160)
size_fact = nGr*refFactor

lBox = str(int(L))+'Mpc'


#--------x0, x1, x2 -------------------

dLoad = '/Users/nesar/Desktop/Streams/Gadget2npy/'+lBox+'/'+str(nGr)+'/'+'Snapshots/'

dLoad = './npy/'

x0_3d = np.load(dLoad+'x0'+'_snap_051.npy')*size_fact/L 
x1_3d = np.load(dLoad+'x1'+'_snap_051.npy')*size_fact/L
x2_3d = np.load(dLoad+'x2'+'_snap_051.npy')*size_fact/L

print 
#print 'stream     Volume fraction   Mass fraction     Density'

x0_1d = np.ravel(x0_3d)
x1_1d = np.ravel(x1_3d)
x2_1d = np.ravel(x2_3d)

macro = rho(x0_1d, x1_1d, x2_1d)

#macro = np.around(macro, decimals = 3)


np.save("npy/DensityCIC_051_"+str(refFactor), macro)
  


