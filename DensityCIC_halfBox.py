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
        
        a1 = x[particle] - i*dx
        b1 = y[particle] - j*dy
        c1 = z[particle] - k*dz
        
        a2 = dx - a1
        b2 = dy - b1
        c2 = dz - c1
        
        wx1 = a1/dx
        wx2 = a2/dx
        wy1 = b1/dy
        wy2 = b2/dy
        wz1 = c1/dz
        wz2 = c2/dz
        
        
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
L = 100.
nGr = 128
nGr2 = nGr/2
refFactor = 4
Dn = 60      # 60 for all except 100Mpc-512 (160)
size_fact = nGr2*refFactor

lBox = str(int(L))+'Mpc'


#--------x0, x1, x2 -------------------

dLoad = '/Users/nesar/Desktop/Streams/Gadget2npy/'+lBox+'/'+str(nGr)+'/'+'Snapshots/'

dLoad = './npy/'

x0_3d = np.load(dLoad+'x0'+'_snap_032.npy')*size_fact/L 
x1_3d = np.load(dLoad+'x1'+'_snap_032.npy')*size_fact/L
x2_3d = np.load(dLoad+'x2'+'_snap_032.npy')*size_fact/L

print 
#print 'stream     Volume fraction   Mass fraction     Density'

boxCenterMpc = np.array([L/2.,L/2.,L/2.]) # Small Box center 
print 'box Center', boxCenterMpc
boxSizeMpc = np.array([L/2.,L/2.,L/2.])   # Small Box  Size 
#boxSize = np.int32(np.around(boxSizeMpc*nGr/L))

xBeg = (boxCenterMpc - boxSizeMpc/2)*size_fact/L # Limits on E-grid
xEnd = (boxCenterMpc + boxSizeMpc/2)*size_fact/L

#x_limMpc =np.array([xBeg[0],xEnd[0],xBeg[1],xEnd[1],xBeg[2],xEnd[2]])

#print 
#print 'stream     Volume fraction   Mass fraction     Density'

x0_1d = np.ravel(x0_3d)
x1_1d = np.ravel(x1_3d)
x2_1d = np.ravel(x2_3d)

x0Cut = (np.abs(x0_1d - (xBeg[0] + xEnd[0])/2. ) <  (xEnd[0] - xBeg[0])/2. )
x1Cut = (np.abs(x1_1d - (xBeg[1] + xEnd[1])/2. ) <  (xEnd[1] - xBeg[1])/2. )
x2Cut = (np.abs(x2_1d - (xBeg[2] + xEnd[2])/2. ) <  (xEnd[2] - xBeg[2])/2. )

xyzCut = np.where( x0Cut & x1Cut & x2Cut )

x0_1d = x0_1d[xyzCut] 
x1_1d = x1_1d[xyzCut] 
x2_1d = x2_1d[xyzCut] 

macro = rho(x0_1d, x1_1d, x2_1d)





np.save("./npy/Half/DensityCIC_032_"+str(refFactor), macro)
  #


