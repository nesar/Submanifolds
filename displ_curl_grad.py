import numpy as np
#import time
#import numb_streamfull as ns
import matplotlib.pyplot as plt
#import os
#from mpi4py import MPI
import time


L = 1
nGr = 128
dirIn = 'npy/'+str(L)+'Mpc'+str(nGr)+'/'


fileOut = dirIn+'s0_'+'snap_051.npy'
#s0 = np.ravel(np.load(fileOut), order ='C')
s0 = np.load(fileOut)
fileOut = dirIn+'s1_'+'snap_051.npy'
#s1 = np.ravel(np.load(fileOut), order ='C')
s1 = np.load(fileOut)
fileOut = dirIn+'s2_'+'snap_051.npy'
#s2 = np.ravel(np.load(fileOut), order ='C')
s2 = np.load(fileOut)

def symlog(x):
    return np.where(x>0, np.log10(x), -np.log10(-x))
    # if (x==0): return 0
    # elif (x>0): return np.log10(x)
    # else: return -np.log10(-x)

# def curl1(Vx,Vy,Vz, dx, dy, dz):
#     Cx = np.array(np.gradient(Vz, dy)[1]) - np.array(np.gradient(Vy, dz)[2])
#     Cy = - np.array(np.gradient(Vx, dz)[2]) + np.array(np.gradient(Vz, dx)[0])
#     Cz = np.array(np.gradient(Vy, dx)[0]) - np.array(np.gradient(Vx, dy)[1])
#     return Cx, Cy, Cz

# def curl(f,x):
#     jac = nd.Jacobian(f)(x)
#     return sp.array([jac[2,1]-jac[1,2],jac[0,2]-jac[2,0],jac[1,0]-jac[0,1]])

def curl(f):  # faster implementation
    # num_dims = len(f)
    Cx = np.ufunc.reduce(np.subtract, [np.gradient(f[i], axis= i%2 + 1 ) for i in [2,1] ])
    Cy = - np.ufunc.reduce(np.subtract, [np.gradient(f[i], axis= (i+2)%4 ) for i in [0,2] ])
    Cz = np.ufunc.reduce(np.subtract, [np.gradient(f[i], axis= (i+1)%2 ) for i in [1,0] ])
    return Cx, Cy, Cz

# def div(Vx, Vy, Vz, dx, dy, dz):
#     div = np.array(np.gradient(Vx, dx)[0]) + np.array(np.gradient(Vy, dy)[1])  + np.array(np.gradient(Vz, dz)[2])
#     return div


def div(f):
    """
    Computes the divergence of the vector field f, corresponding to dFx/dx + dFy/dy + ...
    :param f: List of ndarrays, where every item of the list is one dimension of the vector field
    :return: Single ndarray of the same shape as each of the items in f, which corresponds to a scalar field
    """
    num_dims = len(f)
    return np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) for i in range(num_dims) ])



# curlSx, curlSy, curlSz = curl(s0, s1, s2, 1, 1, 1)
# divS = div(s0, s1, s2, 1, 1, 1)
# CurlMag = curlSx**2 + curlSy**2 + curlSz**2


divS = div([s0, s1, s2])
curlSx, curlSy, curlSz = curl([s0, s1, s2])
CurlMag = curlSx**2 + curlSy**2 + curlSz**2


refFactor = 1
nstream = np.load(dirIn+'numField_051_'+str(refFactor)+'.npy')
flip = np.load(dirIn+'flip_snap_051.npy')


f, ax = plt.subplots(2,2, figsize = (18,20), sharex = True, sharey = True)
f.subplots_adjust(  wspace = 0.02, hspace = 0.2, bottom = 0.15, left = 0.1, right = 0.9)

sliceNo = 125

plt.sca(ax[0,0])
plt.imshow(  symlog(divS[sliceNo,:,:]))
plt.title('log(div(s))', fontsize = 50)

plt.sca(ax[0,1])
plt.imshow( symlog(CurlMag[sliceNo,:,:]) )
plt.colorbar()
plt.title('log(|Curl(s)|)', fontsize = 50)


plt.sca(ax[1,0])
plt.imshow( np.log10(nstream[sliceNo,:,:]) )
plt.title('log(nstr)', fontsize = 50)


plt.sca(ax[1,1])
plt.imshow( np.log10(flip[sliceNo,:,:]+1) )
plt.title('log(ff)', fontsize = 50)

plt.savefig('plots/div_curlS_ff_nstr'+str(sliceNo)+'.png')
plt.show()



plt.figure(13)
plt.imshow( symlog(CurlMag[sliceNo,:,:]) > -3 )


plt.figure(10)
x = np.linspace(0, L, nGr)*nGr
y = np.linspace(0, L, nGr)*nGr
X, Y = np.meshgrid(x, y)


plt.quiver(x, y, curlSx[0], curlSy[0], curlSz[0], cmap = 'nipy_spectral' )
plt.imshow(flip[0], alpha = 0.5)
# plt.streamplot(x, y, curlSx[0], curlSy[0], density=0.6 )

plt.show()

# selecting parts from curl/div and mapping to eulerian coordinates
q0_3d, q1_3d, q2_3d = np.mgrid[0:nGr, 0:nGr, 0:nGr]*L/nGr   # q_i  in Mpc
q_select = np.where( symlog(CurlMag) > -3 )
q0_3d[q_select] - s0[q_select]


import sys
sys.exit()

q_select = np.where( symlog(CurlMag) > -3 )


f, ax = plt.subplots(2,2, figsize = (20,6))
f.subplots_adjust(  wspace = 0.05, hspace = 0.02, bottom = 0.15, left = 0.1, right = 0.85)

ticksLoc = np.linspace(0, img3d_n.shape[0], 6)
ticksLabel = [ '', '0.2', '0.4', '0.6', '0.8', '' ]
    
#cmap = colors.ListedColormap(['white', 'gray', 'red'])
#bounds=[0,1, 2,   np.max(nstream)]
#norm = colors.BoundaryNorm(bounds, cmap.N)
cmap= 'nipy_spectral_r'
cmap= 'CMRmap_r'
    

plt.sca(ax[2])
imshow(img3d_n[5,:,:], origin='lower', interpolation='nearest', cmap = cmap)
#colorbar()
title(r"$n_{str}(z = 0)$")
#plt.minorticks_on()
plt.xlabel(r" $h^{-1} Mpc$")
plt.xticks(ticksLoc , ticksLabel  )
plt.yticks([])

    
#plt.gca().set_yticks(ticksLabel)
    
#subplot(2,3,4)
plt.sca(ax[0])
imshow(img3d_f[5,:,:], origin='lower', interpolation='nearest', cmap = cmap)
#colorbar()
title(r"$ff(z_{ini})$")
#plt.minorticks_on()
plt.xlabel(r" $h^{-1} Mpc$")
plt.ylabel(r" $h^{-1} Mpc$")
plt.yticks( ticksLoc, ticksLabel  )
plt.xticks( ticksLoc, ticksLabel  )


non0ff = np.where( (flip_1d > 0)  & (x0_1d < 0.1))
plt.sca(ax[1])
#plt.plot(x2_1d[non0ff], x1_1d[non0ff], 'o', alpha = 0.3, markersize = 1)

#cmap = colors.ListedColormap(['gray', 'red', 'white'])
cmap = 'prism'
bounds=[1,3, 5, 10, np.max(flip_1d[non0ff])]
norm = colors.BoundaryNorm(bounds, cmap)
    
    
scatter(x2_1d[non0ff], x1_1d[non0ff], 
c = np.log(flip_1d[non0ff]+1), alpha = 0.5, s = 2, cmap = 'gist_stern', edgecolors = 'none')
#colorbar()
title(r"$ff(z=0)$")
#plt.minorticks_on()
plt.xlabel(r" $h^{-1} Mpc$")
#plt.ylabel(r" $h^{-1} Mpc$")
plt.xlim(0,1)
plt.ylim(0,1)
ticksLoc = ticksLabel
plt.yticks([] )
#plt.xticks( ticksLoc, ticksLabel  )


plt.show()

#x,y,z = np.mgrid[-100:101:25., -100:101:25., -100:101:25.]
#
#V = 2*x**2 + 3*y**2 - 4*z # just a random function for the potential
#
#Ex,Ey,Ez = np.gradient(V)
#
#
#import numdifftools as nd
#
#def h(x):
#    return np.array([3*x[0]**2,4*x[1]*x[2]**3, 2*x[0]])
#
#def curl(f,x):
#    jac = nd.Jacobian(f)(x)
#    return np.array([jac[2,1]-jac[1,2],jac[0,2]-jac[2,0],jac[1,0]-jac[0,1]])
#    
#def div(f,x):
#    jac = nd.Jacobian(f)(x)
#    return np.array([jac[0,0]+jac[1,1],jac[2,2]])
#
#x = np.array([1,2,3])
#curl(h,x)
#div(h,x)