import numpy as np
from enthought.tvtk.api import tvtk
import toParaview as toPara
from scipy.spatial import Delaunay
from plyfile import PlyData, PlyElement

#===================================================
import os
def path_create(path):
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
                os.makedirs(dir)
#===================================================

def exportSimplices(coords,simplices, fileName):
    '''
    coords -- co-ordinates of the points.  Shape: (n_pts, 3)
    simplices -- indices of the points for each simplex. Shape: (n_simplices, 4) 
    
    '''

    dtype_vertex = np.dtype( [('x', 'f4'), ('y', 'f4'), ('z', 'f4')] )
    dvt = [tuple(i) for i in coords] 
    vertex = np.array(dvt  , dtype= dtype_vertex )
    
    dtype_face = np.dtype( [ ('vertex_indices', 'i4', (np.shape(simplices[0])[0],)) ]) 
    dfc = [(i,) for i in simplices]
    face = np.array( dfc , dtype = dtype_face)
    
    
    el = [PlyElement.describe(vertex, 'vertex', comments=['tetrahedron vertices']), PlyElement.describe(face, 'face')]

    path_create(fileName)
    PlyData(el).write(fileName)
    print fileName

#=================================================== 



def make_corrected_coord(q,x,L):
    dx = x - q
    dx = np.where(dx >  L/2, dx-L, dx)
    dx = np.where(dx < -L/2, dx+L, dx)
    return q + dx

L = 100.
lBox = str(int(L))+'Mpc'
nGr = 128 # size of the grid (box)
nGr2 = nGr/2
ngr = nGr
#1   #8007  #4019 #1016 #223  #215     #126   #512,100 - 78098 

dirOut = lBox+'_'+str(nGr)+'/'

#nstream= np.load('/home/nesar/Desktop/128MUSIC/npy/numField_064_'+str(refFactor)+'.npy')


#------- Load Particle Coordinates E-space -------------------------
dLoad = '/home/nesar/Desktop/128MUSIC/npy/'
x0_3d = np.load(dLoad+'x0'+'_snap_064.npy')
x1_3d = np.load(dLoad+'x1'+'_snap_064.npy')
x2_3d = np.load(dLoad+'x2'+'_snap_064.npy')

q0_3d, q1_3d, q2_3d = np.mgrid[0:nGr, 0:nGr, 0:nGr]*L/nGr   # q_i  in Mpc

x0Mpc = make_corrected_coord(q0_3d, x0_3d, L)
x1Mpc = make_corrected_coord(q1_3d, x1_3d, L)
x2Mpc = make_corrected_coord(q2_3d, x2_3d, L)

fof = np.load('/home/nesar/Desktop/128MUSIC/npy/AHF_064.npy')
#minlkl = 20   # minimum linking length
#fof = fof[:,np.where(fof[6,:]>minlkl)][:,0,:]
#maxlkl = np.max(fof[6,:])


x0 = np.array([])
x1 = np.array([])
x2 = np.array([])
R = np.array([])
q0 = np.array([])
q1 = np.array([])
q2 = np.array([])
r0 = np.array([])
r1 = np.array([])
r2 = np.array([])
lkl = np.array([])

tot = 1    # no. of haloes
haloID = [1100]

for i in haloID:
    #n = np.arange(np.size(fof[0,]))  # all halo ID
    
    ih = i 
    print ih
    
    lkl = np.append(lkl, fof[6, ih])
    
    boxCenterMpc = np.array([fof[0,ih],fof[1,ih],fof[2,ih]]) # Small Box center 
    #boxCenterMpc = ngr/2*np.array([1., 1., 1.])*L/ngr      # full Box
    
    boxSizeMpc = fof[10,ih]   # Small Box  Size 
    #boxSizeMpc = L        # Full box
        
    xBeg = boxCenterMpc - boxSizeMpc/2 # Limits on E-grid
    xEnd = boxCenterMpc + boxSizeMpc/2
    
    cond = np.where((x0Mpc-boxCenterMpc[0])**2 + (x1Mpc-boxCenterMpc[1])**2 + (x2Mpc-boxCenterMpc[2])**2 < boxSizeMpc**2)
    
    x0 = np.append(x0 ,x0Mpc[cond])
    x1 = np.append(x1, x1Mpc[cond])
    x2 = np.append(x2, x2Mpc[cond])
    
    R = np.append(R, np.ones(np.size(cond)))
    a = np.array(cond)

    q0 = np.append(q0, a[0,:]*L/ngr)
    q1 = np.append(q1, a[1,:]*L/ngr)
    q2 = np.append(q2, a[2,:]*L/ngr)
    
r0 = -q0 + x0
r1 = -q1 + x1
r2 = -q2 + x2

pts = np.array([x0,x1,x2]).T

tri = Delaunay(pts, furthest_site=False)
    
tpt = tri.points        # Gives coords of points  ##### pts == tpt = (x0, x1, x2)
tvt = tri.simplices      # Gives face - index
    
    #tri.vertices
    
    #print np.shape(tpt)
    #print np.shape(tvt)
    
#from scipy.spatial import ConvexHull
#ch = ConvexHull(pts)
#
#tpt = ch.points        # Gives coords of points
#tvt = ch.simplices      # Gives face - index
##


fileOut = 'vti/xfof_'+str(int(L))+'_'+str(nGr)+'_'+str(ih)
toPara.Points1d(x0, x1, x2 , R, fileOut)

fileOut = 'vti/qfof_'+str(int(L))+'_'+str(nGr)+'_'+str(ih)
toPara.Points1d(q0, q1, q2 , R, fileOut)

fileOut = 'vti/Vecfof_'+str(int(L))+'_'+str(nGr)+'_'+str(ih)+'.vtu'
toPara.UnstructuredVector(q0, q1, q2, x0, x1, x2, fileOut)



print 

fileOut = 'vti/xDelaunayfof_'+str(int(L))+'_'+str(nGr)+'_'+str(ih)+'.ply'
exportSimplices(tpt, tvt, fileOut)


qpt = np.array([q0,q1,q2]).T
fileOut = 'vti/qDelaunayfof_'+str(int(L))+'_'+str(nGr)+'_'+str(ih)+'.ply'
exportSimplices(qpt, tvt, fileOut)

#fileOut = 'vti/DelPoints_fof_'+str(int(L))+'_'+str(nGr)+'_'+str(ih)+'.ply'
#toPara.Points1d(tpt[:,0] , tpt[:,1] , tpt[:,2], np.ones_like(tpt[:,1]), fileOut)


pts = np.array([q0,q1,q2]).T

tri = Delaunay(pts, furthest_site=False)
    
tpt = tri.points        # Gives coords of points  ##### pts == tpt = (x0, x1, x2)
tvt = tri.simplices      # Gives face - index
#  
 
 
#ch = ConvexHull(pts)
#
#tpt = ch.points        # Gives coords of points
#tvt = ch.simplices      # Gives face - index
#

print

qpt = np.array([q0,q1,q2]).T
fileOut = 'vti/qADelaunayfof_'+str(int(L))+'_'+str(nGr)+'_'+str(ih)+'.ply'
exportSimplices(qpt, tvt, fileOut)

qpt = np.array([x0,x1,x2]).T
fileOut = 'vti/xADelaunayfof_'+str(int(L))+'_'+str(nGr)+'_'+str(ih)+'.ply'
exportSimplices(qpt, tvt, fileOut)


 
