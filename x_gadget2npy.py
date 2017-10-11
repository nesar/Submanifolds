import pynbody
import numpy as np
#from enthought.mayavi import mlab


#gadget - C
#z0 - F


def index3dTO1d(ng,i,j,k):
    return i*ng**2 +j*ng +k
    
def index1dTO3d(ng, i1d):
    k = (i1d)%ng
    j = (i1d -k)/ng%ng
    i = (i1d -k-j*ng)/ng**2
    return i,j,k


def make_corrected_coord(q,x,L):
    dx = x - q
    dx = np.where(dx >  L/2, dx-L, dx)
    dx = np.where(dx < -L/2, dx+L, dx)
    return q + dx
#-------------------------------------------------------------- 
#==============================================================================
def parameters(f):     
    print 'loadable keys', f.loadable_keys()  # ['pos', 'vel', 'iord', 'mass']
    print 'ID', np.max(f['iord'])
    print 'position shape', np.shape(f['pos'])
   
#========================================


#==============================================================================
#dirIn = '/Volumes/Scratch/nesar/Streams/cmpc/'
#disIn = './'
#dirOut = '/Users/nesar/Desktop/Streams/Gadget2npy/'
#dirOut = './'
ngr = nGr =   128      # 128#  256#    
L =  100.#   200# 

dir2 = str(L)+'Mpc'
dir3 = str(ngr)

#==============================================================================


dir4 = './GadgetSnapshots/'    # for z = 0           
fn1 = str(int(L))+'Mpc128/'                  
#fileName = fn1+dir2+dir3+'_z0'  
fileIn = 'snapshot_050'

#fileIn = 'ics1'
#fileIn = 'snap100Mpc128_z0'
fileIn = dir4+fn1+fileIn

fsIC =  pynbody.load(fileIn)

print '---------------------------------------------------------'
print
print dir2, dir3, dir4, np.shape(fsIC['pos'])

posIC = fsIC['pos']/1000
indIC = fsIC['iord']
indIC = indIC-1

ngr = nGr = np.int(np.round(indIC.shape[0]**(1/3.)))


x0_1dIC=np.zeros_like(posIC[:,0])
x1_1dIC=np.zeros_like(posIC[:,0])
x2_1dIC=np.zeros_like(posIC[:,0])


#iIC , jIC , kIC = index1dTO3d(nGr, indIC)


idxIC = indIC   
#idxIC = np.argsort(posIC[:,1])
#idxIC = kIC

#
x0_1dIC[idxIC] = posIC[:,0]
x1_1dIC[idxIC] = posIC[:,1]
x2_1dIC[idxIC] = posIC[:,2]



print
print indIC
print

x0_3dIC = np.reshape(x0_1dIC, (ngr,ngr,ngr),order='C')
x1_3dIC = np.reshape(x1_1dIC, (ngr,ngr,ngr),order='C')
x2_3dIC = np.reshape(x2_1dIC, (ngr,ngr,ngr),order='C')


q0_3dIC, q1_3dIC, q2_3dIC = np.mgrid[0:nGr, 0:nGr, 0:nGr]*L/nGr   # q_i  in Mpc
print



q1d = np.ravel(q0_3dIC, order= 'C')
#q1d = x0_1dIC
x1d = x0_1dIC




print
print '-------1d------------------------------------------------'
x0_corr1d = make_corrected_coord(q1d, x1d, L)
dx01d = x0_corr1d - q1d
print np.min(x0_corr1d),'< x0_corr1d <', np.max(x0_corr1d)
print np.min(dx01d),'< d01d <', np.max(dx01d)
print '---------------------------------------------------------'

print







x0_corrIC = make_corrected_coord(q0_3dIC, x0_3dIC, L)
dx0IC = x0_corrIC - q0_3dIC
print np.min(x0_corrIC),'< x0_corrIC <', np.max(x0_corrIC)
print np.min(dx0IC),'< d0IC <', np.max(dx0IC)
print



x1_corrIC = make_corrected_coord(q1_3dIC, x1_3dIC, L)
dx1IC = x1_corrIC - q1_3dIC
print np.min(x1_corrIC),'< x1_corrIC <', np.max(x1_corrIC)
print np.min(dx1IC),'< d1IC <', np.max(dx1IC)
print



x2_corrIC = make_corrected_coord(q2_3dIC, x2_3dIC, L)
dx2IC = x2_corrIC - q2_3dIC
print np.min(x2_corrIC),'< x2_corrIC <', np.max(x2_corrIC)
print np.min(dx2IC),'< d2IC <', np.max(dx2IC)
print
print '------------------------------------------------'
print '=^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^='
print










#
#dir4 = 'Snapshots'    # for z = 0           
#fn1 = 'snap'                   
#fileName = fn1+dir2+dir3+'_z0'  
#fileIn = dirIn+dir2+'/'+dir3+'/'+dir4+'/'+fileName
#
#fileIn = 'snapshot_032'


#fileIn = 'snap100Mpc128_z0'

#fileIn = 'planck_100Mpc_128'

#fileIn = 'ics1'

#fileIn = './128cube1Mpc_rho_ff/snapshot_051.gadget'

fs =  pynbody.load(fileIn)



print dir2, dir3, dir4, np.shape(fs['pos'])

pos = fs['pos']/1000
ind = fs['iord']
ind = ind-1

flip = fs['flip']
pot = fs['pot']  
rho = fs['rho']  


ngr = nGr = np.int(np.round(ind.shape[0]**(1/3.)))
#print
#print indIC
#print

x0_1d=np.zeros_like(pos[:,0])
x1_1d=np.zeros_like(pos[:,0])
x2_1d=np.zeros_like(pos[:,0])

flip_1d=np.zeros_like(pos[:,0])
pot_1d=np.zeros_like(pos[:,0])
rho_1d=np.zeros_like(pos[:,0])


# i , j , k = index1dTO3d(nGr, ind)


idx = ind

x0_1d[idx] = pos[:,0]
x1_1d[idx] = pos[:,1]
x2_1d[idx] = pos[:,2]

flip_1d[idx] = flip
#pot_1d[idx] = pot
#rho_1d[idx] = rho

#x0_1d = pos[:,0]
#x1_1d = pos[:,1]
#x2_1d = pos[:,2]








#x0_3d = np.reshape(x0_1d, (ngr,ngr,ngr),order='C')
#x1_3d = np.reshape(x1_1d, (ngr,ngr,ngr),order='C')
#x2_3d = np.reshape(x2_1d, (ngr,ngr,ngr),order='C')
#
x0_3d = np.reshape(x0_1d, (ngr,ngr,ngr),order='F')
x1_3d = np.reshape(x1_1d, (ngr,ngr,ngr),order='F')
x2_3d = np.reshape(x2_1d, (ngr,ngr,ngr),order='F')

flip_3d = np.reshape(flip_1d, (ngr,ngr,ngr),order='F')
#pot_3d = np.reshape(pot_1d, (ngr,ngr,ngr),order='F')
#rho_3d = np.reshape(rho_1d, (ngr,ngr,ngr),order='F')


#x0_3d = np.sort(x0_3d, axis = 0)
#x1_3d = np.sort(x1_3d, axis = 0)
#x2_3d = np.sort(x2_3d, axis = 0)

print '======== snap =========='
print np.min(x0_3d) , 'x0', np.max(x0_3d)
print np.min(x1_3d) , 'x1', np.max(x1_3d)
print np.min(x2_3d) , 'x2', np.max(x2_3d)
print '========================'





#dirO = dirOut+dir2+'/'+dir3+'/'+dir4

#x0_3d = np.reshape(x0_1d, (ngr,ngr,ngr),order='C')
fileOut = 'npy/'+fn1+'x0_'+'snap_050.npy'
np.save(fileOut,x0_3d)
fileOut = 'npy/'+fn1+'x1_'+'snap_050.npy'
np.save(fileOut,x1_3d)
fileOut = 'npy/'+fn1+'x2_'+'snap_050.npy'
np.save(fileOut,x2_3d)

fileOut = 'npy/'+fn1+'flip_'+'snap_050.npy'
np.save(fileOut,flip_3d)

#fileOut = 'npy/pot_'+'snap_051.npy'
#np.save(fileOut,pot_3d)
#
#fileOut = 'npy/rho_'+'snap_051.npy'
#np.save(fileOut,rho_3d)



#---------------------Correct coordinates such that |x-q|<L/2------------------
q0_3d, q1_3d, q2_3d = np.mgrid[0:nGr, 0:nGr, 0:nGr]*L/nGr   # q_i  in Mpc



q1d = np.ravel(q0_3d, order = 'F')  # 0, C works for 032, F for z0 
x1d = x0_1d     


print
print '--------------------1d--------------------------------'
x0_corr1d = make_corrected_coord(q1d, x1d, L)
dx01d = x0_corr1d - q1d
print np.min(x0_corr1d),'< x0_corr1d <', np.max(x0_corr1d)
print np.min(dx01d),'< d01d <', np.max(dx01d)
print '---------------------------------------------------------'

print

x0_corr = make_corrected_coord(q0_3d, x0_3d, L)
dx0 = x0_corr - q0_3d
print np.min(x0_corr),'< x0_corr <', np.max(x0_corr)
print np.min(dx0),'< d0 <', np.max(dx0)
print



x1_corr = make_corrected_coord(q1_3d, x1_3d, L)
dx1 = x1_corr - q1_3d
print np.min(x1_corr),'< x1_corr <', np.max(x1_corr)
print np.min(dx1),'< d1 <', np.max(dx1)
print



x2_corr = make_corrected_coord(q2_3d, x2_3d, L)
dx2 = x2_corr - q2_3d
print np.min(x2_corr),'< x2_corr <', np.max(x2_corr)
print np.min(dx2),'< d2 <', np.max(dx2)
print
print '---------------------------------------------------------'



fileOut = 'npy/'+fn1+'s0_'+'snap_050.npy'
np.save(fileOut,dx0)
fileOut = 'npy/'+fn1+'s1_'+'snap_050.npy'
np.save(fileOut,dx1)
fileOut = 'npy/'+fn1+'s2_'+'snap_050.npy'
np.save(fileOut,dx2)



