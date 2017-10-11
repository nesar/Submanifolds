#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: Projects11/NumberOfStreams/num_stream_fort.py
#  computes: number of streams on regular grid with arbitrary resolution
#

import numpy as np
import time
import numb_streamfull as ns
#import matplotlib.pyplot as plt
import os
from mpi4py import MPI
#===================================================
def path_create(path):
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
                os.makedirs(dir)
#===================================================
def make_corrected_coord(q,x,L):
    dx = x - q
    dx = np.where(dx >  L/2, dx-L, dx)
    dx = np.where(dx < -L/2, dx+L, dx)
    return q + dx
#-------------------------------------------------------------- 
def extend_box(Dn,comp,x):
    """ E-coordinate x in box nGr^3 is extended in box (nGr+2*Dn)^3
        using periodic conditions, comp is '0', '1', or '2'
        returns X = x in extended box
    """
    X = np.zeros((nGr+2*Dn,nGr+2*Dn,nGr+2*Dn), dtype=np.float64)
    
    # central part
    X[Dn:nGr+Dn,Dn:nGr+Dn, Dn:nGr+Dn]= x
    
    # extension along 0-axis
    X[0:Dn,            Dn:nGr+Dn, Dn:nGr+Dn] = x[nGr-Dn:nGr, :, :]
    X[nGr+Dn:nGr+2*Dn, Dn:nGr+Dn, Dn:nGr+Dn] = x[0:Dn,       :, :]
    if comp == 0:
        X[0:Dn,            Dn:nGr+Dn, Dn:nGr+Dn] -= L
        X[nGr+Dn:nGr+2*Dn, Dn:nGr+Dn, Dn:nGr+Dn] += L
        
    # extension  along 1-axis  
    X[:, 0:Dn,            Dn:nGr+Dn] = X[:,nGr:nGr+Dn, Dn:nGr+Dn]
    X[:, nGr+Dn:nGr+2*Dn, Dn:nGr+Dn] = X[:,Dn:2*Dn,   Dn:nGr+Dn]
    if comp == 1:
        X[:, 0:Dn,            Dn:nGr+Dn] -= L
        X[:, nGr+Dn:nGr+2*Dn, Dn:nGr+Dn] += L
        
    # extension  along 2-axis    
    X[:, :, 0:Dn]            = X[:,:,nGr:nGr+Dn]
    X[:, :, nGr+Dn:nGr+2*Dn] = X[:,:,Dn:2*Dn]
    if comp == 2:
        X[:, :, 0:Dn]            -= L
        X[:, :, nGr+Dn:nGr+2*Dn] += L
    return X
#===================================================================

#===================================================
def find_qbox(x_lim):
    xcond = (X0-x_lim[0])*(X0-x_lim[1])
    ycond = (X1-x_lim[2])*(X1-x_lim[3])
    zcond = (X2-x_lim[4])*(X2-x_lim[5])
    mask = np.ones_like(xcond, dtype=np.int32)
    mask0 = np.where( xcond < 0, mask, 0)
    mask1 = np.where( ycond < 0, mask0, 0)
    mask = np.where( zcond < 0, mask1, 0)
    ind_s = np.where(mask == 1)
    q_lim = np.array([np.min(ind_s[0]), np.max(ind_s[0]),
                      np.min(ind_s[1]), np.max(ind_s[1]),
                      np.min(ind_s[2]), np.max(ind_s[2])])
    return q_lim
#--------------------------------------------------- 
    
start = time.clock()
refFactor = 10 ;  # Refinment factor

L = 100.
#num = 9
factSph = 50  # multiplication factor for box



nGr = 128 # size of the grid (box)
nGr2 = nGr/2
ngr = nGr
Dn = 60


#------- Particle Coordinates E-space on 128^3 grid -------------------------

#--------- case 100Mpc/128 ---

lBox = str(int(L))+'Mpc'
dirOut = lBox+'/'+str(nGr)+'/'

dLoad = '../../Streams/Gadget2npy/'+lBox+'/'+str(nGr)+'/'+'Snapshots/'
x0_3d = np.load(dLoad+'x0'+'_snap_'+lBox+'_'+str(nGr)+'.npy')
x1_3d = np.load(dLoad+'x1'+'_snap_'+lBox+'_'+str(nGr)+'.npy')
x2_3d = np.load(dLoad+'x2'+'_snap_'+lBox+'_'+str(nGr)+'.npy')

#---------------------Correct coordinates such that |x-q|<L/2------------------
q0_3d, q1_3d, q2_3d = np.mgrid[0:nGr, 0:nGr, 0:nGr]*L/nGr   # q_i  in Mpc
x0_corr = make_corrected_coord(q0_3d, x0_3d, L)
dx0 = x0_corr - q0_3d

print np.min(q0_3d),'< q0_3d <', np.max(q0_3d),'   ', np.min(x0_corr),'< x0_corr <',\
      np.max(x0_corr),'   ', np.min(dx0),'< d0 <',\
      np.max(dx0),'mean(d0)=',np.mean(dx0),'std(d0)=',np.std(dx0)
print np.min(q0_3d),'< q0_3d <', np.max(q0_3d),'   ', np.min(x0_3d),'< x0_3d <',\
      np.max(x0_3d)
print

x1_corr = make_corrected_coord(q1_3d, x1_3d, L)
dx1 = x1_corr - q1_3d

print np.min(q1_3d),'< q1_3d <', np.max(q1_3d),'   ', np.min(x1_corr),'< x1 <',\
      np.max(x1_corr),'   ', np.min(dx1),'< d1 <',\
      np.max(dx1),'mean(d1)=',np.mean(dx1),'std(d1)=',np.std(dx1)
print

x2_corr = make_corrected_coord(q2_3d, x2_3d, L)
dx2 = x2_corr - q2_3d

print np.min(q2_3d),'< q2_3d <', np.max(q2_3d),'   ', np.min(x2_corr),'< x2 <',\
      np.max(x2_corr),'   ', np.min(dx2),'< d2 <',\
      np.max(dx2),'mean(d2)=',np.mean(dx2),'std(d2)=',np.std(dx2)
#-------------------------------------------------------------------------------

print 
print '-'*20+'Test for initial coordinates after correction '+'-'*20
print '             min         max'
print np.min(x0_corr), ' < x0_corr < ', np.max(x0_corr)
print np.min(x1_corr), ' < x1_corr < ', np.max(x1_corr)
print np.min(x2_corr), ' < x2_corr < ', np.max(x2_corr)
print  '-'*60

x0_64 = np.float64(x0_corr)
x1_64 = np.float64(x1_corr)
x2_64 = np.float64(x2_corr)



#------------FOF----------------------------


fof = np.load('../../Streams/gadget2npy/'+'FOF/'+'FOF_'+lBox+'_'+str(nGr)+'.npy')
print 'fof', np.min(fof[0]), np.max(fof[0])
print 'fof:', np.shape(fof), np.min(fof[7]), np.max(fof[7])



#htot = 1      # no. of haloes
#h_idx = np.arange(np.size(fof[0,]))       # all halo ID
#
#for i in range(htot):
#    
#    #ih = h_idx[i]  # largest tot haloes
#    ih = h_idx[np.size(fof[0,])/2-5:np.size(fof[0,])/2 + 5][i]   # middle haloes
#    #ih = h_idx[np.size(fof[0,]):np.size(fof[0,])-(tot+1):-1][i]   # smallest haloes
#    print 'Halo index', ih
#
#

#hidx = range(3, 6)
#hidx = range(79, 82)
#hidx = range(706, 709)
#hidx = range(4322, 4325)

#hidx = range(900, 901)

#hidx = np.array([3, 79, 706, 4322])  #128
hidx = np.array([627, 3435, 94])  #256
#hidx = np.array([3, 109, 715, 6536]) #512
hidx = np.arange(10,2000,20)

#hidx = np.array([1234,1235,1236,1237, 1238, 1239])


comm = MPI.COMM_WORLD
#print type(comm)
rank = comm.Get_rank()
size = comm.Get_size()
#print "hello world from process ", rank, "size:", size

if (np.size(hidx)%size) != 0: 
    print "No. of jobs is:", np.size(hidx)
    print "Change the no. of processes"
    comm.Abort()








niter = np.size(hidx)/size   # no. iteration for each processor(rank)
for ite in range(niter):

    ih =  hidx[rank +ite*size]









#for hid in range(np.size(hidx)):
    #ih =  hidx[hid]  
    print 'Halo index', ih
    
    
    print fof[0,ih],fof[1,ih],fof[2,ih]
    boxCenterMpc = np.array([fof[0,ih],fof[1,ih],fof[2,ih]]) # Small Box center 
    print 'box Center', boxCenterMpc
    
    
    boxSizeMpc = factSph*np.array([fof[10,ih],fof[10,ih],fof[10,ih]])   # Small Box  Size 
    boxSize = np.int32(np.around(boxSizeMpc*nGr/L))
    #print boxSizeMpc, boxSize
    
    print lBox, nGr,
    print '%6d %12.4e %12.4f (%12.4f %12.4f %12.4f)'% (ih, 
                fof[7,ih],fof[10,ih],fof[0,ih],fof[1,ih],fof[2,ih])
                
                
    #------------FOF----------------------------
    
    
    #
    #xmin = 0; xmax = L
    #xegr = np.linspace(xmin,xmax,n_fine_grid[0]) 
    #yegr = np.linspace(xmin,xmax,n_fine_grid[1])
    #zegr = np.linspace(xmin,xmax,n_fine_grid[2])
    #
    #dgr = np.array([xegr[1]-xegr[0],yegr[1]-yegr[0],zegr[1]-zegr[0]])
    #print 'n_fine_grid', n_fine_grid, 'dgr=',dgr
    
    
    #==============================BIG BOX==========================

    X0 = extend_box(Dn, 0, x0_64)  # in Mpc
    X1 = extend_box(Dn, 1, x1_64)
    X2 = extend_box(Dn, 2, x2_64)
    #--------------------------------------------------------------------
    
    #------------------------ Parameters of Small Box in E-space  
    #cubeSize = np.array(boxSize)
    xBeg = boxCenterMpc - boxSizeMpc/2 # Limits on E-grid
    xEnd = boxCenterMpc + boxSizeMpc/2
    #print 'E-box:', 'boxSize=', boxSize
    
    #print 'indBeg=', indBeg
    #print 'indEnd=', indEnd
    x_limMpc =np.array([xBeg[0],xEnd[0],xBeg[1],xEnd[1],xBeg[2],xEnd[2]])
    
    #-------------------------- Find limits in L-space 
    print
    print 'boxCenterMpc=', boxCenterMpc
    #print 'boxSize=', boxSize, 'refFactor=', refFactor
    #print 'x_lim(grid)=',  x_limMpc *nGr/L
    print 'x_lim(Mpc)=',  x_limMpc
    print 'BOX Mpc', x_limMpc[1]-x_limMpc[0],x_limMpc[3]-x_limMpc[2],x_limMpc[5]-x_limMpc[4]
    print 'boxCenterMpc(Mpc) =',  fof[0,ih],fof[1,ih],fof[2,ih]
    print
    x_limMpc_64=np.float64(x_limMpc)
    
    
    q_lim = find_qbox(x_limMpc)
    #raw_input()
    dn_l = 2 # Add dn_l layers to selected box
    print 'dn_l=', dn_l
    q_lim[0] -=dn_l; q_lim[2] -=dn_l; q_lim[4] -=dn_l   
    q_lim[1] +=dn_l; q_lim[3] +=dn_l; q_lim[5] +=dn_l
    #print 'small_cube.f done-------------------------------'
    print 'q_lim=', q_lim
    
    #for iq in [0,1,2,3,4,5]:
    #    if q_lim[iq] < 0:
    #        q_lim[iq] =0 
    #        print '=============> q_lim set to Zero', iq, q_lim[iq]
    #        raw_input()
    #    if  q_lim[iq] > nGr-1:
    #        q_lim[iq] = nGr-1
    #        print '=============> q_lim set to nGr', iq, q_lim[iq]
    #        raw_input()                
    ## ================Limits in L-space found ===============================
    #
    nf0 = np.int32(np.around(refFactor*boxSize[0]))    # Sizes of Refined Grid
    nf1 = np.int32(np.around(refFactor*boxSize[1]))
    nf2 = np.int32(np.around(refFactor*boxSize[2]))
    n_fine_grid = np.array([nf0,nf1,nf2])
                                         # Coordinates on fine grid
    xegr = np.linspace(x_limMpc[0],x_limMpc[1],n_fine_grid[0]) 
    yegr = np.linspace(x_limMpc[2],x_limMpc[3],n_fine_grid[1])
    zegr = np.linspace(x_limMpc[4],x_limMpc[5],n_fine_grid[2])
    
    dgr = np.array([xegr[1]-xegr[0],yegr[1]-yegr[0],zegr[1]-zegr[0]])
    #print 'n_fine_grid', n_fine_grid, 'dgr=',dgr
    
    q_lim_64 = np.int64(q_lim)
    dgr_64 = np.float64(dgr)
    nf0_64 = np.int64(nf0)
    nf1_64 = np.int64(nf1)
    nf2_64 = np.int64(nf2)
    #print 'before numb_stream'
    
    #==================================================================
    #q_lim_64 = [0, nGr+2*Dn, 0, nGr+2*Dn,0, nGr+2*Dn]
    #x_lim_64 = np.array([0.,L, 0.,L,0.,L])
    
    
    
    nstream = ns.numb_streams(q_lim_64,X0,X1,X2, x_limMpc,dgr_64,nf0_64,nf1_64,nf2_64) # FORTRAN SUBR.(computes n_str field)
    #
    print ' after numb_stream'
    nTot =  np.size(nstream)
    print 'size(nstream)=', nTot, 'shape(nstream)=', np.shape(nstream)
    nTot = np.float64(nTot)
    print 
    print 'min(nstream)=',np.min(nstream),  'max(nstream)=',np.max(nstream)
    print '-'*50


    ns_max = np.max(nstream)
    nstream_64 = np.int64(nstream)
    ns_max_64 = np.int64(ns_max)
    #count = ns.count_streams(nstream_64,ns_max_64)     # FORTRAN SUBR.
    #count_out = np.zeros((ns_max_64+1,2), dtype = np.int32)
    #print 'shape(count)=', np.shape(count)
    #print count[0:22]
    #print np.sum(count), nGr**3
    #print '-'*60


    dirOutput = '../HaloData/npy/stat/'
    haloDir = str(ih)+'/'
    fileOut = dirOutput+dirOut+haloDir+'halo_'+str(ih)+'_numField_'+str(int(L))+'_'+str(nGr)+'_'+str(Dn)+'_'+str(refFactor)+'_'+str(factSph)+'.npy'
    print 'fileOut=', fileOut
    
    path_create(fileOut)
    np.save(fileOut,np.int32(nstream))
    
#fileOut = dirOutput+dirOut+'strFrac_'+str(int(L))+'_'+str(nGr)+'_'+str(Dn)+'_'+str(refFactor)+'.npy'
#print 'fileOut=', fileOut
#np.save(fileOut,np.float32(count))

print
end = time.clock()
ElapsedTime = (end - start)/60
print 'Execution time: %4.2e min' %ElapsedTime

##------------ 3d plot -----------------
#from enthought.mayavi import mlab
#nx1 = x_limMpc[0]; nx2=x_limMpc[1]
#ny1 = x_limMpc[2]; ny2=x_limMpc[3]
#nz1 = x_limMpc[4]; nz2=x_limMpc[5]
#im = nf0*1j
#jm = nf1*1j
#km = nf2*1j
#print 'q_lim=', q_lim
#X, Y, Z = np.mgrid[nx1:nx2:im,ny1:ny2:jm,nz1:nz2:km]
#title ='N-streams'
#
#levels=np.array([10., 100., 1000., 10000.])
##levels = np.array([30, 60, 70, 90, 200, 500])
#
#levels= list(np.log10(levels))
##levels = list(levels)
#
#print 'levels=', levels
#
#p_c0 =  [boxCenterMpc[0]]
#p_c1 =  [boxCenterMpc[1]]
#p_c2 =  [boxCenterMpc[2]]
#s=[0.05]
#
#field =  np.log10(nstream )
##field = nstream
#
#
##for ilev in xrange(len(levels)):
#mlab.figure(1,bgcolor=(1,1,1), fgcolor=(0,0,0), size = (400,350))
#mlab.clf()  
#mlab.get_engine()               
#mlab.contour3d(X,Y,Z, field, contours=levels, opacity=0.4, 
#      transparent = True, colormap='jet')
#mlab.points3d(p_c0,p_c1,p_c2, s, scale_factor=1, opacity=1)
#
#mlab.axes(extent = [nx1,nx2,ny1,ny2,nz1,nz2])
#mlab.outline(extent = [nx1,nx2,ny1,ny2,nz1,nz2])
#    #mlab.savefig('GadgetHalos/Tests/image'+str(ilev)+'.png')
#    #mlab.colorbar(title=title, orientation='vertical', nb_labels=2)
#mlab.show()