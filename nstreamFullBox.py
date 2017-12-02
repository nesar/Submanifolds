#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: Projects11/NumberOfStreams/num_stream_fort.py
#  computes: number of streams on regular grid with arbitrary resolution
#

import numpy as np
from time import *
import numb_streamfull as ns
#import matplotlib.pyplot as plt
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
    
time1=int(time())
refFactor = 4   # Refinment factor

L = 100.

nGr = 128 # size of the grid (box)
nGr2 = nGr/2
ngr = nGr
Dn = 60
#------- Particle Coordinates E-space on 128^3 grid -------------------------

#--------- case 100Mpc/128 ---

lBox = str(int(L))+'Mpc'
dirOut = lBox+str(nGr)+'/'

dLoad = './npy/'+dirOut
x0_3d = np.load(dLoad+'x0'+'_snap_050.npy')
x1_3d = np.load(dLoad+'x1'+'_snap_050.npy')
x2_3d = np.load(dLoad+'x2'+'_snap_050.npy')

#---------------------Correct coordinates such that |x-q|<L/2------------------
q0_3d, q1_3d, q2_3d = np.mgrid[0:nGr, 0:nGr, 0:nGr]*L/nGr   # q_i  in Mpc

#q0_3d = np.load(dLoad+'x0'+'_snap_000.npy')
#q1_3d = np.load(dLoad+'x1'+'_snap_000.npy')
#q2_3d = np.load(dLoad+'x2'+'_snap_000.npy')

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

#---------------------------- Sizes of Refined Grid
nf0 = np.int16(refFactor*nGr)    
nf1 = np.int16(refFactor*nGr)
nf2 = np.int16(refFactor*nGr)

x_lim = np.array([0,nf0,0,nf1,0,nf2])
n_fine_grid = np.array([nf0,nf1,nf2])
#--------------------------- Coordinates on fine grid
xmin = 0; xmax = L
xegr = np.linspace(xmin,xmax,n_fine_grid[0]) 
yegr = np.linspace(xmin,xmax,n_fine_grid[1])
zegr = np.linspace(xmin,xmax,n_fine_grid[2])

dgr = np.array([xegr[1]-xegr[0],yegr[1]-yegr[0],zegr[1]-zegr[0]])
print 'n_fine_grid', n_fine_grid, 'dgr=',dgr


#==============================BIG BOX==========================

X0 = extend_box(Dn, 0, x0_64)  # in Mpc
X1 = extend_box(Dn, 1, x1_64)
X2 = extend_box(Dn, 2, x2_64)
#--------------------------------------------------------------------

#==================================================================
q_lim_64 = [0, nGr+2*Dn, 0, nGr+2*Dn,0, nGr+2*Dn]
x_lim_64 = np.array([0.,L, 0.,L,0.,L])

#
#q_lim_64 = [Dn, nGr+Dn, Dn, nGr+Dn,Dn, nGr+Dn]    # Wrong for nstreams
#x_lim_64 = np.array([Dn,nGr+Dn, Dn,nGr+Dn,Dn,nGr+Dn])/ngr*L


dgr_64 = np.float64([L/(refFactor*nGr),L/(refFactor*nGr),L/(refFactor*nGr)])
nf0_64 = np.int64(nf0)
nf1_64 = np.int64(nf1)
nf2_64 = np.int64(nf2)
print '-'*50
print 'before numb_stream'
print 'q_lim=', q_lim_64
#X0 = X0/L*nGr; X1 = X1/L*nGr; X2 = X2/L*nGr
print  np.min(X0),' < X0 <', np.max(X0)
print  np.min(X1),' < X1 <', np.max(X1)
print  np.min(X2),' < X2 <', np.max(X2)
print 'x_lim=', x_lim_64
print 'dgr_64=', dgr_64
print  'Sizes of refined grid:   nf0,nf1,nf2=', nf0_64,nf1_64,nf2_64
print '-'*50
#raw_input()
nstream = ns.numb_streams(q_lim_64,X0,X1,X2, x_lim_64,dgr_64,nf0_64,nf1_64,nf2_64) # FORTRAN SUBR.(computes n_str field)
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
count = ns.count_streams(nstream_64,ns_max_64)     # FORTRAN SUBR.
count_out = np.zeros((ns_max_64+1,2), dtype = np.int32)
print 'shape(count)=', np.shape(count)
print count[0:22]
print np.sum(count), nGr**3
print '-'*60



#dirOutput = '../fullData/npy/'
#fileOut = dirOutput+dirOut+'numField_'+str(int(L))+'_'+str(nGr)+'_'+str(Dn)+'_'+str(refFactor)+'.npy'


fileOut = dLoad+'numField_050_'+str(refFactor)+'.npy'
print 'fileOut=', fileOut
np.save(fileOut,np.int32(nstream))

#fileOut = dirOutput+dirOut+'strFrac_'+str(int(L))+'_'+str(nGr)+'_'+str(Dn)+'_'+str(refFactor)+'.npy'
#print 'fileOut=', fileOut
#np.save(fileOut,np.float32(count))

print 'numb_stream time', int(time())-time1, 'sec'
time1=int(time())

############################################################


import sys
sys.exit()

nx1 = ny1 = nz1 = 0
nx2 = ny2 = nz2 = int(L)
im = jm = km = nGr*refFactor*1j

nf0 = nf1 = nf2 = refFactor*nGr

Xg, Yg, Zg = np.mgrid[nx1:nx2:im,ny1:ny2:jm,nz1:nz2:km]
# Make the data.
dims = np.array((nf0, nf1, nf2))
vol = np.array((0,L,0,L,0,L))
origin = vol[::2]
spacing = (vol[1::2] - origin)/(dims -1)
xmin, xmax, ymin, ymax, zmin, zmax = vol                
x, y, z = np.ogrid[xmin:xmax:dims[0]*1j,
                ymin:ymax:dims[1]*1j,
                zmin:zmax:dims[2]*1j]
x, y, z = [t.astype('f') for t in (x, y, z)]



from enthought.tvtk.api import tvtk



spoints = tvtk.StructuredPoints(origin=origin, spacing=spacing,
                                dimensions=dims)
                                

#scalars = np.log10(den +1)
scalars = nstream

# Make the tvtk dataset.
spoints = tvtk.StructuredPoints(origin=origin, spacing=spacing,
                                dimensions=dims)
                                
s = scalars.transpose().copy()
spoints.point_data.scalars = np.ravel(s) 
spoints.point_data.scalars.name = 'scalars'


dirOutput = "./vti/"
fileOut = dirOutput+dirOut+'nst_'+str(int(L))+'_'+str(nGr)+'_'+str(Dn)+'_'+str(refFactor)+'.vti'

fileOut = 'vti/nst_051_'+str(refFactor)+'.vti'
w = tvtk.XMLImageDataWriter(input=spoints, file_name=fileOut)
w.write()
