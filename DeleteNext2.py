file_nstr = './npy/numField_051_'+str(refFactor)+'.npy'
nstream = np.load(file_nstr).astype(np.float64)

from matplotlib import colors
slice_noX = 6
ybeg = zbeg = 0
yend = zend = 127

"""
labels        l3 > 0 $ nstr > 1
labels1       nstream cut 
labels2       Volume cut            - meh
labels3       Density cut           - meh
labels0       Mass cut  - Final
"""
f, ax = plt.subplots( 2,2, figsize = (12,12))
f.subplots_adjust( hspace = 0.0, wspace = 0.05)
#f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)


yticks = zticks = np.array([10, 20, 30, 40, 50])/50.
#zticks = [10, 20, 30, 40, 50]

#plt.subplot(2,2,1)
plt.sca(ax[0,0]) 
plt.gca().set_yticks(yticks)
plt.gca().set_xticks([])
plt.ylabel(r" $h^{-1} Mpc$")


nstream_2d_a = nstream[slice_noX,ybeg:yend,zbeg:zend]

#nstream_2d_a = nstream[slice_noX-2:slice_noX+2,ybeg:yend,zbeg:zend]

nstream_2d_a[nstream_2d_a >= 7] = -1

nstream_2d_a[nstream_2d_a ==  1] = 0 #0 -- Void
#nstream_2d_a[nstream_2d_a >=  7] = 100 #2 -- Filament
nstream_2d_a[nstream_2d_a >=  3] = 1 #1 -- Wall
nstream_2d_a[nstream_2d_a == -1] = 3 #3 -- Halo

#nstream_2d_a = np.sum(nstream_2d_a, axis = 0)

#nstream_2d = np.where(nstream[slice_noX,ybeg:yend,zbeg:zend] > 1, 1, 0)
cmap = colors.ListedColormap(['white', 'darkgray', 'navy'])
#bounds= np.array([0, nstrCutMin, nstrCutMax]) + 0.5
#norm = colors.BoundaryNorm(bounds, cmap.N)
#im2 = plt.imshow(nstream_2d,  alpha = 0.5, cmap = cmap , norm = norm)

im3 = ax[0,0].imshow(nstream_2d_a,  alpha = 1.0, cmap = cmap )
im3.set_extent([ ybeg*L/size_fact , yend*L/size_fact, zbeg*L/size_fact, zend*L/size_fact])


#plt.subplot(2,2,2)
plt.sca(ax[0,1])
plt.gca().set_xticks([])
plt.gca().set_yticks([])

l3_slice = l3[slice_noX,ybeg:yend,zbeg:zend]
l3binary = np.where(  ( (l3_slice > 0) &  (nstream_2d_a > 0) ) , 1, 0)
cmap = colors.ListedColormap(['white', 'black'])
img2 = ax[0,1].imshow(l3binary, cmap = cmap)
img2.set_extent([ ybeg*L/size_fact , yend*L/size_fact, zbeg*L/size_fact, zend*L/size_fact])
img2 = plt.imshow(l3_slice, alpha = 0.3, cmap = 'prism')

img2.set_extent([ ybeg*L/size_fact , yend*L/size_fact, zbeg*L/size_fact, zend*L/size_fact])






#plt.subplot(2,2,3)
plt.sca(ax[1, 0])
plt.gca().set_xticks(zticks)
plt.gca().set_yticks(yticks)
plt.xlabel(r" $h^{-1} Mpc$")
plt.ylabel(r" $h^{-1} Mpc$")


labels_2d = np.where( labels[slice_noX,ybeg:yend,zbeg:zend] > 0, 1, 0) 
labels1_2d = np.where( labels1[slice_noX,ybeg:yend,zbeg:zend] > 0, 2, 0)
#labels2_2d = np.where( labels2[slice_noX,ybeg:yend,zbeg:zend] > 0, 4, 0)
#labels3_2d = np.where( labels3[slice_noX,ybeg:yend,zbeg:zend] > 0, 8, 0)
#labels0_2d = np.where( labels0[slice_noX,ybeg:yend,zbeg:zend] > 0, 16, 0)
#labelEffect = labels_2d + labels1_2d + labels2_2d + labels3_2d + labels0_2d
#cmap = colors.ListedColormap(['white', 'green', 'blue', 'red'])
labelEffect = labels_2d + labels1_2d 

cmap = colors.ListedColormap(['white', 'red', 'black'])
bounds= np.unique(labelEffect)  
norm = colors.BoundaryNorm(bounds, cmap.N)
im1 = ax[1, 0].imshow( labelEffect, cmap = cmap)
im1.set_extent([ ybeg*L/size_fact , yend*L/size_fact, zbeg*L/size_fact, zend*L/size_fact])
#print np.unique(labelEffect)


#nstream_2d = nstream[slice_noX,ybeg:yend,zbeg:zend]
nstream_2d = np.where(nstream_2d_a > 0,  1 , 0)

#nstream_2d = np.where(nstream[slice_noX,ybeg:yend,zbeg:zend] > 1, 1, 0)
#cmap = colors.ListedColormap(['white', 'grey', 'red'])
#bounds= np.array([0, nstrCutMin, nstrCutMax]) + 0.5
#norm = colors.BoundaryNorm(bounds, cmap.N)
#im2 = plt.imshow(nstream_2d,  alpha = 0.5, cmap = cmap , norm = norm)

im2 = ax[1, 0].imshow(nstream_2d,  alpha = 0.25, cmap = 'cubehelix_r' )
im2.set_extent([ ybeg*L/size_fact , yend*L/size_fact, zbeg*L/size_fact, zend*L/size_fact])


#plt.subplot(2, 2, 4)
plt.sca(ax[1, 1])
plt.gca().set_xticks(zticks)
plt.gca().set_yticks([])
plt.xlabel(r" $h^{-1} Mpc$")


labels2_binary = np.where( labels0[slice_noX,ybeg:yend,zbeg:zend] > 0, 1, 0)
cmap = colors.ListedColormap(['white', 'black'])
img2 = ax[1,1].imshow(labels2_binary, cmap = cmap)
img2.set_extent([ ybeg*L/size_fact , yend*L/size_fact, zbeg*L/size_fact, zend*L/size_fact])

#nstream_2d = nstream[slice_noX,ybeg:yend,zbeg:zend]
#nstream_2d = np.where(nstream[slice_noX,ybeg:yend,zbeg:zend] > 1,  1 , 0)

im3 = ax[1, 1].imshow(nstream_2d,  alpha = 0.15, cmap = 'cubehelix_r' )
im3.set_extent([ ybeg*L/size_fact , yend*L/size_fact, zbeg*L/size_fact, zend*L/size_fact])


plt.savefig('plots/labels123.pdf', bbox_inches='tight')
