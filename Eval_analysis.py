import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt


L = 1.
nGr = 128
refFactor = 1
Dn = 60      # 60 for all except 100Mpc-512 (160)  # 0 for small box
size_fact = nGr*refFactor
#hidx = 4322
dirOutput = "Out"

sig = 0.0
plt.clf()



#============================= DENSITY fiel ===============================================================================


#l = np.load("/Users/nesar/Desktop/Evals/FullBox/Evals3_"+str(int(L))+'_'+str(int(nGr))+'_'+str(refFactor)++'_'+str(int(100*sig))+'.npy') 
#l = np.load("./npy/Half/Evals3_032_"+str(refFactor)+".npy")
l = np.load("./npy/Evals3_051.npy")


l1 = np.ravel(l[:,:,:,0])
l2 = np.ravel(l[:,:,:,1])
l3 = np.ravel(l[:,:,:,2])



#nbins = np.linspace(-xlimits, xlimits, nbins)

#nbins1 = np.logspace(np.log10(xlimits), np.log10(1e-5), nbins/2)
#nbins = np.append(-nbins1, nbins1[::-1])

#nbins1 = np.logspace(np.log10(1e-5), np.log10(xlimits), nbins/2)
#nbins = np.append(-nbins1[::-1], nbins1)


plt.figure(12324)

#y,binEdges = np.histogram( l1, bins = nbins, density= False)
#bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
#
#plt.plot(bincenters, y,'r-', lw = 2, label = r"$\lambda_1$")
#
#y,binEdges = np.histogram( l2, bins = nbins, density= False)
#bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
#
#plt.plot(bincenters, y,'g-', lw = 2, label = r"$\lambda_2$")
#
#y,binEdges = np.histogram( l3, bins = nbins, density= False)
#bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
#
#plt.plot(bincenters, y,'b-', lw = 2, label = r"$\lambda_3$")


#plt.errorbar(bincenters, y, 
#xerr = np.sqrt(y),  fmt='--b',  elinewidth=2 , lw = 2,
#label = r"$\lambda_1$")

#np.cumsum()

xlimits = 0.1
nbins = 200

l1 = l1[ np.abs(l1) < xlimits  ]
l2 = l2[ np.abs(l2) < xlimits  ]
l3 = l3[ np.abs(l3) < xlimits  ]

plt.hist(l1, bins = nbins,  log = False, 
#normed = 1, 
facecolor = 'r', histtype='stepfilled',alpha=0.5)

plt.hist(l2, bins = nbins, log = False, 
#normed = 1, 
facecolor = 'g', histtype='stepfilled',alpha=0.5)

plt.hist(l3, bins = nbins, log = False, 
#normed = 1, 
facecolor = 'b', histtype='stepfilled',alpha=0.5)
#plt.ylim(1e-4, )
#plt.xlim(-xlimits, xlimits)



#plt.yscale('log')
#plt.xscale('log')
plt.ylabel(r"pdf")
plt.xlabel(r'$\lambda$s') 
plt.legend(loc = "upper right")
plt.savefig('plots/Eval_lambda.pdf', bbox_inches='tight')

plt.show()


#import sys
#sys.exit()

plt.figure(3453)

plt.subplot(2,3,1)

y,binEdges = np.histogram(l1, bins = nbins, density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'r-', lw = 2, label = r"$\lambda_1$")
plt.yscale('log')


plt.ylabel("pdf")
plt.xlabel(r"$\lambda$")
plt.legend(loc = "upper left")

plt.subplot(2,3,2)

y,binEdges = np.histogram(l2,bins = nbins, density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'g-', lw = 2, label = r"$\lambda_2$")
plt.yscale('log')

plt.ylabel("pdf")
plt.xlabel(r"$\lambda$")
plt.legend(loc = "upper left")


plt.subplot(2,3,3)


y,binEdges = np.histogram(l3, bins = nbins, density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'b-', lw = 2, label = r"$\lambda_3$")
plt.yscale('log')

plt.ylabel("pdf")
plt.xlabel(r"$\lambda$")
plt.legend(loc = "upper left")


plt.subplot(2,3,4)
#hist, bin_edges = np.histogram(para3, density=True)
y,binEdges = np.histogram(l1, bins = nbins, density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'r-', lw = 2, label = r"$\lambda_1$")
plt.xscale('symlog')
plt.yscale('log')


plt.ylabel("pdf")
plt.xlabel(r"$\lambda$")
plt.legend(loc = "upper left")


plt.subplot(2,3,5)

y,binEdges = np.histogram(l2, bins = nbins, density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'g-', lw = 2, label = r"$\lambda_2$")
plt.xscale('symlog')
plt.yscale('log')

plt.ylabel("pdf")
plt.xlabel(r"$\lambda$")
plt.legend(loc = "upper left")

plt.subplot(2,3,6)


y,binEdges = np.histogram(l3, bins = nbins, density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'b-', lw = 2, label = r"$\lambda_3$")
plt.xscale('symlog')
plt.yscale('log')

plt.ylabel("pdf")
plt.xlabel(r"$\lambda$")
plt.legend(loc = "upper left")





plt.figure(2)
plt.subplot(2,2,1)

y,binEdges = np.histogram(l1, bins = np.linspace(-xlimits, xlimits, nbins), density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'r-', lw = 2, label = r"$\lambda_1$")


y,binEdges = np.histogram(l2,bins = np.linspace(-xlimits, xlimits, nbins), density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'g-', lw = 2, label = r"$\lambda_2$")

y,binEdges = np.histogram(l3, bins = np.linspace(-xlimits, xlimits, nbins), density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'b-', lw = 2, label = r"$\lambda_3$")


#plt.xlim(-xlimits, xlimits)
#plt.ylim(1e-2, 10)
plt.yscale('log')
plt.ylabel("pdf")
plt.xlabel(r"$\lambda$")
plt.legend(loc = "upper left")

plt.figure(2)
plt.subplot(2,2,2)
#hist, bin_edges = np.histogram(para3, density=True)
y,binEdges = np.histogram(l1, bins = nbins, density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'r-', lw = 2, label = r"$\lambda_1$")


y,binEdges = np.histogram(l2, bins = nbins, density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'g-', lw = 2, label = r"$\lambda_2$")


y,binEdges = np.histogram(l3, bins = nbins, density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'b-', lw = 2, label = r"$\lambda_3$")
plt.xscale('symlog')
plt.yscale('log')

plt.ylabel("pdf")
plt.xlabel(r"$\lambda$")
plt.legend(loc = "upper left")



plt.figure(2)
plt.subplot(2,2,3)
#hist, bin_edges = np.histogram(para3, density=True)
y,binEdges = np.histogram(np.ravel(l[:,:,:,0] - l[:,:,:,1]), bins = nbins, density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'r-', lw = 2, label = r"$\lambda_3 - \lambda_2 $")



y,binEdges = np.histogram(np.ravel(l[:,:,:,1] - l[:,:,:,2]), bins =  nbins, density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'g-', lw = 2, label = r"$\lambda_2 - \lambda_1$")


#y,binEdges = np.histogram(l3, bins = np.logspace(np.min(l[:,:,:,2]), np.max(l[:,:,:,2]), 100),  density=True)
##y,binEdges = np.histogram(l3, bins = 500, density=True)
#bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
#plt.plot(bincenters,y,'b-o', lw = 2, label = r"$\lambda_3$")


#plt.xscale('symlog')
plt.yscale('log')
plt.ylabel("pdf")
plt.xlabel(r"$\lambda_{diff}$")
plt.legend(loc = "upper right")


plt.figure(2)
plt.subplot(2,2,4)
#hist, bin_edges = np.histogram(para3, density=True)
y,binEdges = np.histogram(l1, bins = np.logspace(np.min(l[:,:,:,0]), np.max(l[:,:,:,0]), nbins),  density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
y = np.cumsum(y)
plt.plot(bincenters,y,'r-o', lw = 2, label = r"$\lambda_1$")
#plt.xscale('symlog')
plt.yscale('log')

y,binEdges = np.histogram(l2, bins = np.logspace(np.min(l[:,:,:,1]), np.max(l[:,:,:,1]), nbins),  density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
y = np.cumsum(y)
plt.plot(bincenters,y,'g-o', lw = 2, label = r"$\lambda_2$")
#plt.xscale('symlog')
plt.yscale('log')

y,binEdges = np.histogram(l3, bins = np.logspace(np.min(l[:,:,:,2]), np.max(l[:,:,:,2]), nbins),  density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
y = np.cumsum(y)
plt.plot(bincenters,y,'b-o', lw = 2, label = r"$\lambda_3$")
#plt.xscale('symlog')
plt.yscale('log')

plt.ylabel("cdf")
plt.xlabel(r"$\lambda$")
plt.legend(loc = "upper right")


plt.savefig("plots/123Evals.pdf",bbox_inches="tight")



plt.show()

#import sys
#sys.exit()
#============================= MULTISTREAM field ===============================================================================

# Halo - SmallBox

#lBox = str(int(L))+'Mpc'
#dir1 = lBox+'/'+str(nGr)+'/'
#dirIn = '/Volumes/nesar/para/HaloData/npy/'
#haloDir = str(hidx)+'/'
#
#fOut = dirIn+dir1+haloDir+'Eigen/'
#l = np.load(fOut+'Eval3_'+str(hidx)+'.npy')
#dirOutput = str(hidx)

#======================


#####===============================#FullBox - refFactor = 1


l = np.load("npy/Evals3_032.npy")
#l = np.load('Out/Evals3_100_128_1.npy')  # Dropbox - Trial

l1 = np.ravel(l[:,:,:,0])
l2 = np.ravel(l[:,:,:,1])
l3 = np.ravel(l[:,:,:,2])

plt.figure(1)
plt.subplot(2,2,1)
#hist, bin_edges = np.histogram(para3, density=True)
y,binEdges = np.histogram(l1, bins = np.linspace(-xlimits, xlimits, nbins), density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'r-', lw = 2, label = r"$\lambda_1$")


y,binEdges = np.histogram(l2, bins = np.linspace(-xlimits, xlimits, nbins), density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'g-', lw = 2, label = r"$\lambda_2$")

y,binEdges = np.histogram(l3, bins = np.linspace(-xlimits, xlimits, nbins), density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'b-', lw = 2, label = r"$\lambda_3$")

#plt.yscale('log')
plt.ylabel("pdf")
plt.xlabel(r"$\lambda$")
plt.legend(loc = "upper left")

plt.figure(1)
plt.subplot(2,2,2)
#hist, bin_edges = np.histogram(para3, density=True)
y,binEdges = np.histogram(l1, bins = nbins, density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'r-', lw = 2, label = r"$\lambda_1$")


y,binEdges = np.histogram(l2, bins = nbins, density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'g-', lw = 2, label = r"$\lambda_2$")


y,binEdges = np.histogram(l3, bins = nbins, density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'b-', lw = 2, label = r"$\lambda_3$")
plt.xscale('symlog')
plt.yscale('log')

plt.ylabel("pdf")
plt.xlabel(r"$\lambda$")
plt.legend(loc = "upper left")



plt.figure(1)
plt.subplot(2,2,3)
#hist, bin_edges = np.histogram(para3, density=True)
y,binEdges = np.histogram(np.ravel(l[:,:,:,0] - l[:,:,:,1]), bins = nbins, density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'r-', lw = 2, label = r"$\lambda_3 - \lambda_2 $")



y,binEdges = np.histogram(np.ravel(l[:,:,:,1] - l[:,:,:,2]), bins = nbins, density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'g-', lw = 2, label = r"$\lambda_2 - \lambda_1$")


#y,binEdges = np.histogram(l3, bins = np.logspace(np.min(l[:,:,:,2]), np.max(l[:,:,:,2]), 100),  density=True)
##y,binEdges = np.histogram(l3, bins = 500, density=True)
#bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
#plt.plot(bincenters,y,'b-o', lw = 2, label = r"$\lambda_3$")


#plt.xscale('symlog')
plt.yscale('log')
plt.ylabel("pdf")
plt.xlabel(r"$\lambda_{diff}$")
plt.legend(loc = "upper right")


plt.figure(1)
plt.subplot(2,2,4)
#hist, bin_edges = np.histogram(para3, density=True)
y,binEdges = np.histogram(l1, bins = np.logspace(np.min(l[:,:,:,0]), np.max(l[:,:,:,0]), nbins), density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
y = np.cumsum(y)
plt.plot(bincenters,y,'r-o', lw = 2, label = r"$\lambda_1$")
#plt.xscale('symlog')
plt.yscale('log')

y,binEdges = np.histogram(l2, bins = np.logspace(np.min(l[:,:,:,1]), np.max(l[:,:,:,1]), nbins), density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
y = np.cumsum(y)
plt.plot(bincenters,y,'g-o', lw = 2, label = r"$\lambda_2$")
#plt.xscale('symlog')
plt.yscale('log')

y,binEdges = np.histogram(l3, bins = np.logspace(np.min(l[:,:,:,2]), np.max(l[:,:,:,2]), nbins),  density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
y = np.cumsum(y)
plt.plot(bincenters,y,'b-o', lw = 2, label = r"$\lambda_3$")
#plt.xscale('symlog')
plt.yscale('log')

plt.ylabel("cdf")
plt.xlabel(r"$\lambda$")
plt.legend(loc = "upper right")


plt.savefig("plots/MultistreamEvals.pdf",bbox_inches="tight")

#====================================================================================================================================================================

plt.figure(3)


plt.subplot(3,1,1)
y,binEdges = np.histogram(l1, bins = np.linspace(-xlimits, xlimits, nbins), density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'r-', lw = 2, label = r"$\lambda_1 - Multi$")

y,binEdges = np.histogram(l1, bins = np.linspace(-xlimits, xlimits, nbins), density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'b-', lw = 2, label = r"$\lambda_1 - Den$")

#plt.yscale('log')
plt.ylabel("pdf")
plt.xlabel(r"$\lambda$")
plt.legend(loc = "upper left")


plt.subplot(3,1,2)
y,binEdges = np.histogram(l2, bins = np.linspace(-xlimits, xlimits, nbins), density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'r-', lw = 2, label = r"$\lambda_2 - Multi$")

y,binEdges = np.histogram(l2, bins = np.linspace(-xlimits, xlimits, nbins), density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'b-', lw = 2, label = r"$\lambda_2 - Den$")

#plt.yscale('log')
plt.ylabel("pdf")
plt.xlabel(r"$\lambda$")
plt.legend(loc = "upper left")

plt.subplot(3,1,3)
y,binEdges = np.histogram(l3, bins = np.linspace(-xlimits, xlimits, nbins), density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'r-', lw = 2, label = r"$\lambda_3 - Multi$")

y,binEdges = np.histogram(l3, bins = np.linspace(-xlimits, xlimits, nbins), density=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'b-', lw = 2, label = r"$\lambda_3 - Den$")

#plt.yscale('log')
plt.ylabel("pdf")
plt.xlabel(r"$\lambda$")
plt.legend(loc = "upper left")


plt.savefig("plots/den_Multistream_compare.pdf",bbox_inches="tight")



print '================================='
print np.correlate(l1,l1)
print np.correlate(l2,l2)
print np.correlate(l3,l3)
print '================================='
print np.corrcoef(l1,l1)
print np.corrcoef(l2,l2)
print np.corrcoef (l3,l3)
print '================================='
#print np.correlate(l1,l1)
#print np.correlate(l2,l2)
#print np.correlate(l3,l3)



plt.figure(4)
l1_red = l1[::10]
l2_red = l2[::10]
l3_red = l3[::10]
plt.subplot(2,2,1)
plt.scatter(l1_red, l2_red, s = 100)
plt.plot([-6,0,2],[-6,0,2], 'r', lw = 2)
plt.xlabel(r"$\lambda_1$")
plt.ylabel(r"$\lambda_2$")


plt.subplot(2,2,2)
plt.scatter(l2_red, l3_red, s = 100)
plt.plot([-10,0,2],[-10,0,2], 'r', lw = 2)
plt.xlabel(r"$\lambda_2$")
plt.ylabel(r"$\lambda_3$")

plt.subplot(2,2,3)
plt.scatter(l1_red, l3_red, s = 100)
#plt.plot([-10,0,2],[-10,0,2], 'r', lw = 3)
plt.xlabel(r"$\lambda_1$")
plt.ylabel(r"$\lambda_3$")

plt.show()



plt.savefig("plots/EvalScatter.pdf",bbox_inches="tight")



import sys
sys.exit("Check")


from evtk.hl import pointsToVTK 

	
npoints = np.size(l1) 
     
scalars = np.ones_like(l1)    
    
pointsToVTK("./points", l1,l2,l3, data = {"scalars" : scalars})



#==================================================================================================================================================================



import toParaview as ToPara


OutFileName = dirOutput+"/npy/Eval1"
ToPara.StructuredScalar(l[:,:,:,0], OutFileName, 0, L)


OutFileName = dirOutput+"/npy/Eval2"
ToPara.StructuredScalar(l[:,:,:,1], OutFileName, 0, L)

OutFileName = dirOutput+"/npy/Eval3"
ToPara.StructuredScalar(l[:,:,:,2], OutFileName, 0, L)

