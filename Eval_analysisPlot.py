import matplotlib.pylab as plt
import numpy as np


def set_pub():
    """ Pretty plotting changes in rc for publications
    Might be slower due to usetex=True
    
    
    plt.minorticks_on()  - activate  for minor ticks in each plot

    """
    plt.rc('font', weight='bold')    # bold fonts are easier to see
    #plt.rc('font',family='serif')
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rc('text', usetex=True)   # Slower
    plt.rc('font',size=18)
    
    plt.rc('lines', lw=1, color='k', markeredgewidth=1.5) # thicker black lines
    #plt.rc('grid', c='0.5', ls='-', lw=0.5)  # solid gray grid lines
    plt.rc('savefig', dpi=300)       # higher res outputs
    
    plt.rc('xtick', labelsize='x-large')
    plt.rc('ytick', labelsize='x-large')
    plt.rc('axes',labelsize= 30)
    
    plt.rcParams['xtick.major.size'] = 12
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['xtick.minor.size'] = 8
    plt.rcParams['xtick.minor.width'] = 1
    
    plt.rcParams['ytick.major.size'] = 12
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['ytick.minor.size'] = 8
    plt.rcParams['ytick.minor.width'] = 1
    
    plt.rcParams['axes.color_cycle'] = ['navy', 'forestgreen', 'darkred']

set_pub()




def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None):
    
    #https://tonysyu.github.io/plotting-error-bars.html
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color = color, lw = 1.5, label = strLabel)
    #ax.plot(x, y, color+'o', label = strLabel)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)

refFactor = 1
nGr = 128
size_fact = refFactor*nGr
L = 1.

l = np.load("./npy/Evals3_051.npy")

#l = np.load("/home/nesar/Dropbox/Evals3_100_128_2.npy")


l1 = np.ravel(l[:,:,:,0])
l2 = np.ravel(l[:,:,:,1])
l3 = np.ravel(l[:,:,:,2])



plt.figure(4, figsize = (8,6))

xlimits = 100000

nbins = 100000

#l1 = l1[np.abs(l1) < xlimits]
#l2 = l2[np.abs(l2) < xlimits]
#l3 = l3[np.abs(l3) < xlimits]

#nbins = np.linspace(np.min(l), np.max(l), nbins)

#nbins1 = np.linspace(np.min(l), -xlimits, 10)
#nbins2 = np.linspace(-xlimits, xlimits, nbins)
#nbins3 = np.linspace(xlimits, np.max(l), 10)
#nbins = np.hstack((nbins1, nbins2, nbins3))


#plt.subplot(3,1,1)
y,binEdges = np.histogram(l1, bins = nbins, density=False)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'-', lw = 1.5, label = r"$\lambda_1$")
plt.yscale('log')


#plt.yscale('log')
#plt.xscale('symlog')
#plt.ylabel("PDF")
#plt.xlabel(r"$\lambda$")
#plt.legend(loc = "upper left")
#plt.xlim(-xlimits, xlimits)
#plt.ylim(1e-5,)

#plt.subplot(3,1,2)
y,binEdges = np.histogram(l2, bins = nbins, density=False)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'-', lw = 1.5, label = r"$\lambda_2$")


#plt.yscale('log')
##plt.xscale('symlog')
#plt.ylabel("PDF")
#plt.xlabel(r"$\lambda$")
#plt.legend(loc = "upper left")
#plt.xlim(-xlimits, xlimits)
#plt.ylim(1e-5,)

#plt.subplot(3,1,3)
y,binEdges = np.histogram(l3, bins = nbins, density=False)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'-', lw = 1.5, label = r"$\lambda_3$")

#strLabel = r"$\lambda_3$"
#errorfill(bincenters, y, np.sqrt(y), color='r', alpha_fill=0.3)

plt.yscale('log')
#plt.xscale('symlog')
plt.ylabel("PDF")
plt.xlabel(r"$\lambda$")
plt.legend(loc = "upper left")



plt.xlim(-xlimits, xlimits)
plt.minorticks_on()

#plt.ylim(1e-5,)

#plt.show()
plt.savefig("plots/Evalcompare.pdf",bbox_inches="tight")
#



# ------------------- Non-Void distribution ------------------------------------

#file_nstr = './npy/Half/numFieldHalf_032_'+ str(refFactor)+'.npy'
file_nstr = './npy/numField_051_'+str(refFactor)+'.npy'
nstream = np.load(file_nstr).astype(np.float64)



#l3 = np.sum(l, axis = 3)
l1 = l[:,:,:, 0]
l2 = l[:,:,:, 1]
l3 = l[:,:,:, 2]

nonVoid = np.where(nstream > 1)

l1_nonVoid = l1[nonVoid]
l2_nonVoid = l2[nonVoid]
l3_nonVoid = l3[nonVoid]


Void = np.where(nstream == 1)

l1_Void = l1[Void]
l2_Void = l2[Void]
l3_Void = l3[Void]


plt.figure(6, figsize = (8,6))


#l1 = l1[np.abs(l1) < xlimits]
#l2 = l2[np.abs(l2) < xlimits]
#l3 = l3[np.abs(l3) < xlimits]

#nbins = np.linspace(np.min(l), np.max(l), nbins)

#nbins1 = np.linspace(np.min(l), -xlimits, 10)
#nbins2 = np.linspace(-xlimits, xlimits, nbins)
#nbins3 = np.linspace(xlimits, np.max(l), 10)
#nbins = np.hstack((nbins1, nbins2, nbins3))


#plt.subplot(3,1,1)
y,binEdges = np.histogram(l1_nonVoid, bins = nbins, density=False)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'-', lw = 1.5, label = r"$\lambda_1$ (non-void)")


#plt.yscale('log')
#plt.xscale('symlog')
#plt.ylabel("PDF")
#plt.xlabel(r"$\lambda$")
#plt.legend(loc = "upper left")
#plt.xlim(-xlimits, xlimits)
#plt.ylim(1e-5,)

#plt.subplot(3,1,2)
y,binEdges = np.histogram(l2_nonVoid, bins = nbins, density=False)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'-', lw = 1.5, label = r"$\lambda_2$ (non-void)")


#plt.yscale('log')
##plt.xscale('symlog')
#plt.ylabel("PDF")
#plt.xlabel(r"$\lambda$")
#plt.legend(loc = "upper left")
#plt.xlim(-xlimits, xlimits)
#plt.ylim(1e-5,)

#plt.subplot(3,1,3)
y,binEdges = np.histogram(l3_nonVoid, bins = nbins, density=False)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'-', lw = 1.5, label = r"$\lambda_3$ (non-void)")


plt.yscale('log')
#plt.xscale('symlog')
plt.ylabel("PDF")
plt.xlabel(r"$\lambda$")
plt.legend(loc = "lower right")
plt.minorticks_on()


plt.xlim(-xlimits, xlimits)

plt.savefig("plots/EvalcompareNonVoid.pdf",bbox_inches="tight")



plt.show()


cutfr = 10**np.arange( 0, 6, 0.25)
frl1 = frl2 = frl3 = frl1NonVoid = frl2NonVoid = frl3NonVoid = []

for i in cutfr:

    frl1a = (np.shape(np.where( np.abs(l1) <= i)[0])[0]/(1.*size_fact**3))*L
    frl2a = (np.shape(np.where( np.abs(l2) <= i)[0])[0]/(1.*size_fact**3))*L
    frl3a = (np.shape(np.where( np.abs(l3) <= i)[0])[0]/(1.*size_fact**3))*L
    
    frl1 = np.append(frl1, frl1a)
    frl2 = np.append(frl2, frl2a)    
    frl3 = np.append(frl3, frl3a)   
    
    frl1NonVoida = (np.shape(np.where( np.abs(l1_nonVoid) <= i)[0])[0]/(1.*size_fact**3))*L
    frl2NonVoida = (np.shape(np.where( np.abs(l2_nonVoid) <= i)[0])[0]/(1.*size_fact**3))*L
    frl3NonVoida = (np.shape(np.where( np.abs(l3_nonVoid) <= i)[0])[0]/(1.*size_fact**3))*L
    
    frl1NonVoid = np.append(frl1NonVoid, frl1NonVoida)
    frl2NonVoid = np.append(frl2NonVoid, frl2NonVoida)    
    frl3NonVoid = np.append(frl3NonVoid, frl3NonVoida)  
    

#f, axarr = plt.subplots(2, sharex=True)
#axarr[0].plot(x, y)
#axarr[0].set_title('Sharing X axis')
#axarr[1].scatter(x, y)

#plt.figure(5)
f, axarr = plt.subplots(2, figsize = (8,12),  sharex=True)
f.subplots_adjust( hspace = 0.02)
plt.sca(axarr[0])       
axarr[0].plot(cutfr, frl1, '--', lw = 1.5, label = r"$\lambda_1$")
axarr[0].plot(cutfr, frl2, '--', lw = 1.5, label = r"$\lambda_2$")
axarr[0].plot(cutfr, frl3, '--', lw = 1.5, label = r"$\lambda_3$")
plt.xscale('log')
#plt.xlim(np.min(cutfr), 4)
#plt.ylim(9,95)
#plt.yticks( [0.1, 0.5, 0.9] )
axarr[0].legend(loc = "upper left")
plt.ylabel(" Volume fraction of $|\lambda| < \lambda_{th}$ ")
plt.minorticks_on()

#plt.xlabel(r" $ \lambda_{th} $")


#plt.subplot(2,1,2)
plt.sca(axarr[1])
axarr[1].plot(cutfr, frl1NonVoid, '-', lw = 1.5, label = r"$\lambda_1$ (non-void)")
axarr[1].plot(cutfr, frl2NonVoid, '-', lw = 1.5, label = r"$\lambda_2$ (non-void)")
axarr[1].plot(cutfr, frl3NonVoid, '-', lw = 1.5, label = r"$\lambda_3$ (non-void)")
plt.xscale('log')
plt.xlim(np.min(cutfr), np.max(cutfr))
#plt.ylim(-0.5,8.5)

#plt.yticks([0, 2,  4, 6 , 8])
axarr[1].legend(loc = "upper left")
plt.ylabel(" Volume fraction of $|\lambda| < \lambda_{th}$ ")
plt.xlabel(r" $ \lambda_{th} $")
plt.minorticks_on()

plt.savefig("plots/EvalSmall.pdf",bbox_inches="tight")


plt.show()
    
