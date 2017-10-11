import numpy as np
import toParaview as toPara
import matplotlib.pylab as plt

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

set_pub()

def errorfill(x, y, yerr, color=None, alpha_fill=0.25, ax=None):
    
    #https://tonysyu.github.io/plotting-error-bars.html
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color, lw = 1.5, label = strLabel)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)
    
    

def revcumsum(x):
    return np.cumsum(x[::-1])[::-1] 

refFactor = 1
nGr = 128
L = 1.
size_fact = nGr*refFactor

x_ahf, y_ahf, z_ahf, id_ahf = np.load('npy/particlesAHF.npy')

#x_fof, y_fof, z_fof, id_fof = np.load('npy/particlesFOF.npy')

x_l3,  y_l3, z_l3, id_l3 = np.load('npy/particlesL3Halo.npy')

m_particle = 9.94e+09

freqlabel = np.bincount(id_ahf.astype(int))
non0entries = np.nonzero(freqlabel)[0]
npartHalo_ahf = np.vstack( (non0entries, freqlabel[non0entries]) ).T

#freqlabel = np.bincount(id_fof.astype(int))
#non0entries = np.nonzero(freqlabel)[0]
#npartHalo_fof = np.vstack( (non0entries, freqlabel[non0entries]) ).T

freqlabel = np.bincount(id_l3.astype(int))
non0entries = np.nonzero(freqlabel)[0]
npartHalo_l3 = np.vstack( (non0entries, freqlabel[non0entries]) ).T


#toPara.Points1d(x_fof*size_fact/L , y_fof*size_fact/L , z_fof*size_fact/L , id_fof, 'vti/xLabeledFOF_032_'+str(refFactor))
#toPara.Points1d(x_ahf*size_fact/L, y_ahf*size_fact/L, z_ahf*size_fact/L, id_ahf, 'vti/xLabeledAHF_032_'+str(refFactor))
#toPara.Points1d(x_l3*size_fact/L , y_l3*size_fact/L , z_l3*size_fact/L , id_l3 , 'vti/xLabeledL3_032_'+str(refFactor))


nbins = 50

#nbins = 20

xlim1 = np.min([npartHalo_ahf[:,1].min(), npartHalo_l3[:,1].min()])
xlim2= np.max([npartHalo_ahf[:,1].max(), npartHalo_l3[:,1].max()])
#


#plt.figure(2)
#
#
#y,binEdges = np.histogram( npartHalo_ahf[:,1]*m_particle
#, bins = np.linspace(xlim1*m_particle, xlim2*m_particle, nbins), density=False)
#bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
#plt.plot(bincenters,y/(L**3.0),'r-', lw = 2, label = r"$ AHF  $")
#
#
#y,binEdges = np.histogram( npartHalo_l3[:,1]*m_particle
#, bins = np.linspace(xlim1*m_particle, xlim2*m_particle, nbins), density=False)
#bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
#plt.plot(bincenters,y/(L**3.0),'g-', lw = 2, label = r"$\lambda_3>0 $")
#
#
#y,binEdges = np.histogram( npartHalo_fof[:,1]*m_particle
#, bins = np.linspace(xlim1*m_particle, xlim2*m_particle, nbins), density=False)
#bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
#plt.plot(bincenters,y/(L**3.0), 'b-', lw = 2, label = r"$ FOF$")
#
#
#plt.yscale('log')
#plt.xscale('log')
#plt.ylabel("n(M) ( Mpc^-3)  ")
#plt.xlabel("M") 
#plt.legend(loc = "upper right")
#
#plt.xlim(xlim1*m_particle, xlim2*m_particle)
##plt.ylim(1e-2, 10)
#

#plt.figure(3)
#
#
#y,binEdges = np.histogram( npartHalo_ahf[:,1], bins = np.linspace(xlim1, npartHalo_ahf[:,1].max(), nbins), density=False)
#bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
#plt.plot(bincenters,y,'r-', lw = 2, label = r"$ AHF  $")
#
#
#y,binEdges = np.histogram( npartHalo_l3[:,1], bins = np.linspace(xlim1, npartHalo_l3[:,1].max(), nbins), density=False)
#bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
#plt.plot(bincenters,y,'g-', lw = 2, label = r"$\lambda_3>0 $")
#
#
#y,binEdges = np.histogram( npartHalo_fof[:,1], bins = np.linspace(xlim1, xlim2, nbins), density=False)
#bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
#plt.plot(bincenters,y,'b-', lw = 2, label = r"$ FOF$")
#
#
#
##plt.xlim(-xlimits, xlimits)
##plt.ylim(1e-2, 10)
#plt.yscale('log')
#plt.xscale('log')
#plt.ylabel("n(M)")
#plt.xlabel("M")
#plt.legend(loc = "upper right")
#
#
#plt.figure(5)
#
#y,binEdges = np.histogram( npartHalo_ahf[:,1]*m_particle
#, bins = np.linspace(xlim1*m_particle, xlim2*m_particle, nbins), density=False)
#bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
#plt.plot(bincenters,revcumsum(y*bincenters/(L**3.0)),'r-', lw = 2, label = r"$ AHF  $")
#
#
#y,binEdges = np.histogram( npartHalo_l3[:,1]*m_particle
#, bins = np.linspace(xlim1*m_particle, xlim2*m_particle, nbins), density=False)
#bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
#plt.plot(bincenters,revcumsum(y*bincenters/(L**3.0)),'g-', lw = 2, label = r"$\lambda_3>0 $")
#
#
#y,binEdges = np.histogram( npartHalo_fof[:,1]*m_particle
#, bins = np.linspace(xlim1*m_particle, xlim2*m_particle, nbins), density=False)
#bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
#plt.plot(bincenters,revcumsum(y*bincenters/(L**3.0)),'b-', lw = 2, label = r"$ FOF$")
#
#
#plt.yscale('log')
#plt.xscale('log')
#plt.ylabel(r"n(M)*M  (Msol/Mpc^3)")
#plt.xlabel("M")
#plt.legend(loc = "upper right")
#
#plt.xlim(xlim1*m_particle, xlim2*m_particle)
plt.clf()

plt.figure(6)


#xlim2 = npartHalo_ahf[:,1].max()
y,binEdges = np.histogram( npartHalo_ahf[:,1]*m_particle
, bins = np.logspace(np.log10(xlim1*m_particle), np.log10(xlim2*m_particle), nbins), density= False)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
#plt.plot(bincenters, revcumsum(y*bincenters)/(L**3.0), 'r--', lw = 2, label = r"$ AHF  $")
plt.errorbar(bincenters, revcumsum(y*bincenters/(L**3.0)), 
xerr = np.sqrt(revcumsum(y*bincenters)/(L**3.0)),  fmt='--r',  elinewidth=2 , lw = 2,
label = r"$ AHF-Haloes$")


#xlim2 = npartHalo_l3[:,1].max()
y,binEdges = np.histogram( npartHalo_l3[:,1]*m_particle
, bins = np.logspace(np.log10(xlim1*m_particle), np.log10(xlim2*m_particle), nbins), density= False)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
#plt.plot(bincenters, revcumsum(y*bincenters)/(L**3.0),'g-', lw = 2, label = r"$\lambda_3>0 $")
plt.errorbar(bincenters, revcumsum(y*bincenters/(L**3.0)), 
xerr = np.sqrt( revcumsum(y*bincenters)/(L**3.0) ),  fmt= '-g',  elinewidth=2 , lw = 2,
label = r"$\lambda_3-Haloes$")


#xlim2 = npartHalo_fof[:,1].max()
#y,binEdges = np.histogram( npartHalo_fof[:,1]*m_particle
#, bins = np.logspace(np.log10(xlim1*m_particle), np.log10(xlim2*m_particle), nbins), density= False)
#bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
##plt.plot(bincenters, revcumsum(y*bincenters)/(L**3.0),'b-', lw = 2, label = r"$ FOF$")
#plt.errorbar(bincenters, revcumsum(y*bincenters/(L**3.0)), 
#xerr = np.sqrt(revcumsum(y*bincenters)/(L**3.0)),  fmt='--b',  elinewidth=2 , lw = 2,
#label = r"$ FOF-Haloes$")

#np.cumsum()

plt.xlim(xlim1*m_particle, xlim2*m_particle)

plt.yscale('log')
plt.xscale('log')
plt.ylabel(r"n(>M)*M  (Msol/Mpc^3)")
plt.xlabel("M") 
plt.legend(loc = "upper right")
#plt.savefig('plots/HMF_3.pdf', bbox_inches='tight')


plt.figure(7)


####xlim2 = npartHalo_ahf[:,1].max()
y,binEdges = np.histogram( npartHalo_ahf[:,1]*m_particle
, bins = np.logspace(np.log10(xlim1*m_particle), np.log10(xlim2*m_particle), nbins), density= False)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
###plt.plot(bincenters, revcumsum(y*bincenters)/(L**3.0), 'r--', lw = 2, label = r"$ AHF  $")
#plt.errorbar(bincenters, revcumsum(y/(L**3.0)), 
#yerr = np.sqrt(revcumsum(y))/(L**3.0),  fmt='--r',  elinewidth=2 , lw = 2, errorevery=1, 
#label = r"$ AHF  $")


strLabel = r"AHF-haloes"
errorfill(bincenters, revcumsum(y/(L**3.0)), np.sqrt(revcumsum(y))/(L**3.0), color='darkred')



####xlim2 = npartHalo_l3[:,1].max()
y,binEdges = np.histogram( npartHalo_l3[:,1]*m_particle
, bins = np.logspace(np.log10(xlim1*m_particle), np.log10(xlim2*m_particle), nbins), density= False)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
###plt.plot(bincenters, revcumsum(y*bincenters)/(L**3.0),'g-', lw = 2, label = r"$\lambda_3>0 $")
#plt.errorbar(bincenters, revcumsum(y/(L**3.0)), 
#yerr = np.sqrt(revcumsum(y))/(L**3.0),  fmt= '-g',  elinewidth=2 , lw = 2, errorevery=1, 
#label = r"$\lambda_3>0 $")


strLabel = r"$\lambda_3$-haloes"
errorfill(bincenters, revcumsum(y/(L**3.0)), np.sqrt(revcumsum(y))/(L**3.0), color='darkgreen')


####xlim2 = npartHalo_fof[:,1].max()
#y,binEdges = np.histogram( npartHalo_fof[:,1]*m_particle
#, bins = np.logspace(np.log10(xlim1*m_particle), np.log10(xlim2*m_particle), nbins), density= False)
#bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
###plt.plot(bincenters, revcumsum(y*bincenters)/(L**3.0),'b-', lw = 2, label = r"$ FOF$")
#plt.errorbar(bincenters, revcumsum(y/(L**3.0)), 
#yerr = np.sqrt(revcumsum(y))/(L**3.0),  fmt='--b',  elinewidth=2 , lw = 2, errorevery=1, 
#label = r"$ FOF$")
#

#
#strLabel = r"FOF-haloes"
#errorfill(bincenters, revcumsum(y/(L**3.0)), np.sqrt(revcumsum(y))/(L**3.0), color='navy')



plt.xlim(xlim1*m_particle, xlim2*m_particle)
plt.minorticks_on()
plt.yscale('log')
plt.xscale('log')
plt.ylim(1,)
#plt.xlim(2.5e11,2.5e14)
plt.ylabel(r"$n(> M)(h^3/Mpc^3)$")
plt.xlabel("M $(M_\odot)$")
plt.legend(loc = "upper right")
plt.savefig('plots/HMF_2.pdf', bbox_inches='tight')

plt.figure(9)


#xlim2 = npartHalo_ahf[:,1].max()
y,binEdges = np.histogram( npartHalo_ahf[:,1]*m_particle
, bins = np.linspace((xlim1*m_particle), (xlim2*m_particle), nbins), density= False)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
#plt.plot(bincenters, revcumsum(y*bincenters)/(L**3.0), 'r--', lw = 2, label = r"$ AHF  $")
plt.errorbar(bincenters, revcumsum(y*bincenters/(L**3.0)), 
xerr = np.sqrt(revcumsum(y*bincenters)/(L**3.0)),  fmt='--r',  elinewidth=2 , lw = 2,
label = r"$ AHF  $")

#xlim2 = npartHalo_l3[:,1].max()
y,binEdges = np.histogram( npartHalo_l3[:,1]*m_particle
, bins = np.linspace((xlim1*m_particle), (xlim2*m_particle), nbins), density= False)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
#plt.plot(bincenters, revcumsum(y*bincenters)/(L**3.0),'g-', lw = 2, label = r"$\lambda_3>0 $")
plt.errorbar(bincenters, revcumsum(y*bincenters/(L**3.0)), 
xerr = np.sqrt(revcumsum(y*bincenters)/(L**3.0)),  fmt= 'g-',  elinewidth=2 , lw = 2,
label = r"$\lambda_3>0 $")

##xlim2 = npartHalo_fof[:,1].max()
#y,binEdges = np.histogram( npartHalo_fof[:,1]*m_particle
#, bins = np.linspace((xlim1*m_particle), (xlim2*m_particle), nbins), density= False)
#bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
##plt.plot(bincenters, revcumsum(y*bincenters)/(L**3.0),'b-', lw = 2, label = r"$ FOF$")
#plt.errorbar(bincenters, revcumsum(y*bincenters/(L**3.0)), 
#xerr = np.sqrt(revcumsum(y*bincenters)/(L**3.0)),  fmt='--b',  elinewidth=2 , lw = 2,
#label = r"$ FOF$")

#np.cumsum()

plt.xlim(xlim1*m_particle, xlim2*m_particle)

plt.yscale('log')
plt.xscale('log')
plt.ylabel(r"n(>M)*M  (Msol/Mpc^3)")
plt.xlabel("M") 
plt.legend(loc = "upper right")

#plt.minorticks_on()
#plt.tick_params('both', length=20, width=2, which='major')
#plt.tick_params('both', length=10, width=1, which='minor')


#plt.savefig('plots/HMF_1.pdf', bbox_inches='tight')

plt.show()