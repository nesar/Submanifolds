#================ Radius calculation - Farthest point ============
import matplotlib.pylab as plt
import numpy as np
import SetPub
SetPub.set_pub()

m_particle = 9.94e+09
nGr = 128
refFactor = 2
L = 100.

size_fact = nGr*refFactor

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

def CenterCal(x0_1d, x1_1d, x2_1d, label_1d):

    xCen = np.zeros_like(np.unique(label_1d))
    yCen = np.zeros_like(np.unique(label_1d))
    zCen = np.zeros_like(np.unique(label_1d))
        
    for idx in range(label_1d.max().astype(int)):   # Halo ID
        indP = np.where(label_1d == idx+1)    # particle ID
        #print indP
        x0_idx = x0_1d[indP]
        x1_idx = x1_1d[indP]
        x2_idx = x2_1d[indP]
        
        xCen[idx] = np.sum(x0_idx)/((x0_idx.size))
        yCen[idx] = np.sum(x1_idx)/((x1_idx.size))
        zCen[idx] = np.sum(x2_idx)/((x2_idx.size))
        
        # Check if Halo has points in the border
        
    return xCen, yCen, zCen
        

def RadiusCal(x0_1d, x1_1d, x2_1d, label_1d, xCen, yCen, zCen):
    #i = 0
    r_idx = np.zeros_like(np.unique(label_1d))
        
    for idx in range(label_1d.max().astype(int)):   # Halo ID
        indP = np.where(label_1d == idx+1)    # particle ID
        #print indP
        x0_idx = x0_1d[indP]
        x1_idx = x1_1d[indP]
        x2_idx = x2_1d[indP]
        
        com = np.array([xCen[idx], yCen[idx], zCen[idx]])
        
        a_r = x0_idx - com[0]
        b_r = x1_idx - com[1]
        c_r = x2_idx - com[2]
        
        #a_r[np.abs(a_r) >= L/2.] = np.abs(L - a_r[np.abs(a_r) >= L/2.])
        #b_r[np.abs(b_r) >= L/2.] = np.abs(L - b_r[np.abs(b_r) >= L/2.])
        #c_r[np.abs(c_r) >= L/2.] = np.abs(L - c_r[np.abs(c_r) >= L/2.])
        
        
        r_idx[idx] = np.sqrt( np.sum(a_r**2.0 + b_r**2.0 + c_r**2.0)/ np.size(a_r)) 
        #   sqrt(Sum( Xi - Xc)^2 / N)
        
    print 'discarded (temperorily)', np.shape(r_idx[r_idx > 5])
    r_idx[r_idx > 5 ] = -1  # Ignoring border haloes for now
        
    return r_idx
        


def RadiusCal200(x0_1d, x1_1d, x2_1d, label_1d, xCen, yCen, zCen):
    #i = 0
    r_idx = np.zeros_like(np.unique(label_1d))
        
    for idx in range(label_1d.max().astype(int)):   # Halo ID
        indP = np.where(label_1d == idx+1)    # particle ID
        #print indP
        x0_idx = x0_1d[indP]
        x1_idx = x1_1d[indP]
        x2_idx = x2_1d[indP]
        
        com = np.array([xCen[idx], yCen[idx], zCen[idx]])
        
        a_r = x0_idx - com[0]
        b_r = x1_idx - com[1]
        c_r = x2_idx - com[2]
        
        #a_r[np.abs(a_r) >= L/2.] = np.abs(L - a_r[np.abs(a_r) >= L/2.])
        #b_r[np.abs(b_r) >= L/2.] = np.abs(L - b_r[np.abs(b_r) >= L/2.])
        #c_r[np.abs(c_r) >= L/2.] = np.abs(L - c_r[np.abs(c_r) >= L/2.])
        
        
        r_idx[idx] = np.sqrt( np.sum(a_r**2.0 + b_r**2.0 + c_r**2.0)/ np.size(a_r)) 
        #   sqrt(Sum( Xi - Xc)^2 / N)
        
    print 'discarded (temperorily)', np.shape(r_idx[r_idx > 5])
    r_idx[r_idx > 5 ] = 0   # Ignoring border haloes for now
        
    return r_idx
        
#====================================================================== 


def RadiusCalMax(x0_1d, x1_1d, x2_1d, label_1d, xCen, yCen, zCen):
    #i = 0
    r_idx = np.zeros_like(np.unique(label_1d))
        
    for idx in range(label_1d.max().astype(int)):   # Halo ID
        indP = np.where(label_1d == idx+1)    # particle ID
        #print indP
        x0_idx = x0_1d[indP]
        x1_idx = x1_1d[indP]
        x2_idx = x2_1d[indP]
        
        com = np.array([xCen[idx], yCen[idx], zCen[idx]])
    
        a_r = x0_idx - com[0]
        b_r = x1_idx - com[1]
        c_r = x2_idx - com[2]
        
    
        a_r[np.abs(a_r) >= L/2.] = np.abs(L - a_r[np.abs(a_r) >= L/2.])
        b_r[np.abs(b_r) >= L/2.] = np.abs(L - b_r[np.abs(b_r) >= L/2.])
        c_r[np.abs(c_r) >= L/2.] = np.abs(L - c_r[np.abs(c_r) >= L/2.])

   
    
        r_idx[idx] = np.sqrt(  np.max(a_r**2.0 + b_r**2.0 + c_r**2.0) )
    
    print 'discarded (temperorily)', np.shape(r_idx[r_idx > 5])
    r_idx[r_idx > 5 ] = -1  # Ignoring border haloes for now
        
    return r_idx
    



x_ahf, y_ahf, z_ahf, id_ahf = np.around(np.load('npy/particlesAHF.npy'), decimals = 5)

#x_fof, y_fof, z_fof, id_fof = np.around(np.load('npy/particlesFOF.npy'), decimals = 5) 

x_l3,  y_l3, z_l3, id_l3 = np.around(np.load('npy/particlesL3Halo.npy'), decimals = 5)

 
#xCen_fof, yCen_fof, zCen_fof = CenterCal(x_fof, y_fof, z_fof, id_fof)
#r_fof = RadiusCal(x_fof, y_fof, z_fof, id_fof, xCen_fof, yCen_fof, zCen_fof)
#fofxyzr = np.vstack((xCen_fof, yCen_fof, zCen_fof, r_fof))


xCen_ahf, yCen_ahf, zCen_ahf = CenterCal(x_ahf, y_ahf, z_ahf, id_ahf)
r_ahf = RadiusCalMax(x_ahf, y_ahf, z_ahf, id_ahf, xCen_ahf, yCen_ahf, zCen_ahf)
ahfxyzr = np.vstack((xCen_ahf, yCen_ahf, zCen_ahf, r_ahf))


xCen_l3, yCen_l3, zCen_l3 = CenterCal(x_l3, y_l3, z_l3, id_l3)
r_l3 = RadiusCal(x_l3, y_l3, z_l3, id_l3, xCen_l3, yCen_l3, zCen_l3)
l3xyzr = np.vstack((xCen_l3, yCen_l3, zCen_l3, r_l3))


# BECAUSE EDGE HALOES ARENOT FIXED YET
#fofxyzr = fofxyzr[:, fofxyzr[3] >0 ]
#r_fof = fofxyzr[3]

ahfxyzr = ahfxyzr[:, ahfxyzr[3] >0 ]
r_ahf = ahfxyzr[3]

l3xyzr = l3xyzr[:, l3xyzr[3] >0 ]
r_l3 = l3xyzr[3]

#
#np.save('npy/CenRad_fof1', fofxyzr)
np.save('npy/CenRad_ahf1', ahfxyzr)
np.save('npy/CenRad_l31', l3xyzr)


plt.figure(224)

nbins = 100

#xlim1 = np.min([r_fof.min(), r_ahf.min(), r_l3.min()])
#xlim2= np.max([r_fof.max(), r_ahf.max(), r_l3.max()])
xlim1 = 0.0
xlim2 = 0.5

alp = 0.001
wd = 0.001


y,binEdges = np.histogram( r_ahf , bins = np.linspace(xlim1, xlim2, nbins), density=False)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.bar(bincenters, y , width = wd, color = 'darkred', alpha = 2.0*alp, label = r"$\rm{AHF-haloes}$")

y,binEdges = np.histogram( r_l3 , bins = np.linspace(xlim1, xlim2, nbins), density=False)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.bar(bincenters, y ,width = wd, color = 'forestgreen', alpha = alp, label = r"$ \lambda_3\rm{-haloes}$")


#y,binEdges = np.histogram( r_fof , bins = np.linspace(xlim1, xlim2, nbins), density=False)
#bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
##plt.plot(bincenters, y ,'b-', lw = 2, 
#plt.bar(bincenters, y, width = wd, color = 'navy',  alpha = alp, label = r"$\rm{FOF-haloes}$")

#plt.xlim(xlim1, xlim2)
plt.ylim(0.1,)
#plt.xlim(xlim1, 1.5)
#plt.yscale('log')
###plt.xscale('log')
plt.ylabel("Number of haloes")
plt.xlabel(r"Radius ($h^{-1}$ Mpc)") 
plt.legend(loc = "upper right")
#plt.savefig('plots/RadiusRMS1.pdf', bbox_inches='tight')






#def errorfill(x, y, yerr, color=None, alpha_fill=0.25, ax=None)

plt.figure(424)

nbins = 100

#xlim1 = np.min([r_fof.min(), r_ahf.min(), r_l3.min()])
#xlim2= np.max([r_fof.max(), r_ahf.max(), r_l3.max()])

xlim1 = 0.0
xlim2 = 0.2

alp = 0.005

y,binEdges = np.histogram( r_ahf , bins = np.linspace(xlim1, xlim2, nbins), density = False)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
strLabel = r"$\rm{AHF-haloes}$"
errorfill(bincenters, (y).astype(float)/np.sum(y), np.sqrt(y)/np.sum(y), color= 'darkred', alpha_fill= alp)

y,binEdges = np.histogram( r_l3 , bins = np.linspace(xlim1, xlim2, nbins), density=False)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
strLabel = r"$ \lambda_3\rm{-haloes}$"
errorfill(bincenters, (y).astype(float)/np.sum(y) , np.sqrt(y)/np.sum(y), color = 'forestgreen', alpha_fill = alp)


#y,binEdges = np.histogram( r_fof , bins = np.linspace(xlim1, xlim2, nbins), density= False)
#bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
##plt.plot(bincenters, y ,'b-', lw = 2, 
#strLabel = r"$\rm{FOF-haloes}$"
#errorfill(bincenters, (y).astype(float)/np.sum(y), np.sqrt(y)/np.sum(y), color = 'navy',  alpha_fill = alp)

#plt.xlim(xlim1, xlim2)
#plt.ylim(0.1,)
#plt.xlim(xlim1, 1.5)
#plt.yscale('log')
###plt.xscale('log')
plt.ylabel("Fraction of haloes")
plt.xlabel(r"Radius ($h^{-1}$ Mpc)") 
plt.legend(loc = "upper right")
#plt.savefig('plots/RadiusRMS1.pdf', bbox_inches='tight')







plt.show()





#
import sys
sys.exit()
import toParaview as toPara
#toPara.UnstructuredGrid(xCen_fof*size_fact/L,yCen_fof*size_fact/L,zCen_fof*size_fact/L,r_fof*size_fact/L, 'vti/FOFSph_051_'+str(refFactor))
toPara.UnstructuredGrid(xCen_ahf*size_fact/L,yCen_ahf*size_fact/L,zCen_ahf*size_fact/L,r_ahf*size_fact/L, 'vti/AHFSph_051_'+str(refFactor))
toPara.UnstructuredGrid(xCen_l3*size_fact/L,yCen_l3*size_fact/L,zCen_l3*size_fact/L,r_l3*size_fact/L, 'vti/L3Sph_051_'+str(refFactor))

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




#plt.figure()

idx = 1

x_ahfidx = x_ahf[id_ahf == idx]
y_ahfidx = y_ahf[id_ahf == idx]
z_ahfidx = z_ahf[id_ahf == idx]

#x_fofidx = x_fof[id_fof == idx]
#y_fofidx = y_fof[id_fof == idx]
#z_fofidx = z_fof[id_fof == idx]

x_l3idx = x_l3[id_l3 == idx]
y_l3idx = y_l3[id_l3 == idx]
z_l3idx = z_l3[id_l3 == idx]

plt.figure(figsize = (8,6))

#plt.scatter(x_fofidx, z_fofidx, color = 'navy', alpha = 0.2, marker = 'o', s = 20, label = r"FOF-halo")
plt.scatter(x_ahfidx, z_ahfidx, color = 'darkred', alpha = 0.2, marker = 'o', s = 20,label = r"AHF-halo")
plt.scatter(x_l3idx, z_l3idx, color = 'forestgreen', alpha = 0.2, marker = 'o' , s = 20, label = r"$\lambda_3$-halo")

plt.ylabel(r" $h^{-1} Mpc$")
plt.xlabel(r" $h^{-1} Mpc$")

plt.minorticks_on()
#leg = plt.gca().get_legend()
#ltext  = leg.get_texts()
#llines = leg.get_lines()  # all the lines.Line2D instance in the legend
#plt.setp(ltext, fontsize='large')    # the legend text fontsize
#plt.setp(llines, linewidth= 10)      # the legend linewidth
#plt.legend(loc = "upper right")

plt.legend(loc= 1,prop={'size': 15}, title = 'Halo particles')
#plt.ylim(84, 88)
#plt.xlim(32, 36)
#plt.xlim(2.1, 6.8)
#plt.xticks( [3,4,5,6] )
#plt.yticks([5,6,7,8])
#plt.ylim(4.7, 8.5)
plt.savefig('plots/ScatterHaloIdx'+str(idx)+'.pdf', bbox_inches='tight')
