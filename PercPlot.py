import numpy as np
import matplotlib.pylab as plt
from matplotlib import ticker
np.set_printoptions(precision=3)
import SetPub

SetPub.set_pub()



    
def func(a, x, n):
    return a*(x**n)
    
    
#def yerrX(x):
#    dx = np.sqrt(x*totVol)/totVol
#    #dx = np.sqrt(x)
#    return dx    
#    
#def yerrXY(x,y):
#    
#    xt = x*totVol
#    yt = y*totVol
#    
#    dx = np.sqrt(xt)
#    dy = np.sqrt(yt)
#    
#    dxdy = (xt/yt)*np.sqrt( (dx/xt)**2 + (dy/yt)**2 )
#    return dxdy
        
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


def errorfilllog(x, y, yerr, color=None, alpha_fill=0.3, ax=None):
    # https://tonysyu.github.io/plotting-error-bars.html
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - (yerr/y)
        ymax = y + (yerr/y)
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color, lw=1.5, label=strLabel)
    # ax.yscale('log')
    # ax.plot(x, y, color+'o', label = strLabel)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)


def PowerLawFit(x1d, y1d):
    ''' returns k, n for y = kx^n
    '''
    z = np.polyfit(  np.log10( x1d ) , np.log10( y1d ), 1) 
    n = z[0]           # slope
    k = 10.0**z[1]       # intercept
    
    return  k , n    
    
    
refFactor = 2

for refFactor in [4]:
    
    #Outfile = np.load('npy/Half/nstrfrESfr1LargestLabel_'+str(refFactor)+'.npy')
    
    Outfile = np.load('npy/nstrfrESfr1LargestLabel_032.npy')
    OutAll = np.hstack([1.0, 1.0, 1.0, 1.0])
    Outfile = np.vstack( [OutAll, Outfile] )
    

    nstrCut = Outfile[:,0]
    frES = Outfile[:,1]
    fr1 = Outfile[:,2]
    LargestLabel = Outfile[:,3]
    
    print    
    print nstrCut[0:10]
    print (fr1/frES)[0:10]
    
    # fCut = 0.3
    # f = (fr1/frES)[0:100]
    # n = nstrCut[0:100]
    # n_fCut = n[  np.argmin(  np.abs( f- fCut)  ) ]
    # print
    # print '%d nstream @ f1/fES = %4.1f'%(n_fCut, fCut)
    #
    # fCut = 0.4
    # f = (fr1/frES)[0:100]
    # n = nstrCut[0:100]
    # n_fCut = n[  np.argmin(  np.abs( f- fCut)  ) ]
    # print
    # print '%d nstream @ f1/fES = %4.1f'%(n_fCut, fCut)
    #
    # fCut = 0.5
    # f = (fr1/frES)[0:100]
    # n = nstrCut[0:100]
    # n_fCut = n[  np.argmin(  np.abs( f- fCut)  ) ]
    # print
    # print '%d nstream @ f1/fES = %4.1f'%(n_fCut, fCut)
    # fp_n = frES[ np.argmin(  np.abs( f- fCut)  ) ]

    
    
    
#----------------------#--------------------------#-------------------
    
    
    plt.figure(1, figsize = (8,6))
    #plt.subplot(1,3,3)  
    plt.plot( nstrCut, 100.*(fr1/frES),'-o', lw = 1.5, label = r'$f_{1}/f_{ES}$')
    # strLabel = r'$f_{1}/f_{ES}$'
    # errorfilllog(nstrCut, 100.*(fr1/frES), 100.*(np.sqrt(fr1)/frES), color='b')


    plt.xlabel(r'$n_{str}$')
    plt.ylabel('Volume fraction')
    #plt.legend(loc = "upper right")
    plt.xlim(nstrCut.max(), nstrCut.min())

    plt.ylim(1e-1,1.4e2)
    # plt.axvline(x = n_fCut,  ls = '--', linewidth= 1.5, color='k', ymax = 0.86)
    plt.axhline(y = 50,  ls = '-', linewidth= 1, color='darkred')

    #plt.savefig('plots/f1fESnst.pdf', bbox_inches='tight')
    
    plt.figure(66, figsize = (8,6))
    #plt.subplot(1,2,1)   
    plt.plot( frES, (fr1/frES),'-o', lw = 1.5, label = r'$n_{str}(x)$')
    # strLabel = r'$n_{str}(x)$'
    # errorfilllog(frES, (fr1/frES), (np.sqrt(fr1)/frES), color='forestgreen')

    plt.xlabel(r'$f_{ES}$')
    plt.ylabel(r'$f_{1}/f_{ES}$')
    #plt.legend(loc = "upper right")
    #plt.xlim(1e-3, 0.1)
    #plt.invert_xaxis()
    #plt.savefig('plots/f1byfESnst.pdf', bbox_inches='tight')


    plt.figure(1)
    ax = plt.subplot(111)

    plt.plot(nstrCut, 100. * (frES), '-o', lw=1.5, label=r'$f_{ES}$')
    # strLabel = r'$f_{ES}$'
    # errorfilllog(nstrCut, 100.*(frES), 100.*(np.sqrt(frES)), color='darkred')

    plt.xlabel(r'$n_{ff}$')
    # plt.ylabel()
    plt.legend(loc="lower left")
    # plt.ylim(1e-3,)
    plt.xlim(nstrCut.max(), nstrCut.min())
    plt.yscale('log')


    plt.minorticks_on()
    plt.savefig('plots/fnstr.pdf', bbox_inches='tight')

    

# plt.show()
# import sys
# sys.exit()

#totVol = 256.**3 #np.sum(sortedLabelFr[:,1][:]) = 256^3 = 16777216


refFactor = 2

for refFactor in [8]:
    
    #Outfile = np.load('npy/Half/densfrESfr1LargestLabel_'+str(refFactor)+'.npy')
    
    # Outfile = np.load('npy/densfrESfr1LargestLabel_032_5.npy')
    Outfile = np.load('npy/ff_frESfr1LargestLabel_032.npy')[:8, :]
    OutAll = np.hstack([0.0, 1.0, 1.0, 1.0])
    Outfile = np.vstack( [OutAll, Outfile] )


    ffCut = Outfile[:,0]
    frES = Outfile[:,1]
    fr1 = Outfile[:,2]
    LargestLabel = Outfile[:,3]
    
    #dfr1 = yerrX(fr1)
    #dfrES = yerrX(frES) 
    #dfr1frES = yerrXY(fr1, frES)   
    
    print
    print ffCut[0:7]
    print (fr1/frES)[0:7]

    # fCut = 0.3
    # f = (fr1/frES)[:1000]
    # d = ffCut[:1000]
    # argMin = np.argmin(  np.abs(f- fCut)  )
    # ff_fCut = d[ argMin ]
    # print
    # print '%4.2e ff @ f1/fES = %4.1f'%(ff_fCut, fCut)
    # print ff_fCut
    #
    # fCut = 0.4
    # f = (fr1/frES)[:1000]
    # d = fCut[:1000]
    # argMin = np.argmin(  np.abs(f- fCut)  )
    # ff_fCut = d[ argMin ]
    # print
    # print '%4.2e ff @ f1/fES = %4.1f'%(ff_fCut, fCut)
    # print ff_fCut
    #
    # fCut = 0.5
    # f = (fr1/frES)[000:1000]
    # d = fCut[0000:1000]
    # argMin = np.argmin(  np.abs(f- fCut)  )
    # ff_fCut = d[ argMin ]
    # print
    # print '%4.2e density @ f1/fES = %4.1f'%(ff_fCut, fCut)
    # fp_d = frES[000:1000][ argMin ]
 
    #print d_fCut
    
    #----------------------#--------------------------#-------------------
    
    plt.figure(2, figsize = (8,6))  
    #plt.subplot(1,3,3) 
    plt.plot( ffCut, 100.*(fr1/frES),'-o', lw = 1.5, label = r'$f_{1}/f_{ES}$')

    # strLabel = r'$f_{1}/f_{ES}$'
    # errorfilllog(ffCut, 100.*(fr1/frES), 100.*(np.sqrt(fr1)/frES), color='navy')
    
    plt.xlabel(r'$n_{ff}$')
    plt.ylabel('Volume fraction')
    #plt.legend(loc = "upper right")
    # plt.xlim(15.0, 0.0)     # UNCOMMENT
    # plt.ylim(1e-1,1.4e2)              # UNCOMMENT
    #plt.axvline(x = d_fCut, ymin = 1e-4, ymax = 0.5, ls = '--', linewidth=2, color='k')
    #plt.axhline(y = 0.5, xmin = 1e-6, xmax = d_fCut, ls = '--', linewidth=2, color='k')
    # plt.axvline(x = ff_fCut,  ls = '--', linewidth= 1.5, color='k', ymax = 0.86)
    plt.axhline(y = 50,  ls = '-', linewidth=1, color='darkred')
    #plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useLocale = True, useOffset = False)
    ###ax.xaxis.major.formatter._useMathText = True
    ###plt.gca().get_xaxis().get_major_formatter().set_useOffset(True)

    #plt.savefig('plots/f1fESDen.pdf', bbox_inches='tight')
    
    plt.figure(66)
    #plt.subplot(1,2,2)   
    plt.plot( frES, (fr1/frES),'-o', lw = 1.5, label = r'$n_{ff}(q)$')
    # strLabel = r'$n_{ff}(q)$'
    # errorfilllog( frES, (fr1/frES) , (np.sqrt(fr1)/frES), color='navy')

    plt.axhline(y = 0.5,  ls = '-', linewidth= 1, color='darkred')
    # plt.axvline(x = fp_d,  ls = '--', linewidth= 1.5, color='k', ymax = 0.5)
    # plt.axvline(x = fp_n,  ls = '--', linewidth=1.5, color='k', ymax = 0.5)
    
    #strLabel = r'$\rho/ \rho_{b}$'
    #errorfill(frES, (fr1/frES), dfr1frES, color='b')
    
    
    plt.xlabel(r'$f_{ES}$')
    plt.ylabel(r'$f_{1}/f_{ES}$')
    plt.legend(loc = "lower right")
    #plt.xlim(0, 1.01*np.max(frES))
    # plt.xlim(0, 0.15)
    # plt.xticks( np.arange(0.01, 0.15, 0.03))
    # plt.xlim(35, 0)
    #plt.xlim(1e-5, 1)
    #plt.xscale('log')
    #plt.invert_xaxis()
    plt.minorticks_on()
    
    plt.savefig('plots/f1byfESLog.pdf', bbox_inches='tight')
    

    
    plt.figure(2)
    ax = plt.subplot(111)   
    
    plt.plot( ffCut, 100.*(frES),'-o', lw = 1.5, label = r'$f_{ES}$')
    # strLabel = r'$f_{ES}$'
    # errorfilllog(ffCut, 100.*(frES), 100.*(np.sqrt(frES)), color='darkred')

    plt.xlabel(r'$n_{ff}$')
    #plt.ylabel()
    plt.legend(loc = "lower left")
    #plt.ylim(1e-3,)
    plt.xlim(ffCut.max(), ffCut.min())
    plt.yscale('log')

    
    plt.minorticks_on()
    plt.savefig('plots/fFlip.pdf', bbox_inches='tight')
    


plt.show()
