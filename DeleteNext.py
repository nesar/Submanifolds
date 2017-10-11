import numpy as np
import matplotlib.pyplot as plt


labels3d = labels0
label1d = np.ravel(labels3d)

def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None, strLabel=None):
    
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


def PDFplot(array1d, strLabel, Strcolor):
    
    print '--------------------------------------------'
    
    print strLabel+' Statistics'
    print '#Haloes:     ', np.shape(array1d)
    print '        Min:  %4.2e' %(array1d.min())
    print '        Max:  %4.2e' %(array1d.max())
    print '        Mean: %4.2e' %(array1d.mean())
    print '        std  : %4.2e' %(array1d.std())
    print '        Median %4.2e' %np.median(array1d)
    print '--------------------------------------------'
    
    plt.figure(143)
    nbins = 35
    xlim1 = np.min(array1d)
    xlim2 = np.max(array1d)
    
    xlim1 = 1e4
    xlim2 = 1e9
    
    y,binEdges = np.histogram( array1d
    , bins = np.logspace(np.log10(xlim1), np.log10(xlim2), nbins), density= False)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    #
    #y,binEdges = np.histogram( array1d
    #, bins = np.linspace((xlim1), (xlim2), nbins), density= False)
    #bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    
    
    #plt.plot(bincenters, y, 'ko', lw = 2, label = strLabel)
    
    
#    plt.errorbar(bincenters, y, 
#    yerr = np.sqrt(y),  fmt='.'+Strcolor, markersize = 5,  alpha = 0.35, elinewidth=2 , lw = 2)
#
#    plt.plot(bincenters[y!=0], y[y!=0], Strcolor, alpha = 0.8, lw = 2, label = strLabel)

    
    errorfill(bincenters, y, np.sqrt(y), color = Strcolor, strLabel= strLabel)
    
    
    #plt.xlim(binEdges[1:][0], xlim2)
    #plt.ylim(1, )

    #plt.yscale('log')
    #if (strLabel != '#Grids in Halo'): plt.xscale('log')
    #plt.xlabel(strLabel)
    #plt.ylabel("pdf") 
    #plt.legend(loc = "upper right")
    
    #plt.savefig('plots/'+strLabel+'.pdf', bbox_inches='tight')

lmax = l[:,:,:,0]

maxl3InLabel = np.array(ndi.maximum( lmax , labels= labels3d, index= label1d )) 
PDFplot(maxl3InLabel, r'max($\lambda_1$) in Halo', 'navy')


lmax = l[:,:,:,1]

maxl3InLabel = np.array(ndi.maximum( lmax , labels= labels3d, index= label1d )) 
PDFplot(maxl3InLabel, r'max($\lambda_2$) in Halo', 'forestgreen')


lmax = l[:,:,:,2]

maxl3InLabel = np.array(ndi.maximum( lmax , labels= labels3d, index= label1d )) 
PDFplot(maxl3InLabel, r'max($\lambda_3$) in Halo', 'darkred')

plt.figure(143, figsize = (8,6))

#plt.xlim(binEdges[1:][0], xlim2)
plt.ylim(1, )
#plt.xlim(1e-1, 1e3)
plt.yticks( np.arange(100, 1000, 200), np.arange(100, 1000, 200))
plt.minorticks_on()
#plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'max($\lambda$)')
plt.ylabel("PDF") 
plt.legend(loc = "upper right")

plt.savefig('plots/maxl1l2l3.pdf', bbox_inches='tight')