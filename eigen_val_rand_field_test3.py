import math
import numpy as np
import matplotlib.pyplot  as plt
import numpy.random as nprand

def f3(I1,I2,I3,lam):
    """cubic polynomial: its roots are eigen values
    """
    return lam**3 - I1*lam**2 + I2*lam - I3
    
def f2(I1,I2,I3,lam):
    """firs derivative of cubic polynomial; its roots define three lam_i sectors
    """
    return 3.*lam**2 - 2.*I1*lam + I2

def lam_min_max(I1,I2):
    """ lam_max < lam_min define boundaries of three lam_i sectors
    """
    discr = np.sqrt(I1**2-3.*I2)
    return 1./3.*(I1+discr), 1./3.*(I1-discr)
    
def eigen_val_field_1d(Np,mean=0., sig=3.):
    """generator of ordered lam[0]>lam[1]>lam[2] normal random numbers
    """
    lam = np.zeros((3,Np), dtype=np.float32)
    for ip in range(Np):
        eigen_val = nprand.normal(mean,sig, 3)
        lam[:,ip] = np.sort(eigen_val)[::-1]  # decsending order 
    return lam

def find_lam123_above_thresh(I1,I2,I3,lam_c):
    """ returns three arrays sign_lam[i,Np] arrays:
        if sign_lam[i] > 0 then lam[i] > lam_c 
        otherwise lam[i] < lam_c
    """
    lam_sign = -np.ones((3,Np), dtype=np.int8)
    lam_min, lam_max = lam_min_max(I1,I2)
    fc = f3(I1,I2,I3,lam_c)
#----------------------------------------------------------      
    cond1 = (lam_min < lam_c)  # lam_c is in lam1 sector
    ind1 = np.where(cond1)
    cond11 = (fc[ind1] < 0)
    ind1f = ind1[0][cond11]
    lam_sign[0,ind1f] = 1
#----------------------------------------------------------
    cond2 =np.logical_and(lam_c > lam_max, lam_c < lam_min, ) # lam_c is in lam2 sector
    ind2 = np.where(cond2)   
    lam_sign[0,ind2] = 1     # if lam_c is in lam2 sector then lam1 > lam_c
    cond21 = (fc[ind2] > 0)
    ind2f = ind2[0][cond21]
    lam_sign[1,ind2f] = 1
#----------------------------------------------------------
    cond3 = (lam_c < lam_max)  # lam_c is in lam3 sector
    ind3 = np.where(cond3)
    lam_sign[1,ind3] = 1   # if lam_c is in lam3 sector then both lam2 and lam1 > lam_c
    lam_sign[0,ind3] = 1
    cond31 = (fc[ind3] < 0)
    ind3f = ind3[0][cond31]
    lam_sign[2,ind3f] = 1
    return lam_sign
#==============================================================  
    
Np = 50
randSeed = nprand.seed(2)
lam = (eigen_val_field_1d(Np)) # eigen values lam1=lam[0] > lam2=lam[1] > lam3=lam[2]
I1 = lam[0]+lam[1]+lam[2]
I2 = lam[0]*lam[1]+lam[0]*lam[2]+lam[1]*lam[2]
I3 = lam[0]*lam[1]*lam[2]

lam_c = 1.

lam_sign = find_lam123_above_thresh(I1,I2,I3,lam_c)

x = range(Np)


plt.figure()
plt.plot(x, lam[0], 'b')
plt.hlines(0,0,Np,'k', linestyle='dashed')
plt.hlines(lam_c, 0,Np,'k')
plt.plot(x,lam_sign[0],'ob')
#plt.plot(x,lam[0],'ob')
plt.xlabel('part ID')
plt.ylabel('Eigen Values')

plt.figure()
plt.plot(x, lam[1],'g')
plt.hlines(0,0,Np,'k', linestyle='dashed')
plt.hlines(lam_c, 0,Np,'k')
plt.plot(x,lam_sign[1],'og')
plt.xlabel('part ID')
plt.ylabel('Eigen Values')

plt.figure()
plt.plot(x, lam[2], 'r')
plt.hlines(0,0,Np,'k', linestyle='dashed')
plt.hlines(lam_c, 0,Np,'k')
plt.plot(x,lam_sign[2],'or')
plt.xlabel('part ID')
plt.ylabel('Eigen Values')

plt.show()