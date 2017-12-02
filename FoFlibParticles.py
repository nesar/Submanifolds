"""
./FoF_Special . snapshot 005

./FoF_Special . snap100Mpc128_z0 000



"""
import numpy as np
#from numpy import *


def read_groups_catalogue(filename):
  """
  Read the "fof_special_catalogue" files and return 4 arrays:
  
  GroupLen	: the size of each group
  GroupOffset	: the offset of the first particle in each group
                  as it will be found in the file fof_special_particles
  GroupMass	: the mass of each group (in code unit)
  GroupCM	: the center of mass of each group
  """


  f = open(filename,'r')

  Ngroups = np.fromstring(f.read(4),np.int32)[0]

  GroupLen    = np.fromstring(f.read(4*Ngroups),np.int32)
  GroupOffset = np.fromstring(f.read(4*Ngroups),np.int32)
  GroupMass   = np.fromstring(f.read(4*Ngroups),np.float32)

  GroupCM     = np.fromstring(f.read(3*4*Ngroups),np.float32)
  GroupCM.shape  = (Ngroups,3)


  GroupNspecies = np.fromstring(f.read(3*4*Ngroups),np.int32)
  GroupNspecies.shape  = (Ngroups,3)

  GroupMspecies = np.fromstring(f.read(3*4*Ngroups),np.float32)
  GroupMspecies.shape  = (Ngroups,3)

  GroupSfr   = np.fromstring(f.read(4*Ngroups),np.float32)

  GroupMetallicities = np.fromstring(f.read(2*4*Ngroups),np.float32)
  GroupMetallicities.shape  = (Ngroups,2)

  Mcold = np.fromstring(f.read(4*Ngroups),np.float32) 

  SigmaStars= np.fromstring(f.read(4*Ngroups),np.float32) 
  SigmaDM= np.fromstring(f.read(4*Ngroups),np.float32) 

  f.close()

  return GroupLen,GroupOffset,GroupMass,GroupCM, GroupNspecies, GroupSfr
  
  
  
  
def read_groups_particles(filename):
  """
  Read the "fof_special_particles" files and return
  an array of the positions of each particles belonging
  to a group.
  """
  
  f = open(filename,'r')

  Ntot = np.fromstring(f.read(4),np.int32)[0]
  Pos	  = np.fromstring(f.read(3*4*Ntot),np.float32)
  Pos.shape  = (Ntot,3)
  f.close()
  
  return Pos, Ntot
  
  
def read_groups_indexlist(filename):
  """
  Read the "fof_special_particles" files and return
  an array of the positions of each particles belonging
  to a group.
  """
  
  f = open(filename,'r')
  
  Ntot = np.fromstring(f.read(4),np.int32)[0]
  idx	  = np.fromstring(f.read(3*4*Ntot),np.float32)
  
  f.close()
  
  return Ntot, idx
  
  
  
fName = '/home/nesar/Desktop/128MUSIC/FOF/groups_catalogue/fof_special_catalogue_064'


aGroupLen, aGroupOffset, aGroupMass, aGroupCM, aGroupNspecies, aGroupSfr = read_groups_catalogue(fName)

fName1 = '/home/nesar/Desktop/128MUSIC/FOF/groups_particles/fof_special_particles_064'

aPos, aNtot = read_groups_particles(fName1)

fName2 = '/home/nesar/Desktop/128MUSIC/FOF/groups_indexlist/fof_special_indexlist_064'

ntot, idx = read_groups_indexlist(fName2)
print aGroupLen.sum()



  
print aPos.shape
print
print aGroupMass.shape
idx = [] 
for i in range(1, aGroupLen.size+1):
    
    idx = np.append(idx, i*np.ones(aGroupLen[i-1]).astype(int))
  
  
x, y, z = (aPos/1000)[:,0], (aPos/1000)[:,1], (aPos/1000)[:,2]



np.save('npy/particlesFOF', np.vstack((x, y, z, idx)))



  
  
  
  
