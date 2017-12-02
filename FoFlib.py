"""
./FoF_Special . snapshot 005

./FoF_Special . snap100Mpc128_z0 000



"""
from numpy import *


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

  Ngroups = fromstring(f.read(4),int32)[0]

  GroupLen    = fromstring(f.read(4*Ngroups),int32)
  GroupOffset = fromstring(f.read(4*Ngroups),int32)
  GroupMass   = fromstring(f.read(4*Ngroups),float32)

  GroupCM     = fromstring(f.read(3*4*Ngroups),float32)
  GroupCM.shape  = (Ngroups,3)


  GroupNspecies = fromstring(f.read(3*4*Ngroups),int32)
  GroupNspecies.shape  = (Ngroups,3)

  GroupMspecies = fromstring(f.read(3*4*Ngroups),float32)
  GroupMspecies.shape  = (Ngroups,3)

  GroupSfr   = fromstring(f.read(4*Ngroups),float32)

  GroupMetallicities = fromstring(f.read(2*4*Ngroups),float32)
  GroupMetallicities.shape  = (Ngroups,2)

  Mcold = fromstring(f.read(4*Ngroups),float32) 

  SigmaStars= fromstring(f.read(4*Ngroups),float32) 
  SigmaDM= fromstring(f.read(4*Ngroups),float32) 

  f.close()

  return GroupLen,GroupOffset,GroupMass,GroupCM, GroupNspecies, GroupSfr
  
  
  
  
def read_groups_particles(filename):
  """
  Read the "fof_special_particles" files and return
  an array of the positions of each particles belonging
  to a group.
  """
  
  f = open(filename,'r')

  Ntot = fromstring(f.read(4),int32)[0]
  Pos	  = fromstring(f.read(3*4*Ntot),float32)
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
  
  Ntot = fromstring(f.read(4),int32)[0]
  idx	  = fromstring(f.read(3*4*Ntot),float32)
  
  f.close()
  
  return Ntot, idx
  
  
  
fName = '/home/nesar/Desktop/032/groups_catalogue/fof_special_catalogue_032'


aGroupLen, aGroupOffset, aGroupMass, aGroupCM, aGroupNspecies, aGroupSfr = read_groups_catalogue(fName)

fName1 = '/home/nesar/Desktop/032/groups_particles/fof_special_particles_032'

aPos, aNtot = read_groups_particles(fName1)

fName2 = '/home/nesar/Desktop/032/groups_indexlist/fof_special_indexlist_032'

ntot, idx = read_groups_indexlist(fName2)

  
print aPos.shape
print
print aGroupMass.shape
  
  
  
  
x, y, z = (aPos/1000)[:,0], (aPos/1000)[:,1], (aPos/1000)[:,2]








  
  
  
  
