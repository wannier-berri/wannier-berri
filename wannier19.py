#!/usr/bin/env python3
import numpy as np
import sys
from aux import str2bool
from scipy.io import FortranFile as FF
from DOS import E_to_DOS
import wan_ham as wham

seedname=sys.argv[1]
NK=np.array([25,25,25])

f=open(seedname+"_HH_save.info","r")
l=f.readline().split()[:3]
num_wann,nRvec,spinors=int(l[0]),int(l[1]),str2bool(l[2])
real_lattice=np.array([f.readline().split()[:3] for i in range(3)],dtype=float)
iRvec=np.array([f.readline().split()[:4] for i in range(nRvec)],dtype=int)
f.close()

cell_vollume=np.linalg.det(real_lattice)
Ndegen=iRvec[:,3]
iRvec=iRvec[:,:3]

print ("Number of wannier functions:",num_wann)
print ("Number of R points:", nRvec)
print ("Real-space lattice:\n",real_lattice)
print ("R - points and dege=neracies:\n",iRvec)

f=FF(seedname+"_HH_R.dat")
HH_R=np.array([[np.array(f.read_record('2f8'),dtype=float) for m in range(num_wann)] for n in range(num_wann)])
HH_R=HH_R[:,:,:,0]+1j*HH_R[:,:,:,1]
f.close()
HH_R=HH_R/Ndegen[None,None,:]

E_K=wham.get_eig_deleig(NK,HH_R,iRvec)[0]

print ("Energies calculated")

print(E_K.min(),E_K.max(),E_K[:,6:].min(),E_K[:,:6].max())

edos,DOS=E_to_DOS(E_K,sigma=0.05)#,emin=5.7,emax=6.3)
DOS/=np.prod(NK)*cell_vollume


print(E_K.min(),E_K.max(),E_K[:,6:].min(),E_K[:,:6].max())
open("DOS.dat","w").write("".join("{0:10.5f} {1:20.8e}\n".format(e,d) for e,d in zip(edos,DOS)))

