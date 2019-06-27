#!/usr/bin/env python3
import numpy as np
import sys
from aux import str2bool
from scipy.io import FortranFile as FF
from DOS import E_to_DOS
seedname=sys.argv[1]
NK=np.array([25,25,25])

f=open(seedname+"_HH_save.info","r")
l=f.readline().split()[:3]
num_wann,nRpts,spinors=int(l[0]),int(l[1]),str2bool(l[2])
real_lattice=np.array([f.readline().split()[:3] for i in range(3)],dtype=float)
iRpts=np.array([f.readline().split()[:4] for i in range(nRpts)],dtype=int)
f.close()

cell_vollume=np.linalg.det(real_lattice)
Ndegen=iRpts[:,3]
iRpts=iRpts[:,:3]

print ("Number of wannier functions:",num_wann)
print ("Number of R points:", nRpts)
print ("Real-space lattice:\n",real_lattice)
#print ("R - points and dege=neracies:\n",iRpts)

f=FF(seedname+"_HH_R.dat")
HH_R=np.array([[np.array(f.read_record('2f8'),dtype=float) for m in range(num_wann)] for n in range(num_wann)])
HH_R=HH_R[:,:,:,0]+1j*HH_R[:,:,:,1]
f.close()
HH_R=HH_R/Ndegen[None,None,:]

HH_K=np.zeros( (num_wann,num_wann,NK[0],NK[1],NK[2]), dtype=complex )
for iR in range(nRpts):
    if np.all(np.abs(iRpts[iR])<=NK/2):
        HH_K[:,:,iRpts[iR,0],iRpts[iR,1],iRpts[iR,2]]=HH_R[:,:,iR]

for m in range(num_wann):
   for n in range(num_wann):
      HH_K[m,n]=np.fft.fftn(HH_K[m,n])

print ("FFT performed")

HH_K=HH_K.reshape( num_wann,num_wann,-1)

##### check hermicity:
check=np.max([ np.abs(HH_K[:,:,ik]-HH_K[:,:,ik].T.conj()).max()  for ik in range (np.prod(NK))])
if check>1e-12 :  print ("WARNING:Hermicity is not good : ",check) 

E=np.array([np.linalg.eigvalsh(HH_K[:,:,ik]) for ik in range(np.prod(NK))])

print ("Energies calculated")

edos,DOS=E_to_DOS(E,sigma=0.05)#,emin=5.7,emax=6.3)
DOS/=np.prod(NK)*cell_vollume


print(E.min(),E.max(),E[:,6:].min(),E[:,:6].max())
open("DOS.dat","w").write("".join("{0:10.5f} {1:20.8e}\n".format(e,d) for e,d in zip(edos,DOS)))

