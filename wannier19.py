#!/usr/bin/env python2
import numpy as np
import sys
import get_data
from DOS import E_to_DOS,E_to_DOS_slow
import wan_ham as wham
import berry


def main():
    seedname="Fe"
    NK=np.array([10]*3)
    Efermi=12.6
    Data=get_data.Data(seedname,getAA=True)
    berry.calcAHC(NK,Data,Efermi=Efermi,evalJ0=True,evalJ1=True,evalJ2=True)

#    Data.write_tb()
    exit()
#    E_K=wham.get_eig(NK,Data.HH_R,Data.iRvec)[0]
#    E_K=wham.get_eig1(NK,Data.HH_R,Data.iRvec)
#    E_K=wham.get_eig_slow(NK,Data.HH_R,Data.iRvec)
#    np.savetxt("energies.dat",E_K)
#    np.savetxt("energies-slow.dat",E_K2)
    edos,DOS=E_to_DOS(E_K,sigma=0.001,emin=0,emax=10)
    DOS/=np.prod(NK)*Data.cell_volume
    open("DOS.dat","w").write("".join("{0:10.5f} {1:20.8e}\n".format(e,d) for e,d in zip(edos,DOS)))
    edos,DOS=E_to_DOS_slow(E_K,sigma=0.001,emin=0,emax=10)
    DOS/=np.prod(NK)*Data.cell_volume
    open("DOS-slow.dat","w").write("".join("{0:10.5f} {1:20.8e}\n".format(e,d) for e,d in zip(edos,DOS)))
    

    #,emin=5.7,emax=6.3)
#    berry.calcAHC(NK,Data,Efermi=Efermi,evalJ0=False,evalJ1=False,evalJ2=True)




if __name__ == '__main__':
    main()
    
#E_K=wham.get_eig(NK,Data.HH_R,Data.iRvec)[0]

#print(E_K.min(),E_K.max(),E_K[:,6:].min(),E_K[:,:6].max())

#edos,DOS=E_to_DOS(E_K,sigma=0.05)#,emin=5.7,emax=6.3)
#DOS/=np.prod(NK)*Data.cell_vollume




#print(E_K.min(),E_K.max(),E_K[:,6:].min(),E_K[:,:6].max())
#open("DOS.dat","w").write("".join("{0:10.5f} {1:20.8e}\n".format(e,d) for e,d in zip(edos,DOS)))

