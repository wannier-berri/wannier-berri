#!/usr/bin/env python3
import numpy as np
import sys
import get_data
from DOS import E_to_DOS
import wan_ham as wham
import berry


def main():
    seedname="Fe"
    NK=np.array([50,50,50])
    Efermi=12.6
    Data=get_data.Data(seedname,getAA=True)
    berry.calcAHC(NK,Data,Efermi=Efermi)




if __name__ == '__main__':
    main()
    
#E_K=wham.get_eig(NK,Data.HH_R,Data.iRvec)[0]

#print(E_K.min(),E_K.max(),E_K[:,6:].min(),E_K[:,:6].max())

#edos,DOS=E_to_DOS(E_K,sigma=0.05)#,emin=5.7,emax=6.3)
#DOS/=np.prod(NK)*Data.cell_vollume




#print(E_K.min(),E_K.max(),E_K[:,6:].min(),E_K[:,:6].max())
#open("DOS.dat","w").write("".join("{0:10.5f} {1:20.8e}\n".format(e,d) for e,d in zip(edos,DOS)))

