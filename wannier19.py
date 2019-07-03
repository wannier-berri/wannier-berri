#!/usr/bin/env python2
import numpy as np
import sys
import get_data
#from DOS import E_to_DOS,E_to_DOS_slow
#import wan_ham as wham
import berry

import functools

from parallel import eval_integral_BZ


def main():
    seedname="Fe"
    NK=np.array([10]*3)
    NKdiv=np.array([2]*3)
    Efermi=12.6
    Data=get_data.Data(seedname,getAA=True)
    eval_func=functools.partial(  berry.calcAHC, Efermi=Efermi )
    AHC=eval_integral_BZ(eval_func,Data,NKdiv,NK=NK,parallel=True,nproc=8)
    print "Anomalous Hall conductivity: (in S/cm ) :\n {0}  {1}  {2}".format(*tuple(AHC))
   




def main2():
    seedname="Fe"
    NK=np.array([10]*3)
    NKdiv=np.array([2]*3)
    
    

    eval_func=functools.partial(
        berry.calcAHC_dk, NK=NK,data=Data,Efermi=Efermi )
    


if __name__ == '__main__':
    main()




def main1():
    seedname="Fe"
    NK=np.array([10]*3)
    NKdiv=np.array([2]*3)
    
    dk1=1./(NK*NKdiv)
    dk_list=[dk1*np.array([x,y,z]) for x in range(NKdiv[0]) for y in range(NKdiv[1]) for z in range(NKdiv[2]) ]

    
    Efermi=12.6
    Data=get_data.Data(seedname,getAA=True)
    
    def berry_dk(dk):
        return berry.calcAHC(NK,copy.deepcopy(Data),Efermi=Efermi,evalJ0=True,evalJ1=True,evalJ2=True,dk=dk)
    
    print dk_list
    p=Pool(100)
    AHC=p.map(berry_dk,dk_list)
#    AHC=sum(berry_dk(dk) for dk in dk_list)/len(dk_list)

#        for dk in dk_list )
    print "Anomalous Hall conductivity: (in S/cm ) :\n {0}  {1}  {2}".format(*tuple(AHC))
    
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



    
#E_K=wham.get_eig(NK,Data.HH_R,Data.iRvec)[0]

#print(E_K.min(),E_K.max(),E_K[:,6:].min(),E_K[:,:6].max())

#edos,DOS=E_to_DOS(E_K,sigma=0.05)#,emin=5.7,emax=6.3)
#DOS/=np.prod(NK)*Data.cell_vollume




#print(E_K.min(),E_K.max(),E_K[:,6:].min(),E_K[:,:6].max())
#open("DOS.dat","w").write("".join("{0:10.5f} {1:20.8e}\n".format(e,d) for e,d in zip(edos,DOS)))

