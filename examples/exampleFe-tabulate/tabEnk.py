#!/usr/bin/env python3

DO_profile=True

import sys
sys.path.append('../../modules/')
import numpy as np
import get_data
import tabulateXnk as tab
import functools
from integrate import eval_integral_BZ
from time import time
import symmetry as SYM
from utility import smoother


def main():
    t0=time()
    seedname="Fe"
    NKFFT=np.array([int(sys.argv[1])]*3)
    NKdiv=np.array([int(sys.argv[2])]*3)
    
    
#    NKFFT=np.array([20,10,10])
#    NKdiv=np.array([1,1,1])
    
    Data=get_data.Data(tb_file='Fe_tb.dat',getAA=True)
    generators=[SYM.Inversion,SYM.C4z,SYM.TimeReversal*SYM.C2x]
#    generators=[]
    t1=time()
    
    
    quant="o"
    eval_func=functools.partial( tab.tabXnk, quantities=quant,ibands=(4,5,6,7,8,9) )

    res=eval_integral_BZ(eval_func,Data,NKdiv,NKFFT=NKFFT,nproc=0,
            adpt_num_iter=0,adpt_nk=2,
                fout_name="",symmetry_gen=generators,
                GammaCentered=True,restart=False)
    t2=time()

    print ("V=",res.dEnk)
    res=res.to_grid(NKFFT*NKdiv)
    print ("V=",res.dEnk)
    t3=time()
    
    
    open("Fe_E-{0}.frmsf".format(NKdiv[0]),"w").write(
          res.fermiSurfer(quantity="",efermi=12.6) )
    
    for Q in quant:
     for comp in "xyzsn":
        open("Fe_{2}{0}-{1}.frmsf".format(comp,NKdiv[0],Q),"w").write(
           res.fermiSurfer(quantity=Q+comp,efermi=12.6)
           )
    t4=time()

          
    print ("time for reading     : {0} s ".format(t1-t0))
    print ("time for integration : {0} s ".format(t2-t1))
    print ("for bringing to grid : {0} s ".format(t3-t2))
    print ("time for printing    : {0} s ".format(t4-t3))
    print ("total time           : {0} s ".format(t4-t0))
     



 
if __name__ == '__main__':
    if DO_profile:
        import cProfile
        cProfile.run('main()')
    else:
        main()



