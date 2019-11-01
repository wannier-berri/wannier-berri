#!/usr/bin/env python3

DO_profile=False

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
    
    eval_func=functools.partial( tab.tabEVnk, ibands=[0] )

    res=eval_integral_BZ(eval_func,Data,NKdiv,NKFFT=NKFFT,nproc=0,
            adpt_num_iter=0,adpt_nk=2,
                fout_name="",symmetry_gen=generators,
                GammaCentered=True,restart=False)
    print ("V=",res.dEnk)
    res=res.to_grid(NKFFT*NKdiv)
    print ("V=",res.dEnk)
    
    
    for comp in "xyzsn":
        open("Fe_V{0}-{1}.frmsf".format(comp,NKdiv[0]),"w").write(
           res.fermiSurfer(quantity="v"+comp,efermi=12.6)
           )
    t2=time()

          
    print ("time for reading     : {0} s ".format(t1-t0))
    print ("time for integration : {0} s ".format(t2-t1))
    print ("total time           : {0} s ".format(t2-t0))
     



 
if __name__ == '__main__':
    if DO_profile:
        import cProfile
        cProfile.run('main()')
    else:
        main()



