#!/usr/bin/env python3

DO_profile=True

import sys
sys.path.append('../../modules/')
import numpy as np
import get_data
import berry
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
    
    name1="NKFFT={0}_NKdiv={1}_adptmesh=2-sym-smooth10+TR".format(*tuple(sys.argv[1:4]))
    name=seedname+"_w19_ahc_"+name1
    Efermi=np.linspace(12.,13.,1001)
#    Data=get_data.Data(tb_file='Fe_tb.dat',getAA=True)
    Data=get_data.Data(seedname,getAA=True)
    generators=[SYM.Inversion,SYM.C4z,SYM.TimeReversal*SYM.C2x]
    t1=time()
    eval_func=functools.partial(  berry.calcAHC, Efermi=Efermi, smoother=smoother(Efermi,10) )
    AHC_all=eval_integral_BZ(eval_func,Data,NKdiv,NKFFT=NKFFT,nproc=4,
            adpt_num_iter=10,adpt_nk=2,
                fout_name=name,symmetry_gen=generators,
                GammaCentered=False,restart=False)
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



