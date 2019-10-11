#!/usr/bin/env python2
DO_profile=True

import sys
sys.path.append('../../modules/')
import numpy as np
import get_data
import berry
import functools
from parallel import eval_integral_BZ


def main():
    seedname="Fe"
    NKFFT=np.array([int(sys.argv[1])]*3)
    NKdiv=np.array([int(sys.argv[2])]*3)
    Efermi=np.linspace(12.,13.,11)
#    Data=get_data.Data(tb_file='Fe_tb.dat',getAA=True)
    Data=get_data.Data(seedname,getAA=True)
    
    eval_func=functools.partial(  berry.calcAHC, Efermi=Efermi )
    AHC_all=eval_integral_BZ(eval_func,Data,NKdiv,NKFFT=NKFFT,parallel=False,nproc=4)

## now write the result
    open(seedname+"_w19_ahc_fermi_scan.dat","w").write(
       "    ".join("{0:^15s}".format(s) for s in ["EF",]+
                [a+b for a in ["O",] for b in "x","y","z"])+"\n"+
      "\n".join(
       "    ".join("{0:15.6f}".format(x) for x in [ef]+[x for x in ahc ]) 
                      for ef,ahc in zip (Efermi,AHC_all) )
       +"\n")  
          

 
if __name__ == '__main__':
    if DO_profile:
        import cProfile
        cProfile.run('main()')
    else:
        main()



