#!/usr/bin/env python2
DO_profile=True

import sys
sys.path.append('../../modules/')
import numpy as np
import get_data
import berry
import functools
from parallel import eval_integral_BZ
from time import time


def write_result(AHC,name,Efermi):
        open(name,"w").write(
       "    ".join("{0:^15s}".format(s) for s in ["EF",]+
                [a+b for a in ["","J0_","J1_","J2_"] for b in ("x","y","z")])+"\n"+
      "\n".join(
       "    ".join("{0:15.6f}".format(x) for x in [ef]+[x for X in ahc[(3,0,1,2),:] for x in X]) 
                      for ef,ahc in zip (Efermi,AHC) )
       +"\n")  



def main():
    t0=time()
    seedname="Fe"
    NKFFT=np.array([int(sys.argv[1])]*3)
    NKdiv=np.array([int(sys.argv[2])]*3)
    try:
        adpt_mesh=int(sys.argv[3])
    except:
        adpt_mesh=1
    
    name1="NKFFT={0}_NKdiv={1}_adptmesh={2}".format(*tuple(sys.argv[1:4]))
    name=seedname+"_w19_ahc_"+name1
    Efermi=np.linspace(12.,13.,5001)
#    Data=get_data.Data(tb_file='Fe_tb.dat',getAA=True)
    Data=get_data.Data(seedname,getAA=True)
    t1=time()
    eval_func=functools.partial(  berry.calcAHC, Efermi=Efermi )
    eval_integral_BZ(eval_func,Data,NKdiv,NKFFT=NKFFT,nproc=4,
            adpt_mesh=adpt_mesh,adpt_percent=15,adpt_num_iter=10,adpt_thresh=0.1,
                fout_name=name,fun_write=functools.partial(write_result,Efermi=Efermi))
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



