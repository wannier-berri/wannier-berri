#!/usr/bin/env python3

DO_profile=False

import sys
#sys.path.append('../../modules/')
import numpy as np
import wannier19 as w19
from time import time
SYM=w19.symmetry

#from  w19 import symmetry as SYM


def main():
    t0=time()
    seedname="Fe"
    NKFFT=np.array([int(sys.argv[1])]*3)
    NKdiv=np.array([int(sys.argv[2])]*3)
    
    name1="NKFFT={0}_NKdiv={1}_adptmesh=2-sym-smooth10+TR".format(*tuple(sys.argv[1:4]))
    name=seedname+"_w19_"+name1
    Efermi=np.linspace(12.,13.,1001)
    Data=w19.Data(tb_file='Fe_tb.dat',getAA=True)
#    Data=get_data.Data(seedname,getAA=True)
    generators=[SYM.Inversion,SYM.C4z,SYM.TimeReversal*SYM.C2x]
    t1=time()
    w19.integrate(Data,NKdiv=NKdiv,NKFFT=NKFFT,Efermi=Efermi, smearEf=10,quantities=["ahc","dos"],adpt_num_iter=10,fout_name=name,symmetry_gen=generators,restart=False)
    t2=time()

          
    print ("time for reading     : {0} s ".format(t1-t0))
    print ("time for integration : {0} s ".format(t2-t1))
    print ("total time           : {0} s ".format(t2-t0))
     

 
if DO_profile:
    import cProfile
    cProfile.run('main()')
else:
    main()



