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
    NKtot=int(sys.argv[1])*int(sys.argv[2])
    NKFFT=np.array([int(sys.argv[1])]*3)
    NKdiv=np.array([int(sys.argv[2])]*3)
    Efermi=np.linspace(12.0,13.,1000)
#    Efermi=np.array([12.5])
#    Data=get_data.Data(tb_file='Fe_tb.dat',getAA=True)
    Data=get_data.Data(seedname,getAA=True,getBB=True,getCC=True)
    
    eval_func=functools.partial(  berry.calcMorb, Efermi=Efermi )
#    eval_func=functools.partial(  berry.calcImfgh, Efermi=Efermi )
    Morb_all=eval_integral_BZ(eval_func,Data,NKdiv,NKFFT=NKFFT,parallel=False,nproc=4) #.transpose ( (1,0,2,3) )*berry.fac_morb
    print ("shape",Morb_all.shape)

## now write the result
    for name,arr in zip(['tot','LC','IC'],Morb_all):
#    for name,arr in zip(['imf','img','imh'],Morb_all):
      open(seedname+"_w19_Morb_{0}_NK={1}.dat".format(name,NKtot),"w").write(
       "    ".join("{0:^15s}".format(s) for s in ["EF",]+
                [a+b for a in ["M",] for b in "x","y","z"])+"\n"+
      "\n".join(
       "    ".join("{0:15.6f}".format(x) for x in [ef]+[x for x in ahc ]) 
                      for ef,ahc in zip (Efermi,arr) )
       +"\n")

 
if __name__ == '__main__':
    if DO_profile:
        import cProfile
        cProfile.run('main()')
    else:
        main()



