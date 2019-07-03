#!/usr/bin/env python2
import sys
sys.path.append('../../modules/')

import numpy as np
import get_data
import berry
import functools
from parallel import eval_integral_BZ


def main():
    seedname="Fe"
    NKFFT=np.array([int(sys.argv[2])]*3)
    NKdiv=np.array([int(sys.argv[3])]*3)
    Efermi=12.6
    if sys.argv[1].lower()=="tb":
        Data=get_data.Data(tb_file='Fe_tb.dat',getAA=True)
    elif sys.argv[1].lower()=="aa":
        Data=get_data.Data(seedname,getAA=True)
    
    eval_func=functools.partial(  berry.calcAHC, Efermi=Efermi )
    AHC=eval_integral_BZ(eval_func,Data,NKdiv,NKFFT=NKFFT,parallel=False,nproc=4)
    print "Anomalous Hall conductivity: (in S/cm ) :"
    print "J0 term :    {0:7.4f}    {1:7.4f}    {2:7.4f}".format(*tuple(AHC[0]))
    print "J1 term :    {0:7.4f}    {1:7.4f}    {2:7.4f}".format(*tuple(AHC[1]))
    print "J2 term :    {0:7.4f}    {1:7.4f}    {2:7.4f}".format(*tuple(AHC[2]))
    print "-"*50
    print "Total   :    {0:7.4f}    {1:7.4f}    {2:7.4f}".format(*tuple(AHC[3]))
   
if __name__ == '__main__':
    main()


