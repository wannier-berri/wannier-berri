#!/usr/bin/env python3


## these linesline if you want to use the git version of the code, instead of the one installed by pip
local_code=False


import os

if local_code:
   if 'wannier19' not in os.listdir() :
       os.symlink("../wannier19","wannier19")
else:
   if 'wannier19' in os.listdir() :
       os.remove('wannier19')

if 'Fe_tb.dat' not in os.listdir():
    os.system('tar -xvf ../data/Fe_tb.dat.tar.gz') 


import wannier19 as w19
import numpy as np


SYM=w19.symmetry

seedname="Fe"
NKFFT=np.array([10]*3)
NKdiv=np.array([2]*3)

name=seedname
Efermi=np.linspace(12.,13.,1001)
Data=w19.Data(tb_file='Fe_tb.dat',getAA=True)

generators=[SYM.Inversion,SYM.C4z,SYM.TimeReversal*SYM.C2x]




w19.tabulate(Data,
             NKdiv=NKdiv,
             NKFFT=NKFFT,
             quantities=["V",'berry'],
             symmetry_gen=generators,
             fout_name=name,
             Ef0=0,
             restart=False)


w19.integrate(Data,
    NKdiv=NKdiv,
    NKFFT=NKFFT,
    Efermi=Efermi, 
    smearEf=10,
    quantities=["ahc","dos"],
    adpt_num_iter=10,
    fout_name=name,
    symmetry_gen=generators,
    restart=False)
