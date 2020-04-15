#!/usr/bin/python3


## these linesline if you want to use the git version of the code, instead of the one installed by pip
local_code=True #False
num_proc=32
DO_profile=False

import os
from sys import argv
if local_code:
   if 'wannierberri' not in os.listdir() :
       os.symlink("../../wannierberri","wannierberri")
else:
   if 'wannierberri' in os.listdir() :
       os.remove('wannierberri')



seedname="Te"

name=seedname

import wannierberri as wberri
import numpy as np

system=wberri.System( seedname='Te' , getAA=True , # getBB=True , getCC=True , getSS=True , 
         frozen_max=-np.Inf, degen_thresh=0.00001 , random_gauge=False , use_ws=True )

SYM=wberri.symmetry


Efermi=np.linspace(-10,10,2001)
#Efermi=np.linspace(-10,10,21)
Efermi=np.linspace(5.7,6.3,1001)

generators=[SYM.C3z,SYM.C2x,SYM.TimeReversal]
#generators=[]

#term=argv[1]
term='understand'

def main():
    for NK in 250,500:
   # NK = 5
        wberri.integrate(system,
            NKdiv=[NK//9,NK//9,(3*NK)//16],
            NKFFT=[9,9,4],
            Efermi=Efermi, 
            smearEf=50,
#        quantities=[ "berry_dipole_sea_D","berry_dipole_sea_D_old",],#  "berry_dipole_sea_D2",  "berry_dipole_sea_D3"        ],#  "berry_dipole_sea_ext2_5"],#"berry_dipole_sea_D","berry_dipole_sea_ext2_6","berry_dipole_sea_ext2_3","berry_dipole_sea_ext2_4",  "berry_dipole_sea_ext1_2", "berry_dipole_sea_ext1_1",     ] ,#"berry_dipole_sea_ext2_3", "berry_dipole_sea_ext2_4",  "berry_dipole_sea_ext2_5", "berry_dipole_sea_ext2_6",    ],#"berry_dipole_D","berry_dipole_ext1","berry_dipole_sea_ext1","berry_dipole_ext2","berry_dipole_sea_ext2",],
#        quantities=["berry_dipole_sea_ext2_3","berry_dipole_sea_ext2_4","berry_dipole_sea_ext2_5","berry_dipole_sea_ext2_6"],
#        quantities=["berry_dipole_sea_ext2","berry_dipole_sea_ext1","berry_dipole_sea_D"  ,  "berry_dipole_ext2","berry_dipole_ext1","berry_dipole_D"   ],
            quantities=['orbital_mag_sea'],#"berry_dipole_sea_ext2","berry_dipole_sea_D" ] , # "berry_dipole_ext2","berry_dipole_ext1","berry_dipole_D"   ],
#        quantities=['nonabelian_velvel'],   ['nonabelian_curvmorb','nonabelian_curvspin','nonabelian_curvvel','nonabelian_spinvel','nonabelian_morbvel','nonabelian_velvel'],  #["ahc","dos"],#,"ahc_band"],
            numproc=num_proc,
            adpt_num_iter= 10 if NK>=100 else 0 ,
            fout_name=name,
            suffix="NK={}".format(NK),
            symmetry_gen=generators,
            restart=False,
            file_klist="klist_{}_NK={}".format(term,NK),
            )

if DO_profile:
    import cProfile
    cProfile.run('main()')
else:
    main()

