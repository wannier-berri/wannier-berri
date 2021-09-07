#!/usr/bin/env python

from ase import Atoms
from gpaw import GPAW
from wannierberri import System_ASE
import wannierberri as wberri
import numpy as np


from ase.dft import Wannier
calc = GPAW('Te.gpw')
wan = Wannier( nwannier=12,calc = calc,file='wannier.pickle')

for kz in np.linspace(0,1,21):
    print (f"{kz:8.5f}",np.linalg.eigvalsh(wan.get_hamiltonian_kpoint([0,0,kz])))


system = System_ASE(wan)
path=wberri.Path(system,
#                 k_nodes=[[1./3,1./3,0],[1./3,1./3,0.5],None,[1./3,1./3,0.5],[1./3,1./3,1],],
                 k_nodes=[[0,0,0],[0,0,1]],
                 labels=["G","A"],
#                 labels=["K1","H","H","K2"],
                 length=100)
path_result=wberri.tabulate(system,
                 grid=path,
                 quantities=[])
k=path.getKline()
E=path_result.get_data(quantity='E',iband=np.arange(12))
for _k,_e in zip(k,E):
    print (_k,_e)
    

exit()
system = System_ASE(wan)


SYM=wberri.symmetry

Efermi=np.linspace(-10,10,1001)

#generators=[SYM.Inversion,SYM.C4z,SYM.TimeReversal*SYM.C2x]
#system.set_symmetry(generators)
grid=wberri.Grid(system,length=50,NKFFT=10)
print ("grid : ",grid.FFT, grid.div)
parallel=wberri.Parallel(method="ray",num_cpus=4)

wberri.integrate(system,
            grid=grid,
            Efermi=Efermi, 
            smearEf=300,
            quantities=["dos","cumdos"],
            parallel=parallel,
            adpt_num_iter=10,
            fftlib='fftw', #default.  alternative  option - 'numpy'
            fout_name='Fe',
            file_Klist = None,
            restart=False,
            )


