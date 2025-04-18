#!/usr/bin/env python

from ase import Atoms
from gpaw import GPAW
from wannierberri import System_ASE
import wannierberri as wberri
import numpy as np

from ase.dft import Wannier

calc = GPAW('Te.gpw')
ik1 = 7
ik2 = 8
G = [0, 0, 0]
NB = 3
# print (calc.get_wannier_integrals(  0, ik1,ik2, G, NB))

# exit()


wan = Wannier(nwannier=3, calc=calc, file='wannier-s.pickle')

print(wan.Gdir_dc)
print(wan.kklst_dk)
# print (wan.kpt_kc)
# for k,kpt in enumerate(wan.kpt_kc):
#    for d,G in enumerate(wan.Gdir_dc):
#        print (k,kpt,G,wan.kklst_dk[d,k],wan.kpt_kc[wan.kklst_dk[d,k]])
# system = System_ASE(wan,ase_calc=calc, berry=True,ase_R_vectors = False,transl_inv_MV = True)

# for k,kpt in enumerate(system.kpt_red):
#    for d,G in enumerate(system.mmn.bk_red):
#        print (k,kpt,G,system.mmn.neighbours[k,d],system.kpt_red[system.mmn.neighbours[k,d]],system.mmn.G[k,d])


# exit()
# for kz in np.linspace(0,1,21):
#    print (f"{kz:8.5f}",np.linalg.eigvalsh(wan.get_hamiltonian_kpoint([1./3,1./3,kz])))

# system = System_ASE(wan,ase_calc=calc, berry=False)
# exit()


# for kz  in np.linspace(0,1,21):
#    print (f"{kz:8.5f}",np.linalg.eigvalsh(wan.get_hamiltonian_kpoint([1./3,1./3,kz])))

k1 = k2 = 1. / 3

system = System_ASE(wan, ase_calc=calc, use_ws=False, berry=True, ase_R_vectors=False, transl_inv=True)

print(system.wannier_centres_cart)
print(system.AA_R[:, :, system.iRvec0, 0].diagonal())
print(system.AA_R[:, :, system.iRvec0, 1].diagonal())
print(system.AA_R[:, :, system.iRvec0, 2].diagonal())

exit()

path = wberri.Path(system,
                   #                 k_nodes=[[1./3,1./3,0],[1./3,1./3,0.5],None,[1./3,1./3,0.5],[1./3,1./3,1],],
                   k_nodes=[[k1, k2, 0.35], [k1, k2, 0.5], [k1, k2, 0.65]],
                   labels=["K", "H", "K"],
                   #                 labels=["K1","H","H","K2"],
                   length=5000)
path_result = wberri.tabulate(system,
                              grid=path,
                              quantities=["berry"])

path_result.plot_path_fat(path,
                          quantity='berry',
                          component='z',
                          save_file="Te-berry-VB.pdf",
                          Eshift=0,
                          Emin=5, Emax=6,
                          iband=None,
                          mode="fatband",
                          fatfactor=0.01,
                          fatmax=200,
                          cut_k=True
                          )

exit()
from matplotlib import pyplot as plt

kz = np.linspace(0, 1, 101)
E = np.array([np.linalg.eigvalsh(wan.get_hamiltonian_kpoint([k1, k2, _])) for _ in kz])
for e in E.T:
    plt.plot(kz, e, c='blue')
    plt.plot(1. - kz, e, c='red')

plt.ylim(5, 6)
plt.xlim(0.35, 0.65)

plt.show()

exit()

k = path.getKline()
E = path_result.get_data(quantity='E', iband=np.arange(12))
for _k, _e in zip(k, E):
    print(_k, _e)

exit()
# system = System_ASE(wan)


SYM = wberri.point_symmetry

Efermi = np.linspace(-10, 10, 10001)

# generators=[SYM.Inversion,SYM.C4z,SYM.TimeReversal*SYM.C2x]
# system.set_symmetry(generators)
grid = wberri.Grid(system, length=50, NKFFT=10)
print("grid : ", grid.FFT, grid.div)
parallel = wberri.Parallel(method="ray", num_cpus=4)

wberri.integrate(system,
                 grid=grid,
                 Efermi=Efermi,
                 smearEf=300,
                 quantities=["dos", "cumdos"],
                 parallel=parallel,
                 adpt_num_iter=10,
                 fftlib='fftw',  # default.  alternative  option - 'numpy'
                 fout_name='Fe',
                 file_Klist=None,
                 restart=False,
                 )
