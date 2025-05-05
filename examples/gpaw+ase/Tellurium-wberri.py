#!/usr/bin/env python

from matplotlib import pyplot as plt
from gpaw import GPAW
from wannierberri import System_ASE
import wannierberri as wberri
import numpy as np

from ase.dft.wannier import Wannier

calc = GPAW('Te.gpw')
ik1 = 7
ik2 = 8
G = [0, 0, 0]
NB = 3
# print (calc.get_wannier_integrals(  0, ik1,ik2, G, NB))

# exit()


wan = Wannier(nwannier=12, calc=calc, file='wannier-sp.pickle')

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

system = System_ASE(wan, berry=True)

print(system.wannier_centers_cart)


path, bands = wberri.evaluate_k_path(system, k_nodes=[[k1, k2, 0], [k1, k2, 0.5], [k1, k2, 1.]],
                   labels=["K<-", "H", "->K"],
    quantities=["berry_curvature"]
)


bands.plot_path_fat(path,
                    quantity='berry_curvature',
                    component='z',
                    # save_file="Te-berry-VB.pdf",
                    close_fig=False,
                    show_fig=False,
                    Eshift=0,
                    Emin=5, Emax=6,
                    iband=None,
                    mode="fatband",
                    fatfactor=20,
                    fatmax=200,
                    cut_k=True
                          )


kz = np.linspace(0, 1., 101)
fac = system.recip_lattice[2, 2]
E = np.array([np.linalg.eigvalsh(wan.get_hamiltonian_kpoint([k1, k2, _])) for _ in kz])
for e in E.T:
    plt.plot(kz * fac, e, '--', c='blue')
    plt.plot((1. - kz) * fac, e, '--', c='red')

# plt.ylim(5, 6)
# plt.xlim(0.35, 0.65)

plt.savefig('Te-berry-VB_ase.pdf')
