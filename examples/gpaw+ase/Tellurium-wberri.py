#!/usr/bin/env python

from matplotlib import pyplot as plt
from gpaw import GPAW
from wannierberri import System_R
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

k1 = k2 = 1. / 3

system = System_R.from_ase(wan, berry=True)

print(system.wannier_centers_cart)


path, bands = wberri.evaluate_k_path(system, nodes=[[k1, k2, 0], [k1, k2, 0.5], [k1, k2, 1.]],
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
