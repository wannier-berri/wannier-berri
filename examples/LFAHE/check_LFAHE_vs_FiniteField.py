from matplotlib import pyplot as plt
import ray
from wannierberri.calculators.static import LFAHE_spin_fsea, AHC_Zeeman_spin, AHC
from wannierberri import run
from wannierberri.grid import Grid
from wannierberri.system import System_R
import numpy as np
from wannierberri import factors
from copy import deepcopy
from wannierberri.logging import configure_logging
configure_logging(loglevel="INFO")
print(f"factor_spin: {factors.factor_spin}")
ray.init(num_cpus=16)



def Haldane_ptb_double(delta=0.2, hop1=-1.0, hop2=0.15, phi=np.pi / 2, soc_term=0):
    """same as :func:`~wannierberri.models.Haldane_tbm`, but uses `PythTB <http://www.physics.rutgers.edu/pythtb/>`__

    Notes
    -----
    PythTB should be installed to use this (`pip install pythtb`)
    """
    import pythtb
    lat = [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
    orb = [[1. / 3., 1. / 3.], [1. / 3., 1. / 3.], [2. / 3., 2. / 3.], [2. / 3., 2. / 3.]]
    lattice = pythtb.Lattice(lat_vecs=lat, orb_vecs=orb, periodic_dirs=[0, 1])
    my_model = pythtb.TBModel(lattice)
    delta = 0.2
    t2 = hop2 * np.exp(1.j * phi)
    t2c = t2.conjugate()

    my_model.set_hop(soc_term, 0, 1, [0, 0])
    my_model.set_hop(soc_term, 2, 3, [0, 0])

    for i, _t2 in enumerate([t2, t2c]):
        _t2c = _t2.conjugate()
        my_model.set_onsite([-delta, -delta, delta, delta])
        my_model.set_hop(hop1, 0 + i, 2 + i, [0, 0])
        my_model.set_hop(hop1, 2 + i, 0 + i, [1, 0])
        my_model.set_hop(hop1, 2 + i, 0 + i, [0, 1])
        my_model.set_hop(_t2, 0 + i, 0 + i, [1, 0])
        my_model.set_hop(_t2, 2 + i, 2 + i, [1, -1])
        my_model.set_hop(_t2, 2 + i, 2 + i, [0, 1])
        my_model.set_hop(_t2c, 2 + i, 2 + i, [1, 0])
        my_model.set_hop(_t2c, 0 + i, 0 + i, [1, -1])
        my_model.set_hop(_t2c, 0 + i, 0 + i, [0, 1])

    return my_model


model = Haldane_ptb_double(delta=0.2, hop1=-1.0, hop2=0.15, phi=np.pi / 2, soc_term=0.5)
system = System_R.from_pythtb(model)
system.set_spin_eigenstates([1, -1, 1, -1])
system.set_pointgroup(["C3z"])

B_tesla = 100

system_magnetic = deepcopy(system)
system_magnetic.add_Zeemann_spin(B=np.array([0, 0, B_tesla]))
EFermi = np.linspace(-2, 2, 1001)
kwargs = dict(Efermi=EFermi, tetra=True)
grid = Grid(system, length=100, NKFFT=(6, 6, 1))


calculators_mag = {}
calculators_nomag = {}
calculators_mag["AHC"] = AHC(**kwargs)
calculators_nomag["LFAHE_spin_fsea"] = LFAHE_spin_fsea(**kwargs)
calculators_nomag["AHC_Zeeman_spin"] = AHC_Zeeman_spin(**kwargs)

result_non_magnetic = run(system, grid, calculators=calculators_nomag)
result_magnetic = run(system_magnetic, grid, calculators=calculators_mag)

ahc_Z_fsurf = -result_non_magnetic.results["AHC_Zeeman_spin"].data[:, 2, 2] * B_tesla
ahc_Z_fsea = result_non_magnetic.results["LFAHE_spin_fsea"].data[:, 2, 2] * B_tesla
AHC_finiteB = result_magnetic.results["AHC"].data[:, 2]




plt.plot(EFermi, AHC_finiteB, label=f"AHC z with B={B_tesla} T")
plt.plot(EFermi, ahc_Z_fsea, label="fermi-sea")
plt.plot(EFermi, ahc_Z_fsurf, label="fermi-surface (correcteed sign)")
plt.plot(EFermi, (ahc_Z_fsurf + ahc_Z_fsea), label="total", linestyle="--")
plt.legend()
plt.xlabel("Energy (eV)")
plt.ylabel("AHC (S/m)")
plt.legend()

plt.savefig("LFAHE_vs_FiniteField.png", dpi=300)
plt.show()
