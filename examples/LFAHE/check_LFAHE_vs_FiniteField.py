from matplotlib import pyplot as plt
import ray
from wannierberri.calculators.static import LFAHE_spin_fsea, AHC_Zeeman_spin, AHC
from wannierberri import run, symmetry
from wannierberri.grid import Grid
import numpy as np
from wannierberri import factors
from copy import deepcopy
from wannierberri.logging import configure_logging
configure_logging(loglevel="INFO")
print(f"factor_spin: {factors.factor_spin}")
ray.init(num_cpus=16)
symmetry = True

from wannierberri.models import Haldane_ptb_double


system = Haldane_ptb_double(delta=0.2, hop1=-1.0, hop2=0.15, phi=np.pi / 2, soc_term=0.3)
B_tesla = 100

system_magnetic = deepcopy(system)
system_magnetic.add_Zeeman_spin(B=np.array([0, 0, B_tesla]))
EFermi = np.linspace(-2, 2, 1001)
kwargs = dict(Efermi=EFermi, tetra=True)
kwargs_ext = dict(kwargs_formula=dict(internal_terms=False, external_terms=True), **kwargs)
kwargs_int = dict(kwargs_formula=dict(internal_terms=True, external_terms=False), **kwargs)
grid = Grid(system, length=100, NKFFT=(6, 6, 1))


calculators_mag = {}
calculators_nomag = {}
calculators_mag["AHC_int"] = AHC(**kwargs_int)
calculators_mag["AHC_ext"] = AHC(**kwargs_ext)
calculators_nomag["LFAHE_spin_fsea_int"] = LFAHE_spin_fsea(**kwargs_int)
calculators_nomag["LFAHE_spin_fsea_ext"] = LFAHE_spin_fsea(**kwargs_ext)

calculators_nomag["AHC_Zeeman_spin_int"] = AHC_Zeeman_spin(**kwargs_int)
calculators_nomag["AHC_Zeeman_spin_ext"] = AHC_Zeeman_spin(**kwargs_ext)

result_non_magnetic = run(system, grid, calculators=calculators_nomag).results
result_magnetic = run(system_magnetic, grid, calculators=calculators_mag).results



fig, axes = plt.subplots(2, 1, figsize=(8, 6))

correct_sign_AHC_Z = -1

for ax, term in zip(axes, ["int", "ext"]):
    ahc_Z_fsurf = correct_sign_AHC_Z * result_non_magnetic[f"AHC_Zeeman_spin_{term}"].data[:, 2, 2] * B_tesla
    ahc_Z_fsea = result_non_magnetic[f"LFAHE_spin_fsea_{term}"].data[:, 2, 2] * B_tesla
    AHC_finiteB = result_magnetic[f"AHC_{term}"].data[:, 2]
    ax.plot(EFermi, AHC_finiteB, label=f"AHC_{term} with B={B_tesla} T")
    ax.plot(EFermi, ahc_Z_fsea, label=f"fermi-sea {term}")  
    ax.plot(EFermi, ahc_Z_fsurf, label=f"fermi-surface {term} (corrected sign)")
    ax.plot(EFermi, (ahc_Z_fsurf + ahc_Z_fsea), label=f"total {term}", linestyle="--")
    ax.set_title(f"AHC {term}")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("AHC (S/m)")
    ax.legend()


plt.savefig(f"LFAHE_vs_FiniteField-symmetry={symmetry}.png", dpi=300)
plt.show()
