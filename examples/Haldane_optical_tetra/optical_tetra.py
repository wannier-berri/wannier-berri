from matplotlib import pyplot as plt
import numpy as np
from wannierberri.models import Chiral, Haldane_ptb, Haldane_double
from wannierberri.system import System_R
from wannierberri.calculators.dynamic import JDOS, OpticalConductivity, SHC
from wannierberri.calculators.static import DOS
from wannierberri import Grid, run
from wannierberri import configure_logging
import ray
ray.init(num_cpus=16)
configure_logging()

# model = Chiral()
model = Haldane_ptb()
system = System_R.from_pythtb(model)
system.set_pointgroup(["C3z"])

# system = Haldane_double()

omega = np.linspace(-30, 30, 1000)
Efermi = np.linspace(-5, 5, 1000)
Efermi0 = np.array([0.0])
kwargs_shc = dict(Efermi=Efermi0, omega=omega, SHC_type="simple")
calculators = {
    # "jdos": JDOS(Efermi=Efermi0, omega=omega, kBT=0.001, smr_fixed_width=0.005),
    # "jdos_tetra": JDOS(Efermi=Efermi0, omega=omega, tetra_delta=True),
    "optical": OpticalConductivity(Efermi=Efermi0, omega=omega, smr_fixed_width=0.01),
    "optical_tetra": OpticalConductivity(Efermi=Efermi0, omega=omega, tetra_delta=True, tetra_reactive=True),
    # "SHC": SHC(**kwargs_shc, smr_fixed_width=0.05),
    # "SHC_tetra": SHC(**kwargs_shc, tetra=True),
    # "dos": DOS(Efermi=Efermi, tetra=True),
}

grid = Grid(system, NK=120)

results = run(system, grid=grid, calculators=calculators)  # , use_irred_kpt=False, symmetrize=False)  # do not use symmetry because tetrahedron is not very symmetric


key = "optical"
component = (0, 0)

# key = "SHC"
# component = (0,1,2)

data_smr = results.results[f"{key}"].data[0]
data_tetra = results.results[f"{key}_tetra"].data[0]
component_str = "".join(["xyz"[i] for i in component])

fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axes[0].plot(omega, data_smr[:, *component].real, label=f"{key}_{component_str}_real")
axes[0].plot(omega, data_tetra[:, *component].real, label=f"{key}_{component_str}_real_tetra", linestyle="--")
axes[1].plot(omega, data_smr[:, *component].imag, label=f"{key}_{component_str}_imag")
axes[1].plot(omega, data_tetra[:, *component].imag, label=f"{key}_{component_str}_imag_tetra", linestyle="--")
axes[0].set_ylabel(f"Re Sigma")
axes[1].set_ylabel(f"Im Sigma")
axes[1].set_xlabel("omega")
axes[0].legend()
axes[1].legend()
# plt.show()
plt.savefig(f"optical_tetra_{key}_{component_str}.png", dpi=300)
