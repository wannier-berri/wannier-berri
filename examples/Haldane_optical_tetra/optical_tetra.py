from matplotlib import pyplot as plt
import numpy as np
from wannierberri.models import Haldane_ptb
from wannierberri.system import System_R
from wannierberri.calculators.dynamic import OpticalConductivity
from wannierberri import Grid, run
from wannierberri import configure_logging
import ray
ray.init(num_cpus=16)
configure_logging()

model = Haldane_ptb()
system = System_R.from_pythtb(model)
# system.set_pointgroup(["C3z"])


omega = np.linspace(-30, 30, 1000)
Efermi = np.linspace(-5, 5, 1000)
Efermi0 = np.array([-2.0])
kwargs_shc = dict(Efermi=Efermi0, omega=omega, SHC_type="simple")
calculators = {
    "optical": OpticalConductivity(Efermi=Efermi0, omega=omega, smr_fixed_width=0.1),
    "optical_tetra_delta": OpticalConductivity(Efermi=Efermi0, omega=omega, tetra_delta=True, tetra_reactive=False, tetra_occupation=False),
    "optical_tetra_reactive": OpticalConductivity(Efermi=Efermi0, omega=omega, tetra_delta=False, tetra_reactive=True, tetra_occupation=False),
    "optical_tetra_all": OpticalConductivity(Efermi=Efermi0, omega=omega, tetra_delta=True, tetra_reactive=True, tetra_occupation=True),
}

grid = Grid(system, NK=9)




results_tetra = run(system, grid=grid, calculators=calculators, parameters_K={"use_tetra_in_low_D":True})  # , use_irred_kpt=False, symmetrize=False)  # do not use symmetry because tetrahedron is not very symmetric
results_triangle = run(system, grid=grid, calculators=calculators, parameters_K={"use_tetra_in_low_D":False})  # , use_irred_kpt=False, symmetrize=False)  # do not use symmetry because tetrahedron is not very symmetric
key = "optical"
component = (0, 0)


data_smr = results_tetra.results[f"{key}"].data[0]
data_tetra = results_tetra.results[f"{key}_tetra"].data[0]
data_tetra_occ = results_tetra.results[f"{key}_tetra_occ"].data[0]
data_triangle = results_triangle.results[f"{key}_tetra"].data[0]
data_triangle_occ = results_triangle.results[f"{key}_tetra_occ"].data[0]
component_str = "".join(["xyz"[i] for i in component])

fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
# axes[0].plot(omega, data_smr[:, *component].real, label=f"{key}_{component_str}_real")
# axes[0].plot(omega, data_tetra[:, *component].real, label=f"{key}_{component_str}_real_tetra", linestyle="--")
# axes[0].plot(omega, data_tetra_occ[:, *component].real, label=f"{key}_{component_str}_real_tetra_occ", linestyle=":")
axes[0].plot(omega, data_triangle[:, *component].real, label=f"{key}_{component_str}_real_triangle", linestyle="-.")
axes[0].plot(omega, data_triangle_occ[:, *component].real, label=f"{key}_{component_str}_real_triangle_occ", linestyle=":")
# axes[1].plot(omega, data_smr[:, *component].imag, label=f"{key}_{component_str}_imag")
# axes[1].plot(omega, data_tetra[:, *component].imag, label=f"{key}_{component_str}_imag_tetra", linestyle="--")
# axes[1].plot(omega, data_tetra_occ[:, *component].imag, label=f"{key}_{component_str}_imag_tetra_occ", linestyle=":")
axes[1].plot(omega, data_triangle[:, *component].imag, label=f"{key}_{component_str}_imag_triangle", linestyle="-.")
axes[1].plot(omega, data_triangle_occ[:, *component].imag, label=f"{key}_{component_str}_imag_triangle_occ", linestyle=":") 
axes[0].set_ylabel(f"Re Sigma")
axes[1].set_ylabel(f"Im Sigma")
axes[1].set_xlabel("omega")
axes[0].legend()
axes[1].legend()
# plt.show()
plt.savefig(f"optical_tetra_{key}_{component_str}.png", dpi=300)
