from matplotlib import pyplot as plt
import numpy as np
theta = 90
phi = 90
EF = 9.22085

fig, axes = plt.subplots(2, 1, sharey=False, sharex=True, figsize=(10, 8))
for i in range(0, 101, 10):
    color = (0, 1 - i / 100, i / 100)
    res = np.load(f"results/Fe-irred-soc-ahc-th{theta}-phi{phi}-ahc_int_iter-{i:04d}.npz")
    Efermi = res['Energies_0']
    ahc = res["data"]
    axes[0].plot(Efermi - EF, ahc[:, 2] / 100, label=f"iter={i}", color=color)

res = np.load("../MnTe/results/soc-ahc_int_iter-0100.npz")
Efermi = res['Energies_0']
ahc = res["data"]
axes[0].plot(Efermi - EF, ahc[:, 2] / 100, label="not irred", color="black", linestyle="--")

axes[0].set_title("AHC internal terms convergence")
axes[0].set_ylim(-400, 400)
axes[0].set_xlabel("E-EF (eV)")
axes[0].set_ylabel("AHC (S/cm)")
axes[0].legend()
for i in range(0, 51, 10):
    color = (0, 1 - i / 50, i / 50)
    res = np.load(f"results/Fe-irred-soc-ahc-th{theta}-phi{phi}-ahc_ext_iter-{i:04d}.npz")
    Efermi = res['Energies_0']
    ahc = res["data"]
    axes[1].plot(Efermi - EF, ahc[:, 2] / 100, label=f"iter={i}", color=color)
axes[1].set_title("AHC external terms convergence")
axes[1].set_xlabel("E-EF (eV)")
axes[1].set_ylabel("AHC (S/cm)")
axes[1].legend()
plt.tight_layout()
plt.savefig("ahc_convergence.png", dpi=300)
