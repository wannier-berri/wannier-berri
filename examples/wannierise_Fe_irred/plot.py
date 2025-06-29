from matplotlib import pyplot as plt
import numpy as np

seedname = "result"


def plot_quantity(quantity, axes=None, suffix=""):
    if axes is None:
        axes = plt.gca()
    # data = np.load(f"results/{seedname}-{quantity}-{suffix}_iter-0100.npz")
    data = np.load(f"{seedname}-{quantity}-{suffix}_iter-0000.npz")
    x = data['Energies_0']
    y = data["data"]
    if y.ndim == 2:
        y = y[:, -1]
    axes.plot(x, y, label=f"{quantity}-{suffix}")
    axes.legend()
    if quantity == "dos":
        axes.set_ylim(0, 4)
    return data



quantities = ["dos", "spin", "ahc_internal", "ahc_external"]
nq = len(quantities)
fig, axes = plt.subplots(1, nq, figsize=(5 * nq, 5))
for ax, quantity in zip(axes, quantities):  # "ahc_internal", "ahc_external"]):
    d1 = plot_quantity(quantity, suffix="irr", axes=ax)
    d2 = plot_quantity(quantity, suffix="full", axes=ax)
    diff = np.max(abs(d1["data"] - d2["data"]))
    ax.set_title(f"{quantity} diff: {diff:.3e}")

plt.show()
