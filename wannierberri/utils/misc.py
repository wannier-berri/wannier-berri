"""
Here I will put some utility functions that are not directly related to the WannierBerri code, but may be useful (at least to myself)
please use on your own risk, and report if you find any bugs
"""
from matplotlib import pyplot as plt
import numpy as np



def plot_data_BZ(data, basis, vmax, axes, to_BZ=True):
    """
    Plot data, given on a regular grid in the reciprocal space, 
    in the first Brillouin zone of the reciprocal lattice.
    
    Parameters
    ----------
    data : 2D array
        The data to plot.
    basis : 2D array (2x2)
        The basis vectors of the reciprocal lattice.
    vmax : float
        The maximum value for the colormap.
    axes : matplotlib.axes.Axes
        The axes to plot on.
    to_BZ : bool
        If True, the data will be folded back into the first Brillouin zone. If False, the data will be plotted as is.
    """

    g1 = basis[0, :2]
    g2 = basis[1, :2]
    k1 = np.linspace(0, 1, data.shape[0], endpoint=False)
    k2 = np.linspace(0, 1, data.shape[1], endpoint=False)
    K1, K2 = np.meshgrid(k1, k2)
    KX = K1 * g1[0] + K2 * g2[0]
    KY = K1 * g1[1] + K2 * g2[1]

    if to_BZ:
        for i in range(10):
            finish = True
            for direction in [(-1, 1), (0, 1), (1, 0)]:
                direction_cart = basis[0, :2] * direction[0] + basis[1, :2] * direction[1]
                direction_cart_norm = direction_cart / np.linalg.norm(direction_cart) ** 2
                proj = KX * direction_cart_norm[0] + KY * direction_cart_norm[1]
                selection = proj > 0.5
                if np.any(selection):
                    finish = False
                KX[selection] -= direction_cart[0]
                KY[selection] -= direction_cart[1]
                selection = proj < -0.5
                if np.any(selection):
                    finish = False
                KX[selection] += direction_cart[0]
                KY[selection] += direction_cart[1]
            if finish:
                break

    # Flatten the arrays for use with tricontourf or tripcolor
    KX_flat = KX.flatten()
    KY_flat = KY.flatten()
    jdos0_flat = data.flatten()

    # plt.figure(figsize=(8, 6))
    tricontour = axes.tricontourf(KX_flat, KY_flat, jdos0_flat, levels=100, cmap='viridis', vmin=0, vmax=vmax)
    plt.colorbar(tricontour,label='JDOS')
    axes.set_xlabel('KX')
    axes.set_ylabel('KY')
    axes.set_title('Colormap of JDOS')


import time
from joblib import Parallel, delayed

def JDOSq(Enk, grid, Emin, Emax, nE=100, smear=0.1, use_FFT=True, use_smoother=False):
    """
    Calculate the joint density of states 
        JDOS(E, q) = 1/Nk \sum_k \sum_n \delta(E - E_n(k)) \delta(E-E_n(k+q))

    Parameters:
    -----------
    Enk : np.array(float)
        The band energies in eV. Shape (NK, NB)
    grid : tuple(int)
        The grid size, e.g. (100,100), (100,100,100)
    Emin : float
        The minimum energy to calcualte
    Emax : float
        The maximum energy to calculate
    nE : int
        The number of energy points to calculate
    smear : float
        The smearing parameter (in eV)
    use_smoother:
        If True, use the Smoother class is used, to speed up the calculation
    use_FFT:
        If True, use the FFT to speed up the calculation. If False, use the brute force method (very slow, just for testing on smmall grids)

    Returns:
    --------
    E : np.array(float)
        The energy points in eV
    DOS : np.array(float)
        The density of states. Shape  (*grid,nE)
    JDOS : np.array(float)
        The joint density of states. Shape (*grid,nE)   
    """
    nd = len(grid)
    grid = tuple(grid) + (1,)*(3-nd)
    E = np.linspace(Emin, Emax, nE)
    nk = Enk.shape[0]
    def broaden(_E):
        return np.exp(-(_E / smear) ** 2) / smear / np.sqrt(np.pi)
    print ("calculating DOS")
    DOS = []
    if use_smoother:
        nE_loc = 
        for En in Enk:
            dos = np.zeros(nE)
            for e in En:
                dos += broaden(e-E)
    else:
        for En in Enk:
            dos = np.zeros(nE)
            for e in En:
                dos += broaden(e-E)
            DOS.append(dos)
        DOS = np.array(DOS)
    DOS = np.reshape(DOS, grid + (nE,))
    DOS_roll = DOS.copy()
    JDOS = np.zeros(grid + (nE,))
    print ("calculating JDOS")
    t0 = time.time()
    if use_FFT:
        DOS_r = np.fft.fftn(DOS, axes=(0,1,2))
        DOS_mr = np.fft.ifftn(DOS, axes=(0,1,2))
        DOS_rr = DOS_r * DOS_mr
        JDOS = np.fft.ifftn(DOS_rr, axes=(0,1,2))
        JDOS_real = np.real(JDOS)
        JDOS_imag = np.imag(JDOS)
        print (f"minimum of JDOS_real = {JDOS_real.min()}")
        print (f"maximum of JDOS_real = {JDOS_real.max()}")
        print (f"maximum abs of JDOS_imag = {np.abs(JDOS_imag).max()}")
        JDOS = np.real(JDOS)
        JDOS[JDOS < 0] = 0
    
    else:
        for k1 in range(grid[0]):
            print (f" k1 = {k1} of {grid[0]}")
            DOS_roll = np.roll(DOS_roll, 1, axis=0)
            for k2 in range(grid[1]):
                DOS_roll = np.roll(DOS_roll, 1, axis=1)
                for k3 in range(grid[2]):
                    DOS_roll = np.roll(DOS_roll, 1, axis=2)
                    JDOS[k1, k2, k3] = np.einsum('ijkl,ijkl->l', DOS, DOS_roll)
            print (f"time per k1 = {time.time()-t0}")
            t0 = time.time()
        JDOS /= nk
    
    # Reshape the results back into the JDOS array
    JDOS = JDOS.reshape(grid + (nE,))
    
    DOS = np.reshape(DOS, grid[:nd] + (nE,))
    JDOS = np.reshape(JDOS, (grid[:nd]) + (nE,))
    return E, DOS, JDOS
