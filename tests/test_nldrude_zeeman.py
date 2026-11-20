"""Test the orbital Zeeman term of the nonlinear Drude conductivity against a derivative in the magnetic field"""

import os

import numpy as np
import pytest

import wannierberri as wberri
from wannierberri import factors
from wannierberri.calculators import tabulate
from wannierberri.calculators.static import NLDrude_Zeeman_orb
from wannierberri.formula import covariant as frml
from wannierberri.fourier.rvectors import Rvectors
from wannierberri.system.system_R import System_R

from .common import OUTPUT_DIR
from .test_formula import get_datak


def system_QWZ_broken(m0=1.0, a=0.4, b=0.2, external_terms=False):
    """two-band QWZ model plus a sin(kx) + b sin(kx + ky) (identity). Time reversal and inversion are broken, and the
    bands are separated by a gap everywhere, so the energies and the orbital moments are smooth. With
    external_terms=True, random AA, BB and CC matrices are added, so that the orbital moment has all three
    components."""
    sx, sy, sz, I2 = np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1.0, -1.0]), np.eye(2)
    ham = {(0, 0, 0): m0 * sz,
           (1, 0, 0): -0.5j * sx + 0.5 * sz - 0.5j * a * I2, (-1, 0, 0): 0.5j * sx + 0.5 * sz + 0.5j * a * I2,
           (0, 1, 0): -0.5j * sy + 0.5 * sz, (0, -1, 0): 0.5j * sy + 0.5 * sz,
           (1, 1, 0): -0.5j * b * I2, (-1, -1, 0): 0.5j * b * I2}
    system = System_R(force_internal_terms_only=not external_terms, silent=True)
    system.real_lattice = np.eye(3)
    system.num_wann = 2
    system.set_R_mat("Ham", np.array(list(ham.values())))
    system.set_wannier_centers(wannier_centers_red=np.zeros((2, 3)))
    system.rvec = Rvectors(lattice=np.eye(3), iRvec=np.array(list(ham.keys())), shifts_left_red=system.wannier_centers_red)
    if external_terms:
        rng = np.random.default_rng(0)
        shape = (len(ham), 2, 2, 3)
        for key, hermitian in ("AA", True), ("BB", False), ("CC", True):
            system.set_R_mat(key, 0.05 * (rng.normal(size=shape) + 1j * rng.normal(size=shape)), Hermitian=hermitian)
    system.do_at_end_of_init()
    return system


@pytest.mark.parametrize("external_terms", [False, True])
def test_NLDrude_Zeeman_orb_vs_dB(external_terms):
    """The orbital Zeeman field shifts the band energies by E -> E - mu m.B, with mu = fac_orb_Z and the orbital
    moment of the band m = H - E Omega. The response of the Fermi-sea nonlinear Drude conductivity
    factor_nldrude * int f(E) d_a d_b d_c E to this shift is computed with a finite difference in B and with k
    derivatives by FFT (spectrally accurate for these smooth, gapped bands), at kBT = 0.05 eV. NLDrude_Zeeman_orb (a
    Fermi-surface formula, with the sign of -d/dB) is evaluated at T = 0 on a fine grid of Fermi levels and averaged
    with -f'."""
    N, kBT, EF = 60, 0.05, -1.3
    system = system_QWZ_broken(external_terms=external_terms)

    data_k = get_datak(system, k=[0, 0, 0], NKFFT=[N, N, 1])
    k = data_k.kpoints_all
    order = np.lexsort((k[:, 1], k[:, 0]))
    assert np.allclose(k[order, 0].reshape(N, N)[:, 0], np.arange(N) / N)

    def on_grid(X):
        return X[order].reshape((N, N) + X.shape[1:])

    E = on_grid(data_k.E_K)
    morb = on_grid(tabulate.Tabulator(frml.morb)(data_k).data)
    ik = 1j * np.fft.fftfreq(N, 1 / N)
    ik = [ik[:, None] + 0 * ik[None, :], ik[None, :] + 0 * ik[:, None]]  # d/dk_x, d/dk_y (real lattice = identity)

    def d_abc(F, a, b, c):
        return np.real(np.fft.ifft2(np.fft.fft2(F, axes=(0, 1)) * (ik[a] * ik[b] * ik[c])[:, :, None], axes=(0, 1)))

    def fermi_sea(B, mu, a, b, c):
        E_B = E - factors.fac_orb_Z * morb[..., mu] * B
        f = 0.5 * (1 - np.tanh((E_B - EF) / (2 * kBT)))
        return factors.factor_nldrude * np.sum(f * d_abc(E_B, a, b, c)) / (N**2 * data_k.cell_volume)

    dE = 0.005
    efermi = EF + np.arange(-0.6, 0.6 + 1e-9, dE) + 1e-7
    grid = wberri.Grid(system, NKdiv=1, NKFFT=[N, N, 1])
    calculators = {"orb": NLDrude_Zeeman_orb(Efermi=efermi),
                   "orb_1": NLDrude_Zeeman_orb(Efermi=efermi, constant_factor=1.),
                   "orb_nofactor": NLDrude_Zeeman_orb(Efermi=efermi, use_factor=False)}
    results = wberri.run(system, grid=grid, calculators=calculators, parallel=False, use_irred_kpt=False,
                         symmetrize=False, fout_name=os.path.join(OUTPUT_DIR, f"NLDrude_Zeeman_orb_vs_dB_{external_terms}")
                         ).results
    result = results["orb"].data
    factor = factors.fac_orb_Z * factors.factor_nldrude
    np.testing.assert_allclose(results["orb_1"].data * factor, result, rtol=1e-12, atol=0)
    np.testing.assert_allclose(results["orb_nofactor"].data, np.sign(factor) * results["orb_1"].data, rtol=1e-12, atol=0)
    weights = dE / (4 * kBT * np.cosh((efermi - EF) / (2 * kBT))**2)  # -f'
    orb = np.tensordot(weights, result, axes=(0, 0))  # [b, c, a, mu]

    dB = 1e-3 / factors.fac_orb_Z  # mu m B ~ 1 meV
    components = [(mu, a, b, c) for mu in ([0, 1, 2] if external_terms else [2]) for a in range(2)
                  for b, c in [(0, 0), (0, 1), (1, 0), (1, 1)]]
    direct = np.array([-(fermi_sea(dB, *comp) - fermi_sea(-dB, *comp)) / (2 * dB) for comp in components])
    calculated = np.array([orb[b, c, a, mu] for mu, a, b, c in components])
    # each component separately, with a floor for the components that are much smaller than the largest one
    tolerance = 1e-2 * np.abs(direct) + 2e-3 * np.max(np.abs(direct))
    assert np.all(np.abs(calculated - direct) < tolerance), (
        f"NLDrude_Zeeman_orb {calculated} does not match the derivative in B {direct} for (mu, a, b, c) in {components}")
