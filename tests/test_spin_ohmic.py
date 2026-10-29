"""Tests of the spin conductivity without time-reversal symmetry: SHC (even part) and SpinOhmic_FermiSurf (odd part).

Model: Rashba square lattice with an exchange field M, H = t (cos kx + cos ky) + alpha (sin ky sx - sin kx sy) + M.s,
two spin orbitals at the origin. The reference is the Kubo formula for the spin current j = {s, v}/2, with all
matrix elements, written out k-point by k-point.
"""

import contextlib
import copy
import io
import os
import tempfile

import numpy as np
import pytest
from scipy.constants import elementary_charge, hbar, angstrom

import wannierberri as wberri
from wannierberri import factors
from wannierberri.calculators.static import SHC, Ohmic_FermiSurf, SpinOhmic_FermiSurf
from wannierberri.fourier.rvectors import Rvectors
from wannierberri.system import System_R

S0 = np.eye(2, dtype=complex)
SIGMA = [np.array([[0, 1], [1, 0]], dtype=complex), np.array([[0, -1j], [1j, 0]]), np.diag([1., -1]).astype(complex)]
T, ALPHA, C = -1.0, 0.6, 10.0  # hopping and Rashba coupling (eV), interlayer distance (Angstrom)
MU, KBT, N = -0.5, 0.05, 40
KWARGS_FORMULA = {"spin_current_type": "simple"}


def system(M):
    sx, sy, _ = SIGMA
    hoppings = {(0, 0, 0): sum(m * s for m, s in zip(M, SIGMA)),
                (1, 0, 0): T / 2 * S0 - ALPHA * sy / 2j, (-1, 0, 0): T / 2 * S0 + ALPHA * sy / 2j,
                (0, 1, 0): T / 2 * S0 + ALPHA * sx / 2j, (0, -1, 0): T / 2 * S0 - ALPHA * sx / 2j}
    iRvec = list(hoppings)
    s = System_R(force_internal_terms_only=True, silent=True)
    s.real_lattice = np.diag([1., 1., C])
    s.num_wann = 2
    s.set_R_mat("Ham", np.array([hoppings[R] for R in iRvec]))
    SS = np.zeros((len(iRvec), 2, 2, 3), dtype=complex)
    for a in range(3):
        SS[0, :, :, a] = SIGMA[a]
    s.set_R_mat("SS", SS)
    s.set_wannier_centers(wannier_centers_red=np.zeros((2, 3)))
    s.rvec = Rvectors(lattice=s.real_lattice, iRvec=np.array(iRvec), shifts_left_red=s.wannier_centers_red)
    s.do_at_end_of_init()
    return s


def kubo(M, eta=None):
    """even (interband) and odd (intraband, times eta) parts of the spin conductivity [a, b, gamma] in S/m at
    temperature KBT, and the full Kubo formula with broadening eta (eV) if given."""
    sx, sy, _ = SIGMA
    k = 2 * np.pi * np.arange(N) / N
    kx, ky = [x.ravel() for x in np.meshgrid(k, k, indexing="ij")]
    H = ((T * (np.cos(kx) + np.cos(ky)))[:, None, None] * S0 +
         ALPHA * (np.sin(ky)[:, None, None] * sx - np.sin(kx)[:, None, None] * sy) + sum(m * s for m, s in zip(M, SIGMA)))
    vx = (-T * np.sin(kx))[:, None, None] * S0 - ALPHA * np.cos(kx)[:, None, None] * sy
    vy = (-T * np.sin(ky))[:, None, None] * S0 + ALPHA * np.cos(ky)[:, None, None] * sx
    V = [vx, vy, np.zeros_like(vx)]
    E, U = np.linalg.eigh(H)

    def rotate(X):
        return U.conj().transpose(0, 2, 1) @ X @ U

    f = 0.5 * (1 - np.tanh((E - MU) / (2 * KBT)))
    df = -f * (1 - f) / KBT
    dE = E[:, :, None] - E[:, None, :]
    intra = np.abs(dE) < 1e-8
    # (f_n - f_m) / (E_n - E_m), and f'(E_n) for n = m
    F = np.where(intra, df[:, :, None] + 0 * dE, (f[:, :, None] - f[:, None, :]) / np.where(intra, 1, dE))
    # the spin current of electrons (charge -e) has the opposite sign of the charge current with v replaced by j
    factor = -elementary_charge ** 2 / hbar / angstrom / 2 / (N * N * C)
    even, odd, full = (np.zeros((3, 3, 3)) for _ in range(3))
    for g in range(3):
        for a in range(3):
            j = rotate(0.5 * (SIGMA[g] @ V[a] + V[a] @ SIGMA[g]))
            for b in range(3):
                X = j * rotate(V[b]).transpose(0, 2, 1)  # j_nm v_mn
                even[a, b, g] = factor * np.where(intra, 0, F * X.imag / np.where(intra, 1, dE)).sum()
                odd[a, b, g] = factor * np.where(intra, -F * X.real, 0).sum()
                if eta is not None:
                    full[a, b, g] = factor * (-1j * F * X / (dE + 1j * eta)).sum().real
    return even, odd, full


def run(s, use_symmetry=False, Efermi=None):
    if Efermi is None:
        Efermi = MU + np.arange(-1, 1 + 1e-9, 1e-3) + 1e-7
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        r = wberri.run(s, grid=wberri.Grid(s, NKFFT=[N, N, 1], NKdiv=1), parallel=False,
                       use_irred_kpt=use_symmetry, symmetrize=use_symmetry,
                       calculators={"even": SHC(Efermi=Efermi, kwargs_formula=KWARGS_FORMULA),
                                    "odd": SpinOhmic_FermiSurf(Efermi=Efermi, kwargs_formula=KWARGS_FORMULA)},
                       fout_name=os.path.join(tmp, "x"))
    return Efermi, {key: r.results[key].data for key in ("even", "odd")}


def at_temperature(Efermi, data):
    """from T = 0 on a fine grid of Fermi levels to the temperature KBT at MU"""
    weight = 0.25 / KBT / np.cosh((Efermi - MU) / (2 * KBT)) ** 2 * (Efermi[1] - Efermi[0])
    return np.tensordot(weight, data, axes=(0, 0))


ETA_TAU = hbar / elementary_charge / factors.TAU_UNIT  # the broadening (eV) for the unit of the relaxation time


@pytest.mark.parametrize("M", [(0.3, 0.2, 0.4), (0, 0, 0.4)])
def test_against_kubo(M):
    Efermi, r = run(system(M))
    even, odd = (at_temperature(Efermi, r[key]) for key in ("even", "odd"))
    ref_even, ref_odd, _ = kubo(M)
    np.testing.assert_allclose(even, ref_even, rtol=0, atol=1e-3 * np.abs(ref_even).max())
    np.testing.assert_allclose(odd * ETA_TAU, ref_odd, rtol=0, atol=2e-3 * np.abs(ref_odd).max())
    assert np.abs(ref_odd).max() > 0.1 * np.abs(ref_even).max()  # the odd part is not negligible


def test_sum_is_the_kubo_formula():
    """at a finite broadening eta, the Kubo formula is the sum of the two parts with tau = hbar / eta, up to O(eta)"""
    M, eta = (0.3, 0.2, 0.4), 0.01
    Efermi, r = run(system(M))
    even, odd = (at_temperature(Efermi, r[key]) for key in ("even", "odd"))
    full = kubo(M, eta)[2]
    assert np.abs(full - even).max() > 0.5 * np.abs(full).max()
    np.testing.assert_allclose(even + odd * ETA_TAU / eta, full, rtol=0, atol=2e-3 * np.abs(full).max())


def test_odd_part_vanishes_with_time_reversal():
    Efermi, r = run(system((0, 0, 0)))
    assert np.abs(r["even"]).max() > 1e3
    assert np.abs(r["odd"]).max() < 1e-8 * np.abs(r["even"]).max()


@pytest.mark.parametrize("generators", [["C4z", "TimeReversal*Mx"], ["C4z", "TimeReversal*C2x"]])
def test_symmetrization(generators):
    """symmetrization with magnetic symmetries (time reversal times a rotation) gives the same result as the full
    grid only if the odd part is treated as odd under time reversal"""
    M = (0, 0, 0.4)
    Efermi = np.linspace(-1.0, 0.0, 101)
    full = run(system(M), Efermi=Efermi)[1]
    s = system(M)
    s.set_pointgroup(generators)
    sym = run(s, use_symmetry=True, Efermi=Efermi)[1]
    for key in full:
        np.testing.assert_allclose(sym[key], full[key], rtol=0, atol=1e-10 * np.abs(full[key]).max())


def run_Fe(system, use_symmetry, calculators):
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        r = wberri.run(system, grid=wberri.Grid(system, NKFFT=6, NKdiv=2), parallel=False,
                       use_irred_kpt=use_symmetry, symmetrize=use_symmetry, calculators=calculators,
                       fout_name=os.path.join(tmp, "x"))
    return {key: r.results[key].data for key in calculators}


EFERMI_FE = np.linspace(17, 18, 11)


@pytest.mark.parametrize("spin_current_type", ["ryoo", "qiao", "simple"])
def test_symmetrization_Fe(system_Fe_sym_W90, spin_current_type):
    """symmetrization with the magnetic symmetry of Fe (time reversal times C2x), for each type of spin current
    and with the external terms, on a symmetrized model"""
    calculators = {"odd": SpinOhmic_FermiSurf(Efermi=EFERMI_FE, kwargs_formula={"spin_current_type": spin_current_type})}
    full = run_Fe(system_Fe_sym_W90, False, calculators)["odd"]
    sym = run_Fe(system_Fe_sym_W90, True, calculators)["odd"]
    assert np.abs(full).max() > 1e4
    np.testing.assert_allclose(sym, full, rtol=0, atol=1e-10 * np.abs(full).max())


def test_identity_spin_is_ohmic(system_Fe_W90):
    """with the spin operator replaced by the identity, the spin current is the velocity, so the result is
    -1/2 of the ohmic conductivity for every spin component (here with the external terms)"""
    system = copy.deepcopy(system_Fe_W90)
    SS = np.zeros_like(system.get_R_mat("SS"))
    SS[system.rvec.iR0] = np.eye(system.num_wann)[:, :, None]
    system.set_R_mat("SS", SS, reset=True)
    r = run_Fe(system, False, {"spin": SpinOhmic_FermiSurf(Efermi=EFERMI_FE, kwargs_formula={"spin_current_type": "simple"}),
                               "ohmic": Ohmic_FermiSurf(Efermi=EFERMI_FE)})
    for g in range(3):
        np.testing.assert_allclose(r["spin"][..., g], -r["ohmic"] / 2, rtol=0, atol=1e-10 * np.abs(r["ohmic"]).max())
