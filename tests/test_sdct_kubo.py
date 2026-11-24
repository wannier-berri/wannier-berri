"""
SDCT against the q derivative of the Kubo formula at finite q.

For a tight-binding model with point orbitals (internal terms only), sigma_ab(q, omega) is computed directly from the
Kubo formula with the Peierls coupling, and differentiated numerically in q. The multipole form of SDCT and this
derivative agree as the k mesh gets denser. The model is a chain along a1 (hoppings R = a1, 2 a1, orbitals off the
chain axis), where a dense mesh is cheap, with an insulator and a metal and frequencies across the interband
transitions.
"""

import contextlib
import io
import itertools
import os
import tempfile

import numpy as np
import pytest
from scipy.constants import elementary_charge, hbar

import wannierberri as wberri
from wannierberri.calculators import sdct
from wannierberri.fourier.rvectors import Rvectors
from wannierberri.system.system_R import System_R

LATTICE = np.array([[2.5, 0.0, 0.0], [-1.25, 2.165, 0.0], [0.3, 0.4, 7.0]])  # Angstrom
RECIPROCAL = 2 * np.pi * np.linalg.inv(LATTICE).T
VOLUME = abs(np.linalg.det(LATTICE))
CENTRES = np.array([[0.1, 0.2, 0.05], [0.4, -0.1, 0.3], [-0.25, 0.35, -0.1]])  # fractional
NK = [600, 1, 1]
ETA = 0.2  # broadening (eV): omega + i ETA in both
# bands -9.6 .. -2.9, -1.1 .. 1.3, 2.6 .. 9.4 eV; E_F, kBT (eV) and frequencies (eV) at the transitions
CASES = {"insulator": (-2.0, 0.05, [4.0, 7.5, 13.0]), "metal": (0.2, 0.3, [2.0, 5.0])}


def chain():
    """H(R) (nR, n, n) with random complex hoppings on R = +-a1, +-2 a1."""
    rng = np.random.default_rng(21)
    half = [(1, 0, 0), (2, 0, 0)]
    irvec = np.array([(0, 0, 0)] + half + [tuple(-np.array(R)) for R in half])
    H = np.zeros((len(irvec), 3, 3), dtype=complex)
    X = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    H[0] = X + X.conj().T + np.diag([-4.0, 0.0, 4.0])
    X = 0.6 * (rng.normal(size=(2, 3, 3)) + 1j * rng.normal(size=(2, 3, 3)))
    H[1:3] = X
    H[3:] = X.conj().swapaxes(1, 2)
    return irvec, H


def system(irvec, H, centres):
    s = System_R(force_internal_terms_only=True, silent=True)
    s.real_lattice = LATTICE
    s.num_wann = H.shape[1]
    s.set_R_mat("Ham", H)
    s.set_wannier_centers(wannier_centers_red=centres)
    s.rvec = Rvectors(lattice=LATTICE, iRvec=irvec, shifts_left_red=s.wannier_centers_red)
    s.do_at_end_of_init()
    return s


def run_sdct(s, ef, kBT, omega, nk=NK, eta=ETA, **kwargs):
    """SDCT (all terms but spin) in S/cm per 1/Angstrom, as [omega, a, b, c]."""
    calc = sdct.SDCT(Efermi=np.array([ef]), omega=np.array(omega), kBT=kBT, smr_fixed_width=eta,
                     kwargs_formula=dict(external_terms=False), **kwargs)
    grid = wberri.Grid(s, NKdiv=1, NKFFT=nk)
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        result = wberri.run(s, grid=grid, calculators={"sdct": calc}, parallel=False, use_irred_kpt=False,
                            symmetrize=False, fout_name=os.path.join(tmp, "sdct"))
    return result.results["sdct"].data[0] * 1e8  # S/m per 1/m -> S/cm per 1/Angstrom


def S(x):
    """(e^{ix} - 1) / (ix), the average of e^{i q.r} along a bond."""
    x = np.asarray(x, dtype=complex)
    out = np.ones_like(x)
    m = np.abs(x) > 1e-12
    out[m] = (np.exp(1j * x[m]) - 1) / (1j * x[m])
    return out


def kubo(irvec, H, q, omega, ef, kBT, nk=NK, eta=ETA):
    """
    sigma_ab(q, omega) (S/cm) for the vector potential A e^{i(q.r - omega t)}: the hopping t_ij(d) from orbital j to
    orbital i, at d = r_i - r_j, takes the Peierls phase exp(-i e/hbar int A.dl); the current J(q) = -dH/dA(-q) and
    its retarded correlation with J(-q), plus the diamagnetic term -<d^2 H / dA(-q) dA(q)>, at omega + i eta.
    """
    tau = CENTRES @ LATTICE
    d = tau[None, :, None] - (irvec @ LATTICE)[:, None, None] - tau[None, None, :]  # (R, i, j, 3)

    def M(g, k):
        """sum_d t_ij(d) g_ij(d) e^{-i k.d}."""
        return np.einsum("Rij,Rij,Rij->ij", np.exp(-1j * d @ k), g, H)

    one = np.ones(H.shape)
    f = lambda E: 0.5 * (1 - np.tanh((E - ef) / (2 * kBT)))  # noqa: E731
    kpoints = np.array(list(itertools.product(*[np.arange(n) / n for n in nk]))) @ RECIPROCAL
    K = np.zeros((len(omega), 3, 3), dtype=complex)
    for k in kpoints:
        E0, U0 = np.linalg.eigh(M(one, k))
        E1, U1 = np.linalg.eigh(M(one, k + q))
        X = np.array([-U0.conj().T @ M(-1j * d[..., a] * S(-(d @ q)), k) @ U1 for a in range(3)])
        Y = np.array([U1.conj().T @ M(-1j * d[..., b] * S(d @ q), k + q) @ U0 for b in range(3)])
        D = np.array([[U0.conj().T @ M(-d[..., a] * d[..., b] * S(d @ q) * S(-(d @ q)), k) @ U0 for b in range(3)]
                      for a in range(3)])
        dia = -np.einsum("n,abnn->ab", f(E0), D)
        df = f(E0)[:, None] - f(E1)[None, :]
        for i, w in enumerate(omega):
            K[i] += dia + np.einsum("nm,anm,bmn->ab", df / (w + 1j * eta + E0[:, None] - E1[None, :]), X, Y)
    # j = K A with A in units of hbar / (e Angstrom); E = i omega A
    omega = np.array(omega)
    return (elementary_charge**2 / hbar * 1e10 * K / (len(kpoints) * VOLUME) /
            (1j * (omega[:, None, None] + 1j * eta)) / 100)


def kubo_derivative(irvec, H, omega, ef, kBT):
    """d sigma_ab(q, omega) / d q_c at q = 0 by central differences, as [omega, a, b, c]."""
    h = 1e-4
    ref = np.zeros((len(omega), 3, 3, 3), dtype=complex)
    for c in range(3):
        q = h * np.eye(3)[c]
        ref[..., c] = (kubo(irvec, H, q, omega, ef, kBT) - kubo(irvec, H, -q, omega, ef, kBT)) / (2 * h)
    return ref


def reactive(x):
    """the part Hermitian in a, b."""
    return 0.5 * (x + x.swapaxes(1, 2).conj())


def dissipative(x):
    """the part anti-Hermitian in a, b."""
    return 0.5 * (x - x.swapaxes(1, 2).conj())


@pytest.mark.parametrize("case", CASES)
def test_sdct_against_kubo(case):
    irvec, H = chain()
    ef, kBT, omega = CASES[case]
    ref = kubo_derivative(irvec, H, omega, ef, kBT)

    sigma = run_sdct(system(irvec, H, CENTRES), ef, kBT, omega)

    for part in (reactive, dissipative):
        r, x = part(ref), part(sigma)
        assert np.abs(r).max() > 100
        np.testing.assert_allclose(x, r, rtol=0, atol=1e-3 * np.abs(r).max())


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("degen_thresh", [1e-4, -1])
def test_sdct_degenerate_copies(case, degen_thresh):
    """
    two identical copies of the model (all bands doubly degenerate) give twice the result, with the degenerate bands
    in one group, or in separate groups of the same energy (negative degen_thresh)
    """
    irvec, H = chain()
    ef, kBT, omega = CASES[case]
    H2 = np.array([np.kron(np.eye(2), h) for h in H])
    one = run_sdct(system(irvec, H, CENTRES), ef, kBT, omega, nk=[60, 1, 1])
    two = run_sdct(system(irvec, H2, np.vstack([CENTRES, CENTRES])), ef, kBT, omega, nk=[60, 1, 1],
                   degen_thresh=degen_thresh)
    assert np.abs(one).max() > 0
    np.testing.assert_allclose(two, 2 * one, rtol=0, atol=1e-8 * np.abs(one).max())


def test_sdct_low_temperature():
    """bands far above the Fermi level at a low temperature: finite, the same as at a higher one for an insulator."""
    irvec, H = chain()
    ef, _, omega = CASES["insulator"]
    low = run_sdct(system(irvec, H, CENTRES), ef, 0.002, omega, nk=[60, 1, 1])
    assert np.all(np.isfinite(low))
    high = run_sdct(system(irvec, H, CENTRES), ef, 0.02, omega, nk=[60, 1, 1])
    np.testing.assert_allclose(low, high, rtol=0, atol=1e-10 * np.abs(high).max())
