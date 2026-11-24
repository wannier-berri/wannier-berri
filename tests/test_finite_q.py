"""
Tests of OpticalConductivity_q, the optical conductivity at a finite wave vector in the tight-binding approximation.

The reference is the Kubo formula with the Peierls coupling written out directly, k point by k point, for a small
three-dimensional model with orbitals away from the lattice points. It checks the gauge of the Bloch states, the
Fourier transforms and the units, but it shares the Peierls vertices with the code. Those are checked independently by
gauge invariance (charge conservation), the reality of the response and Onsager reciprocity, for q on the k grid.
"""

import contextlib
import io
import itertools
import os
import tempfile
import types

import numpy as np
import pytest
from scipy.constants import elementary_charge, hbar, angstrom

import wannierberri as wberri
from wannierberri.calculators.dynamic import OpticalConductivity
from wannierberri.calculators.finite_q import OpticalConductivity_q
from wannierberri.data_K import Data_K_R
from wannierberri.fourier.rvectors import Rvectors
from wannierberri.parallel import check_ray_initialized
from wannierberri.system.system_R import System_R

LATTICE = np.array([[2.5, 0.0, 0.0], [-1.25, 2.165, 0.0], [0.3, 0.4, 4.0]])  # Angstrom
RECIPROCAL = 2 * np.pi * np.linalg.inv(LATTICE).T
CENTRES = np.array([[0.1, 0.2, 0.05], [0.4, -0.1, 0.3], [-0.25, 0.35, -0.1]])  # fractional
NK = [6, 5, 4]
ETA = 0.2  # eV
# bands -11.0 .. -4.7, -3.9 .. 3.0, 3.3 .. 10.7 eV; E_F, kBT (eV) and frequencies (eV)
CASES = {"insulator": (-4.3, 0.02, [3.0, 7.0]), "metal": (0.3, 0.1, [0.5, 4.0])}
Q = {"zero": [0, 0, 0], "x": [0.3, 0, 0], "general": [0.2, -0.15, 0.25]}  # 1/Angstrom
Q_GRID = RECIPROCAL[0] / 6 + 2 * RECIPROCAL[1] / 5 - RECIPROCAL[2] / 4  # on the k grid NK


def model():
    """H(R) (nR, n, n) with random complex hoppings, including hoppings to R = 0 between different orbitals."""
    rng = np.random.default_rng(7)
    half = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, -1, 1)]
    irvec = np.array([(0, 0, 0)] + half + [tuple(-np.array(R)) for R in half])
    H = np.zeros((len(irvec), 3, 3), dtype=complex)
    X = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    H[0] = 0.5 * (X + X.conj().T) + np.diag([-7.5, 0.0, 7.5])
    X = 0.25 * (rng.normal(size=(len(half), 3, 3)) + 1j * rng.normal(size=(len(half), 3, 3)))
    H[1:len(half) + 1] = X
    H[len(half) + 1:] = X.conj().swapaxes(1, 2)
    return irvec, H


def system(irvec, H, centres=CENTRES, periodic=(True, True, True)):
    s = System_R(force_internal_terms_only=True, silent=True, periodic=periodic)
    s.real_lattice = LATTICE
    s.num_wann = H.shape[1]
    s.set_R_mat("Ham", H)
    s.set_wannier_centers(wannier_centers_red=centres)
    s.rvec = Rvectors(lattice=LATTICE, iRvec=irvec, shifts_left_red=s.wannier_centers_red)
    s.do_at_end_of_init()
    return s


def run(s, calculators, nk=NK, NKdiv=1, grid=None, **kwargs):
    if grid is None:
        grid = wberri.Grid(s, NKdiv=NKdiv, NKFFT=nk)
    kwargs = dict(dict(use_irred_kpt=False, symmetrize=False, parallel=False), **kwargs)
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        result = wberri.run(s, grid=grid, calculators=calculators, fout_name=os.path.join(tmp, "x"), **kwargs)
    return {key: val.data for key, val in result.results.items()}


def bond_average(x):
    x = np.asarray(x, dtype=complex)
    out = np.ones_like(x)
    m = np.abs(x) > 1e-12
    out[m] = (np.exp(1j * x[m]) - 1) / (1j * x[m])
    return out


def kubo(irvec, H, q, omega, ef, kBT, nk=NK, eta=ETA):
    """
    sigma_ab(q, omega) (S/m) for the vector potential A e^{i(q.r - omega t)}: the hopping t_ij(d) from orbital j to
    orbital i, at d = r_i - r_j, takes the Peierls phase exp(-i e/hbar int A.dl) along the bond; the current
    J(q) = -dH/dA(-q) and its retarded correlation with J(-q), plus the diamagnetic term -<d^2 H / dA(-q) dA(q)>, at
    omega + i eta. Bloch states sum_R e^{ik.(R + tau)} |R + tau>.
    """
    q = np.array(q, dtype=float)
    tau = CENTRES @ LATTICE
    d = tau[None, :, None] - (irvec @ LATTICE)[:, None, None] - tau[None, None, :]  # (R, i, j, 3)

    def M(g, k):
        """sum_d t_ij(d) g_ij(d) e^{-i k.d}."""
        return np.einsum("Rij,Rij,Rij->ij", np.exp(-1j * d @ k), g, H)

    one = np.ones(H.shape)
    f = lambda E: 0.5 * (1 - np.tanh((E - ef) / (2 * kBT))) if kBT > 0 else 1.0 * (E <= ef)  # noqa: E731
    kpoints = np.array(list(itertools.product(*[np.arange(n) / n for n in nk]))) @ RECIPROCAL
    K = np.zeros((len(omega), 3, 3), dtype=complex)
    for k in kpoints:
        E0, U0 = np.linalg.eigh(M(one, k))
        E1, U1 = np.linalg.eigh(M(one, k + q))
        X = np.array([-U0.conj().T @ M(-1j * d[..., a] * bond_average(-(d @ q)), k) @ U1 for a in range(3)])
        Y = np.array([U1.conj().T @ M(-1j * d[..., b] * bond_average(d @ q), k + q) @ U0 for b in range(3)])
        D = np.array([[U0.conj().T @ M(-d[..., a] * d[..., b] * abs(bond_average(d @ q)) ** 2, k) @ U0
                       for b in range(3)] for a in range(3)])
        dia = -np.einsum("n,abnn->ab", f(E0), D)
        df = f(E0)[:, None] - f(E1)[None, :]
        for i, w in enumerate(omega):
            K[i] += dia + np.einsum("nm,anm,bmn->ab", df / (w + 1j * eta + E0[:, None] - E1[None, :]), X, Y)
    omega = np.array(omega)
    volume = abs(np.linalg.det(LATTICE))
    # j = K A with A in units of hbar / (e Angstrom); E = i omega A
    return (elementary_charge ** 2 / hbar / angstrom * K / (len(kpoints) * volume) /
            (1j * (omega[:, None, None] + 1j * eta)))


def calculator(q, case, **kwargs):
    """the calculator for a case; q may be off the k grid unless allow_q_off_grid=False is given."""
    ef, kBT, omega = CASES[case]
    kwargs = dict(dict(allow_q_off_grid=True), **kwargs)
    return OpticalConductivity_q(q=q, Efermi=[ef], omega=omega, kBT=kBT, smr_fixed_width=ETA, **kwargs)


@pytest.mark.parametrize("q", Q)
@pytest.mark.parametrize("case", CASES)
def test_against_kubo(case, q):
    irvec, H = model()
    ef, kBT, omega = CASES[case]
    ref = kubo(irvec, H, Q[q], omega, ef, kBT)
    sigma = run(system(irvec, H), {"q": calculator(Q[q], case)})["q"][0]
    assert np.abs(ref).max() > 1e4
    np.testing.assert_allclose(sigma, ref, rtol=0, atol=1e-10 * np.abs(ref).max())


def test_zero_temperature_and_several_fermi_levels():
    irvec, H = model()
    omega = [0.5, 3.0]
    efermi = [CASES["insulator"][0], CASES["metal"][0]]
    calc = OpticalConductivity_q(q=Q["general"], Efermi=efermi, omega=omega, smr_fixed_width=ETA, allow_q_off_grid=True)
    sigma = run(system(irvec, H), {"q": calc})["q"]
    for i, ef in enumerate(efermi):
        ref = kubo(irvec, H, Q["general"], omega, ef, 0)
        np.testing.assert_allclose(sigma[i], ref, rtol=0, atol=1e-10 * np.abs(ref).max())


def test_derivative_at_small_q():
    """
    numerical stability: the q derivative at q = 0 by central differences is the same with a step of 1e-8 and 1e-4
    (1/Angstrom).
    """
    irvec, H = model()
    s = system(irvec, H)
    q = np.array(Q["general"]) / np.linalg.norm(Q["general"])

    def derivative(h):
        res = run(s, {"+": calculator(h * q, "metal"), "-": calculator(-h * q, "metal")})
        return (res["+"] - res["-"]) / (2 * h)

    small, large = derivative(1e-8), derivative(1e-4)
    assert np.abs(large).max() > 1e3
    np.testing.assert_allclose(small, large, rtol=0, atol=1e-5 * np.abs(large).max())


@pytest.mark.parametrize("ef, kBT", [(CASES["insulator"][0], 0), (CASES["metal"][0], 0.1)])
def test_gauge_invariance(ef, kBT):
    """
    charge conservation: the static response to a pure gauge A = q chi e^{iq.r} vanishes, K_ab(q, 0) q_b = 0 and
    q_a K_ab(q, 0) = 0, with K = -i (omega + i eta) sigma, up to O(eta). It tests the paramagnetic and diamagnetic
    vertices and their relative sign, independently of the reference above.
    """
    irvec, H = model()
    eta = 1e-6
    sigma = run(system(irvec, H), {"q": OpticalConductivity_q(q=Q_GRID, Efermi=[ef], omega=[0.0], kBT=kBT,
                                                              smr_fixed_width=eta)})["q"][0, 0]
    K = -eta * sigma
    scale = np.abs(K).max() * np.linalg.norm(Q_GRID)
    assert np.abs(K).max() > 0
    assert np.abs(K @ Q_GRID).max() < 1e-5 * scale
    assert np.abs(Q_GRID @ K).max() < 1e-5 * scale


@pytest.mark.parametrize("case", CASES)
def test_reality(case):
    """the response to a real field is real: sigma_ab(q, omega + i eta)^* = sigma_ab(-q, -omega + i eta)."""
    irvec, H = model()
    ef, kBT, omega = CASES[case]
    s = system(irvec, H)
    plus = run(s, {"q": OpticalConductivity_q(q=Q_GRID, Efermi=[ef], omega=omega, kBT=kBT,
                                              smr_fixed_width=ETA)})["q"]
    minus = run(s, {"q": OpticalConductivity_q(q=-Q_GRID, Efermi=[ef], omega=-np.array(omega), kBT=kBT,
                                               smr_fixed_width=ETA)})["q"]
    np.testing.assert_allclose(plus.conj(), minus, rtol=0, atol=1e-10 * np.abs(plus).max())


@pytest.mark.parametrize("case", CASES)
def test_onsager_reciprocity(case):
    """with time-reversal symmetry (real hoppings), sigma_ab(q, omega) = sigma_ba(-q, omega), and differs from
    sigma_ab(-q, omega)."""
    irvec, H = model()
    s = system(irvec, H.real)
    plus = run(s, {"q": calculator(Q_GRID, case, allow_q_off_grid=False)})["q"]
    minus = run(s, {"q": calculator(-Q_GRID, case, allow_q_off_grid=False)})["q"]
    np.testing.assert_allclose(plus, minus.swapaxes(-1, -2), rtol=0, atol=1e-10 * np.abs(plus).max())
    assert np.abs(plus - minus).max() > 1e-3 * np.abs(plus).max()


@pytest.mark.parametrize("case, kBT, sign, nk, NKdiv, q", [
    ("insulator", None, 1, NK, 1, Q_GRID),
    ("insulator", None, -1, NK, 1, Q_GRID),
    ("insulator", 0, 1, NK, 1, Q_GRID),
    ("metal", None, 1, NK, 1, Q_GRID),
    ("metal", None, -1, NK, 1, Q_GRID),
    ("metal", None, 1, [3, 5, 4], [2, 1, 1], RECIPROCAL[0] / 6),  # q on the grid NKFFT * NKdiv, not on NKFFT
])
def test_continuity(case, kBT, sign, nk, NKdiv, q):
    """
    charge conservation at finite frequency, independently of the Peierls vertices: q_a K_ab(q, z) q_b
    = -z^2 chi(q, z) with z = omega + i eta, K = i z sigma (in units of e^2 / (hbar Angstrom V)) and the density
    response chi from the density vertex <u_n(k)| e^{-iq.tau} |u_m(k+q)>.
    """
    irvec, H = model()
    ef, kBT_case, omega = CASES[case]
    kBT = kBT_case if kBT is None else kBT
    q = sign * q
    z = np.array(omega) + 1j * ETA
    calc = OpticalConductivity_q(q=q, Efermi=[ef], omega=omega, kBT=kBT, smr_fixed_width=ETA)
    sigma = run(system(irvec, H), {"q": calc}, nk=nk, NKdiv=NKdiv)["q"][0]
    K = 1j * z[:, None, None] * sigma / (elementary_charge ** 2 / hbar / angstrom / abs(np.linalg.det(LATTICE)))
    tau = CENTRES @ LATTICE
    f = lambda E: 0.5 * (1 - np.tanh((E - ef) / (2 * kBT))) if kBT > 0 else 1.0 * (E <= ef)  # noqa: E731
    dense = np.array(nk) * NKdiv
    kpoints = np.array(list(itertools.product(*[np.arange(n) / n for n in dense]))) @ RECIPROCAL
    chi = np.zeros(len(z), dtype=complex)
    for k in kpoints:
        E0, U0 = np.linalg.eigh(np.einsum("R,Rij->ij", np.exp(1j * (irvec @ LATTICE) @ k), H))
        E1, U1 = np.linalg.eigh(np.einsum("R,Rij->ij", np.exp(1j * (irvec @ LATTICE) @ (k + q)), H))
        rho = U0.conj().T @ np.diag(np.exp(-1j * tau @ q)) @ U1
        df = f(E0)[:, None] - f(E1)[None, :]
        chi += [np.sum(df / (w + E0[:, None] - E1[None, :]) * np.abs(rho) ** 2) for w in z]
    chi /= len(kpoints)
    qKq = np.einsum("a,wab,b->w", q, K, q)
    np.testing.assert_allclose(qKq, -z ** 2 * chi, rtol=0, atol=1e-10 * np.abs(qKq).max())


def test_q_off_the_grid_is_refused():
    irvec, H = model()
    s = system(irvec, H)
    ef, kBT, omega = CASES["metal"]
    with pytest.raises(ValueError, match="not on the k grid"):  # by default
        run(s, {"q": OpticalConductivity_q(q=Q["general"], Efermi=[ef], omega=omega, kBT=kBT)})
    calc = calculator(Q_GRID, "metal", allow_q_off_grid=False)
    run(s, {"q": calc})
    with pytest.raises(ValueError, match="not on the k grid"):  # the same calculator on another grid
        run(s, {"q": calc}, nk=[5, 5, 4])
    with pytest.raises(ValueError, match="adaptive refinement"):
        run(s, {"q": calc}, adpt_num_iter=1)
    with pytest.raises(ValueError, match="regular grid"):
        run(s, {"q": calc}, grid=wberri.grid.GridTetra(s, length=5, NKFFT=NK))
    # on the grid NKFFT * NKdiv, but not on NKFFT
    calc = calculator(RECIPROCAL[0] / 6, "metal", allow_q_off_grid=False)
    run(s, {"q": calc}, nk=[3, 5, 4], NKdiv=[2, 1, 1])
    with pytest.raises(ValueError, match="not on the k grid"):
        run(s, {"q": calc}, nk=[3, 5, 4])
    # a large q close to the grid, but not on it
    with pytest.raises(ValueError, match="not on the k grid"):
        run(s, {"q": calculator(50 * RECIPROCAL[0] + 3e-4 * RECIPROCAL[0] / 6, "metal", allow_q_off_grid=False)})


def test_two_dimensional_system():
    """q_z is free along the non-periodic direction: the result is still gauge invariant."""
    irvec, H = model()
    planar = irvec[:, 2] == 0
    s = system(irvec[planar], H[planar], periodic=(True, True, False))
    q = RECIPROCAL[0] / 6 + 2 * RECIPROCAL[1] / 5 + [0, 0, 0.37]
    eta = 1e-6
    calc = OpticalConductivity_q(q=q, Efermi=[CASES["metal"][0]], omega=[0.0], kBT=0.1, smr_fixed_width=eta)
    K = -eta * run(s, {"q": calc}, nk=[6, 5, 1])["q"][0, 0]
    assert np.abs(K).max() > 0
    assert np.abs(K @ q).max() < 1e-5 * np.abs(K).max() * np.linalg.norm(q)
    # hoppings along z, with periodic set to False afterwards: q_z is not free
    s = system(irvec, H)
    s.periodic = [True, True, False]
    with pytest.raises(ValueError, match="not on the k grid"):
        run(s, {"q": calc}, nk=[6, 5, 1])


def test_parallel():
    """with ray workers, the same result as in serial, and the error for q off the grid reaches the caller."""
    if not check_ray_initialized():
        pytest.skip("ray is not initialized")
    irvec, H = model()
    s = system(irvec, H)
    serial = run(s, {"q": calculator(Q_GRID, "metal")}, NKdiv=[2, 1, 1])["q"]
    parallel = run(s, {"q": calculator(Q_GRID, "metal")}, NKdiv=[2, 1, 1], parallel=True)["q"]
    np.testing.assert_allclose(parallel, serial, rtol=0, atol=1e-12 * np.abs(serial).max())
    with pytest.raises(ValueError, match="not on the k grid"):  # ray raises the error of a worker as a ValueError
        run(s, {"q": calculator(Q["general"], "metal", allow_q_off_grid=False)}, NKdiv=[2, 1, 1], parallel=True)


def test_kp_systems_are_refused():
    s = wberri.system.SystemKP(Ham=lambda k: np.array([[np.dot(k, k)]], dtype=complex), kmax=1.)
    with pytest.raises(ValueError, match="real-space matrix elements"):
        run(s, {"q": calculator(Q["zero"], "metal")})


def test_phonons_are_refused():
    data_K = Data_K_R.__new__(Data_K_R)  # only its system is looked at before the error
    data_K.system = types.SimpleNamespace(is_phonon=True)
    with pytest.raises(ValueError, match="phonons"):
        calculator(Q["zero"], "metal")(data_K)


def test_zero_q_is_optical_conductivity():
    """
    at q = 0, an insulator: the same as OpticalConductivity with the internal terms. The current (velocity) form
    used here and the position form of OpticalConductivity agree when the sum over the grid of the second derivative
    of the occupied energies vanishes, which it does exponentially with the density of the grid.
    """
    irvec, H = model()
    ef, kBT, omega = CASES["insulator"]
    result = run(system(irvec, H), {
        "q": calculator([0, 0, 0], "insulator"),
        "opt": OpticalConductivity(Efermi=[ef], omega=np.array(omega), kBT=kBT, smr_fixed_width=ETA,
                                   kwargs_formula=dict(external_terms=False))}, nk=[16, 16, 12])
    assert np.abs(result["opt"]).max() > 1e4
    np.testing.assert_allclose(result["q"], result["opt"], rtol=0, atol=1e-6 * np.abs(result["opt"]).max())


def test_zero_q_is_optical_conductivity_GaAs(system_GaAs_W90):
    """
    the same for GaAs from Wannier90 (E_F in the gap), where the Wannier centres are the shifts of the R vectors. The
    difference is 4e-2 on 8^3, 5e-3 on 12^3 and 1e-3 on 16^3.
    """
    kwargs = dict(Efermi=np.array([7.9]), omega=np.array([2.0, 4.0]), kBT=0.01, smr_fixed_width=0.2)
    result = run(system_GaAs_W90, {
        "q": OpticalConductivity_q(**kwargs),
        "opt": OpticalConductivity(**kwargs, kwargs_formula=dict(external_terms=False))}, nk=[16, 16, 16])
    assert np.abs(result["opt"]).max() > 1e4
    np.testing.assert_allclose(result["q"], result["opt"], rtol=0, atol=2e-3 * np.abs(result["opt"]).max())


def test_grid_division():
    """the FFT grid divided into shifted subgrids (NKdiv) gives the same result."""
    irvec, H = model()
    s = system(irvec, H)
    one = run(s, {"q": calculator(Q["general"], "metal")}, nk=[6, 6, 4])["q"]
    divided = run(s, {"q": calculator(Q["general"], "metal")}, nk=[3, 2, 4], NKdiv=[2, 3, 1])["q"]
    np.testing.assert_allclose(divided, one, rtol=0, atol=1e-10 * np.abs(one).max())


def test_chunks():
    """the sum over k in several chunks, the last one partial, gives the same result."""
    irvec, H = model()
    s = system(irvec, H)
    kwargs = dict(q=Q["general"], Efermi=[CASES["insulator"][0], CASES["metal"][0]], omega=[0.5, 3.0], kBT=0.1,
                  allow_q_off_grid=True)
    one = run(s, {"q": OpticalConductivity_q(**kwargs)})["q"]
    chunked = OpticalConductivity_q(**kwargs)
    chunked.k_chunk_elements = 9 * 9 * 7  # 7 k-points per chunk, of 120
    np.testing.assert_allclose(run(s, {"q": chunked})["q"], one, rtol=0, atol=1e-12 * np.abs(one).max())
    chunked.k_chunk_elements = 1  # less than one k-point: one k-point per chunk
    np.testing.assert_allclose(run(s, {"q": chunked})["q"], one, rtol=0, atol=1e-12 * np.abs(one).max())
    # several Fermi levels, fewer and more than the 9 components of sigma
    for nEF in (5, 12):
        kwargs["Efermi"] = np.linspace(-4.5, 0.5, nEF)
        one = run(s, {"q": OpticalConductivity_q(**kwargs)})["q"]
        chunked = OpticalConductivity_q(**kwargs)
        chunked.k_chunk_elements = 9 * max(nEF, 9) * 7  # 7 k-points per chunk
        np.testing.assert_allclose(run(s, {"q": chunked})["q"], one, rtol=0, atol=1e-12 * np.abs(one).max())


def test_uniform_shift_of_centres():
    """moving all Wannier centres by the same vector changes nothing."""
    irvec, H = model()
    one = run(system(irvec, H), {"q": calculator(Q["general"], "metal")})["q"]
    shifted = run(system(irvec, H, CENTRES + [0.13, -0.4, 0.27]), {"q": calculator(Q["general"], "metal")})["q"]
    np.testing.assert_allclose(shifted, one, rtol=0, atol=1e-10 * np.abs(one).max())


@pytest.mark.parametrize("kwargs", [dict(use_irred_kpt=True), dict(symmetrize=True)])
def test_symmetrization_is_refused(kwargs):
    irvec, H = model()
    with pytest.raises(ValueError, match="not symmetric"):
        run(system(irvec, H), {"q": calculator(Q["x"], "metal")}, **kwargs)
    run(system(irvec, H), {"q": calculator(Q["zero"], "metal")}, **kwargs)  # allowed at q = 0


@pytest.mark.parametrize("kwargs, match", [
    (dict(omega=[1.0]), "Efermi and omega"),
    (dict(Efermi=[0.0], omega=[]), "omega should be"),
    (dict(Efermi=[np.nan], omega=[1.0]), "Efermi should be"),
    (dict(Efermi=[0.0], omega=[1.0], q=[np.inf, 0, 0]), "q should be"),
    (dict(Efermi=[0.0], omega=[1.0], kBT=-0.1), "kBT"),
    (dict(Efermi=[0.0], omega=[1.0], smr_fixed_width=0), "smr_fixed_width"),
    (dict(Efermi=[0.0], omega=[1.0], smr_fixed_width=np.nan), "smr_fixed_width"),
])
def test_invalid_input(kwargs, match):
    with pytest.raises(ValueError, match=match):
        OpticalConductivity_q(**kwargs)
