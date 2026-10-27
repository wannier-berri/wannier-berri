"""Tests of the tetrahedron method for the delta functions of the frequency in the dynamic calculators (tetra=True)."""

import contextlib
import io
import os
import tempfile

import numpy as np
import pytest

import wannierberri as wberri
from wannierberri.calculators import dynamic
from wannierberri.calculators.sdct import SDCT_sym
from wannierberri.fourier.rvectors import Rvectors
from wannierberri.grid.tetrahedron import (TetraWeights, TetraWeightsParal, _edges_in_range, delta_weights_parallel,
                                           delta_weights_tetra, weights_tetra)
from wannierberri.system import System_R

S0 = np.eye(2, dtype=complex)
SIGMA = [np.array([[0, 1], [1, 0]], dtype=complex), np.array([[0, -1j], [1j, 0]]), np.diag([1., -1]).astype(complex)]
A, M, T0, ALPHA = 3.0, 3.5, 0.1, 0.2  # lattice constant (Angstrom), and the parameters of the two-band model (eV)
EFERMI = [0.3]


def make_system(hoppings, num_wann):
    iRvec = list(hoppings)
    system = System_R(force_internal_terms_only=True, silent=True)
    system.real_lattice = np.eye(3) * A
    system.num_wann = num_wann
    system.set_R_mat("Ham", np.array([hoppings[R] for R in iRvec]))
    system.set_wannier_centers(wannier_centers_red=np.zeros((num_wann, 3)))
    system.rvec = Rvectors(lattice=system.real_lattice, iRvec=np.array(iRvec), shifts_left_red=system.wannier_centers_red)
    system.do_at_end_of_init()
    return system


def two_band_system():
    """H(k) = T0 sum_a cos k_a + ALPHA sin k_z + sin k_x s_x + sin k_y s_y + (M - sum_a cos k_a) s_z"""
    hoppings = {(0, 0, 0): M * SIGMA[2]}
    for a in range(3):
        R = tuple(int(a == b) for b in range(3))
        odd = SIGMA[a] if a < 2 else ALPHA * S0  # the coefficient of sin k_a
        hoppings[R] = T0 / 2 * S0 - SIGMA[2] / 2 + odd / 2j
        hoppings[tuple(-r for r in R)] = T0 / 2 * S0 - SIGMA[2] / 2 - odd / 2j
    return make_system(hoppings, 2)


def two_band_energies(N):
    """the two bands of `two_band_system` on an N x N x N grid, written out"""
    k = 2 * np.pi * np.arange(N) / N
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    cos = np.cos(kx) + np.cos(ky) + np.cos(kz)
    d = np.sqrt(np.sin(kx) ** 2 + np.sin(ky) ** 2 + (M - cos) ** 2)
    e0 = T0 * cos + ALPHA * np.sin(kz)
    return e0 - d, e0 + d


def random_system(num_wann=3, seed=1, copies=1, two_dimensional=False):
    """random hoppings to the nearest and next-nearest neighbours, without any symmetry. With copies > 1, the model
    is repeated, so that all bands are degenerate. With two_dimensional=True, only in the xy plane (and halved)."""
    rng = np.random.default_rng(seed)
    hoppings = {}
    if two_dimensional:
        Rlist, amplitude = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (1, -1, 0)], 0.5
    else:
        Rlist, amplitude = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, -1, 0)], 1
    for R in Rlist:
        h = rng.normal(size=(num_wann, num_wann)) + 1j * rng.normal(size=(num_wann, num_wann))
        if R == (0, 0, 0):
            h = h + h.conj().T + np.diag([-3., 0., 3.])
        h = h * amplitude
        hoppings[R] = np.kron(np.eye(copies), h)
        hoppings[tuple(-r for r in R)] = np.kron(np.eye(copies), h.conj().T)
    num_wann *= copies
    system = make_system(hoppings, num_wann)
    SS = np.zeros((system.rvec.nRvec, num_wann, num_wann, 3), dtype=complex)
    for a in range(3):
        s = rng.normal(size=(num_wann, num_wann)) + 1j * rng.normal(size=(num_wann, num_wann))
        SS[system.rvec.iR0, :, :, a] = s + s.conj().T
    system.set_R_mat("SS", SS)
    return system


def run(system, calculators, NK=10, grid=None, parallel=False):
    """NK : int or list of 3 ints, the FFT grid"""
    if grid is None:
        grid = wberri.Grid(system, NKFFT=NK, NKdiv=1)
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        result = wberri.run(system, grid=grid, parallel=parallel, calculators=calculators,
                            fout_name=os.path.join(tmp, "x"))
    return {key: result.results[key].data for key in calculators}


@pytest.mark.parametrize("shape", ["parallelepiped", "tetrahedron"])
def test_delta_weights_linear(shape):
    """for a linear function dE, the weights of delta(dE - omega) summed up to an edge of omega give the fraction of
    the volume where dE is below it (Monte Carlo)"""
    rng = np.random.default_rng(0)
    edges = np.linspace(-30, 30, 6001)
    for _ in range(5):
        c, g = rng.normal(), rng.normal(size=3) * 2
        if shape == "parallelepiped":
            corners = np.array([[[c + g @ (np.array([i, j, k]) - 0.5) for k in (0, 1)] for j in (0, 1)] for i in (0, 1)])
            weights = delta_weights_parallel(edges, np.array([c]), corners[None])[0]
            x = rng.random((400000, 3)) - 0.5
        else:
            vertices = rng.normal(size=(4, 3))
            weights = delta_weights_tetra(edges, (c + vertices @ g)[None])[0]
            x = rng.dirichlet(np.ones(4), size=400000) @ vertices
        values = c + x @ g
        cumulative = np.cumsum(weights * np.diff(edges))
        for w in (-1.0, -0.2, 0.0, 0.5, 1.5):
            i = np.searchsorted(edges, w)
            assert abs(cumulative[i - 1] - (values < edges[i]).mean()) < 4e-3
        assert abs(cumulative[-1] - 1) < 1e-12


@pytest.mark.parametrize("dE", [[1, 1, 1, 1], [1, 1, 1, 1.0001], [1, 1, 2, 2], [1, 1.001, 1.002, 3]])
def test_delta_weights_narrow(dE):
    """a delta function narrower than the intervals of omega keeps its weight"""
    edges = np.arange(0, 4.001, 0.1)
    weights = delta_weights_tetra(edges, np.array([dE], dtype=float))[0]
    assert np.sum(weights * np.diff(edges)) == pytest.approx(1, abs=1e-12)
    assert np.all(weights >= -1e-12)


@pytest.mark.parametrize("dE", [[1.0, 1.0, 1.0, 1.0], [1.0, 1.2, 1.4, 2.0], [-3.0, -2.0, -1.0, 0.5], [3.5, 4.0, 4.5, 6.0],
                                [-9.0, -8.0, -7.0, -6.0], [7.0, 8.0, 8.5, 9.0], [0.5, 0.5, 0.5, 1.0 + 1e-13]])
def test_delta_weights_range(dE):
    """the weights are computed only between the edges around the range of dE: the same as from all edges, also
    with dE on the edges, extending beyond them or outside them"""
    edges = np.arange(-5.0, 5.001, 0.5)
    dE = np.array(dE)
    occ = weights_tetra(edges, dE[0], dE[1], dE[2], dE[3], 0)
    np.testing.assert_allclose(delta_weights_tetra(edges, dE[None])[0], np.diff(occ) / np.diff(edges), rtol=0, atol=1e-12)
    corners = np.empty((2, 2, 2))
    corners[0, 0, 0], corners[1, 1, 1], corners[0, 1, 0], corners[1, 0, 1] = dE
    corners[1, 0, 0] = corners[0, 1, 1] = corners[0, 0, 1] = corners[1, 1, 0] = dE.mean()
    occ = np.zeros(len(edges))
    for axis in range(3):
        for iface in range(2):
            face = np.take(corners, iface, axis=axis)
            occ += weights_tetra(edges, dE.mean(), face[0, 0], face[0, 1], face[1, 1], 0)
            occ += weights_tetra(edges, dE.mean(), face[0, 0], face[1, 0], face[1, 1], 0)
    np.testing.assert_allclose(delta_weights_parallel(edges, np.array([dE.mean()]), corners[None])[0],
                               np.diff(occ) / np.diff(edges) / 12, rtol=0, atol=1e-12)


def test_python_versions():
    """the compiled functions give the same results as their Python versions (which the coverage report can trace)"""
    rng = np.random.default_rng(4)
    edges = np.arange(-5.0, 5.001, 0.5)
    dE = np.concatenate([rng.normal(size=(6, 4)) * 2, [[1.0, 1.0, 1.0, 1.0], [7.0, 8.0, 8.5, 9.0]]])
    np.testing.assert_allclose(delta_weights_tetra.py_func(edges, dE), delta_weights_tetra(edges, dE), rtol=0, atol=1e-12)
    centre = dE.mean(axis=1)
    corners = centre[:, None, None, None] + rng.normal(size=(len(dE), 2, 2, 2))
    np.testing.assert_allclose(delta_weights_parallel.py_func(edges, centre, corners),
                               delta_weights_parallel(edges, centre, corners), rtol=0, atol=1e-12)
    for emin, emax in [(-6.0, -5.5), (-1.0, 1.0), (1.0, 1.0), (4.0, 6.0), (6.0, 7.0)]:
        assert _edges_in_range.py_func(edges, emin, emax) == _edges_in_range(edges, emin, emax)


@pytest.mark.parametrize("cls", [TetraWeights, TetraWeightsParal])
def test_delta_weights_groups(cls):
    """the weights for pairs of band groups are the averages over the pairs of bands, with the energies at the
    corners of each band (here not degenerate)"""
    rng = np.random.default_rng(3)
    nb = 4
    eCenter = np.sort(rng.normal(size=(1, nb)) * 3, axis=1)
    shape = (1, 4, nb) if cls is TetraWeights else (1, 2, 2, 2, nb)
    eCorners = eCenter.reshape((1,) + (1,) * (len(shape) - 2) + (nb,)) + rng.normal(size=shape) * 0.5
    edges = np.linspace(-15, 15, 301)
    pairs = [((2, 4), (0, 2)), ((0, 2), (2, 4)), ((3, 4), (1, 2)), ((1, 3), (1, 3))]
    weights = cls(eCenter=eCenter, eCorners=eCorners)
    result = weights.delta_weights(0, pairs, edges)
    # empty pairs (used for the pairs that do not contribute) give zeros, and do not change the others
    with_empty = weights.delta_weights(0, [((0, 0), (0, 0))] + pairs, edges)
    np.testing.assert_array_equal(with_empty[0], 0)
    np.testing.assert_allclose(with_empty[1:], result, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(weights.delta_weights(0, [((0, 0), (0, 0))], edges), 0)

    def one(i, j):
        if cls is TetraWeights:
            return delta_weights_tetra(edges, (eCorners[0][:, i] - eCorners[0][:, j])[None])[0]
        return delta_weights_parallel(edges, eCenter[0, i:i + 1] - eCenter[0, j:j + 1],
                                      (eCorners[0][..., i] - eCorners[0][..., j])[None])[0]

    for (gi, gj), res in zip(pairs, result):
        expected = np.mean([one(i, j) for i in range(*gi) for j in range(*gj)], axis=0)
        np.testing.assert_allclose(res, expected, rtol=0, atol=1e-12)
        assert np.abs(res).max() > 0


def jdos_reference():
    """JDOS of `two_band_system` (an insulator) from a histogram of the band gaps on a dense grid"""
    Elow, Ehigh = two_band_energies(160)
    assert Elow.max() < EFERMI[0] < Ehigh.min()
    bins = np.arange(0.5, 14.0 + 1e-9, 0.25)
    return bins, np.histogram(Ehigh - Elow, bins=bins)[0] / Elow.size / (bins[1] - bins[0]) / A ** 3


def test_jdos_against_reference():
    """JDOS against a histogram of the band gaps on a dense grid. On the same grid, the tetrahedron method is closer to
    it than the Gaussian smearing with any of the widths tried."""
    system = two_band_system()
    bins, reference = jdos_reference()
    domega = 0.005
    omega = np.arange(bins[0], bins[-1], domega) + domega / 2
    widths = (0.05, 0.1, 0.2)
    calculators = {"tetra": dynamic.JDOS(Efermi=EFERMI, omega=omega, tetra=True)}
    calculators.update({f"smear{w}": dynamic.JDOS(Efermi=EFERMI, omega=omega, smr_type="Gaussian", smr_fixed_width=w)
                        for w in widths})
    result = run(system, calculators, NK=16)

    def error(jdos):
        binned = jdos[0].reshape(len(bins) - 1, -1).mean(axis=1)
        return np.abs(binned - reference).sum() / np.abs(reference).sum()

    assert error(result["tetra"]) < 0.03
    assert error(result["tetra"]) < min(error(result[f"smear{w}"]) for w in widths)


CALCULATORS = [
    (dynamic.JDOS, {}),
    (dynamic.OpticalConductivity, {}),
    (dynamic.SHC, {"SHC_type": "simple"}),
    (dynamic.ShiftCurrent, {"sc_eta": 0.1}),
    (dynamic.InjectionCurrent, {}),
]


@pytest.mark.parametrize("copies", [1, 2])
@pytest.mark.parametrize("calculator, parameters", CALCULATORS)
def test_frequency_integral(calculator, parameters, copies):
    """the delta functions of the tetrahedron method and of the smearing are normalised. The two results differ only
    in the delta functions (the principal part of the optical and spin Hall conductivities is the same Lorentzian
    in both), so the integral of their difference over frequencies that cover all the band gaps vanishes, for each
    component, while the difference itself does not. With copies=2, all bands are degenerate, and the weights are
    averaged over the pairs of bands of two groups."""
    system = random_system(copies=copies)
    domega = 0.01
    omega = np.arange(0.005, 60, domega)  # the band gaps of the model are below 40 eV
    kwargs = dict(Efermi=EFERMI, omega=omega, **parameters)
    result = run(system, {"tetra": calculator(tetra=True, **kwargs),
                          "smear": calculator(smr_type="Gaussian", smr_fixed_width=0.1, **kwargs)}, NK=6)
    difference = result["tetra"] - result["smear"]
    scale = np.abs(difference).sum(axis=1).max() * domega
    assert scale > 0.1 * np.abs(result["smear"]).sum(axis=1).max() * domega
    for part in (np.real, np.imag):
        np.testing.assert_allclose(part(difference.sum(axis=1)) * domega, 0, rtol=0, atol=2e-3 * scale)


@pytest.mark.parametrize("calculator, parameters, tolerance", [
    (dynamic.OpticalConductivity, {}, 0.15),
    (dynamic.SHC, {"SHC_type": "simple"}, 0.25),
])
def test_against_dense_smearing(calculator, parameters, tolerance):
    """for a two-dimensional model, against the Gaussian smearing on a much denser grid, averaged over 0.5 eV. The
    differences are dominated by the principal part and by the grid, so this checks the sign and the normalisation
    rather than the accuracy: it fails if the delta function is taken with the wrong sign of the frequency (by 45%
    for the optical conductivity and by 160% for the spin Hall conductivity)."""
    system = random_system(seed=2, two_dimensional=True)
    domega = 0.01
    omega = np.arange(0.005, 20, domega)
    kwargs = dict(Efermi=[0.0], omega=omega, **parameters)
    reference = run(system, {"x": calculator(smr_type="Gaussian", smr_fixed_width=0.05, **kwargs)}, NK=[120, 120, 1])["x"]
    tetra = run(system, {"x": calculator(tetra=True, **kwargs)}, NK=[32, 32, 1])["x"]

    def binned(x):
        return x[0].reshape(-1, 50, *x.shape[2:]).mean(axis=1)

    assert np.abs(binned(tetra) - binned(reference)).max() < tolerance * np.abs(binned(reference)).max()


def test_grid_tetra():
    """JDOS on a grid of tetrahedra (GridTetra), against the histogram of the band gaps on a dense grid. With the band
    gaps at the k-points instead of the corners, the error would be 0.13."""
    system = two_band_system()
    bins, reference = jdos_reference()
    domega = 0.005
    omega = np.arange(bins[0], bins[-1], domega) + domega / 2
    grid = wberri.grid.GridTetra(system, length=40, NKFFT=(12, 12, 12))
    jdos = run(system, {"x": dynamic.JDOS(Efermi=EFERMI, omega=omega, tetra=True)}, grid=grid)["x"]
    binned = jdos[0].reshape(len(bins) - 1, -1).mean(axis=1)
    assert np.abs(binned - reference).sum() < 0.05 * np.abs(reference).sum()
    assert jdos.sum() * domega == pytest.approx(1 / A ** 3, rel=1e-10)


def test_narrow_window():
    """with a range of frequencies narrower than the spread of the band gaps within a cell, the pairs are selected by
    the energies at the corners, not by those at the k-point"""
    system = random_system()
    domega = 0.01
    omega = np.arange(0.005, 60, domega)
    window = slice(1000, 1100)  # 10 to 11 eV
    calculators = {"full": dynamic.JDOS(Efermi=EFERMI, omega=omega, tetra=True),
                   "window": dynamic.JDOS(Efermi=EFERMI, omega=omega[window], tetra=True)}
    result = run(system, calculators, NK=4)
    assert np.abs(result["window"]).max() > 0
    np.testing.assert_allclose(result["window"], result["full"][:, window], rtol=1e-10, atol=0)


def test_parallel():
    """the same result with ray workers as in serial"""
    ray = pytest.importorskip("ray")
    if not ray.is_initialized():
        pytest.skip("ray is not initialized")
    system = random_system()
    omega = np.arange(0.005, 40, 0.05)
    calculator = dict(Efermi=EFERMI, omega=omega, tetra=True)
    grid = wberri.Grid(system, NKFFT=4, NKdiv=2)
    serial = run(system, {"x": dynamic.OpticalConductivity(**calculator)}, grid=grid)["x"]
    parallel = run(system, {"x": dynamic.OpticalConductivity(**calculator)}, grid=grid, parallel=True)["x"]
    np.testing.assert_allclose(parallel, serial, rtol=0, atol=1e-10 * np.abs(serial).max())


def test_nkdiv_fermi_levels_temperature():
    """the grid divided into shifted subgrids (NKdiv) gives the same result as the full FFT grid, and several Fermi
    levels at a finite temperature give the same results as separate calculations"""
    system = random_system()
    omega = np.arange(0.005, 40, 0.05)
    kwargs = dict(omega=omega, kBT=0.05, tetra=True)
    efermi = [0.0, 0.5]
    divided = run(system, {"x": dynamic.OpticalConductivity(Efermi=efermi, **kwargs)},
                  grid=wberri.Grid(system, NKFFT=4, NKdiv=2))["x"]
    full = run(system, {"x": dynamic.OpticalConductivity(Efermi=efermi, **kwargs)}, NK=8)["x"]
    scale = np.abs(full).max()
    np.testing.assert_allclose(divided, full, rtol=0, atol=1e-10 * scale)
    for i, ef in enumerate(efermi):
        single = run(system, {"x": dynamic.OpticalConductivity(Efermi=[ef], **kwargs)}, NK=8)["x"]
        np.testing.assert_allclose(full[i], single[0], rtol=0, atol=1e-12 * scale)


def test_tetra_not_implemented():
    with pytest.raises(ValueError, match="tetrahedron method"):
        SDCT_sym(Efermi=EFERMI, omega=np.linspace(0.1, 1, 3), tetra=True)


@pytest.mark.parametrize("dE", [0.97, 1.03])
def test_tetra_omega_intervals(dE):
    """the result at each frequency is the average over the interval between the midpoints to its neighbours"""
    omega = np.array([0.9, 1.0, 1.1, 1.3])
    edges = dynamic.JDOS(Efermi=EFERMI, omega=omega, tetra=True).omega_edges
    np.testing.assert_allclose(edges, [0.85, 0.95, 1.05, 1.2, 1.4])
    weights = delta_weights_tetra(edges, np.full((1, 4), dE))[0]
    np.testing.assert_allclose(weights, [0, 10, 0, 0])


@pytest.mark.parametrize("omega", [[0.5], [1.0, 0.5, 2.0], [0.0, np.nan, 2.0], [0.0, 1.0, np.inf]])
def test_tetra_omega_increasing(omega):
    with pytest.raises(ValueError, match="omega must be finite and increasing"):
        dynamic.JDOS(Efermi=EFERMI, omega=np.array(omega), tetra=True)
