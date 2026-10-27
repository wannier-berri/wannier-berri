"""Tests of the tetrahedron method for the delta functions of the frequency in the dynamic calculators (tetra_delta=True)."""

import contextlib
import io
import os
import tempfile

import numpy as np
import pytest

import wannierberri as wberri
from wannierberri.calculators import dynamic
from wannierberri import factors
from wannierberri.calculators.sdct import SDCT_sym
from wannierberri.fourier.rvectors import Rvectors
from wannierberri.grid.tetrahedron import (FLAT_FRACTION, TetraWeights, _edges_in_range,
                                           _omega_in_range,
                                           delta_tetra, delta_values_tetra, delta_weights_tetra,
                                           weights_tetra)
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
    system.test_hoppings = hoppings  # for the reference written out below
    system.test_spin = SS[system.rvec.iR0]
    return system


def run(system, calculators, NK=10, grid=None, parallel=False):
    """NK : int or list of 3 ints, the FFT grid"""
    if grid is None:
        grid = wberri.Grid(system, NKFFT=NK, NKdiv=1)
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        result = wberri.run(system, grid=grid, parallel=parallel, calculators=calculators,
                            fout_name=os.path.join(tmp, "x"))
    return {key: result.results[key].data for key in calculators}


def parallel_delta(omega, center, corners, edges=None):
    """delta(dE - omega) over the 12 tetrahedra of a parallelepiped (values at omega, or averages over the intervals
    between the edges), for dE given at its centre and at its corners (2, 2, 2): the weights of a pair of bands,
    the upper one with the energies dE and the lower one at zero"""
    eCenter = np.array([[0.0, center]])
    eCorners = np.stack([np.zeros((2, 2, 2)), corners], axis=-1)[None]
    weights = TetraWeights(eCenter=eCenter, eCorners=eCorners, kind="parallelepiped")
    return weights.delta_weights(0, [((1, 2), (0, 1))], omega, edges)[0]


def parallelepiped_tetrahedra(center, corners):
    """the values at the 4 vertices of each of the 12 tetrahedra of a parallelepiped"""
    weights = TetraWeights(eCenter=np.array([[center]]), eCorners=corners[None, ..., None], kind="parallelepiped")
    return [vertices[0, :, 0] for vertices in weights.all_tetrahedra_vertices]


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
            weights = parallel_delta((edges[1:] + edges[:-1]) / 2, c, corners, edges)
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
    np.testing.assert_allclose(parallel_delta((edges[1:] + edges[:-1]) / 2, dE.mean(), corners, edges),
                               np.diff(occ) / np.diff(edges) / 12, rtol=0, atol=1e-12)


def test_python_versions():
    """the compiled functions give the same results as their Python versions (which the coverage report can trace)"""
    rng = np.random.default_rng(4)
    edges = np.arange(-5.0, 5.001, 0.5)
    dE = np.concatenate([rng.normal(size=(6, 4)) * 2, [[1.0, 1.0, 1.0, 1.0], [7.0, 8.0, 8.5, 9.0]]])
    np.testing.assert_allclose(delta_weights_tetra.py_func(edges, dE), delta_weights_tetra(edges, dE), rtol=0, atol=1e-12)
    for emin, emax in [(-6.0, -5.5), (-1.0, 1.0), (1.0, 1.0), (4.0, 6.0), (6.0, 7.0)]:
        assert _edges_in_range.py_func(edges, emin, emax) == _edges_in_range(edges, emin, emax)
        assert _omega_in_range.py_func(edges, emin, emax) == _omega_in_range(edges, emin, emax)
    np.testing.assert_allclose(delta_values_tetra.py_func(edges, dE), delta_values_tetra(edges, dE), rtol=0, atol=1e-12)
    np.testing.assert_allclose(delta_tetra.py_func(edges, *dE[0]), delta_tetra(edges, *dE[0]), rtol=0, atol=1e-12)


@pytest.mark.parametrize("average", [False, True])
@pytest.mark.parametrize("kind", ["single", "parallelepiped"])
def test_delta_weights_groups(kind, average):
    """the weights for pairs of band groups are the averages over the pairs of bands, with the energies at the
    corners of each band (here not degenerate)"""
    rng = np.random.default_rng(3)
    nb = 4
    eCenter = np.sort(rng.normal(size=(1, nb)) * 3, axis=1)
    shape = (1, 4, nb) if kind == "single" else (1, 2, 2, 2, nb)
    eCorners = eCenter.reshape((1,) + (1,) * (len(shape) - 2) + (nb,)) + rng.normal(size=shape) * 0.5
    edges = np.linspace(-15, 15, 301)
    omega = (edges[1:] + edges[:-1]) / 2
    if not average:
        edges = None
    pairs = [((2, 4), (0, 2)), ((0, 2), (2, 4)), ((3, 4), (1, 2)), ((1, 3), (1, 3))]
    weights = TetraWeights(eCenter=eCenter, eCorners=eCorners, kind=kind)
    result = weights.delta_weights(0, pairs, omega, edges)
    # empty pairs (used for the pairs that do not contribute) give zeros, and do not change the others
    with_empty = weights.delta_weights(0, [((0, 0), (0, 0))] + pairs, omega, edges)
    np.testing.assert_array_equal(with_empty[0], 0)
    np.testing.assert_allclose(with_empty[1:], result, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(weights.delta_weights(0, [((0, 0), (0, 0))], omega, edges), 0)

    def one(i, j):
        if kind == "single":
            dE = (eCorners[0][:, i] - eCorners[0][:, j])[None]
            return (delta_weights_tetra(edges, dE) if average else delta_values_tetra(omega, dE))[0]
        return parallel_delta(omega, eCenter[0, i] - eCenter[0, j], eCorners[0][..., i] - eCorners[0][..., j], edges)

    for (gi, gj), res in zip(pairs, result):
        expected = np.mean([one(i, j) for i in range(*gi) for j in range(*gj)], axis=0)
        np.testing.assert_allclose(res, expected, rtol=0, atol=1e-12)
        assert np.abs(res).max() > 0


@pytest.mark.parametrize("dE", [[0.3, -1.2, 2.0, 0.7], [1.0, 1.0, 2.0, 3.0], [1.0, 2.0, 2.0, 3.0], [1.0, 2.0, 3.0, 3.0],
                                [-1.0, 0.0, 0.0, 0.0]])
def test_delta_values(dE):
    """the values of the delta function are the derivative of the cumulative weight, and are continuous unless three
    corners have the same value"""
    omega = np.linspace(-3, 4, 7001)
    values = delta_tetra(omega, *dE)
    h = 1e-6
    derivative = (weights_tetra(omega + h, *dE, der=0) - weights_tetra(omega - h, *dE, der=0)) / (2 * h)
    inside = np.min(np.abs(omega[:, None] - np.array(dE)[None, :]), axis=1) > 2 * h  # not at the kinks
    np.testing.assert_allclose(values[inside], derivative[inside], rtol=0, atol=1e-5 * np.abs(derivative).max())
    if len(set(dE)) > 2:
        assert np.abs(np.diff(values)).max() < 0.01 * np.abs(values).max()
    assert np.all(values >= 0)


@pytest.mark.parametrize("dE", [[1.0, 1.2, 1.4, 2.0], [-3.0, -2.0, -1.0, 0.5], [3.5, 4.0, 4.5, 6.0],
                                [-9.0, -8.0, -7.0, -6.0], [0.5, 0.5, 0.5, 1.0 + 1e-13]])
def test_delta_values_range(dE):
    """the values are computed only at the frequencies within the range of dE: the same as at all frequencies (for
    a flat dE, see `test_delta_values_flat`)"""
    omega = np.arange(-5.0, 5.001, 0.25)
    dE = np.array(dE)
    np.testing.assert_allclose(delta_values_tetra(omega, dE[None])[0], delta_tetra(omega, *dE), rtol=0, atol=1e-12)
    corners = np.empty((2, 2, 2))
    corners[0, 0, 0], corners[1, 1, 1], corners[0, 1, 0], corners[1, 0, 1] = dE
    corners[1, 0, 0] = corners[0, 1, 1] = corners[0, 0, 1] = corners[1, 1, 0] = dE.mean()
    expected = sum(delta_tetra(omega, *t) for t in parallelepiped_tetrahedra(dE.mean(), corners)) / 12
    np.testing.assert_allclose(parallel_delta(omega, dE.mean(), corners), expected,
                               rtol=0, atol=1e-12)


class KuboReference:
    """the dynamic calculators for `random_system` (internal terms only, kBT = 0) written out k-point by k-point
    with numpy, independently of wannierberri: the Kubo formulas with the delta function given either by the Gaussian
    smearing or by the linear tetrahedron method (the band energies at the corners of the parallelepiped around each
    k-point, divided into 12 tetrahedra, and the density of a linear function on a tetrahedron by Bloechl et al.,
    PRB 49, 16223 (1994))"""

    def __init__(self, system, N, efermi, omega, kBT=0):
        self.hoppings, self.spin = system.test_hoppings, system.test_spin
        self.lattice = system.real_lattice
        self.nw = system.num_wann
        self.N, self.omega = N, omega
        self.k = np.array(np.meshgrid(*[np.arange(N) / N] * 3, indexing="ij")).reshape(3, -1).T
        H, v = self.hamiltonian(self.k)
        self.E, U = np.linalg.eigh(H)
        V = np.einsum("kin,akij,kjm->aknm", U.conj(), v, U)
        self.V = V
        self.S = np.einsum("kin,ija,kjm->aknm", U.conj(), self.spin, U)
        gap = self.E[:, None, :] - self.E[:, :, None]  # E_m - E_n [k, n, m]
        offdiagonal = ~np.eye(self.nw, dtype=bool)
        self.A = np.zeros_like(V)  # the Berry connection A_nm = i v_nm / (E_m - E_n) for n != m
        self.A[:, :, offdiagonal] = 1j * V[:, :, offdiagonal] / gap[None][:, :, offdiagonal]
        if kBT == 0:
            self.f = (self.E < efermi).astype(float)
        else:
            self.f = 1 / (1 + np.exp((self.E - efermi) / kBT))
        self.corners = np.zeros((2, 2, 2, len(self.k), self.nw))
        for i in np.ndindex(2, 2, 2):
            self.corners[i] = np.linalg.eigvalsh(self.hamiltonian(self.k + (np.array(i) - 0.5) / N)[0])
        self.volume = abs(np.linalg.det(self.lattice))

    def hamiltonian(self, k):
        """H and dH/dk (Cartesian) at the reduced k-points"""
        H = np.zeros((len(k), self.nw, self.nw), dtype=complex)
        v = np.zeros((3,) + H.shape, dtype=complex)
        for R, h in self.hoppings.items():
            phase = np.exp(2j * np.pi * k @ np.array(R))[:, None, None]
            H += phase * h
            for a, Ra in enumerate(np.array(R) @ self.lattice):
                v[a] += 1j * Ra * phase * h
        return H, v

    def delta(self, ik, n, m, method, edges=None):
        """delta(E_n - E_m - omega) at the k-point ik"""
        if method == "smearing":
            x = (self.E[ik, n] - self.E[ik, m] - self.omega) / 0.1
            return np.exp(-x ** 2) / np.sqrt(np.pi) / 0.1
        center = self.E[ik, n] - self.E[ik, m]
        corners = self.corners[..., ik, n] - self.corners[..., ik, m]
        result = 0
        for axis in range(3):
            for side in (0, 1):
                face = np.take(corners, side, axis=axis)
                for triangle in ([face[0, 0], face[0, 1], face[1, 1]], [face[0, 0], face[1, 0], face[1, 1]]):
                    e = np.sort([center] + triangle)
                    if edges is None:
                        result = result + self.bloechl(self.omega, e, density=True)
                    else:
                        result = result + np.diff(self.bloechl(edges, e, density=False)) / np.diff(edges)
        return result / 12

    @staticmethod
    def bloechl(w, e, density):
        """the density (or the fraction below w) of a linear function with the sorted corner values e"""
        e1, e2, e3, e4 = e
        res = np.zeros(len(w)) if density else (w >= e4).astype(float)
        x = w - e1
        inside = (w > e1) & (w < e2)
        c = 1 / ((e2 - e1) * (e3 - e1) * (e4 - e1))
        res[inside] = (3 * x ** 2 * c if density else x ** 3 * c)[inside]
        x = w - e2
        inside = (w >= e2) & (w < e3)
        c, q = 1 / ((e3 - e1) * (e4 - e1)), (e3 - e1 + e4 - e2) / ((e3 - e2) * (e4 - e2))
        if density:
            res[inside] = ((3 * (e2 - e1) + 6 * x - 3 * q * x ** 2) * c)[inside]
        else:
            res[inside] = (((e2 - e1) ** 2 + 3 * (e2 - e1) * x + 3 * x ** 2 - q * x ** 3) * c)[inside]
        x = e4 - w
        inside = (w >= e3) & (w < e4)
        c = 1 / ((e4 - e1) * (e4 - e2) * (e4 - e3))
        res[inside] = (3 * x ** 2 * c if density else 1 - x ** 3 * c)[inside]
        return res

    def calculate(self, name, method, edges=None):
        """the result [omega, ...] of the calculator `name` with the delta function by `method`"""
        omega, eta = self.omega, 0.1
        shape = {"JDOS": (), "OpticalConductivity": (3, 3), "SHC": (3, 3, 3), "InjectionCurrent": (3, 3, 3)}[name]
        result = np.zeros((len(omega),) + shape, dtype=complex)
        for ik in range(len(self.k)):
            for n in range(self.nw):
                for m in range(self.nw):
                    occupation = self.f[ik, m] - self.f[ik, n]  # f(E2) - f(E1) for the pair (n, m)
                    if n == m or occupation == 0:
                        continue
                    gap = self.E[ik, m] - self.E[ik, n]
                    if name == "JDOS":
                        result += occupation * self.delta(ik, n, m, method, edges)
                    elif name == "OpticalConductivity":
                        cfac = 1 / (gap - omega - 1j * eta)
                        cfac.imag = np.pi * self.delta(ik, m, n, method, edges)
                        element = 1j * self.A[:, ik, n, m][:, None] * self.A[:, ik, m, n][None, :]
                        result += (occupation * gap * cfac)[:, None, None] * element
                    elif name == "InjectionCurrent":  # (v_nn - v_mm)^a A_nm^b A_mn^c delta(E_n - E_m - omega)
                        dv = self.V[:, ik, n, n] - self.V[:, ik, m, m]
                        element = np.einsum("a,b,c->abc", dv, self.A[:, ik, n, m], self.A[:, ik, m, n])
                        result += (occupation * self.delta(ik, n, m, method, edges))[:, None, None, None] * element
                    else:  # SHC with the spin current {S, v} / 2
                        cfac = 1 / (-gap - omega - 1j * eta)
                        cfac.imag = np.pi * self.delta(ik, n, m, method, edges)
                        J = np.array([[(self.V[a, ik] @ self.S[g, ik] + self.S[g, ik] @ self.V[a, ik])[n, m] / 2
                                       for g in range(3)] for a in range(3)])  # [a, gamma]
                        element = np.imag(J[:, None, :] * (-1j * self.A[:, ik, m, n])[None, :, None])
                        result += (occupation * cfac / 2)[:, None, None, None] * element
        factor = {"JDOS": 1, "OpticalConductivity": factors.factor_opt, "SHC": factors.factor_shc,
                  "InjectionCurrent": factors.factor_injection_current}[name]
        return result * factor / (len(self.k) * self.volume)


@pytest.mark.parametrize("average, kBT", [(False, 0), (True, 0), (False, 0.3), (True, 0.3)])
def test_against_kubo_reference(average, kBT):
    """JDOS, optical and spin Hall conductivities and the injection current against the Kubo formulas written out with
    numpy (`KuboReference`), to 1e-10, also at a finite temperature (where all pairs of bands contribute). With the
    smearing, the reference reproduces the calculators, which checks the reference itself; with the tetrahedron
    method, the results differ from those with the smearing by more than half of the result."""
    system = random_system()
    N = 4
    omega = np.arange(0.05, 40, 0.1)
    reference = KuboReference(system, N, EFERMI[0], omega, kBT=kBT)
    edges = dynamic.JDOS(Efermi=EFERMI, omega=omega, tetra_delta=True, tetra_delta_average=True).omega_edges
    for name, parameters in [("JDOS", {}), ("OpticalConductivity", {}), ("SHC", {"SHC_type": "simple"}),
                             ("InjectionCurrent", {})]:
        calculator = getattr(dynamic, name)
        kwargs = dict(Efermi=EFERMI, omega=omega, kBT=kBT, **parameters)
        result = run(system, {"smear": calculator(smr_type="Gaussian", smr_fixed_width=0.1, **kwargs),
                              "tetra": calculator(tetra_delta=True, tetra_delta_average=average, **kwargs)}, NK=N)
        smear = reference.calculate(name, "smearing")
        tetra = reference.calculate(name, "tetra", edges if average else None)
        scale = np.abs(tetra).max()
        np.testing.assert_allclose(result["smear"][0], smear, rtol=0, atol=1e-10 * np.abs(smear).max())
        np.testing.assert_allclose(result["tetra"][0], tetra, rtol=0, atol=1e-10 * scale)
        assert np.abs(tetra - smear).max() > 0.5 * scale


REFERENCE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference", "dynamic_tetra.npz")


def stored_results():
    """the results of the five calculators with the tetrahedron method (values at the frequencies, and averages over
    the intervals) for `random_system`, compared with `REFERENCE_FILE`. To write the file again, run this module."""
    system = random_system()
    omega = np.arange(0.05, 40, 0.2)
    calculators = {}
    for calculator, parameters in CALCULATORS:
        for average in (False, True):
            calculators[f"{calculator.__name__}-{'average' if average else 'values'}"] = calculator(
                Efermi=EFERMI, omega=omega, tetra_delta=True, tetra_delta_average=average, **parameters)
    return run(system, calculators, NK=4)


def test_against_stored_reference():
    """all five calculators against stored results, to 1e-10. These were computed with this implementation, so they
    guard against changes; for JDOS, OpticalConductivity, SHC and InjectionCurrent they agree with `KuboReference`
    (ShiftCurrent is checked independently only by `test_frequency_integral`)."""
    reference = np.load(REFERENCE_FILE)
    result = stored_results()
    assert sorted(result) == sorted(reference.files)
    for key, value in result.items():
        np.testing.assert_allclose(value, reference[key], rtol=0, atol=1e-10 * np.abs(reference[key]).max(),
                                   err_msg=key)


@pytest.mark.parametrize("spread", [0, 1e-13, 1e-9, 0.5 * FLAT_FRACTION * 0.1])
def test_delta_values_flat(spread):
    """for a (nearly) flat dE, the delta function cannot be sampled at the frequencies: its weight is put into the
    interval around the nearest one (here 0.1 wide), also over a parallelepiped"""
    omega = np.arange(0.05, 2, 0.1)
    for dE in (0.73, 0.75 + 1e-13, 0.8):  # inside an interval, at an edge, at a frequency
        corners = dE + spread * np.array([0, 0.3, 0.7, 1.0])
        values = delta_values_tetra(omega, corners[None])[0]
        assert values.sum() * 0.1 == pytest.approx(1, rel=1e-12)
        assert np.count_nonzero(values) == 1 and abs(omega[np.argmax(values)] - corners.mean()) <= 0.05 + 1e-12
        parallel = parallel_delta(omega, dE, dE + spread * np.arange(8).reshape(2, 2, 2) / 7)
        assert parallel.sum() * 0.1 == pytest.approx(1, rel=1e-12)
    # a single frequency has no interval: nothing is taken as flat
    np.testing.assert_array_equal(delta_values_tetra(np.array([0.75]), np.full((1, 4), 0.8)), 0)
    single = delta_values_tetra(np.array([0.75]), np.array([[0.7, 0.72, 0.78, 0.8]]))
    assert single[0, 0] == pytest.approx(delta_tetra(np.array([0.75]), 0.7, 0.72, 0.78, 0.8)[0], rel=1e-14)


def test_delta_values_flat_intervals():
    """the interval of a flat dE, on a nonuniform grid (edges -0.125, 0.125, 0.375, 0.75, 1.25), by the middle of the
    range of dE: explicit values, at the edges, outside the ends, and at the threshold of the spread"""
    omega = np.array([0.0, 0.25, 0.5, 1.0])
    widths = np.array([0.25, 0.25, 0.375, 0.5])

    def expected(i):
        res = np.zeros(4)
        if i is not None:
            res[i] = 1 / widths[i]
        return res

    for dE, i in [(0.1, 0), (0.125, 1), (0.374, 1), (0.375, 2), (0.8, 3), (1.249, 3), (-0.125, 0),
                  (-0.2, None), (1.25, None), (1.3, None)]:
        np.testing.assert_allclose(delta_values_tetra(omega, np.full((1, 4), dE))[0], expected(i), rtol=1e-14)
    # the middle of the range (0.3749997, below the edge 0.375), not the mean of the corners (above it), selects the
    # interval
    corners = np.array([[0.374999, 0.3750004, 0.3750004, 0.3750004]])
    assert (corners.min() + corners.max()) / 2 < 0.375 <= corners.mean()
    np.testing.assert_allclose(delta_values_tetra(omega, corners)[0], expected(1), rtol=1e-14)
    # just below and above the threshold, FLAT_FRACTION times the width of the interval
    for factor, flat in ((0.99, True), (1.01, False)):
        spread = factor * FLAT_FRACTION * widths[2]
        dE = np.array([[0.6, 0.6 + spread / 3, 0.6 + spread / 2, 0.6 + spread]])
        values = delta_values_tetra(omega, dE)[0]
        if flat:
            np.testing.assert_allclose(values, expected(2), rtol=1e-14)
        else:
            np.testing.assert_allclose(values, delta_tetra(omega, *dE[0]), rtol=1e-14, atol=0)
    # a parallelepiped where only some of the tetrahedra are flat: those through the flat face x = 0 (corners at
    # 0.6) with the centre (also 0.6), while the others extend to 0.9
    corners = np.full((2, 2, 2), 0.6)
    corners[1] = 0.9
    fine = np.linspace(-0.125, 1.25, 1101)[:-1] + 0.000625
    values = parallel_delta(fine, 0.6, corners)
    tetrahedra = parallelepiped_tetrahedra(0.6, corners)
    expected_values = sum(delta_values_tetra(fine, t[None]) for t in tetrahedra)[0] / 12
    np.testing.assert_allclose(values, expected_values, rtol=1e-14, atol=0)
    flat_tetrahedra = sum(np.ptp(t) == 0 for t in tetrahedra)
    assert flat_tetrahedra == 2
    i = np.searchsorted(fine, 0.6) - 1  # the flat ones are in one interval around 0.6
    assert values.sum() * 0.00125 == pytest.approx(1, rel=1e-3)
    assert values[i:i + 2].max() * 0.00125 >= flat_tetrahedra / 12 - 1e-12


def test_jdos_constant_splitting():
    """two bands with a constant splitting 2J: the JDOS is (1/V) delta(omega - 2J), with the weight in the interval
    around 2J, without the tetrahedron method being able to resolve it, also with omega exactly at 2J"""
    J = 0.4
    hoppings = {(0, 0, 0): J * SIGMA[2]}
    for a in range(3):
        R = tuple(int(a == b) for b in range(3))
        hoppings[R] = hoppings[tuple(-r for r in R)] = T0 / 2 * S0  # bands T0 sum_a cos k_a -/+ J, an insulator
    system = make_system(hoppings, 2)
    domega = 0.01
    for omega in (np.arange(0.005, 2, domega), np.arange(0, 2, domega)):
        jdos = run(system, {"x": dynamic.JDOS(Efermi=[0.0], omega=omega, tetra_delta=True)}, NK=4)["x"][0]
        assert jdos.sum() * domega == pytest.approx(1 / A ** 3, rel=1e-10)
        assert abs(omega[np.argmax(jdos)] - 2 * J) <= domega / 2 + 1e-12


def jdos_reference():
    """JDOS of `two_band_system` (an insulator) from a histogram of the band gaps on a dense grid"""
    Elow, Ehigh = two_band_energies(160)
    assert Elow.max() < EFERMI[0] < Ehigh.min()
    bins = np.arange(0.5, 14.0 + 1e-9, 0.25)
    return bins, np.histogram(Ehigh - Elow, bins=bins)[0] / Elow.size / (bins[1] - bins[0]) / A ** 3


@pytest.mark.parametrize("average", [False, True])
def test_jdos_against_reference(average):
    """JDOS against a histogram of the band gaps on a dense grid. On the same grid, the tetrahedron method is closer to
    it than the Gaussian smearing with any of the widths tried."""
    system = two_band_system()
    bins, reference = jdos_reference()
    domega = 0.005
    omega = np.arange(bins[0], bins[-1], domega) + domega / 2
    widths = (0.05, 0.1, 0.2)
    calculators = {"tetra": dynamic.JDOS(Efermi=EFERMI, omega=omega, tetra_delta=True, tetra_delta_average=average)}
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


@pytest.mark.parametrize("copies, average", [(1, False), (2, False), (1, True)])
@pytest.mark.parametrize("calculator, parameters", CALCULATORS)
def test_frequency_integral(calculator, parameters, copies, average):
    """the delta functions of the tetrahedron method and of the smearing are normalised. The two results differ only
    in the delta functions (the principal part of the optical and spin Hall conductivities is the same Lorentzian
    in both), so the integral of their difference over frequencies that cover all the band gaps vanishes, for each
    component, while the difference itself does not. With copies=2, all bands are degenerate, and the weights are
    averaged over the pairs of bands of two groups."""
    system = random_system(copies=copies)
    domega = 0.01
    omega = np.arange(0.005, 60, domega)  # the band gaps of the model are below 40 eV
    kwargs = dict(Efermi=EFERMI, omega=omega, **parameters)
    smearing = dict(smr_type="Gaussian", smr_fixed_width=0.1)  # the same on both sides: if the tetrahedron method
    # were not used, the difference would vanish
    result = run(system, {"tetra": calculator(tetra_delta=True, tetra_delta_average=average, **smearing, **kwargs),
                          "smear": calculator(**smearing, **kwargs)}, NK=6)
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
    differences are dominated by the principal part and by the grid, so this is only a rough check (e.g. of the
    sign): it fails if the delta function is taken with the wrong sign of the frequency (by 45% for the optical
    conductivity and by 160% for the spin Hall conductivity). The accurate comparison is with `KuboReference`."""
    system = random_system(seed=2, two_dimensional=True)
    domega = 0.01
    omega = np.arange(0.005, 20, domega)
    kwargs = dict(Efermi=[0.0], omega=omega, **parameters)
    reference = run(system, {"x": calculator(smr_type="Gaussian", smr_fixed_width=0.05, **kwargs)}, NK=[120, 120, 1])["x"]
    tetra = run(system, {"x": calculator(tetra_delta=True, **kwargs)}, NK=[32, 32, 1])["x"]

    def binned(x):
        return x[0].reshape(-1, 50, *x.shape[2:]).mean(axis=1)

    assert np.abs(binned(tetra) - binned(reference)).max() < tolerance * np.abs(binned(reference)).max()


@pytest.mark.parametrize("average", [False, True])
def test_grid_tetra(average):
    """JDOS on a grid of tetrahedra (GridTetra), against the histogram of the band gaps on a dense grid. With the band
    gaps at the k-points instead of the corners, the error would be 0.13. The frequency integral is 1/V, exactly
    with the averages over the intervals; with the values at the frequencies, the sum over them misses 0.5% here,
    from the features of the small tetrahedra that are narrower than the intervals."""
    system = two_band_system()
    bins, reference = jdos_reference()
    domega = 0.005
    omega = np.arange(bins[0], bins[-1], domega) + domega / 2
    grid = wberri.grid.GridTetra(system, length=40, NKFFT=(12, 12, 12))
    jdos = run(system, {"x": dynamic.JDOS(Efermi=EFERMI, omega=omega, tetra_delta=True, tetra_delta_average=average)},
               grid=grid)["x"]
    binned = jdos[0].reshape(len(bins) - 1, -1).mean(axis=1)
    assert np.abs(binned - reference).sum() < 0.05 * np.abs(reference).sum()
    assert jdos.sum() * domega == pytest.approx(1 / A ** 3, rel=1e-10 if average else 1e-2)


@pytest.mark.parametrize("average", [False, True])
def test_narrow_window(average):
    """with a range of frequencies narrower than the spread of the band gaps within a cell, the pairs are selected by
    the energies at the corners, not by those at the k-point"""
    system = random_system()
    domega = 0.01
    omega = np.arange(0.005, 60, domega)
    window = slice(1000, 1100)  # 10 to 11 eV
    kwargs = dict(Efermi=EFERMI, tetra_delta=True, tetra_delta_average=average)
    calculators = {"full": dynamic.JDOS(omega=omega, **kwargs), "window": dynamic.JDOS(omega=omega[window], **kwargs)}
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
    calculator = dict(Efermi=EFERMI, omega=omega, tetra_delta=True)
    grid = wberri.Grid(system, NKFFT=4, NKdiv=2)
    serial = run(system, {"x": dynamic.OpticalConductivity(**calculator)}, grid=grid)["x"]
    parallel = run(system, {"x": dynamic.OpticalConductivity(**calculator)}, grid=grid, parallel=True)["x"]
    np.testing.assert_allclose(parallel, serial, rtol=0, atol=1e-10 * np.abs(serial).max())


def test_nkdiv_fermi_levels_temperature():
    """the grid divided into shifted subgrids (NKdiv) gives the same result as the full FFT grid, and several Fermi
    levels at a finite temperature give the same results as separate calculations"""
    system = random_system()
    omega = np.arange(0.005, 40, 0.05)
    kwargs = dict(omega=omega, kBT=0.05, tetra_delta=True)
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
        SDCT_sym(Efermi=EFERMI, omega=np.linspace(0.1, 1, 3), tetra_delta=True)


def test_tetra_delta_average_requires_tetra_delta():
    with pytest.raises(ValueError, match="requires tetra_delta=True"):
        dynamic.JDOS(Efermi=EFERMI, omega=np.linspace(0.1, 1, 3), tetra_delta_average=True)


def test_independent_of_frequency_grid():
    """the values at the frequencies do not depend on the other frequencies: every second point of a twice denser
    grid, or a single frequency, gives the same result. With the averages over the intervals, they do."""
    system = random_system()
    omega = np.arange(0.005, 30, 0.1)
    dense = np.arange(0.005, 30, 0.05)
    assert np.allclose(dense[::2], omega)

    def jdos(w, average=False):
        return run(system, {"x": dynamic.OpticalConductivity(Efermi=EFERMI, omega=w, tetra_delta=True,
                                                             tetra_delta_average=average)}, NK=4)["x"]

    coarse = jdos(omega)
    scale = np.abs(coarse).max()
    np.testing.assert_allclose(jdos(dense)[:, ::2], coarse, rtol=0, atol=1e-12 * scale)
    np.testing.assert_allclose(jdos(omega[100:101]), coarse[:, 100:101], rtol=0, atol=1e-12 * scale)
    assert np.abs(jdos(dense, True)[:, ::2] - jdos(omega, True)).max() > 1e-3 * scale


@pytest.mark.parametrize("dE", [0.97, 1.03])
def test_tetra_omega_intervals(dE):
    """the result at each frequency is the average over the interval between the midpoints to its neighbours"""
    omega = np.array([0.9, 1.0, 1.1, 1.3])
    edges = dynamic.JDOS(Efermi=EFERMI, omega=omega, tetra_delta=True, tetra_delta_average=True).omega_edges
    np.testing.assert_allclose(edges, [0.85, 0.95, 1.05, 1.2, 1.4])
    weights = delta_weights_tetra(edges, np.full((1, 4), dE))[0]
    np.testing.assert_allclose(weights, [0, 10, 0, 0])


@pytest.mark.parametrize("omega, average", [([0.5], True), ([1.0, 0.5, 2.0], False), ([1.0, 1.0], False),
                                            ([0.0, np.nan, 2.0], False), ([0.0, 1.0, np.inf], True)])
def test_tetra_omega_increasing(omega, average):
    with pytest.raises(ValueError, match="omega must be finite and increasing"):
        dynamic.JDOS(Efermi=EFERMI, omega=np.array(omega), tetra_delta=True, tetra_delta_average=average)


def test_single_frequency():
    """without the averages over the intervals, a single frequency is allowed"""
    assert dynamic.JDOS(Efermi=EFERMI, omega=np.array([0.5]), tetra_delta=True).omega_edges is None


if __name__ == "__main__":
    np.savez_compressed(REFERENCE_FILE, **stored_results())
