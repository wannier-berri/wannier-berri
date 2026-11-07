import contextlib
import decimal
import os
import tempfile
import io
import numpy as np
from wannierberri.grid.grid_tetra import GridTetra
from wannierberri.grid.tetrahedron import (weights_tetra, PRINCIPAL_FAR,
                                           _separate_values, _add_principal_3d)
from wannierberri.grid.ttweights import TetraWeights, TriangleWeights
from wannierberri.grid.triangle import (weights_triangle, _add_principal_2d, _mean_log, occupied_triangles,
                                        PRINCIPAL_FAR as PRINCIPAL_FAR_2D)
from wannierberri.grid.tetratriangle import delta_tetra
from scipy.integrate import quad
from wannierberri.grid import Grid
from wannierberri.data_K import get_data_k_class_from_system
import pytest
from pytest import approx

from wannierberri.run_grid import run


def weights_tetra_reactive(omega, E):
    res = np.zeros(omega.shape[0])
    _add_principal_3d(res, omega, E)
    return res


def weights_triangle_reactive(omega, E):
    res = np.zeros(omega.shape[0])
    _add_principal_2d(res, omega, E)
    return res


def check_tetra_derivatives(E, Efermi=np.linspace(-0.6, 0.6, 100001), acc1=1e-5, acc2=1e-3, acc3=1e-2, plot=False):
    E = E - 0.5
    # Ecorner = E[1:].reshape(2, 2, 2)
    E = E[:4]

    occ_sea = weights_tetra(Efermi, E[0], E[1], E[2], E[3], der=0)
    occ_der1 = weights_tetra(Efermi, E[0], E[1], E[2], E[3], der=1)
    occ_der2 = weights_tetra(Efermi, E[0], E[1], E[2], E[3], der=2)
    occ_der3 = weights_tetra(Efermi, E[0], E[1], E[2], E[3], der=3)
    norm1 = np.abs(occ_der1).max()
    norm2 = np.abs(occ_der2).max()
    norm3 = np.abs(occ_der3).max()
    occ_der1 /= norm1
    occ_der2 /= norm2
    occ_der3 /= norm3

    e1, e2, e3, e4 = sorted(E)
    C23 = (e3 + e4 - e1 - e2) / (3 * (e4 - e2) * (e3 - e2))
    # Ecenter = e2+1/(3*C23)
    Ecenter = (e3 * e4 - e1 * e2) / (e3 + e4 - e1 - e2)
    print(f"e1={e1}, e2={e2}, e3={e3}, e4={e4}, C23={C23}, Ecenter={Ecenter}")

    occ_der1_fd = occ_sea * 0
    occ_der1_fd[1:-1] = (occ_sea[2:] - occ_sea[:-2]) / (Efermi[2] - Efermi[0]) / norm1
    occ_der2_fd = occ_sea * 0
    occ_der2_fd[2:-2] = (occ_sea[4:] + occ_sea[:-4] - 2 * occ_sea[2:-2]) / (Efermi[2] - Efermi[0])**2 / norm2
    occ_der3_fd = occ_sea * 0
    occ_der3_fd[3:-3] = (occ_sea[6:] - occ_sea[:-6] - 3 * (occ_sea[4:-2] - occ_sea[2:-4])) / (Efermi[2] - Efermi[0])**3 / norm3

    for e in E:
        idx = np.where(abs(Efermi - e) < 0.001)[0]
        occ_der3_fd[idx] = 0
        occ_der3[idx] = 0

    print(norm1, max(abs(occ_der1 - occ_der1_fd)), norm2, max(abs(occ_der2 - occ_der2_fd)), norm3, max(abs(occ_der3 - occ_der3_fd)))
    error = None
    try:
        assert (occ_der1_fd == approx(occ_der1, abs=acc1)), (
            "finite-diff first derivative of weights_tetra did not match he analytic expression by"
            f"{max(abs(occ_der1 - occ_der1_fd))} > {acc1}, the energies were {E}")
        assert (occ_der2_fd == approx(occ_der2, abs=acc2)), (
            "finite-diff second derivative of weights_tetra did not match he analytic expression by" +
            f"{max(abs(occ_der2 - occ_der2_fd))} > {acc2}, the energies were {E}")
        assert (occ_der3_fd == approx(occ_der3, abs=acc3)), (
            "finite-diff third derivative of weights_tetra did not match he analytic expression by" +
            f"{max(abs(occ_der3 - occ_der3_fd))} > {acc3}, the energies were {E}")
    except Exception as err:
        plot = True
        error = err

    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 1, figsize=(15, 6), sharex=True, sharey=False)
        axes[0].plot(Efermi, occ_sea, c='blue')
        axes[1].scatter(Efermi, occ_der1_fd, c='green')
        axes[2].scatter(Efermi, occ_der2_fd, c='red')
        axes[3].scatter(Efermi, occ_der3_fd, c='purple')
        axes[1].plot(Efermi, occ_der1, c='yellow')
        axes[2].plot(Efermi, occ_der2, c='cyan')
        axes[3].plot(Efermi, occ_der3, c='magenta')
        for ax in axes:
            ax.axvline(Ecenter, c='orange', linestyle='--')
            for x in E:
                ax.axvline(x, c='blue')
        axes[0].set_title("weights_tetra and its derivatives")
        axes[0].set_ylabel("weight")
        axes[1].set_ylabel("first derivative")
        axes[2].set_ylabel("second derivative")
        axes[3].set_ylabel("third derivative")
        axes[3].set_xlabel("Energy (eV)")
        plt.tight_layout()
        plt.xlim(-0.6, 0.6)
        plt.savefig(f"tetra_weight_{E[0]}.png")
        plt.close()

    if error is not None:
        raise error


def test_tetra_derivatives():
    E = np.array(
        [
            [0.2984, 0.4595, 0.5829, 0.7502, 0.7196, 0.9911, 0.7955, 0.1899, 0.3873],
            [0.1452, 0.0907, 0.4754, 0.4508, 0.3953, 0.459, 0.3443, 0.5388, 0.559],
            [0.339, 0.6026, 0.3444, 0.0816, 0.5054, 0.4948, 0.9665, 0.0043, 0.079],
            [0.8992, 0.8343, 0.2925, 0.6543, 0.9982, 0.796, 0.7424, 0.3358, 0.0717],
            [0.214, 0.9312, 0.8754, 0.137, 0.6876, 0.9805, 0.8198, 0.7867, 0.8013],
            [0.6402, 0.6796, 0.5923, 0.3457, 0.2366, 0.629, 0.5247, 0.4212, 0.3966],
            [0.9593, 0.2697, 0.2101, 0.2543, 0.5695, 0.1169, 0.1485, 0.6536, 0.0675],
            [0.3641, 0.1229, 0.7018, 0.0084, 0.5121, 0.0491, 0.2748, 0.0281, 0.9587],
            [0.1439, 0.5332, 0.9988, 0.4064, 0.1984, 0.4195, 0.9107, 0.0021, 0.6467],
            [0.7869, 0.3651, 0.6513, 0.8564, 0.216, 0.6372, 0.8625, 0.5177, 0.247]
        ])
    for e in E:
        check_tetra_derivatives(E=e)


data1 = (5.7337, 5.73353853088151321771, 5.73363955524649249185, 5.73396963801047121478, 5.74032502121022680797, -0.01272583, 0.01325275)
# the last value of data2 was 0.220233965; the exact value for these inputs as floating-point numbers (from the
# formulas evaluated with 50 digits) is 0.220234009710982
data2 = (5.933275, 5.93314105, 5.93327453, 5.93327816, 5.93373734, 0.5, 0.220234009710982)
data2b = (5.933274, 5.93314105, 5.93327453, 5.93327816, 5.93373734, 0.2265625, 0.21533874)
data2c = (5.933200, 5.93314105, 5.93327453, 5.93327816, 5.93373734, 0.02734375, 0.01877191)
data2d = (5.933, 5.93314105, 5.93327453, 5.93327816, 5.93373734, 0.0, 0.0)
data2e = (5.934, 5.93314105, 5.93327453, 5.93327816, 5.93373734, 1.0, 1.0)
data3 = (0.34253665432, 0.121245, 0.2356431, 0.51254, 0.614651, 0.38891045, 0.38891045)


@pytest.mark.parametrize("data", [data1, data2, data2b, data2c, data2d, data2e, data3])
def test_tetra_accurate(data):
    """testing some 'magic' values, for which the old implementation gives errors
        aso some normal inpyuts are checked
        data is (ef,e1,e2,e3,e4,w_old,w_acc)
    """
    ef, e1, e2, e3, e4 = data[:5]
    ef = np.array([ef])
    # weight_old = weights_tetra(ef, e1, e2, e3, e4, der=0, accurate=False)
    weight_acc = weights_tetra(ef, e1, e2, e3, e4, der=0)
    print("weights (acc):", weight_acc)
    # assert weight_old[0] == approx(data[5], abs=1e-8)
    assert weight_acc[0] == approx(data[6], abs=1e-8)


@pytest.mark.parametrize("kpoint_type", ["tetra", "parallel"])
def test_tetra_corners_k(kpoint_type, system_kp_mass_aniso_0):
    check_tetra_corners(system=system_kp_mass_aniso_0, kpoint_type=kpoint_type)


@pytest.mark.parametrize("kpoint_type", ["tetra", "parallel"])
def test_tetra_corners_R(kpoint_type, system_GaAs_W90_JM):
    check_tetra_corners(system=system_GaAs_W90_JM, kpoint_type=kpoint_type)


@pytest.mark.parametrize("kpoint_type", ["tetra", "parallel"])
def test_tetra_corners_soc(kpoint_type, system_Fe_gpaw_soc_z):
    check_tetra_corners(system=system_Fe_gpaw_soc_z, kpoint_type=kpoint_type)


def check_tetra_corners(system, kpoint_type):
    k = np.array([0.123, 0.234, 0.345])
    if kpoint_type == "parallel":
        grid = Grid(system=system, NK=(8, 8, 8), NKFFT=(4, 4, 4))
    elif kpoint_type == "tetra":
        grid = GridTetra(system, length=20, NKFFT=(2, 2, 2))
    kpoint = grid.get_K_list(use_symmetry=False)[3]

    data_k_class = get_data_k_class_from_system(system)
    data_k = data_k_class(system=system, grid=grid, dK=k, Kpoint=kpoint)
    if kpoint_type == "parallel":
        E_corners = data_k.E_K_corners_parallel()
        E_corners_test = data_k.E_K_corners_parallel_test()
    elif kpoint_type == "tetra":
        E_corners = data_k.E_K_corners_tetra()
        E_corners_test = data_k.E_K_corners_tetra_test()
    assert E_corners.shape == E_corners_test.shape, f"for kpoint type {kpoint_type}, got shape {E_corners.shape} vs {E_corners_test.shape}"
    assert E_corners == approx(E_corners_test, abs=1e-8), f"for kpoint type {kpoint_type}, got \n{E_corners} vs {E_corners_test}"


def test_tetra_reactive():
    nomegap = 1000
    nomega_cut = nomegap // 2
    omega = np.linspace(-2, 2, 2 * nomegap + 1)
    rng = np.random.default_rng(0)
    for i in range(10):
        E = rng.random((4,)) - 0.5
        while np.diff(np.sort(E)).min() < 0.02:  # the numerical convolution does not resolve nearly equal values
            E = rng.random((4,)) - 0.5
        tetra_absorptive = weights_tetra(omega, *E, der=1)
        tetra_reactive = weights_tetra_reactive(omega, E)
        assert tetra_absorptive.shape == omega.shape
        assert tetra_reactive.shape == omega.shape
        inv_omega = np.zeros_like(omega)
        select = abs(omega) >= 1e-3
        inv_omega[select] = 1 / omega[select]
        reactive_numerical = np.zeros_like(omega)
        for j in range(-nomegap, nomegap + 1):
            if j > 0:
                reactive_numerical[nomegap + j] = np.sum(tetra_absorptive[j:] * inv_omega[:-j])
            elif j < 0:
                reactive_numerical[nomegap + j] = np.sum(tetra_absorptive[:j] * inv_omega[-j:])
            else:
                reactive_numerical[nomegap + j] = np.sum(tetra_absorptive * inv_omega)
        reactive_numerical *= (omega[1] - omega[0])
        reactive_numerical = reactive_numerical[nomega_cut:-nomega_cut]
        tetra_reactive = tetra_reactive[nomega_cut:-nomega_cut]
        atol = np.max(np.abs(tetra_reactive)) * 0.03
        assert reactive_numerical == approx(tetra_reactive, abs=atol), f"reactive does not match for E={E}, diff={np.max(np.abs(reactive_numerical - tetra_reactive))}"


def test_tetra_reactive_asymptotic():
    E = np.random.random((4,)) - 0.5
    omega1 = np.linspace(1000, 1010, 11)
    omega2 = np.linspace(-1010, -1000, 11)
    tetra_reactive1 = weights_tetra_reactive(omega1, E)
    tetra_reactive2 = weights_tetra_reactive(omega2, E)
    assert tetra_reactive1 == approx(-1 / omega1, rel=1e-3)
    assert tetra_reactive2 == approx(-1 / omega2, rel=1e-3)


@pytest.mark.parametrize("kind", ["single", "parallelepiped"])
def test_dynamic_weights_reactive_pairs(kind):
    """each pair of band groups gets the average of the principal parts of its own pairs of bands"""
    rng = np.random.default_rng(1)
    nk, nb = 3, 4
    eCenter = rng.random((nk, nb)) * 2
    eCorners = rng.random((nk, 4, nb) if kind == "single" else (nk, 2, 2, 2, nb)) * 2
    tetra = TetraWeights(eCenter=eCenter, eCorners=eCorners, kind=kind)
    omega = np.linspace(-3, 3, 31)
    pairs = [((0, 1), (2, 3)), ((2, 3), (0, 1)), ((0, 0), (0, 0)), ((0, 2), (3, 4)), ((3, 4), (0, 2))]
    for ik in range(nk):
        result = tetra.dynamic_weights(ik, pairs, omega, type="reactive")
        assert result.shape == (len(pairs), len(omega))
        for ipair, ((i1, i2), (j1, j2)) in enumerate(pairs):
            expected = np.zeros_like(omega)
            for vertices in tetra.all_tetrahedra_vertices:
                for i in range(i1, i2):
                    for j in range(j1, j2):
                        expected += weights_tetra_reactive(omega, (vertices[ik][:, i] - vertices[ik][:, j]))
            expected *= tetra.factor_num_tetra / max((i2 - i1) * (j2 - j1), 1)  # an empty pair gives 0
            assert result[ipair] == approx(expected, rel=1e-10, abs=1e-12), f"pair {pairs[ipair]} at ik={ik}"


@pytest.mark.parametrize("dE", [[0.3, -1.2, 2.0, 0.7], [0.0, 0.0, 1.0, 2.0], [0.0, 1.0, 1.0, 2.0], [0.0, 1.0, 2.0, 2.0],
                                [0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0], [0.0, 1e-9, 1.0, 2.0]])
def test_principal_values(dE):
    """the principal part against the numerical principal-value integral of the density of dE (`delta_tetra`), at
    frequencies inside and outside the range of dE and far from it; at the corner values, it is continuous"""
    dE = np.array(dE)
    nodes = np.unique(dE)
    omega = np.concatenate([np.linspace(nodes[0] - 3, nodes[-1] + 3, 37) + 0.0123, [-200.0, 300.0]])
    values = weights_tetra_reactive(omega, dE)

    def density(x):
        return delta_tetra(np.array([x]), *dE)[0]

    for w, value in zip(omega, values):
        reference = 0
        for a, b in zip(nodes[:-1], nodes[1:]):
            if a < w < b:
                reference += quad(density, a, b, weight="cauchy", wvar=w, limit=200)[0]
            else:
                reference += quad(lambda x: density(x) / (x - w), a, b, limit=200)[0]
        assert value == pytest.approx(reference, rel=1e-7, abs=1e-9)
    for node in nodes:
        if np.sum(dE == node) < 3:  # continuous unless three corners have the same value
            near = weights_tetra_reactive(node + np.array([-1e-7, 0, 1e-7]), dE)
            assert np.all(np.isfinite(near))
            assert np.abs(np.diff(near)).max() < 1e-4 * (1 + np.abs(near).max())


@pytest.mark.parametrize("values, separated", [
    ([0, 0, 10, 20], [-0.5, 0.5, 10, 20]), ([0, 10, 10, 20], [0, 9.5, 10.5, 20]), ([0, 0, 0, 10], [-1, 0, 1, 10]),
    ([0, 0, 10, 10], [-0.5, 0.5, 9.5, 10.5]), ([0, 0, 0, 0], [-1.5, -0.5, 0.5, 1.5]),
    ([0, 0.3, 0.6, 10], [-0.7, 0.3, 1.3, 10]), ([0, 0, 1, 10], [-2 / 3, 1 / 3, 4 / 3, 10]),
    ([0, 0.3, 10, 20], [-0.35, 0.65, 10, 20]), ([0, 3, 10, 20], [0, 3, 10, 20]), ([0, 0, 1, 2], [-0.75, 0.25, 1.25, 2.25])])
def test_separate_values(values, separated):
    """with the gap 1: values closer than the gap are spread symmetrically around their mean, groups that come closer
    than the gap are merged, and the other values are unchanged"""
    for function in (_separate_values, _separate_values.py_func):
        e = np.array(values, dtype=float)
        function(e, 1.0)
        np.testing.assert_allclose(e, separated, rtol=0, atol=1e-14)
        assert e.sum() == pytest.approx(sum(values), abs=1e-13)


@pytest.mark.parametrize("dE", [[1.0, 1.0, 1.0, 1.0], [3.0, 3.0 + 1e-6, 3.0 + 2e-6, 3.0 + 4e-6]])
def test_principal_values_flat(dE):
    """for (nearly) equal values, 1/(dE - omega), and zero at omega equal to all of them"""
    omega = np.array([-2.0, 0.5, 2.9, 3.1, 10.0])
    np.testing.assert_allclose(weights_tetra_reactive(omega, dE), 1 / (np.mean(dE) - omega), rtol=1e-9)
    assert weights_tetra_reactive(np.array([1.0]), [1.0, 1.0, 1.0, 1.0])[0] == 0


@pytest.mark.parametrize("offset", [1.0, 10.0])
def test_principal_values_narrow(offset):
    """a small spread (3e-8) far from zero, also with omega at the corner values: the same as near zero, and finite"""
    e = np.array([0, 1e-8, 2e-8, 3e-8])
    omega = np.array([0, 1e-8, 1.5e-8, 3e-8, 1e-6, -1e-6])
    near_zero = weights_tetra_reactive(omega, e)
    far = weights_tetra_reactive(omega + offset, e + offset)
    assert np.all(np.isfinite(far))
    np.testing.assert_allclose(far, near_zero, rtol=0, atol=1e-6 * np.abs(near_zero).max())  # 0 at the middle


@pytest.mark.parametrize("dE", [[0.3, -1.2, 2.0, 0.7], [0.0, 1.0, 1.5, 2.0]])
def test_principal_expansion_continuity(dE):
    """continuous where the expansion far from the values (at 1/PRINCIPAL_FAR times their spread from their mean)
    takes over from the integral over the intervals"""
    dE = np.array(dE)
    mean, spread = dE.mean(), np.ptp(dE)
    for side in (-1, 1):
        omega = mean + side * spread / PRINCIPAL_FAR * np.array([1 - 1e-9, 1 + 1e-9])
        inside, outside = weights_tetra_reactive(omega, dE)
        assert outside == pytest.approx(inside, rel=1e-7)


@pytest.mark.parametrize("dE", [[0.3, -1.2, 2.0, 0.7], [0.0, 0.0, 0.1, 2.0], [0.0, 1.9, 2.0, 2.0], [0.0, 1.0, 1.0, 1.0]])
def test_principal_expansion_accuracy(dE):
    """the expansion far from the values, just beyond where it takes over, on both sides, against the numerical
    principal-value integral of the density (also for asymmetric and repeated corner values)"""
    dE = np.array(dE)
    nodes = np.unique(dE)
    mean, spread = dE.mean(), np.ptp(dE)

    def density(x):
        return delta_tetra(np.array([x]), *dE)[0]

    for side in (-1, 1):
        w = mean + side * spread / PRINCIPAL_FAR * (1 + 1e-6)
        reference = sum(quad(lambda x: density(x) / (x - w), a, b, epsabs=1e-14, epsrel=1e-12, limit=200)[0]
                        for a, b in zip(nodes[:-1], nodes[1:]))
        assert weights_tetra_reactive(np.array([w]), dE)[0] == pytest.approx(reference, rel=1e-9)


def test_principal_values_near_corner():
    """omega within a few units of rounding of an interior corner value: continuous (the logarithms of the two
    intervals at the corner cancel)"""
    e = (0.0, 0.13, 0.4, 1.0)
    at = weights_tetra_reactive(np.array([0.4]), e)[0]
    for d in (1e-16, 1e-15, 1e-14, 1e-13, 1e-12):
        near = weights_tetra_reactive(np.array([0.4 - d, 0.4 + d]), e)
        np.testing.assert_allclose(near, at, rtol=1e-8)


def test_principal_values_at_corner():
    """omega exactly at a corner value of a cluster of nearly equal values, and corner values given as integers: the
    same as just beside it"""
    for e, w in (((0., 1., 1. + 1e-6, 1. + 2e-6), 1.), ((0., 0.3, 0.3 + 1e-9, 1.), 0.3), ((0, 0, 1, 2), 1.)):
        values = weights_tetra_reactive(np.array([w - 1e-15, w, w + 1e-15]), e)
        np.testing.assert_allclose(values, values[0], rtol=1e-9)


@pytest.mark.parametrize("kind", ["single", "parallelepiped"])
def test_dynamic_weights_kramers_kronig(kind):
    """the reactive weights are the principal-value integrals of the delta weights over the frequency, for pairs of
    band groups: computed from the averages of the delta weights over fine intervals, with 1/(x - omega) integrated
    exactly over each interval"""
    rng = np.random.default_rng(2)
    nk, nb = 2, 4
    eCenter = rng.random((nk, nb)) * 2
    eCorners = rng.random((nk, 4, nb) if kind == "single" else (nk, 2, 2, 2, nb)) * 2
    tetra = TetraWeights(eCenter=eCenter, eCorners=eCorners, kind=kind)
    pairs = [((0, 1), (2, 3)), ((2, 4), (0, 2)), ((1, 2), (3, 4))]
    edges = np.linspace(-3, 3, 60001)
    omega = np.array([-2.5, -1.0033, -0.31, 0.0, 0.47, 1.2007, 2.6]) + (edges[1] - edges[0]) / 2  # middles of intervals
    for ik in range(nk):
        delta = tetra.dynamic_weights(ik, pairs, (edges[1:] + edges[:-1]) / 2, edges=edges, type="delta")
        reactive = tetra.dynamic_weights(ik, pairs, omega, type="reactive")
        for w, values in zip(omega, reactive.T):
            with np.errstate(divide="ignore"):
                kernel = np.log(np.abs((edges[1:] - w) / (edges[:-1] - w)))
            kernel[~np.isfinite(kernel)] = 0
            np.testing.assert_allclose(values, delta @ kernel, rtol=1e-3, atol=1e-4)


def test_weights_cached_by_value():
    """the weights for the Fermi levels are computed once for equal arrays (each calculator has its own copy), again
    for different ones, and not mixed up when the caller changes its array afterwards"""
    rng = np.random.default_rng(4)
    nk, nb = 2, 3
    eCorners = rng.random((nk, 2, 2, 2, nb))
    tetra = TetraWeights(eCenter=rng.random((nk, nb)), eCorners=eCorners, kind="parallelepiped")
    computed = []
    compute = tetra.weight_1k1b_priv

    def counting(eFermi, ik, ib, der):
        computed.append((tuple(eFermi), ik, ib))
        return compute(eFermi, ik, ib, der)

    tetra.weight_1k1b_priv = counting
    efermi = np.linspace(0.2, 0.8, 7)
    original = efermi.copy()
    first = tetra.weights_all_band_groups(efermi, der=1)
    ncomputed = len(computed)
    assert ncomputed > 0
    second = tetra.weights_all_band_groups(efermi.copy(), der=1)
    assert len(computed) == ncomputed  # nothing computed again
    for w1, w2 in zip(first, second):
        assert w1.keys() == w2.keys()
        for key in w1:
            np.testing.assert_array_equal(w2[key], w1[key])
    shifted = efermi + 0.05
    other = tetra.weights_all_band_groups(shifted, der=1)
    assert len(computed) > ncomputed and all(e == tuple(shifted) for e, _, _ in computed[ncomputed:])
    ncomputed = len(computed)
    efermi[:] = shifted  # the caller changes its array: the weights are those for the new values
    again = tetra.weights_all_band_groups(efermi, der=1)
    assert len(computed) == ncomputed
    # the same ends, other values between them
    between = np.concatenate([shifted[:1], np.sort(rng.uniform(shifted[0], shifted[-1], 5)), shifted[-1:]])
    third = tetra.weights_all_band_groups(between, der=1)
    assert len(tetra.eFermi_stored) == 3
    # one value changed by the smallest step: computed again
    nudged = shifted.copy()
    nudged[3] = np.nextafter(nudged[3], np.inf)
    ncomputed = len(computed)
    fourth = tetra.weights_all_band_groups(nudged, der=1)
    assert len(computed) > ncomputed and len(tetra.eFermi_stored) == len(tetra.weights) == 4
    # a list with the same values: nothing computed again
    ncomputed = len(computed)
    tetra.weights_all_band_groups(list(original), der=1)
    assert len(computed) == ncomputed
    # against a new instance, without the cache
    tetra.weight_1k1b_priv = compute
    for array, results in ((shifted, other), (shifted, again), (between, third), (nudged, fourth), (original, first)):
        fresh = TetraWeights(eCenter=tetra.eCenter, eCorners=eCorners, kind="parallelepiped")
        for w1, w2 in zip(fresh.weights_all_band_groups(array, der=1), results):
            assert w1.keys() == w2.keys()
            for key in w1:
                np.testing.assert_array_equal(w2[key], w1[key])


@pytest.mark.parametrize("dtype, values", [(int, [1, 2, 3]), (np.float32, [1.25, 2.5, 3.75])])
@pytest.mark.parametrize("first", ["other", "float64"])
def test_weights_cached_other_dtype(dtype, values, first):
    """Fermi levels given as integers or float32 (exact in both): the weights are computed in float64 (not truncated
    or rounded), and the same as for the equal float64 values, in either order. A failed conversion leaves the cache
    unchanged"""
    rng = np.random.default_rng(5)
    eCenter, eCorners = rng.random((2, 3)) * 4, rng.random((2, 2, 2, 2, 3)) * 4
    efermi = np.array(values, dtype=dtype)
    reference = TetraWeights(eCenter=eCenter, eCorners=eCorners, kind="parallelepiped")
    tetra = TetraWeights(eCenter=eCenter, eCorners=eCorners, kind="parallelepiped")
    for der in (0, 1, 2, -1):
        expected = reference.weights_all_band_groups(efermi.astype(np.float64), der=der)
        arrays = [efermi, efermi.astype(np.float64)]
        if first == "float64":
            arrays.reverse()
        for array in arrays:
            for w1, w2 in zip(expected, tetra.weights_all_band_groups(array, der=der)):
                assert w1.keys() == w2.keys()
                for key in w1:
                    np.testing.assert_allclose(w2[key], w1[key], rtol=0, atol=1e-14)
    assert len(tetra.eFermi_stored) == 1
    with pytest.raises(ValueError):
        tetra.weights_all_band_groups(np.array(["a", "b"]), der=1)
    assert len(tetra.eFermi_stored) == len(tetra.weights) == 1


def test_tetra_2D(system_Haldane_PythTB):
    from wannierberri.calculators.static import DOS, CumDOS
    system = system_Haldane_PythTB
    Efermi = np.linspace(-5, 5, 1000)
    acc = 0.01
    calculators = {
        "dos": DOS(Efermi=Efermi, tetra=True),
        "cdos": CumDOS(Efermi=Efermi, tetra=True),
    }
    grid = Grid(system, NK=60)


    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        kwargs = dict(system=system, grid=grid, calculators=calculators, fout_name=os.path.join(tmp, "x"))
        results_tetra = run(**kwargs, parameters_K={"use_tetra_in_low_D": True})  # , use_irred_kpt=False, symmetrize=False)  # do not use symmetry because tetrahedron is not very symmetric
        results_triangle = run(**kwargs, parameters_K={"use_tetra_in_low_D": False})  # , use_irred_kpt=False, symmetrize=False)  # do not use symmetry because tetrahedron is not very symmetric
    for key, acc in [("dos", 0.05), ("cdos", 0.01)]:
        res_tetra = results_tetra.results[key].data
        res_triangle = results_triangle.results[key].data
        maxval = max(np.abs(res_tetra).max(), np.abs(res_triangle).max())
        diff = np.abs(res_tetra - res_triangle).max()
        assert diff <= maxval * acc, f"results for {key} differ by {diff} > {maxval * acc} between tetrahedron and triangle methods"


def test_triangle_occupation_metal_jdos(system_Haldane_PythTB):
    """2D: JDOS of the Haldane model with the Fermi level in the upper band, with the triangles, against a histogram of
    the band gaps weighted by the occupations on a 400^2 grid: closer with the occupations within the triangles than
    with those at the k-points"""
    from wannierberri.calculators.dynamic import JDOS
    system = system_Haldane_PythTB
    efermi = 1.0
    n = 400
    k = np.stack(np.meshgrid(np.arange(n) / n, np.arange(n) / n, [0]), axis=-1).reshape(-1, 3)
    E = np.linalg.eigvalsh(np.einsum("kr,rab->kab", np.exp(2j * np.pi * k @ system.rvec.iRvec.T), system.Ham_R))
    bins = np.arange(0.0, 8.0 + 1e-9, 0.25)
    occupation = (E[:, 0] < efermi).astype(float) - (E[:, 1] < efermi)
    reference = np.histogram(E[:, 1] - E[:, 0], bins=bins, weights=occupation)[0] / len(E) / 0.25 / system.cell_volume
    domega = 0.005
    kwargs = dict(Efermi=[efermi], omega=np.arange(bins[0], bins[-1], domega) + domega / 2, tetra_delta=True)
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        result = run(system=system, grid=Grid(system, NK=(24, 24, 1)), fout_name=os.path.join(tmp, "x"),
                     calculators={"k": JDOS(**kwargs), "cells": JDOS(tetra_occupation=True, **kwargs)}).results

    def error(key):
        jdos = result[key].data[0].reshape(len(bins) - 1, -1).mean(axis=1)
        return np.abs(jdos - reference).sum() / np.abs(reference).sum()

    assert error("cells") < 0.025
    assert error("cells") < 0.6 * error("k")


def triangle_principal_reference(e, w):
    """the average of P 1/(dE - omega) over a triangle, by the numerical principal-value integral of the density of dE"""
    e = np.sort(np.array(e, dtype=float))
    result = 0
    for a, b in zip(e[:-1], e[1:]):
        if b > a:
            def density(x):
                return weights_triangle(np.array([x]), *e, der=1)[0]
            if a < w < b:
                result += quad(density, a, b, weight="cauchy", wvar=w, limit=200)[0]
            else:
                result += quad(lambda x: density(x) / (x - w), a, b, limit=200)[0]
    return result


@pytest.mark.parametrize("e", [[0.3, -1.2, 2.0], [0.0, 1.0, 3.0], [-1.0, 0.5, 0.5 + 1e-3], [2.0, 2.0 + 1e-4, 5.0]])
def test_principal_values_2d(e):
    """the principal part over a triangle against the numerical principal-value integral of the density, at
    frequencies within and beyond the range of the values and on both sides of where the expansion far from the values
    takes over; at the corner values, it is continuous"""
    e = np.array(e)
    mean, spread = e.mean(), np.ptp(e)
    omega = np.concatenate([np.linspace(e.min() - 1, e.max() + 1, 17) + 0.0123,
                            mean + spread / PRINCIPAL_FAR_2D * np.array([-1 - 1e-9, -1 + 1e-9, 1 - 1e-9, 1 + 1e-9])])
    for function in (_add_principal_2d, _add_principal_2d.py_func):
        res = np.zeros(len(omega))
        function(res, omega, e)
        for w, value in zip(omega, res):
            assert value == pytest.approx(triangle_principal_reference(e, w), rel=1e-7, abs=1e-9)
    step = 1e-9 * np.diff(np.sort(e)).min()  # small compared with the distance to the other values
    for node in e:
        near = weights_triangle_reactive(node + np.array([-step, 0, step]), e)
        assert np.abs(np.diff(near)).max() < 1e-6 * (1 + np.abs(near).max())


@pytest.mark.parametrize("w", [-1.0, 0.25, 0.5, 0.999, 1.5, 3.0])
def test_principal_values_2d_equal(w):
    """equal corner values, without separating them: the closed forms for two equal values (the density 2 (1 - x)
    on [0, 1], or 2 x), also given as integers, and for three equal values (1/(dE - omega), and zero at omega equal
    to them)"""
    lower = 2 * ((1 - w) * np.log(np.abs((1 - w) / w)) - 1)  # P int_0^1 2 (1 - x) / (x - w) dx
    upper = 2 * (1 + w * np.log(np.abs((1 - w) / w)))  # P int_0^1 2 x / (x - w) dx
    for e, expected in (([0.0, 0.0, 1.0], lower), ((0, 1, 0), lower), ([1.0, 0.0, 1.0], upper),
                        ([7.0, 7.0, 7.0], 1 / (7 - w))):
        assert weights_triangle_reactive(np.array([w]), np.array(e))[0] == pytest.approx(expected, rel=1e-12)
    assert weights_triangle_reactive(np.array([7.0]), np.array([7.0, 7.0, 7.0]))[0] == 0


def test_principal_values_2d_at_corner():
    """omega exactly at the corner values: 2 log 2, 0 and -2 log 2 for the values 0, 1, 2 (the density is continuous
    there). At two equal values, where the principal part diverges logarithmically: the closed forms of
    test_principal_values_2d_equal (shifted and scaled) at the neighbouring floating-point numbers and below, and at
    the value itself the closed form at a distance of the machine epsilon times the scale of the values"""
    e = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(weights_triangle_reactive(e, e), [2 * np.log(2), 0, -2 * np.log(2)], rtol=1e-14,
                               atol=1e-15)

    def lower(w):  # without overflow for the smallest w
        return 2 * ((1 - w) * (np.log(np.abs(1 - w)) - np.log(np.abs(w))) - 1)

    # (values, node, closed form in the distance of omega from the node; for the upper values, the closed form of
    # test_principal_values_2d_equal with 1 - w = -d / 5 without cancellation)
    for e, node, expected in (([0.0, 0.0, 1.0], 0.0, lower), ([5.0, 5.0, 6.0], 5.0, lower),
                              ([-3.0, 2.0, 2.0], 2.0,
                               lambda d: 2 * (1 + (1 + d / 5) * np.log(np.abs(d / 5 / (1 + d / 5)))) / 5)):
        e = np.array(e)
        # the neighbouring floating-point numbers, and distances down to 1e-20 (resolvable near zero)
        omega = np.array([np.nextafter(node, -np.inf), np.nextafter(node, np.inf), node + 1e-20, node - 1e-20,
                          node + 1e-10, node - 1e-6])
        omega = omega[omega != node]
        for function in (_add_principal_2d, _add_principal_2d.py_func):
            res = np.zeros(len(omega) + 1)
            function(res, np.append(omega, node), e)
            np.testing.assert_allclose(res[:-1], expected(omega - node), rtol=1e-8)
            at_zero = 2.2e-16 * (np.ptp(e) + np.abs(e).max())  # the distance taken at the node
            assert res[-1] == pytest.approx(expected(at_zero), rel=1e-8)


def test_mean_log_series():
    """the average of log|t| over a short interval far from zero, where the series is used (half width over the
    distance from zero below 0.1) and around the switch, against the formula in 50-digit decimal arithmetic"""
    for m in (1.0, -3.7, 1e3, 2e-5):
        for r in (1e-4, 0.011, 0.05, 0.0999, 0.1, 0.1001, 0.3):
            a, b = m - r * abs(m), m + r * abs(m)
            with decimal.localcontext() as context:
                context.prec = 50
                A, B = decimal.Decimal(a), decimal.Decimal(b)
                expected = float((B * abs(B).ln() - A * abs(A).ln()) / (B - A) - 1)
            for function in (_mean_log, _mean_log.py_func):
                assert function(a, b, 1.0) == pytest.approx(expected, rel=1e-14, abs=1e-15)


@pytest.mark.parametrize("offset", [1.0, 10.0])
def test_principal_values_2d_narrow(offset):
    """a small spread (2e-8) far from zero, also with omega at the corner values: the same as near zero, and finite"""
    e = np.array([0, 1e-8, 2e-8])
    omega = np.array([0, 1e-8, 1.5e-8, 2e-8, 1e-6, -1e-6])
    near_zero = weights_triangle_reactive(omega, e)
    far = weights_triangle_reactive(omega + offset, e + offset)
    assert np.all(np.isfinite(far))
    np.testing.assert_allclose(far, near_zero, rtol=0, atol=1e-6 * np.abs(near_zero).max())


@pytest.mark.parametrize("energies", [[0.3, -1.2, 2.0], [2.0, -1.2, 0.3], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0],
                                      [1.0, 1.0, 1.0]])
def test_occupied_triangles(energies):
    """the part of a triangle below the Fermi level: the areas are the occupation of `weights_triangle`, the values
    at the corners of the parts are below the Fermi level, and the integrals of the barycentric coordinates over the
    parts are those over the region below the Fermi level, sampled at the centres of the N^2 triangles of a fine
    subdivision; a flat band at the Fermi level is occupied"""
    energies = np.array(energies)
    N = 200
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    lower, upper = i + j < N, i + j < N - 1
    uv = np.concatenate([np.stack([i[lower] + 1 / 3, j[lower] + 1 / 3], axis=-1),
                         np.stack([i[upper] + 2 / 3, j[upper] + 2 / 3], axis=-1)]) / N
    samples = np.column_stack([1 - uv.sum(axis=1), uv])
    values = energies[0] + uv @ (energies[1:] - energies[0])  # exact for a flat band
    for efermi in np.linspace(-2, 3, 51):
        moment = samples[values <= efermi].sum(axis=0) / N ** 2
        for function in (occupied_triangles, occupied_triangles.py_func):
            bary, area = function(energies, efermi)
            np.testing.assert_allclose((area[:, None] * bary.mean(axis=1)).sum(axis=0), moment, rtol=0, atol=2e-4)
            if np.all(energies == efermi):
                assert area.sum() == 1
                continue
            # (weights_triangle separates equal values by 1e-12)
            assert area.sum() == pytest.approx(weights_triangle(np.array([efermi]), *energies, der=0)[0], abs=1e-11)
            for b, a in zip(bary, area):
                if a > 0:
                    assert np.all(b @ energies <= efermi + 1e-12)
                    np.testing.assert_allclose(b.sum(axis=1), 1, rtol=0, atol=1e-14)
    assert occupied_triangles(np.ones(3), 1.0)[1].sum() == 1


@pytest.mark.parametrize("kind", ["0011", "0101"])
def test_triangle_weights_kramers_kronig(kind):
    """for triangles: the reactive weights are the principal-value integrals of the delta weights over the frequency,
    for pairs of band groups (as in test_dynamic_weights_kramers_kronig)"""
    rng = np.random.default_rng(3)
    nk, nb = 2, 4
    eCenter = rng.random((nk, nb)) * 2
    eCorners = rng.random((nk, 2, 2, nb)) * 2
    triangles = TriangleWeights(eCenter=eCenter, eCorners=eCorners, kind=kind)
    pairs = [((0, 1), (2, 3)), ((2, 4), (0, 2)), ((1, 2), (3, 4))]
    edges = np.linspace(-3, 3, 60001)
    omega = np.array([-2.5, -1.0033, -0.31, 0.0, 0.47, 1.2007, 2.6]) + (edges[1] - edges[0]) / 2  # middles of intervals
    for ik in range(nk):
        delta = triangles.dynamic_weights(ik, pairs, (edges[1:] + edges[:-1]) / 2, edges=edges, type="delta")
        reactive = triangles.dynamic_weights(ik, pairs, omega, type="reactive")
        assert np.abs(reactive).max() > 0
        for w, values in zip(omega, reactive.T):
            with np.errstate(divide="ignore"):
                kernel = np.log(np.abs((edges[1:] - w) / (edges[:-1] - w)))
            kernel[~np.isfinite(kernel)] = 0
            np.testing.assert_allclose(values, delta @ kernel, rtol=1e-3, atol=1e-4)
