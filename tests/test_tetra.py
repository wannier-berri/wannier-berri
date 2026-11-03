import contextlib
import os
import tempfile
import io
import numpy as np
from wannierberri.grid.grid_tetra import GridTetra
from wannierberri.grid.tetrahedron import weights_tetra, weights_tetra_reactive, TetraWeights, delta_tetra, PRINCIPAL_FAR
from scipy.integrate import quad
from wannierberri.grid import Grid
from wannierberri.data_K import get_data_k_class_from_system
import pytest
from pytest import approx

from wannierberri.run_grid import run


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
        tetra_reactive = weights_tetra_reactive(omega, *E)
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
    tetra_reactive1 = weights_tetra_reactive(omega1, *E)
    tetra_reactive2 = weights_tetra_reactive(omega2, *E)
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
                        expected += weights_tetra_reactive(omega, *(vertices[ik][:, i] - vertices[ik][:, j]))
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
    values = weights_tetra_reactive(omega, *dE)

    def density(x):
        return delta_tetra(np.array([x]), *dE)[0]

    for w, value in zip(omega, values):
        reference = 0
        for a, b in zip(nodes[:-1], nodes[1:]):
            if a < w < b:
                reference += quad(density, a, b, weight="cauchy", wvar=w, limit=200)[0]
            else:
                reference += quad(lambda x: density(x) / (x - w), a, b, limit=200)[0]
        assert value == pytest.approx(reference, rel=1e-5, abs=1e-6)
    for node in nodes:
        if np.sum(dE == node) < 3:  # continuous unless three corners have the same value
            near = weights_tetra_reactive(node + np.array([-1e-7, 0, 1e-7]), *dE)
            assert np.all(np.isfinite(near))
            assert np.abs(np.diff(near)).max() < 1e-4 * (1 + np.abs(near).max())


@pytest.mark.parametrize("dE", [[1.0, 1.0, 1.0, 1.0], [3.0, 3.0 + 1e-6, 3.0 + 2e-6, 3.0 + 4e-6]])
def test_principal_values_flat(dE):
    """for (nearly) equal values, 1/(dE - omega), and zero at omega equal to all of them"""
    omega = np.array([-2.0, 0.5, 2.9, 3.1, 10.0])
    np.testing.assert_allclose(weights_tetra_reactive(omega, *dE), 1 / (np.mean(dE) - omega), rtol=1e-9)
    assert weights_tetra_reactive(np.array([1.0]), 1.0, 1.0, 1.0, 1.0)[0] == 0


@pytest.mark.parametrize("offset", [1.0, 10.0])
def test_principal_values_narrow(offset):
    """a small spread (3e-8) far from zero, also with omega at the corner values: the same as near zero, and finite"""
    e = np.array([0, 1e-8, 2e-8, 3e-8])
    omega = np.array([0, 1e-8, 1.5e-8, 3e-8, 1e-6, -1e-6])
    near_zero = weights_tetra_reactive(omega, *e)
    far = weights_tetra_reactive(omega + offset, *(e + offset))
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
        inside, outside = weights_tetra_reactive(omega, *dE)
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
        assert weights_tetra_reactive(np.array([w]), *dE)[0] == pytest.approx(reference, rel=1e-9)


def test_principal_values_near_corner():
    """omega within a few units of rounding of an interior corner value: continuous (the logarithms of the two
    intervals at the corner cancel)"""
    e = (0.0, 0.13, 0.4, 1.0)
    at = weights_tetra_reactive(np.array([0.4]), *e)[0]
    for d in (1e-16, 1e-15, 1e-14, 1e-13, 1e-12):
        near = weights_tetra_reactive(np.array([0.4 - d, 0.4 + d]), *e)
        np.testing.assert_allclose(near, at, rtol=1e-8)


def test_principal_values_at_corner():
    """omega exactly at a corner value of a cluster of nearly equal values, and corner values given as integers: the
    same as just beside it"""
    for e, w in (((0., 1., 1. + 1e-6, 1. + 2e-6), 1.), ((0., 0.3, 0.3 + 1e-9, 1.), 0.3), ((0, 0, 1, 2), 1.)):
        values = weights_tetra_reactive(np.array([w - 1e-15, w, w + 1e-15]), *e)
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
        delta = tetra.dynamic_weights(ik, pairs, (edges[1:] + edges[:-1]) / 2, edges, type="delta")
        reactive = tetra.dynamic_weights(ik, pairs, omega, type="reactive")
        for w, values in zip(omega, reactive.T):
            with np.errstate(divide="ignore"):
                kernel = np.log(np.abs((edges[1:] - w) / (edges[:-1] - w)))
            kernel[~np.isfinite(kernel)] = 0
            np.testing.assert_allclose(values, delta @ kernel, rtol=1e-3, atol=1e-4)


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
