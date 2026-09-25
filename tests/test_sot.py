"""Tests for the spin-orbit torque and the torkance, bcc Fe (GPAW) with SOC"""

import copy
import os

import numpy as np
import pytest

import wannierberri as wberri
from wannierberri.calculators.static import TorkanceEven, TorkanceOdd
from wannierberri.formula.covariant import TorqueOmega, TorqueVel
from wannierberri.grid.Kpoint import KpointBZparallel

from .common import OUTPUT_DIR_RUN


def get_datak(system, dK, NKFFT=1):
    """Data_K on the k-points n / NKFFT + dK (reduced coordinates)"""
    grid = wberri.Grid(system=system, NKFFT=NKFFT, NKdiv=1, use_symmetry=False)
    kpoint = KpointBZparallel(K=[0, 0, 0], dK=1. / grid.div, NKFFT=grid.FFT, factor=1., pointgroup=None)
    data_k_class = wberri.data_K.get_data_k_class_from_system(system)
    return data_k_class(system, dK=dK, grid=grid, Kpoint=kpoint, fftlib='numpy')


def band_values(system, k, Formula):
    """formula for every band at the k-point k, and the formula object"""
    data_k = get_datak(system, k)
    formula = Formula(data_k)
    nb = data_k.num_wann
    return np.array([formula.trace(0, [n], [m for m in range(nb) if m != n]) for n in range(nb)]), formula


def test_torque_hellmann_feynman(system_Fe_gpaw_soc_ref):
    """-<T> is the derivative of the band energies with respect to rotating the magnetization"""
    theta, phi, h = np.deg2rad(49), np.deg2rad(33), 1e-5
    k = [0.13, 0.27, 0.41]

    def energies_torque(theta, phi):
        system = copy.deepcopy(system_Fe_gpaw_soc_ref)
        system.set_soc_axis(theta=theta, phi=phi)
        data_k = get_datak(system, k)
        return data_k.E_K[0], np.einsum("nna->na", data_k.Xbar("SOT")[0]).real

    E, T = energies_torque(theta, phi)
    dE_dphi = (energies_torque(theta, phi + h)[0] - energies_torque(theta, phi - h)[0]) / (2 * h)
    dE_dtheta = (energies_torque(theta + h, phi)[0] - energies_torque(theta - h, phi)[0]) / (2 * h)
    assert abs(dE_dphi).max() > 1e-3
    # phi rotates the magnetization about z, theta about e_phi
    assert dE_dphi == pytest.approx(-T[:, 2], abs=1e-7)
    assert dE_dtheta == pytest.approx(-T @ [-np.sin(phi), np.cos(phi), 0], abs=1e-7)


def test_torkance_formulas(system_Fe_gpaw_soc_angle):
    """TorqueOmega and TorqueVel against the explicit sums over bands"""
    k = [0.13, 0.27, 0.41]
    # a generic Berry connection: for this Fe model its contribution to TorqueOmega happens to vanish
    system = copy.deepcopy(system_Fe_gpaw_soc_angle)
    rng = np.random.default_rng(0)
    for system_spin in system.system_up, system.system_down:
        AA = system_spin.get_R_mat("AA")
        system_spin.set_R_mat("AA", 0.1 * (rng.normal(size=AA.shape) + 1j * rng.normal(size=AA.shape)), reset=True, Hermitian=True)
    data_k = get_datak(system, k)
    E, T, V, A = data_k.E_K[0], data_k.Xbar("SOT")[0], data_k.Xbar("Ham", der=1)[0], data_k.Xbar("AA")[0]
    v = V + 1j * (E[:, None] - E[None, :])[:, :, None] * A  # interband velocity including the Berry connection
    dE = E[:, None] - E[None, :]
    np.fill_diagonal(dE, np.inf)
    omega = -2 * np.einsum("nla,lnb,nl->nab", T, v, dE ** -2.).imag
    torque_vel = np.einsum("nna,nnb->nab", T, V).real
    for Formula, expected in [(TorqueOmega, omega), (TorqueVel, torque_vel)]:
        assert band_values(system, k, Formula)[0] == pytest.approx(expected, abs=1e-10)


def test_torkance_symmetry(system_Fe_gpaw_soc_ref):
    """m || z: C2x*T maps k to (-kx, ky, kz) and inversion maps k to -k; the band-resolved torkance
    transforms as declared by transformTR (even: invariant, odd: changes sign) and transformInv (odd)."""
    system = copy.deepcopy(system_Fe_gpaw_soc_ref)
    system.set_soc_axis(theta=0, phi=0)
    k = np.array([0.11, 0.23, 0.07])
    R = np.diag([1, -1, -1])
    B = system.recip_lattice
    k_C2xT = np.linalg.solve(B.T, -R @ (B.T @ k))
    for Formula, sign_TR in [(TorqueOmega, 1), (TorqueVel, -1)]:
        X, formula = band_values(system, k, Formula)
        assert abs(X).max() > 1e-3
        assert formula.transformTR.factor == sign_TR and formula.transformInv.factor == -1
        X_C2xT = band_values(system, k_C2xT, Formula)[0]
        assert X_C2xT == pytest.approx(sign_TR * np.einsum("ab,nbc,dc->nad", R, X, R), abs=1e-8)
        assert band_values(system, -k, Formula)[0] == pytest.approx(-X, abs=1e-8)


def test_torkance_routes(system_Fe_gpaw_soc_angle, system_Fe_gpaw_soc_angle_R):
    """the torque built on the fly (SystemSOC) and as a real-space matrix (get_system_R) give the same torkance"""
    Efermi = np.linspace(8, 10, 5)
    data = [get_datak(system, dK=[0.13, 0.27, 0.41], NKFFT=2) for system in (system_Fe_gpaw_soc_angle, system_Fe_gpaw_soc_angle_R)]
    for Calculator in TorkanceEven, TorkanceOdd:
        t_soc, t_R = (Calculator(Efermi=Efermi, print_comment=False)(data_k).data for data_k in data)
        assert abs(t_soc).max() > 1e-4
        assert t_R == pytest.approx(t_soc, abs=1e-10)
        # bcc Fe has inversion symmetry, so the integrated torkance vanishes
        result = wberri.run(system_Fe_gpaw_soc_angle, grid=wberri.Grid(system_Fe_gpaw_soc_angle, NKFFT=3, NKdiv=1),
                            calculators={"t": Calculator(Efermi=Efermi)}, parallel=False, print_progress_step_time=1000,
                            fout_name=os.path.join(OUTPUT_DIR_RUN, "Fe_gpaw_soc_torkance"))
        assert abs(result.results["t"].data).max() < 1e-10
