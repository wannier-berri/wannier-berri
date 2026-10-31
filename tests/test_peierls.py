import copy

import numpy as np
import pytest
from wannierberri.system.peierls import get_peierls_phases_numerical, get_peierls_phases_linear, \
    get_A_matrix, get_peierls_phase_num
from wannierberri.fourier.rvectors import Rvectors


@pytest.fixture(scope="module")
def system_Chiral_right_slab(system_Chiral_right):
    system = copy.deepcopy(system_Chiral_right)
    system.set_atoms_trivial()
    return system.make_slab(M=np.eye(3), nslab=1)


@pytest.mark.parametrize("num_wannier", [1, 3])
@pytest.mark.parametrize("num_points", [2, 10])
@pytest.mark.parametrize("include_A0", [True, False])
def test_numerical_vs_linear(num_points, include_A0, num_wannier):
    A_matrix = np.random.rand(3, 3) * 100  # Random 3x3 matrix for testing
    if include_A0:
        A0 = np.random.rand(3)  # Random constant vector for testing
    else:
        A0 = np.zeros(3)  # No constant vectordef get_peierls_phase_num(r1, r2, A_cartesian, num_points=100):

    def A(r):
        return A0 + r @ A_matrix

    iRvec_half = np.array([[1, 2, 3], [-1, 3, -2], [2, -1, 1]])
    iRvec = np.vstack([[[0, 0, 0]], iRvec_half, -iRvec_half])
    wannier_centers_red = np.random.rand(num_wannier, 3)  # Random Wannier centers in reduced coordinates
    lattice = np.array([[1, 0, 0], [-1 / 2, np.sqrt(3) / 2, 0], [0, 0, 1.2]])  # Example lattice vectors
    rvectors = Rvectors(lattice=lattice,
                        shifts_left_red=wannier_centers_red,
                        shifts_right_red=wannier_centers_red,
                        iRvec=iRvec)

    phases_numerical = get_peierls_phases_numerical(rvectors, A, num_points=num_points)
    phases_linear = get_peierls_phases_linear(rvectors, A_matrix, A0=A0)
    assert np.allclose(phases_numerical, phases_linear, atol=1e-6), "Numerical and linear Peierls phases do not match"


@pytest.mark.parametrize("num_points", [1, 2, 10])
@pytest.mark.parametrize("include_A0", [True, False])
def test_numerical_vs_linear_system(num_points, include_A0, num_wannier, system_Chiral_right):
    A_matrix = np.random.rand(3, 3) * 1000  # Random 3x3 matrix for testing
    if include_A0:
        A0 = np.random.rand(3)  # Random constant vector for testing
    else:
        A0 = np.zeros(3)  # No constant vectordef get_peierls_phase_num(r1, r2, A_cartesian, num_points=100):

    def A(r):
        return A0 + r @ A_matrix
    system1 = copy.deepcopy(system_Chiral_right)
    system1.set_peierls(A_matrix=A_matrix, A0=A0)
    system2 = copy.deepcopy(system_Chiral_right)
    system2.set_peierls(A_func=A, num_points=num_points)
    assert np.allclose(system1.get_R_mat("Ham"), system2.get_R_mat("Ham"), atol=1e-6), "Numerical and linear Peierls phases do not match in the system"


def test_B_field():
    # Test that the get_A_matrix function returns a matrix that produces the correct B field
    B = np.random.random(3)  # Example magnetic field in Tesla
    B[2] = 0  # Ensure B in the A_direction is zero for this test
    A_matrix = get_A_matrix(B, A_direction=2)  # A changes linearly in the z-direction
    assert np.allclose(A_matrix, [[0, 0, 0],
                                  [0, 0, 0],
                                  [-B[1], B[0], 0]]), "A_matrix does not produce the correct B field for z-direction"
    B = np.roll(B, 1)  # Rotate B to test another direction
    A_matrix = get_A_matrix(B, A_direction=0)  # A changes linearly in the x-direction
    assert np.allclose(A_matrix, [[0, -B[2], B[1]],
                                  [0, 0, 0],
                                  [0, 0, 0]]), "A_matrix does not produce the correct B field for x-direction"
    B = np.roll(B, 1)  # Rotate B to test another direction
    A_matrix = get_A_matrix(B, A_direction=1)  # A changes linearly in the y-direction
    assert np.allclose(A_matrix, [[0, 0, 0],
                                  [B[2], 0, -B[0]],
                                  [0, 0, 0]]), "A_matrix does not produce the correct B field for y-direction"


def test_additivity(system_Chiral_right):
    system1 = copy.deepcopy(system_Chiral_right)
    system2 = copy.deepcopy(system_Chiral_right)
    system1.set_peierls(B=np.array([0, 0, 1]), A_direction=1)
    system1.set_peierls(B=np.array([1.3, 0, 0]), A_direction=1)
    system2.set_peierls(B=np.array([1.3, 0, 1]), A_direction=1)
    assert np.allclose(system1.get_R_mat("Ham"), system2.get_R_mat("Ham"), atol=1e-6), "Peierls phases are not additive for combined B fields"


def test_hermitean(system_Chiral_right_slab):
    system = copy.deepcopy(system_Chiral_right_slab)
    print(system.rvec.cRvec_shifted)
    print(system.rvec.shifts_left_red)
    print(system.rvec.shifts_right_red)
    system.set_peierls(B=np.array([0, 100, 0]), A_direction=2)
    Ham = system.get_R_mat("Ham")
    Ham_conj = system.rvec.conj_XX_R(Ham)
    assert np.allclose(Ham, Ham_conj, atol=1e-6), "Peierls phases do not produce a Hermitian Hamiltonian"


def test_reverse_path():
    def A_func(r):
        return np.cos(r) * np.exp(-r)
    r1 = np.random.rand(3)
    r2 = np.random.rand(3)
    phase_forward = get_peierls_phase_num(r1, r2, A_func)
    phase_backward = get_peierls_phase_num(r2, r1, A_func)
    assert np.allclose(phase_forward, -phase_backward, atol=1e-6), "Peierls phase for reverse path is not the negative of the forward path"
