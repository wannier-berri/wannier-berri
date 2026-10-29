


# from numba import njit
import numpy as np
from ..factors import factor_peierls

# @njit


def get_peierls_phase_num(r1, r2, A_func, num_points=100):
    r"""calculates phases for Peierls substitution
    t_ij -> t_ij * exp(i e/hbar * phi_ij)
    where phi_ij = \int_r1^r2 A(r) . dr

    Parameters:
    -----------
    r1 : np.ndarray(3)
        Wannier center of the first site in Cartesian coordinates
    r2 : np.ndarray(3)
        Wannier center of the second site in Cartesian coordinates
    A_cartesian : function np.ndarray(num_points, 3) -> np.ndarray(num_points, 3)
        Function that returns the vector potential A at a given position in Cartesian coordinates,
    num_points : int, optional
        Number of points to use for the numerical integration (default is 100)

    Returns:
    --------
    phase : float
        Peierls phase factor for the hopping from r1 to r2
    """
    t = np.linspace(0, 1, num_points)
    points = np.outer(1 - t, r1) + np.outer(t, r2)
    A_values = A_func(points)
    dr = (r2 - r1) / (num_points - 1)
    return (np.sum(A_values[1:-1], axis=0) + (A_values[0] + A_values[-1]) / 2) @ dr


def get_peierls_phases_numerical(rvectors, A_cartesian_fun, num_points=100):
    r"""calculates phases for Peierls substitution
    t_ij -> t_ij * exp(i e/hbar * phi_ij)
    where phi_ij = \int_r1^{r2+R} A(r) . dr

    Parameters:
    -----------
    rvectors : wannierberri.fourier.rvectors.Rvectors
        RVectors object containing the lattice vectors and their shifts
    A_cartesian_fun : function
        Function that returns the vector potential A at a given position/array of positions in Cartesian coordinates, A(r) = A_cartesian_fun(r)
    """
    assert num_points >= 2, "num_points must be at least 2 for numerical integration"
    nleft = rvectors.nshifts_left
    nright = rvectors.nshifts_right
    nR = rvectors.nRvec
    path_integrals = np.zeros((nR, nleft, nright), dtype=np.complex128)
    for i in range(nleft):
        for j in range(nright):
            for iR, R in enumerate(rvectors.cRvec):
                r1 = rvectors.shifts_left_cart[i]
                r2 = rvectors.shifts_right_cart[j] + R
                path_integrals[iR, i, j] = get_peierls_phase_num(r1, r2, A_cartesian_fun, num_points=num_points)
    return np.exp(path_integrals * factor_peierls)


def get_peierls_phases_linear(rvectors, A_matrix, A0=None):
    """
    same as get_peierls_phases_numerical but the vector potential is assumed to be linear in space, 
    i.e. A(r) = A0 + r @ A_matrix, where A0 is a constant vector and A_matrix is a 3x3 matrix.
    A0 is in Tesla*Angstrom, A_matrix is in Tesla
    """
    delta = rvectors.cRvec_shifted
    mid_value = rvectors.mid_point_hop @ A_matrix
    if A0 is not None:
        mid_value += A0[None, None, None, :]
    return np.exp(factor_peierls * (mid_value * delta).sum(axis=-1))


def get_A_matrix(B, A_direction):
    """
    uses a gauge where A changes linearly in A_direction, e.g. for 
    A_direction = 2 (z-direction), A = (-By*z, Bx*z, 0) for a magnetic field B = (Bx, By, Bz
    """
    i = A_direction
    ip = (i + 1) % 3
    im = (i + 2) % 3
    assert B[i] == 0, "B field in the A_direction should be zero for this gauge"
    A_matrix = np.zeros((3, 3))
    A_matrix[i, ip] = -B[im]
    A_matrix[i, im] = B[ip]
    return A_matrix


def get_peierls_phases_B(rvectors, B, A_direction):
    """
    uses a gauge where A changes linearly in A_direction, e.g. for 
    A_direction = 2 (z-direction), A = (-By*z, Bx*z, 0) for a magnetic field B = (Bx, By, Bz
    """
    A_matrix = get_A_matrix(B, A_direction=A_direction)
    return get_peierls_phases_linear(rvectors, A_matrix=A_matrix)


def get_peierls_phases(rvectors,
                       A_func=None, num_points=100,
                       A_matrix=None, A0=None,
                       B=None, A_direction=None):
    assert (A_func is not None) + (A_matrix is not None) + (B is not None) == 1, "Exactly one of A_cartesian_fun, A_matrix, or B must be provided"
    if A_func is not None:
        return get_peierls_phases_numerical(rvectors, A_func)
    elif A_matrix is not None:
        return get_peierls_phases_linear(rvectors, A_matrix, A0=A0)
    elif B is not None:
        assert A_direction is not None, "A_direction must be provided when B is provided"
        return get_peierls_phases_B(rvectors, B, A_direction=A_direction)
