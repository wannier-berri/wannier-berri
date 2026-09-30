"""Supercells of a :class:`~wannierberri.system.System_R`.

:func:`get_system_supercell` folds all real-space matrices of a system (Ham, AA, SS, BB, CC, ...)
into a supercell, and :func:`add_proximity_potential` adds a potential with the periodicity of the
supercell (e.g. induced by proximity to a substrate) to the Hamiltonian of the supercell system.

The rows of ``supercell_matrix`` are the supercell lattice vectors in units of the primitive ones:
``real_lattice_sc = supercell_matrix @ real_lattice``.
The Wannier functions of the supercell are ordered as (cell, wannier function of the primitive system).
"""

import itertools
import logging
import warnings

import numpy as np

from ..fourier.rvectors import Rvectors
from ..utility import iterate_nd, one2three
from .system_R import System_R

logger = logging.getLogger(__name__)


def _check_supercell_matrix(supercell_matrix):
    """Check that ``supercell_matrix`` is a non-singular 3x3 integer matrix and return it as an integer array."""
    M = np.asarray(supercell_matrix)
    if M.shape != (3, 3):
        raise ValueError(f"supercell_matrix should be a 3x3 integer matrix, found shape {M.shape}")
    M_int = np.round(M).astype(int)
    if not np.allclose(M, M_int):
        raise ValueError(f"supercell_matrix should contain only integers, found\n{M}")
    if int(round(np.linalg.det(M_int))) == 0:
        raise ValueError(f"supercell_matrix should be non-singular, found\n{M_int}")
    return M_int


def _get_iRvec_in_supercell(supercell_matrix):
    """Lattice vectors of the primitive cells inside the supercell.

    Returns
    -------
    iRvec_cells : np.ndarray(shape=(num_cells, 3), dtype=int)
        the integer vectors ``t`` with ``t @ inv(supercell_matrix)`` in [0, 1)^3,
        sorted lexicographically. ``num_cells = |det(supercell_matrix)|``
    """
    M = _check_supercell_matrix(supercell_matrix)
    num_cells = abs(int(round(np.linalg.det(M))))
    # t = f @ M with f in [0, 1)^3, hence |t_j| < sum_i |M_ij|
    bound = np.abs(M).sum(axis=0)
    t = np.array(list(itertools.product(*[range(-b, b + 1) for b in bound])))
    frac = t @ np.linalg.inv(M)
    iRvec_cells = t[np.all((frac > -1e-8) & (frac < 1 - 1e-8), axis=1)]
    assert len(iRvec_cells) == num_cells, f"found {len(iRvec_cells)} cells in the supercell, expected {num_cells}"
    return iRvec_cells


def _split_iRvec(iRvec, supercell_matrix, iRvec_cells):
    """Write primitive lattice vectors as ``iRvec = iRvec_cells[icell] + iRvec_sc @ supercell_matrix``.

    Returns
    -------
    icell : np.ndarray(dtype=int)
        the cell of each vector, shape ``iRvec.shape[:-1]``
    iRvec_sc : np.ndarray(dtype=int)
        the supercell lattice vectors, same shape as ``iRvec``
    """
    iRvec_sc = np.floor(iRvec @ np.linalg.inv(supercell_matrix) + 1e-8).astype(int)
    index_cell = {tuple(t): i for i, t in enumerate(iRvec_cells)}
    t = iRvec - iRvec_sc @ supercell_matrix
    icell = np.array([index_cell[tuple(x)] for x in t.reshape(-1, 3)]).reshape(t.shape[:-1])
    return icell, iRvec_sc


def _get_iRvec_supercell(iRvec, supercell_matrix, iRvec_cells):
    """Map the primitive lattice vectors onto the supercell.

    For the bra in cell ``i`` and a primitive lattice vector ``R`` the ket is in cell ``j`` of the supercell
    ``R_sc``: ``iRvec_cells[i] + R = iRvec_cells[j] + R_sc @ supercell_matrix``.

    Returns
    -------
    iRvec_sc : np.ndarray(shape=(nRvec_sc, 3), dtype=int)
        the supercell lattice vectors, sorted
    iR_sc : np.ndarray(shape=(num_cells, nRvec), dtype=int)
        index of ``R_sc`` in ``iRvec_sc``
    jcell : np.ndarray(shape=(num_cells, nRvec), dtype=int)
        the cell ``j`` of the ket
    """
    jcell, R_sc = _split_iRvec(iRvec_cells[:, None, :] + iRvec[None, :, :], supercell_matrix, iRvec_cells)
    iRvec_sc, iR_sc = np.unique(R_sc.reshape(-1, 3), axis=0, return_inverse=True)
    return iRvec_sc, iR_sc.reshape(jcell.shape), jcell


def _fold_XX_R(XX_R_cells, iR_sc, jcell, nRvec_sc):
    """Fold a real-space matrix into the supercell.

    Parameters
    ----------
    XX_R_cells : np.ndarray(shape=(num_cells, nRvec, num_wann, num_wann, ...))
        the matrix elements with the bra in each cell of the supercell
        (use ``np.broadcast_to`` for a translation-invariant matrix)
    iR_sc, jcell : np.ndarray(shape=(num_cells, nRvec), dtype=int)
        see :func:`_get_iRvec_supercell`
    nRvec_sc : int
        number of the supercell lattice vectors

    Returns
    -------
    XX_R_sc : np.ndarray(shape=(nRvec_sc, num_cells * num_wann, num_cells * num_wann, ...))
    """
    num_cells, _, num_wann = XX_R_cells.shape[:3]
    shape_cart = XX_R_cells.shape[4:]
    XX_R_sc = np.zeros((nRvec_sc, num_cells, num_wann, num_cells, num_wann) + shape_cart, dtype=XX_R_cells.dtype)
    for i in range(num_cells):
        XX_R_sc[iR_sc[i], i, :, jcell[i], :] = XX_R_cells[i]
    return XX_R_sc.reshape((nRvec_sc, num_cells * num_wann, num_cells * num_wann) + shape_cart)


def _add_R_mat(system, key, XX_R, iRvec):
    """Add a real-space matrix given on the lattice vectors ``iRvec`` to the matrix ``key`` of ``system``,
    extending the lattice vectors of the system if needed."""
    iRvec_new = np.unique(np.vstack([system.rvec.iRvec, iRvec]), axis=0)
    index_R = {tuple(R): i for i, R in enumerate(iRvec_new)}

    def remap(XX, iRvec_XX):
        XX_new = np.zeros((len(iRvec_new),) + XX.shape[1:], dtype=XX.dtype)
        XX_new[[index_R[tuple(R)] for R in iRvec_XX]] = XX
        return XX_new

    XX_R_dic = {k: remap(XX, system.rvec.iRvec) for k, XX in system._XX_R.items()}
    XX_R_dic[key] = XX_R_dic.get(key, 0) + remap(XX_R, iRvec)
    system.rvec = Rvectors(lattice=system.real_lattice, shifts_left_red=system.wannier_centers_red, iRvec=iRvec_new)
    for k, XX in XX_R_dic.items():
        system.set_R_mat(k, XX, reset=True)


def get_system_supercell(system, supercell_matrix, **parameters):
    """
    Create a supercell of a system. All real-space matrices of the system (Ham, AA, SS, BB, CC, ...)
    are folded into the supercell.

    Parameters
    ----------
    system : :class:`~wannierberri.system.System_R`
        the primitive system
    supercell_matrix : array(int, shape=(3, 3))
        the rows are the supercell lattice vectors in units of the primitive ones:
        ``real_lattice_sc = supercell_matrix @ system.real_lattice``.
        The non-periodic directions should be left unchanged.
    **parameters
        parameters of :class:`~wannierberri.system.System`. By default they are taken from ``system``

    Returns
    -------
    :class:`~wannierberri.system.System_R`
        the supercell system with ``|det(supercell_matrix)| * system.num_wann`` Wannier functions,
        ordered as (cell, wannier function of the primitive system)

    Notes
    -----
    * To double the spin of a spinless supercell system use :meth:`~wannierberri.system.System_R.double_spin`
    * To add a potential with the periodicity of the supercell use :func:`add_proximity_potential`
    """
    M = _check_supercell_matrix(supercell_matrix)
    parameters_sc = dict(periodic=system.periodic, spinor=system.spinor, name=system.name, silent=system.silent,
                         frozen_max=system.frozen_max, force_internal_terms_only=system.force_internal_terms_only)
    parameters_sc.update(parameters)
    system_sc = System_R(**parameters_sc)
    unit = np.eye(3, dtype=int)
    for i in np.where(np.logical_not(system_sc.periodic))[0]:
        if np.any(M[i] != unit[i]) or np.any(M[:, i] != unit[i]):
            raise ValueError(f"supercell_matrix should not change the non-periodic direction {i}, found\n{M}")
    iRvec_cells = _get_iRvec_in_supercell(M)
    num_cells = len(iRvec_cells)
    iRvec_sc, iR_sc, jcell = _get_iRvec_supercell(system.rvec.iRvec, M, iRvec_cells)

    system_sc.is_phonon = system.is_phonon
    system_sc.real_lattice = M @ system.real_lattice
    system_sc.num_wann = num_cells * system.num_wann
    system_sc.wannier_centers_cart = ((iRvec_cells @ system.real_lattice)[:, None, :]
                                      + system.wannier_centers_cart[None, :, :]).reshape(-1, 3)
    system_sc.rvec = Rvectors(lattice=system_sc.real_lattice, shifts_left_red=system_sc.wannier_centers_red,
                              iRvec=iRvec_sc)
    for key, XX_R in system._XX_R.items():
        XX_R_cells = np.broadcast_to(XX_R, (num_cells,) + XX_R.shape)
        system_sc.set_R_mat(key, _fold_XX_R(XX_R_cells, iR_sc, jcell, len(iRvec_sc)))
    system_sc.do_at_end_of_init()
    logger.info(f"Supercell of {num_cells} cells: {system.num_wann} -> {system_sc.num_wann} "
                f"Wannier functions, {system.rvec.nRvec} -> {system_sc.rvec.nRvec} R-vectors, "
                f"matrices {list(system._XX_R.keys())}")
    return system_sc


def add_proximity_potential(system, VV_qq, mp_grid, supercell_matrix, ws_dist_tol=1e-5):
    """
    Add a potential with the periodicity of the supercell (e.g. induced by proximity to a substrate)
    to the Hamiltonian of a supercell system (in place).

    The potential is given by the matrix elements ``VV_qq[k1, k2, m, n] = <psi_{k1,m}|V|psi_{k2,n}>``
    between the Bloch sums ``psi_{k,n} = N^{-1/2} sum_R exp(i k.R) w_{n,R}`` of the Wannier functions of
    the primitive system, on the Gamma-centered grid ``mp_grid`` of the primitive cell.
    They vanish unless ``k1 - k2`` is a reciprocal lattice vector of the supercell.
    Only the Hermitian part of ``VV_qq`` and its part periodic with the supercell are used
    (a warning is issued if ``VV_qq`` differs from them).
    In real space every matrix element is assigned to the lattice vector of minimal distance
    between the Wannier centers (with degeneracy weights), in the same way as for the Hamiltonian
    of a Wannierised system (see :meth:`~wannierberri.fourier.rvectors.Rvectors.set_Rvec`).
    The lattice vectors of the system are extended if needed.

    Parameters
    ----------
    system : :class:`~wannierberri.system.System_R`
        the supercell system, created by :func:`get_system_supercell` with the same ``supercell_matrix``
    VV_qq : np.ndarray(shape=(NK, NK, num_wann, num_wann))
        the matrix elements of the potential, in the energy units of the Hamiltonian of ``system``.
        ``num_wann`` is the number of Wannier functions of the primitive system, in the same order
        (e.g. :meth:`~wannierberri.system.System_R.double_spin` interlaces the spin index),
        and ``NK = prod(mp_grid)``. The k-points ``(i1/N1, i2/N2, i3/N3)`` are ordered
        with the last index running fastest (as in ``np.ndindex(*mp_grid)``)
    mp_grid : int or tuple(int)
        the grid of k-points of the primitive cell, should be commensurate with the supercell
    supercell_matrix : array(int, shape=(3, 3))
        see :func:`get_system_supercell`
    ws_dist_tol : float
        tolerance for the Wigner-Seitz distance
    """
    M = _check_supercell_matrix(supercell_matrix)
    mp_grid = one2three(mp_grid)
    NK = int(np.prod(mp_grid))
    VV_qq = np.asarray(VV_qq)
    num_wann = VV_qq.shape[2]
    if VV_qq.shape != (NK, NK, num_wann, num_wann):
        raise ValueError(f"VV_qq should have shape (NK, NK, num_wann, num_wann) with NK={NK}, found {VV_qq.shape}")
    VV_qq_dagger = VV_qq.transpose(1, 0, 3, 2).conj()
    deviation = abs(VV_qq - VV_qq_dagger).max()
    if deviation > 1e-6 * abs(VV_qq).max():
        warnings.warn(f"VV_qq is not Hermitian (deviation {deviation:.2e}, max |V| {abs(VV_qq).max():.2e}). "
                      "Only its Hermitian part is used")
    VV_qq = (VV_qq + VV_qq_dagger) / 2
    iRvec_cells = _get_iRvec_in_supercell(M)
    num_cells = len(iRvec_cells)
    if system.num_wann != num_cells * num_wann:
        raise ValueError(f"the system has {system.num_wann} Wannier functions, but a supercell of {num_cells} "
                         f"cells with {num_wann} Wannier functions per cell was expected")
    bvk_sc = np.diag(mp_grid) @ np.linalg.inv(M)
    if not np.allclose(bvk_sc, np.round(bvk_sc)):
        raise ValueError(f"mp_grid={mp_grid} is not commensurate with the supercell_matrix\n{M}")

    # primitive lattice and Wannier centers, recovered from the supercell
    real_lattice = np.linalg.inv(M) @ system.real_lattice
    wannier_centers_cart = (system.wannier_centers_cart.reshape(num_cells, num_wann, 3)
                            - (iRvec_cells @ real_lattice)[:, None, :])
    if not np.allclose(wannier_centers_cart, wannier_centers_cart[0], atol=1e-6):
        raise ValueError("the Wannier centers of the system do not correspond to a supercell with "
                         f"supercell_matrix\n{M}")

    # <w_{m,R1}|V|w_{n,R2}> on the Born-von Karman grid (same convention as Rvectors.qq_to_RR)
    VV_RR = VV_qq.reshape(tuple(mp_grid) * 2 + (num_wann, num_wann))
    VV_RR = np.fft.fftn(np.fft.ifftn(VV_RR, axes=(0, 1, 2)), axes=(3, 4, 5))

    # the matrix elements with the bra in cell i and the ket shifted by R, averaged over the supercells
    iRvec_grid = iterate_nd(mp_grid)
    icell, _ = _split_iRvec(iRvec_grid, M, iRvec_cells)

    def shifted(R1):
        return np.roll(VV_RR[tuple(R1)], shift=tuple(-R1), axis=(0, 1, 2))

    XX_R_grid = np.zeros((num_cells,) + VV_RR.shape[3:], dtype=complex)
    for R1, i in zip(iRvec_grid, icell):
        XX_R_grid[i] += shifted(R1)
    XX_R_grid /= NK // num_cells
    deviation = max(abs(shifted(R1) - XX_R_grid[i]).max() for R1, i in zip(iRvec_grid, icell))
    VV_max = abs(VV_RR).max()
    if deviation > 1e-6 * VV_max:
        warnings.warn(f"the potential is not periodic with the supercell (deviation {deviation:.2e}, "
                      f"max |V| {VV_max:.2e}). Only its periodic part is used")

    rvec = Rvectors(lattice=real_lattice, shifts_left_red=wannier_centers_cart[0] @ np.linalg.inv(real_lattice))
    rvec.set_Rvec(mp_grid, ws_tolerance=ws_dist_tol)
    XX_R_cells = np.array([rvec.remap_XX_from_grid_to_list_R(X) for X in XX_R_grid])
    iRvec_sc, iR_sc, jcell = _get_iRvec_supercell(rvec.iRvec, M, iRvec_cells)
    Ham_R_add = _fold_XX_R(XX_R_cells, iR_sc, jcell, len(iRvec_sc))
    # lattice vectors where the potential vanishes would only enlarge the FFT grid
    nonzero = abs(Ham_R_add).max(axis=(1, 2)) > 1e-12 * VV_max
    _add_R_mat(system, 'Ham', Ham_R_add[nonzero], iRvec_sc[nonzero])
    logger.info(f"Added the proximity potential on the grid {mp_grid}, now {system.rvec.nRvec} R-vectors")
