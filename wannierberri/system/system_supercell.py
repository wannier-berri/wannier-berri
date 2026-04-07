"""Fold a primitive System_R into a supercell System_R.

All R-space matrix elements (Ham, AA, SS, BB, CC, ...) present in the
primitive system are folded into the supercell representation.  Scattering
potentials can be added to the Hamiltonian afterward via
:func:`add_scattering`.
"""

import logging

import numpy as np

from ..fourier.rvectors import Rvectors
from .system_R import System_R

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------


def _validate_M(M):
    """Validate that *M* is a 3x3 integer matrix with non-zero determinant."""
    M = np.asarray(M, dtype=int)
    if M.shape != (3, 3):
        raise ValueError(f"M must be a 3x3 integer matrix, got shape {M.shape}")
    det = int(round(np.linalg.det(M.astype(float))))
    if det == 0:
        raise ValueError(f"M must be non-singular, got det(M) = 0:\n{M}")
    return M


def enumerate_subcells(M):
    """Primitive lattice points inside one supercell (coset representatives).

    Returns integer vectors *t* such that M^{-1} t in [0, 1)^3,
    sorted lexicographically.

    Parameters
    ----------
    M : ndarray, shape [3, 3]
        Integer supercell matrix.

    Returns
    -------
    subcells : ndarray, shape [nsc, 3]
        nsc = |det(M)|.
    """
    M = _validate_M(M)
    nsc = abs(int(round(np.linalg.det(M.astype(float)))))
    Minv = np.linalg.inv(M.astype(float))

    bound = int(np.sum(np.abs(M))) + 1
    subcells = []
    for idx in np.ndindex(2 * bound + 1, 2 * bound + 1, 2 * bound + 1):
        t = np.array(idx) - bound
        frac = Minv @ t
        if np.all(frac > -1e-10) and np.all(frac < 1.0 - 1e-10):
            subcells.append(t)

    if len(subcells) != nsc:
        raise RuntimeError(
            f"Expected {nsc} subcells for det(M)={nsc}, found {len(subcells)}. "
            f"M =\n{M}"
        )

    arr = np.array(subcells)
    return arr[np.lexsort(arr[:, ::-1].T)]


def _build_wannier_centres_sc(subcells, lattice, wc_prim):
    """Tile primitive Wannier centres across all subcells.

    Parameters
    ----------
    subcells : ndarray, shape [nsc, 3]
    lattice : ndarray, shape [3, 3]
        Primitive lattice vectors (same units as *wc_prim*).
    wc_prim : ndarray, shape [nwann, 3]

    Returns
    -------
    wc_sc : ndarray, shape [nsc * nwann, 3]
    """
    nsc = len(subcells)
    nwann = len(wc_prim)
    shifts = subcells @ lattice  # [nsc, 3]
    return (shifts[:, np.newaxis, :] + wc_prim[np.newaxis, :, :]).reshape(nsc * nwann, 3)


# ------------------------------------------------------------------
# Folding helpers
# ------------------------------------------------------------------


def _supercell_rvectors(iRvec_prim, M, subcells):
    """Determine supercell R-vectors from the primitive R-vector set.

    For each primitive R and subcell pair (i, j), compute
    R_sc = M^{-1} (R_prim + tau_i - tau_j).  Collect all integer results.

    Returns sorted integer array of shape [n_R_sc, 3].
    """
    Minv = np.linalg.inv(M.astype(float))
    R_sc_set: set[tuple[int, ...]] = set()

    for R_prim in iRvec_prim:
        for tau_i in subcells:
            for tau_j in subcells:
                R_sc_f = Minv @ (R_prim + tau_i - tau_j)
                R_sc_int = np.round(R_sc_f).astype(int)
                if np.allclose(R_sc_f, R_sc_int, atol=1e-6):
                    R_sc_set.add(tuple(R_sc_int))

    arr = np.array(sorted(R_sc_set))
    return arr[np.lexsort(arr[:, ::-1].T)]


def _fold_matrix(X_R, iRvec_prim, prim_lookup, R_sc, M, subcells, nwann):
    """Fold a single R-space matrix from primitive to supercell.

    Parameters
    ----------
    X_R : ndarray, shape [nR_prim, nwann, nwann, ...]
        Primitive matrix elements.
    iRvec_prim : ndarray, shape [nR_prim, 3]
        Primitive R-vectors.
    prim_lookup : dict
        Mapping tuple(R_prim) -> index in iRvec_prim.
    R_sc : ndarray, shape [n_R_sc, 3]
        Supercell R-vectors.
    M : ndarray, shape [3, 3]
        Integer supercell matrix.
    subcells : ndarray, shape [nsc, 3]
        Subcell positions.
    nwann : int
        Number of Wannier functions in the primitive cell.

    Returns
    -------
    X_sc : ndarray, shape [n_R_sc, norb_sc, norb_sc, ...]
    """
    nsc = len(subcells)
    norb_sc = nsc * nwann
    extra_shape = X_R.shape[3:]
    n_R_sc = len(R_sc)

    X_sc = np.zeros((n_R_sc, norb_sc, norb_sc) + extra_shape, dtype=X_R.dtype)

    for ir, R in enumerate(R_sc):
        for si, tau_i in enumerate(subcells):
            row = slice(si * nwann, (si + 1) * nwann)
            for sj, tau_j in enumerate(subcells):
                col = slice(sj * nwann, (sj + 1) * nwann)
                R_prim = tuple((M @ R + tau_j - tau_i).astype(int))
                idx = prim_lookup.get(R_prim)
                if idx is not None:
                    X_sc[ir, row, col] = X_R[idx]

    return X_sc


def _fold_scattering(T_R, R_sc, subcells, M, grid_arr, norb):
    """Fold the real-space scattering potential into supercell blocks.

    Parameters
    ----------
    T_R : ndarray, shape [*grid_shape, *grid_shape, norb, norb]
        Scattering potential in real space (from double FFT of V_kk).
    R_sc : ndarray, shape [n_R_sc, 3]
        Supercell R-vectors.
    subcells : ndarray, shape [nsc, 3]
        Subcell positions.
    M : ndarray, shape [3, 3]
        Integer supercell matrix.
    grid_arr : ndarray, shape [3]
        Primitive k-grid dimensions.
    norb : int
        Number of orbitals per primitive cell.

    Returns
    -------
    dH : ndarray, shape [n_R_sc, norb_sc, norb_sc]
    """
    nsc = len(subcells)
    norb_sc = nsc * norb
    n_R_sc = len(R_sc)

    dH = np.zeros((n_R_sc, norb_sc, norb_sc), dtype=complex)

    for ir, dRsc in enumerate(R_sc):
        for s1, t1 in enumerate(subcells):
            col = slice(s1 * norb, (s1 + 1) * norb)
            for s2, t2 in enumerate(subcells):
                row = slice(s2 * norb, (s2 + 1) * norb)
                R2 = M @ dRsc + t2
                idx1 = tuple((t1 % grid_arr).astype(int))
                idx2 = tuple((R2 % grid_arr).astype(int))
                dH[ir, row, col] = T_R[idx1 + idx2]

    return dH


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def fold_system(system_prim, M, periodic=None):
    """Fold a primitive WannierBerri System_R into a supercell System_R.

    Every R-space matrix present in ``system_prim`` (Ham, AA, SS, BB, ...)
    is folded into the supercell block structure.

    Parameters
    ----------
    system_prim : :class:`~wannierberri.system.System_R`
        Primitive-cell system, e.g. from ``System_w90()``.
    M : ndarray, shape [3, 3]
        Integer supercell matrix.
    periodic : tuple of bool, optional
        Periodic directions for the supercell system.  If not provided,
        inherits ``system_prim.periodic``.

    Returns
    -------
    system_sc : :class:`~wannierberri.system.System_R`
        Supercell system with all matrix elements folded.
    """
    M = _validate_M(M)
    if periodic is None:
        periodic = system_prim.periodic

    nwann = system_prim.num_wann
    iRvec_prim = system_prim.rvec.iRvec
    real_lattice_prim = system_prim.real_lattice
    wc_prim = system_prim.wannier_centers_cart

    # Subcells and supercell R-vectors
    subcells = enumerate_subcells(M)
    nsc = len(subcells)
    R_sc = _supercell_rvectors(iRvec_prim, M, subcells)

    # Primitive R-vector lookup
    prim_lookup = {tuple(R): i for i, R in enumerate(iRvec_prim)}

    # Fold all matrix elements
    folded = {}
    for key, X_R in system_prim._XX_R.items():
        folded[key] = _fold_matrix(
            X_R, iRvec_prim, prim_lookup, R_sc, M, subcells, nwann
        )
        logger.info("Folded %s: %s -> %s", key, X_R.shape, folded[key].shape)

    # Supercell lattice and Wannier centres
    cell_sc = M @ real_lattice_prim
    norb_sc = nsc * nwann
    wc_sc = _build_wannier_centres_sc(subcells, real_lattice_prim, wc_prim)
    wc_sc_red = wc_sc @ np.linalg.inv(cell_sc)

    # Build supercell System_R
    system_sc = System_R(
        periodic=periodic,
        spinor=system_prim.spinor,
        silent=system_prim.silent,
        name=system_prim.name,
    )
    system_sc.real_lattice = cell_sc
    system_sc.num_wann = norb_sc
    system_sc.wannier_centers_cart = wc_sc
    system_sc.rvec = Rvectors(
        lattice=cell_sc, shifts_left_red=wc_sc_red, iRvec=R_sc
    )

    for key, X_sc in folded.items():
        system_sc.set_R_mat(key, X_sc)

    system_sc.do_at_end_of_init()

    logger.info(
        "fold_system: nwann=%d->%d, nR=%d->%d, matrices=%s",
        nwann, norb_sc, len(iRvec_prim), len(R_sc),
        list(folded.keys()),
    )
    return system_sc


def spin_double_system(system, periodic=None):
    """Double a spinless System_R into a spin-1/2 system with Pauli SS.

    Every R-space matrix ``X`` is expanded as ``I_2 ⊗ X`` (block-diagonal in
    spin), Wannier centres are duplicated, and the on-site Pauli spin operator
    ``SS`` is added at R=0.  Orbital ordering is (spin, wannier): the spin-up
    block comes first, the spin-down block second.

    Parameters
    ----------
    system : :class:`~wannierberri.system.System_R`
        Spinless input system (e.g. from :func:`fold_system`).
    periodic : tuple of bool, optional
        Periodic directions for the new system.  Defaults to ``system.periodic``.

    Returns
    -------
    system_spin : :class:`~wannierberri.system.System_R`
        Spin-doubled system with ``2 * num_wann`` orbitals and an SS matrix.
    """
    if periodic is None:
        periodic = system.periodic

    if getattr(system, "spinor", False):
        raise ValueError("spin_double_system expects a spinless system, got system.spinor=True")
    if "SS" in system._XX_R:
        raise ValueError("spin_double_system expects a system without SS; existing spin information would be overwritten")
    nw = system.num_wann
    nw2 = 2 * nw
    iRvec = system.rvec.iRvec
    nR = len(iRvec)
    real_lattice = system.real_lattice
    wc = system.wannier_centers_cart

    # Expand all R-matrices via I_2 ⊗ X (block-diagonal in spin)
    doubled = {}
    for key, X_R in system._XX_R.items():
        extra = X_R.shape[3:]
        X_new = np.zeros((nR, nw2, nw2) + extra, dtype=X_R.dtype)
        X_new[:, :nw, :nw] = X_R
        X_new[:, nw:, nw:] = X_R
        doubled[key] = X_new

    # Pauli SS at R = 0: σ_c ⊗ I_nw, shape [nR, nw2, nw2, 3]
    R0_idx = None
    for i, R in enumerate(iRvec):
        if np.all(R == 0):
            R0_idx = i
            break
    if R0_idx is None:
        raise ValueError("System Rvectors do not contain R = 0")

    SS = np.zeros((nR, nw2, nw2, 3), dtype=complex)
    I_nw = np.eye(nw, dtype=complex)
    # σ_x
    SS[R0_idx, :nw, nw:, 0] = I_nw
    SS[R0_idx, nw:, :nw, 0] = I_nw
    # σ_y
    SS[R0_idx, :nw, nw:, 1] = -1j * I_nw
    SS[R0_idx, nw:, :nw, 1] = 1j * I_nw
    # σ_z
    SS[R0_idx, :nw, :nw, 2] = I_nw
    SS[R0_idx, nw:, nw:, 2] = -I_nw
    doubled["SS"] = SS

    # Duplicate Wannier centres
    wc_new = np.tile(wc, (2, 1))
    wc_red = wc_new @ np.linalg.inv(real_lattice)

    system_spin = System_R(periodic=periodic, silent=True, spinor=True)
    system_spin.real_lattice = real_lattice
    system_spin.num_wann = nw2
    system_spin.wannier_centers_cart = wc_new
    system_spin.rvec = Rvectors(
        lattice=real_lattice, shifts_left_red=wc_red, iRvec=iRvec
    )

    for key, X in doubled.items():
        system_spin.set_R_mat(key, X)

    system_spin.do_at_end_of_init()

    logger.info(
        "spin_double_system: nwann=%d->%d, matrices=%s",
        nw, nw2, list(doubled.keys()),
    )
    return system_spin


def add_scattering(system_sc, V_kk, grid_shape, M):
    """Add scattering potential to the supercell Hamiltonian in-place.

    Transforms ``V_kk`` to real space via double FFT, folds it into the
    supercell block structure, and adds it to the existing Ham matrix.

    Parameters
    ----------
    system_sc : :class:`~wannierberri.system.System_R`
        Supercell system from :func:`fold_system`.
    V_kk : ndarray, shape [nk, nk, norb, norb]
        Total scattering matrix in k-space.
    grid_shape : tuple of int, length 3
        Primitive k-grid dimensions, e.g. ``(24, 24, 1)``.
    M : ndarray, shape [3, 3]
        Integer supercell matrix.
    """
    V_kk = np.asarray(V_kk, dtype=complex)
    if V_kk.ndim != 4:
        raise ValueError(
            f"V_kk must have shape [nk, nk, norb, norb], got {V_kk.shape}"
        )
    if V_kk.shape[0] != V_kk.shape[1]:
        raise ValueError(
            f"V_kk must be square in k-space, got shape {V_kk.shape}"
        )
    if V_kk.shape[2] != V_kk.shape[3]:
        raise ValueError(
            f"V_kk must be square in orbital space, got shape {V_kk.shape}"
        )

    grid_shape = tuple(int(n) for n in grid_shape)
    if len(grid_shape) != 3:
        raise ValueError(
            f"grid_shape must have length 3, got {len(grid_shape)}"
        )
    M = _validate_M(M)
    grid_arr = np.array(grid_shape, dtype=int)
    nk = V_kk.shape[0]
    norb = V_kk.shape[2]

    if int(np.prod(grid_shape)) != nk:
        raise ValueError(
            f"grid_shape={grid_shape} implies {int(np.prod(grid_shape))} k-points, "
            f"but V_kk has nk={nk}"
        )

    subcells = enumerate_subcells(M)
    expected_num_wann = len(subcells) * norb
    if system_sc.num_wann != expected_num_wann:
        raise ValueError(
            f"system_sc.num_wann={system_sc.num_wann} is inconsistent with "
            f"M={M.tolist()} and norb={norb}; expected {expected_num_wann}"
        )
    if not system_sc.has_R_mat("Ham"):
        raise ValueError("system_sc does not contain a Ham matrix to update")

    R_sc = system_sc.rvec.iRvec

    # Double FFT: V(k1,k2) -> T(R1,R2)
    ax_k1 = (0, 1, 2)
    ax_k2 = (3, 4, 5)
    V_grid = V_kk.reshape(*grid_shape, *grid_shape, norb, norb)
    T_R = np.fft.ifftn(np.fft.fftn(V_grid, axes=ax_k1), axes=ax_k2)

    dH = _fold_scattering(T_R, R_sc, subcells, M, grid_arr, norb)

    if system_sc.get_R_mat("Ham").shape != dH.shape:
        raise ValueError(
            f"Ham has shape {system_sc.get_R_mat('Ham').shape}, "
            f"but folded scattering has shape {dH.shape}"
        )

    system_sc.set_R_mat("Ham", dH, add=True)

    logger.info(
        "add_scattering: added T_R to Ham, grid=%s, norb=%d", grid_shape, norb
    )
