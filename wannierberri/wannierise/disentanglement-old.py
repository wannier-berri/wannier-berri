from copy import deepcopy
import warnings
import numpy as np

from ..__utility import vectorize
from .sitesym import VoidSymmetrizer, orthogonalize, Symmetrizer

DEGEN_THRESH = 1e-2  # for safety - avoid splitting (almost) degenerate states between free/frozen  inner/outer subspaces  (probably too much)


def disentangle(w90data,
                froz_min=np.Inf,
                froz_max=-np.Inf,
                num_iter=1000,
                conv_tol=1e-9,
                num_iter_converge=10,
                mix_ratio=0.5,
                print_progress_every=10,
                sitesym=False,
                kwargs_sitesym={}):
    r"""
    Deprecated - use wannierise() instead

    Performs disentanglement of the bands recorded in w90data, following the procedure described in
    `Souza et al., PRB 2001 <https://doi.org/10.1103/PhysRevB.65.035109>`__
    At the end writes `w90data.chk.v_matrix` and sets `w90data.wannierised = True`


    Parameters
    ----------
    w90data: :class:`~wannierberri.system.Wannier90data`
        the data
    froz_min : float
        lower bound of the frozen window
    froz_max : float
        upper bound of the frozen window
    num_iter : int
        maximal number of iteration for disentanglement
    conv_tol : float
        tolerance for convergence of the spread functional  (in :math:`\mathring{\rm A}^{2}`)
    num_iter_converge : int
        the convergence is achieved when the standard deviation of the spread functional over the `num_iter_converge`
        iterations is less than conv_tol
    mix_ratio : float
        0 <= mix_ratio <=1  - mixing the previous itertions. 1 for max speed, smaller values are more stable
    print_progress_every
        frequency to print the progress
    sitesym : bool
        whether to use the site symmetry (if True, the seedname.dmn file should be present)

    Returns
    -------
    w90data.chk.v_matrix : numpy.ndarray


    Sets
    ------------
    w90data.chk.v_matrix : numpy.ndarray
        the optimized U matrix
    w90data.wannierised : bool
        True
    sets w90data.chk._wannier_centers : numpy.ndarray (nW,3)
        the centers of the Wannier functions
    w90data.chk._wannier_spreads : numpy.ndarray (nW)
        the spreads of the Wannier functions
    """
    warnings.warn("This function is deprecated, use wannierise() instead", DeprecationWarning)
    if froz_min > froz_max:
        print("froz_min > froz_max, nothing will be frozen")
    assert 0 < mix_ratio <= 1
    if sitesym:
        kptirr = w90data.dmn.kptirr
    else:
        kptirr = np.arange(w90data.mmn.NK)

    frozen = vectorize(frozen_nondegen, w90data.eig.data[kptirr], to_array=True,
                       kwargs=dict(froz_min=froz_min, froz_max=froz_max))
    free = vectorize(np.logical_not, frozen, to_array=True)

    if sitesym:
        symmetrizer = Symmetrizer(w90data.dmn, neighbours=w90data.mmn.neighbours,
                                  free=free,
                                  **kwargs_sitesym)
    else:
        symmetrizer = VoidSymmetrizer(NK=w90data.mmn.NK)


    num_bands_free = vectorize(np.sum, free, to_array=True)
    num_bands_frozen = vectorize(np.sum, frozen, to_array=True)
    nWfree = w90data.chk.num_wann - vectorize(np.sum, frozen, to_array=True)

    lst = vectorize(lambda amn, fr: amn[fr, :].dot(amn[fr, :].T.conj()),
                    w90data.amn.data[kptirr], free)
    U_opt_free = vectorize(get_max_eig, lst, nWfree, num_bands_free)  # nBfee x nWfree marrices

    # Maybe too much of rotation and symmetrization...
    U_opt_full = rotate_to_projections(w90data, U_opt_free,
                                       free, frozen, num_bands_frozen,
                                       kptirr)
    U_opt_full_BZ = symmetrizer.symmetrize_U(U_opt_full)

    neighbours_irreducible = np.array([[symmetrizer.kpt2kptirr[ik] for ik in neigh]
                                       for neigh in w90data.mmn.neighbours[kptirr]])
    Mmn_FF = MmnFreeFrozen(w90data.mmn.data[kptirr],
                           free, frozen,
                           neighbours_irreducible,
                           w90data.mmn.wk[kptirr],
                           w90data.chk.num_wann)

    Z_frozen = calc_Z(w90data, Mmn_FF('free', 'frozen'))
    print("Z_frozen ", [z.shape for z in Z_frozen])
    symmetrizer.symmetrize_Z(Z_frozen)

    Omega_I_list = []
    Z_old = None
    for i_iter in range(num_iter):
        U_opt_free_BZ = U_opt_full_to_free_BZ(U_opt_full_BZ, free, symmetrizer.kpt2kptirr)
        Z = [(z + zfr) for z, zfr in zip(calc_Z(w90data, Mmn_FF('free', 'free'),
                                                U_loc=U_opt_free_BZ, kptirr=kptirr), Z_frozen)]
        if i_iter > 0 and mix_ratio < 1:
            Z = vectorize(lambda z, zo: mix_ratio * z + (1 - mix_ratio) * zo,
                          Z, Z_old)
        symmetrizer.symmetrize_Z(Z)

        U_opt_free = vectorize(get_max_eig, Z, nWfree, num_bands_free)
        U_opt_full = rotate_to_projections(w90data, U_opt_free, free, frozen, num_bands_frozen, kptirr)

        Omega_I_list.append(sum(Mmn_FF.Omega_I(U_opt_free)))
        U_opt_full_BZ = symmetrizer.symmetrize_U(U_opt_full, all_k=(i_iter % print_progress_every == 0))

        delta_std = print_progress(i_iter, Omega_I_list, num_iter_converge, print_progress_every,
                                   w90data, U_opt_full_BZ)


        if delta_std < conv_tol:
            print(f"Converged after {i_iter} iterations")
            break
        Z_old = deepcopy(Z)
    if num_iter > 0:
        del Z_old, Z

    U_opt_full = rotate_to_projections(w90data, U_opt_free,
                                       free, frozen, num_bands_frozen,
                                       kptirr)
    print("U_opt_full ", [u.shape for u in U_opt_full])
    U_opt_full_BZ = symmetrizer.symmetrize_U(U_opt_full, all_k=True)

    w90data.chk.v_matrix = np.array(U_opt_full_BZ)
    w90data.chk._wannier_centers, w90data.chk._wannier_spreads = w90data.chk.get_wannier_centers(w90data.mmn, spreads=True)
    print_centers_and_spreads(w90data, U_opt_full_BZ)

    w90data.wannierised = True
    return w90data.chk.v_matrix


def U_opt_full_to_free_BZ(U_opt_full_BZ, free, kpt2kptirr):
    """
    return the U matrix for the free bands

    Parameters
    ----------
    U_opt_full_BZ : list of numpy.ndarray(nB,nW)
        the optimized U matrix for the bands and wannier functions
    free : list of numpy.ndarray(nb, dtype=bool)
        the boolean array of the free bands  (True for free), for each irreducible k-point
    kpt2kptirr : numpy.ndarray(nk)
        the mapping from k-point to irreducible k-point

    Returns
    -------
    list of numpy.ndarray(nBfree,nW)
        the optimized U matrix for the free bands and wannier functions

    Note: Some entries of the U_opt_full_BZ list are set to None. 
          The corresponding entries in the output list are set to None as well.
    """
    lst = []
    for ik, U in enumerate(U_opt_full_BZ):
        if U is None:
            lst.append(None)
        else:
            lst.append(U[free[kpt2kptirr[ik]]])
    return lst


def rotate_to_projections(w90data, U_opt_free, free, frozen, nfrozen, kptirr):
    """
    rotate the U matrix to the projections of the bands
    to better match the initial guess

    Parameters
    ----------
    w90data : Wannier90data
        the data (inputs of wannier90)
    U_opt_free : list of numpy.ndarray(nBfree,nW)
        the optimized U matrix for the free bands and wannier functions

    Returns
    -------
    list of numpy.ndarray(nBfree,nW)
        the rotated U matrix
    """
    def inner(U_opt, E, amn, free, frozen, nfrozen, num_wann, return_free_only=False):
        nband = E.shape[0]
        U = np.zeros((nband, num_wann), dtype=complex)
        U[frozen, range(nfrozen)] = 1.
        U[free, nfrozen:] = U_opt
        ZV = orthogonalize(U.T.conj().dot(amn))
        U_out = U.dot(ZV)
        return U_out
    return vectorize(inner, U_opt_free, w90data.eig.data[kptirr],
                     w90data.amn.data[kptirr], free, frozen, nfrozen,
                     kwargs={"num_wann": w90data.chk.num_wann})


def print_centers_and_spreads(w90data, U_opt_full_BZ):
    """
    print the centers and spreads of the Wannier functions

    Parameters
    ----------
    w90data : Wannier90data
        the data (inputs of wannier90)
    U_opt_free_BZ : list of numpy.ndarray(nBfree,nW)
        the optimized U matrix for the free bands and wannier functions
    """
    w90data.chk.v_matrix = np.array(U_opt_full_BZ)
    w90data.chk._wannier_centers, w90data.chk._wannier_spreads = w90data.chk.get_wannier_centers(w90data.mmn, spreads=True)

    wcc, spread = w90data.chk._wannier_centers, w90data.chk._wannier_spreads
    print("wannier centers and spreads")
    print("-" * 80)
    for wcc, spread in zip(wcc, spread):
        wcc = np.round(wcc, 6)
        print(f"{wcc[0]:10.6f}  {wcc[1]:10.6f}  {wcc[2]:10.6f}   |   {spread:10.8f}")
    print("-" * 80)


def print_progress(i_iter, Omega_I_list, num_iter_converge, print_progress_every,
                   w90data=None, U_opt_full_BZ=None):
    """
    print the progress of the disentanglement

    Parameters
    ----------
    Omega_I_list : list of float
        the list of the spread functional
    num_iter_converge : int
        the number of iterations to check the convergence
    print_progress_every : int
        the frequency to print the progress

    Returns
    -------
    float
        the standard deviation of the spread functional over the last `num_iter_converge` iterations
    """
    Omega_I = Omega_I_list[-1]
    if i_iter > 0:
        delta = f"{Omega_I - Omega_I_list[-2]:15.8e}"
    else:
        delta = "--"

    if i_iter >= num_iter_converge:
        delta_std = np.std(Omega_I_list[-num_iter_converge:])
        delta_std_str = f"{delta_std:15.8e}"
    else:
        delta_std = np.Inf
        delta_std_str = "--"

    if i_iter % print_progress_every == 0:

        print(f"iteration {i_iter:4d} Omega= {Omega_I:15.10f}  delta={delta}, delta_std={delta_std_str}")
        if w90data is not None and U_opt_full_BZ is not None:
            print_centers_and_spreads(w90data, U_opt_full_BZ)

    return delta_std


def calc_Z(w90data, mmn_ff, kptirr=None, U_loc=None):
    """
    calculate the Z matrix for the given Mmn matrix and U matrix

    Z = \sum_{b,k} w_{b,k} M_{b,k} M_{b,k}^{\dagger}
    where M_{b,k} = M_{b,k}^{loc} U_{b,k}

    Parameters
    ----------
    w90data : Wannier90data
        the data (inputs of wannier90)
    mmn_ff : list of numpy.ndarray(nnb,nb,nb)
        the Mmn matrix (either free-free or free-frozen)

    U_loc : list of numpy.ndarray(nBfree,nW)
        the U matrix

    Returns
    -------
    list of numpy.ndarray(nW,nW)
        the Z matrix
    """
    if U_loc is None:
        Mmn_loc_opt = mmn_ff
    else:
        assert kptirr is not None
        Mmn_loc_opt = [[Mmn[ib].dot(U_loc[ikb]) for ib, ikb in enumerate(neigh)] for Mmn, neigh in
                       zip(mmn_ff, w90data.mmn.neighbours[kptirr])]
    return [sum(wb * mmn.dot(mmn.T.conj()) for wb, mmn in zip(wbk, Mmn)) for wbk, Mmn in
            zip(w90data.mmn.wk, Mmn_loc_opt)]


def frozen_nondegen(E, thresh=DEGEN_THRESH, froz_min=np.inf, froz_max=-np.inf):
    """define the indices of the frozen bands, making sure that degenerate bands were not split
    (unfreeze the degenerate bands together)

    Parameters
    ----------
    E : numpy.ndarray(nb, dtype=float)
        the energies of the bands
    thresh : float
        the threshold for the degeneracy

    Returns
    -------
    numpy.ndarray(bool)
        the boolean array of the frozen bands  (True for frozen)
    """
    ind = list(np.where((E <= froz_max) * (E >= froz_min))[0])
    while len(ind) > 0 and ind[0] > 0 and E[ind[0]] - E[ind[0] - 1] < thresh:
        del ind[0]
    while len(ind) > 0 and ind[-1] < len(E) - 1 and E[ind[-1] + 1] - E[ind[-1]] < thresh:
        del ind[-1]
    froz = np.zeros(E.shape, dtype=bool)
    froz[ind] = True
    return froz


def get_max_eig(matrix, nvec, nBfree):
    """ return the nvec column-eigenvectors of matrix with maximal eigenvalues.

    Parameters
    ----------
    matrix : numpy.ndarray(n,n)
        list of matrices
    nvec : int
        number of eigenvectors to return
    nBfree : int
        number of free bands

    Returns
    -------
    numpy.ndarray(n,nvec)
        eigenvectors
    """
    assert matrix.shape[0] == matrix.shape[1]
    assert matrix.shape[0] >= nvec, f"nvec={nvec}, matrix.shape={matrix.shape}"
    e, v = np.linalg.eigh(matrix)
    return v[:, np.argsort(e)[nBfree - nvec:nBfree]]


class MmnFreeFrozen:
    # TODO : make use of irreducible kpoints (maybe)
    """ a class to store and call the Mmn matrix between/inside the free and frozen subspaces,
        as well as to calculate the spreads

        Parameters
        ----------
        Mmn : list of numpy.ndarray(nnb,nb,nb)
            list of Mmn matrices
        free : list of numpy.ndarray(nk,nb)
            list of free bands at each k-point
        frozen : list of numpy.ndarray(nk,nb)
            list of frozen bands at each k-point
        neighbours : list of list of tuple
            list of neighbours for each k-point
        wb : list of numpy.ndarray(nnb)
            list of weights for each neighbour (b-vector)
        NW : int
            number of Wannier functions

        Attributes
        ----------
        Omega_I_0 : float
            the constant term of the spread functional
        Omega_I_frozen : float
            the spread of the frozen bands
        data : dict((str,str),list of numpy.ndarray(nnb,nf,nf)
            the data for the Mmn matrix for each pair of subspaces (free/frozen)
        spaces : dict
            the spaces (free/frozen)
        neighbours : list of list of tuple
            list of neighbours for each k-point
        wk : list of numpy.ndarray(nnb)
            list of weights for each neighbour (b-vector)
        NK : int
            number of k-points
        """

    def __init__(self, Mmn, free, frozen, neighbours, wb, NW):
        self.NK = len(Mmn)
        self.wk = wb
        self.neighbours = neighbours
        self.data = {}
        self.spaces = {'free': free, 'frozen': frozen}
        for s1, sp1 in self.spaces.items():
            for s2, sp2 in self.spaces.items():
                self.data[(s1, s2)] = vectorize(lambda M, s1, neigh: [M[ib][s1, :][:, sp2[ikb]] for ib, ikb in enumerate(neigh)],
                                                Mmn, sp1, self.neighbours)
        self.Omega_I_0 = NW * self.wk[0].sum()
        self.Omega_I_frozen = -sum(sum(wb * np.sum(abs(mmn[ib]) ** 2) for ib, wb in enumerate(WB)) for WB, mmn in
                                   zip(self.wk, self('frozen', 'frozen'))) / self.NK

    def __call__(self, space1, space2):
        """
        return the Mmn matrix between the given subspaces

        Parameters
        ----------
        space1, space2 : str
            the two subspaces (free/frozen)

        Returns
        -------
        list of numpy.ndarray(nnb,nf,nf)
            the Mmn matrix
        """
        assert space1 in self.spaces
        assert space2 in self.spaces
        return self.data[(space1, space2)]

    def Omega_I_free_free(self, U_opt_free):
        """
        calculate the spread of the free bands

        Parameters
        ----------
        U_opt_free : list of numpy.ndarray(nBfree,nW)
            the optimized U matrix for the free bands

        Returns
        -------
        float
            the spread of the free bands (eq. 27 of SMV2001)
        """
        U = U_opt_free
        Mmn = self('free', 'free')
        return -sum(self.wk[ik][ib] * np.sum(abs(U[ik].T.conj().dot(Mmn[ib]).dot(U[ikb])) ** 2)
                    for ik, Mmn in enumerate(Mmn) for ib, ikb in enumerate(self.neighbours[ik])) / self.NK

    def Omega_I_free_frozen(self, U_opt_free):
        """
        calculate the spread between the free and frozen bands

        Parameters
        ----------
        U_opt_free : list of numpy.ndarray(nBfree,nW)
            the optimized U matrix for the free bands

        Returns
        -------
        float
            the spread between the free and frozen bands (eq. 27 of SMV2001)
        """
        U = U_opt_free
        Mmn = self('free', 'frozen')
        return -sum(self.wk[ik][ib] * np.sum(abs(U[ik].T.conj().dot(Mmn[ib])) ** 2)
                    for ik, Mmn in enumerate(Mmn) for ib, ikb in enumerate(self.neighbours[ik])) / self.NK * 2

    def Omega_I(self, U_opt_free):
        """
        calculate the spread of the optimized U matrix

        Parameters
        ----------
        U_opt_free : list of numpy.ndarray(nBfree,nW)
            the optimized U matrix for the free bands

        Returns
        -------
        float, float, float, float
            the spreads: Omega_I_0, Omega_I_frozen, Omega_I_free_frozen, Omega_I_free_free
        """
        return self.Omega_I_0, self.Omega_I_frozen, self.Omega_I_free_frozen(U_opt_free), self.Omega_I_free_free(
            U_opt_free)
