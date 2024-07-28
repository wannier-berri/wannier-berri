import numpy as np

from .kpoint import Kpoint_and_neighbours

from .utility import frozen_nondegen, print_centers_and_spreads, print_progress

from ..__utility import vectorize
from .sitesym import VoidSymmetrizer, Symmetrizer
from .spreadfunctional import SpreadFunctional


def wannierise(w90data,
               froz_min=np.inf,
               froz_max=-np.inf,
               num_iter=1000,
               conv_tol=1e-9,
               num_iter_converge=3,
               mix_ratio_z=0.5,
               mix_ratio_u=0.5,
               print_progress_every=10,
               sitesym=False,
               localise=True,
               kwargs_sitesym={}):
    r"""
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
        the convergence is achieved when the standard deviation of the spread functional over the `num_iter_converge*print_progress_every`
        iterations is less than conv_tol
    mix_ratio_z : float
        0 <= mix_ratio <=1  - mixing the Z matrix (disentanglement) from previous itertions. 1 for max speed, smaller values are more stable
    mix_ratio_u : float
        0 <= mix_ratio <=1  - mixing the U matrix (localization) from previous itertions. 1 for max speed, smaller values are more stable
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
    if froz_min > froz_max:
        print("froz_min > froz_max, nothing will be frozen")
    assert 0 < mix_ratio_z <= 1
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


    neighbours_all = w90data.mmn.neighbours
    neighbours_irreducible = np.array([[symmetrizer.kpt2kptirr[ik] for ik in neigh]
                                       for neigh in w90data.mmn.neighbours[kptirr]])

    kpoints = [Kpoint_and_neighbours(w90data.mmn.data[kpt],
                           frozen[ik], frozen[neighbours_irreducible[ik]],
        w90data.mmn.wk[kpt], w90data.mmn.bk_cart[kpt],
        symmetrizer, ik,
        amn=w90data.amn.data[kpt],
        weight=symmetrizer.ndegen(ik) / symmetrizer.NK
    )
        for ik, kpt in enumerate(kptirr)
    ]
    SpreadFunctional_loc = SpreadFunctional(
        w=w90data.mmn.wk / w90data.mmn.NK,
        bk=w90data.mmn.bk_cart,
        neigh=w90data.mmn.neighbours,
        Mmn=w90data.mmn.data)

    # The _IR suffix is used to denote that the U matrix is defined only on k-points in the irreducible BZ
    U_opt_full_IR = [kpoint.U_opt_full for kpoint in kpoints]
    # the _BZ suffix is used to denote that the U matrix is defined on all k-points in the full BZ
    U_opt_full_BZ = symmetrizer.symmetrize_U(U_opt_full_IR, all_k=True)

    # spreads = getSpreads(kpoints, U_opt_full_BZ, neighbours_irreducible)
    print_centers_and_spreads(w90data, U_opt_full_BZ,
                              spread_functional=SpreadFunctional_loc,
                              comment="Initial  State")
    # print ("  |  ".join(f"{key} = {value:16.8f}" for key, value in spreads.items() if key.startswith("Omega")))


    Omega_list = []
    for i_iter in range(num_iter):
        U_opt_full_IR = []
        for ikirr, kpt in enumerate(kptirr):
            U_neigh = ([U_opt_full_BZ[ib] for ib in neighbours_all[kpt]])
            U_opt_full_IR.append(kpoints[ikirr].update(U_neigh,
                                                       mix_ratio=mix_ratio_z,
                                                       mix_ratio_u=mix_ratio_u,
                                                       localise=localise))

        U_opt_full_BZ = symmetrizer.symmetrize_U(U_opt_full_IR, all_k=True)

        if i_iter % print_progress_every == 0:
            delta_std = print_progress(i_iter, Omega_list, num_iter_converge, 
                                    spread_functional=SpreadFunctional_loc, w90data=w90data, U_opt_full_BZ=U_opt_full_BZ)

            if delta_std < conv_tol:
                print(f"Converged after {i_iter} iterations")
                break

    U_opt_full_BZ = symmetrizer.symmetrize_U(U_opt_full_IR, all_k=True)

    print_centers_and_spreads(w90data, U_opt_full_BZ,
                              spread_functional=SpreadFunctional_loc,
                              comment="Final State")
    w90data.wannierised = True
    return w90data.chk.v_matrix
