import numpy as np

from .kpoint import Kpoint_and_neighbours

from .disentanglement import frozen_nondegen, print_centers_and_spreads, print_progress

from ..__utility import vectorize
from .sitesym import VoidSymmetrizer, Symmetrizer
from .spreadfunctional import SpreadFunctional
DEGEN_THRESH = 1e-2  # for safety - avoid splitting (almost) degenerate states between free/frozen  inner/outer subspaces  (probably too much)


def wannierise(w90data,
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

    
    
    neighbours_irreducible = np.array([[symmetrizer.kpt2kptirr[ik] for ik in neigh]
                                       for neigh in w90data.mmn.neighbours[kptirr]])
    
    kpoints = [ Kpoint_and_neighbours(w90data.mmn.data[kpt],
                           frozen[ik], frozen[neighbours_irreducible[ik]],
                           w90data.mmn.wk[kpt], w90data.mmn.bk_cart[kpt],
                           symmetrizer, ik,
                           amn = w90data.amn.data[kpt],
                           weight=symmetrizer.ndegen(ik)/symmetrizer.NK,
                           mix_ratio=mix_ratio
                           ) 
                for ik, kpt in enumerate(kptirr)
                ]
    SpreadFunctional_loc = SpreadFunctional( 
                                   w=w90data.mmn.wk/w90data.mmn.NK,
                                     bk=w90data.mmn.bk_cart,
                                       neigh=w90data.mmn.neighbours,
                                         Mmn=w90data.mmn.data)
    
    # The _IR suffix is used to denote that the U matrix is defined only on k-points in the irreducible BZ
    U_opt_full_IR = [kpoint.U_opt_full for kpoint in kpoints]
    # the _BZ suffix is used to denote that the U matrix is defined on all k-points in the full BZ
    U_opt_full_BZ = symmetrizer.symmetrize_U(U_opt_full_IR, all_k=True)

    # spreads = getSpreads(kpoints, U_opt_full_BZ, neighbours_irreducible)
    spreads = SpreadFunctional_loc(U_opt_full_BZ)
    print ("Initial state. Spread : ", spreads)
    print ("  |  ".join(f"{key} = {value:16.8f}" for key, value in spreads.items()))


    Omega_list = []
    for i_iter in range(num_iter):
        for ikirr,_ in enumerate(kptirr):
            U_neigh = [U_opt_full_BZ[neigh] for neigh in neighbours_irreducible[ikirr]]
            U_opt_full_IR[ikirr] = kpoints[ikirr].update(U_neigh)

        # spreads = getSpreads(kpoints)
        Omega_list.append(spreads["Omega_tot"]) # so far fake values
        do_print_progress = i_iter % print_progress_every == 0
        U_opt_full_BZ = symmetrizer.symmetrize_U(U_opt_full_IR, all_k=do_print_progress)
        if do_print_progress:
            spreads = SpreadFunctional_loc(U_opt_full_BZ)
            print ("  |  ".join(f"{key} = {value:16.8f}" for key, value in spreads.items()))
        

        delta_std = print_progress(i_iter, Omega_list, num_iter_converge, print_progress_every,
                                   w90data, U_opt_full_BZ)

        if delta_std < conv_tol:
            print(f"Converged after {i_iter} iterations")
            break
    
    print("U_opt_full ", [u.shape for u in U_opt_full_IR])
    U_opt_full_BZ = symmetrizer.symmetrize_U(U_opt_full_IR, all_k=True)

    w90data.chk.v_matrix = np.array(U_opt_full_BZ)
    w90data.chk._wannier_centers, w90data.chk._wannier_spreads = w90data.chk.get_wannier_centers(w90data.mmn, spreads=True)
    print_centers_and_spreads(w90data, U_opt_full_BZ)

    w90data.wannierised = True
    return w90data.chk.v_matrix


