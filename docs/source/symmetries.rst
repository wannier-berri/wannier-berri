Symmetries
==========

Symmetries may be involved in WannierBerri on three different levels

1. Point group symmetries
    are used in the :func:`~wannierberri.run` function to reduce the number of k-points and to make the 
    resulting tensor symmetric.  See :func:`~wannierberri.system.System.set_symmetry`

2. Symmetrization of  the Hamiltonian and matrix elements
    typically the Wannier modelcreated by Wannier90 slightly (or strongly) breaks the symmetries. 
    To fix this a simmetrization procedure is implemented. See :func:`~wannierberri.system.System_R.symmetrize`

3. NEW! Symmetry adapted Wannier functions. 
    following the paper by R. Sakuma `Phys. Rev. B 87, 235109 (2013) <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.87.235109>`__
    the wannier functions can be kept symmetric on every step of wannierisation. See the `sitesym` parameter of
    :func:`~wannierberri.wannierise.wannierise` and :class:`~wannierberri.symmetry.sawf.SymmetrizerSAWF` class.
Note:

    * Any combination can be used in one workflow, and it is best to use all of them.  
      (at the moment it should be done explicitly, later it will be unified)

    * Even if (3) was used, some matrix elements still may be not exactly symmetric, 
      depending on the finite difference scheme used to evaluate those matrix elements. 
      However, the Hamiltonian will be symmetric.

    * if (2) was used, (1) will not affect the results, but will improve performance (due to reduction of 
      the number of k-points)