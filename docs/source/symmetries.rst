Symmetries
==========

Symmetries may be involved in WannierBerri on three different levels:

* **Point group symmetries** are used in :func:`~wannierberri.run` to reduce the
  number of k-points and to make the resulting tensor symmetric. See
  :func:`~wannierberri.system.System.set_pointgroup`.

* **Symmetrization of the Hamiltonian and matrix elements** fixes the symmetry
  breaking that is often present in Wannier models created by Wannier90. See
  :func:`~wannierberri.system.System_R.symmetrize`.

* **NEW! Symmetry-adapted Wannier functions** follow the approach of
  R. Sakuma, `Phys. Rev. B 87, 235109 (2013) <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.87.235109>`__.
  In this workflow the Wannier functions are kept symmetric at every step of
  wannierisation. See the `sitesym` parameter of
  :func:`~wannierberri.wannierise.wannierise` and the
  :class:`~wannierberri.symmetry.sawf.SymmetrizerSAWF` class.

Note:

    * Any combination can be used in one workflow, and it is best to use all of them.
      (at the moment it should be done explicitly, later it will be unified)

    * Even if symmetry-adapted Wannier functions are used, some matrix elements still may be not exactly symmetric,
      depending on the finite difference scheme used to evaluate those matrix elements. 
      However, the Hamiltonian will be symmetric.

    * If Hamiltonian symmetrization is used, point-group reduction will usually not
      change the final result, but it can still improve performance by reducing the
      number of k-points.