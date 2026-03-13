.. _sec-wannierisation:

========================================
|NEW| Wannierisation inside WannierBerri
========================================

Now WannierBerri can construct wannier functions on its own. Both disentanglement and localisation.

Wannierisation can be performed with both `sitesym=True` and **frozen window** (this feature is currently not available in Wannier90).

see examples in `examples/wannierise+sitesym-Fe` and `examples/wannierise+sitesym-diamond`

.. automethod:: wannierberri.wannierise

Example
====================

.. code-block:: python

   # This tutorial shows how to generate symmetry-adapted wannier functions with WannierBerri
   # It may be used with any DFT code that is supported by IrRep (QE, VASP, AINIT, ...)
   # the .dmn and .amn files are NOT needed before the calculation, they are generated on the fly
   # from the Wavefunctions stored by the DFT code (e.g. from the WAVECAR. .save files, etc)

   from wannierberri.symmetry.projections import Projection, ProjectionsSet
   from irrep.bandstructure import BandStructure
   import wannierberri as wb

   path_data = "../../tests/data/Fe-222-pw/"
   w90data = wb.w90files.WannierData.from_w90_files(seedname=path_data + "Fe", files=["mmn", "eig", "win", ])


   bandstructure = BandStructure(code='espresso',
                              prefix=path_data + "/Fe",
                              Ecut=200,
                              normalize=True,
                              magmom=[[0, 0, 1.]],
                              include_TR=True,
                              )
   spacegroup = bandstructure.spacegroup
   spacegroup.show()
   # exit()
   # symmetrizer = wb.symmetry.sawf.SymmetrizerSAWF.from_npz(path_data + f"/Fe_TR={False}.sawf.npz")
   symmetrizer = wb.symmetry.sawf.SymmetrizerSAWF.from_irrep(bandstructure)

   projection_s = Projection(orbital='s', position_num=[0, 0, 0], spacegroup=spacegroup)
   projection_p = Projection(orbital='p', position_num=[0, 0, 0], spacegroup=spacegroup)
   projection_d = Projection(orbital='d', position_num=[0, 0, 0], spacegroup=spacegroup)
   projection_sp3d2 = Projection(orbital='sp3d2', position_num=[0, 0, 0], spacegroup=spacegroup)
   projection_t2g = Projection(orbital='t2g', position_num=[0, 0, 0], spacegroup=spacegroup)

   # projections_set = ProjectionsSet(projections=[projection_s, projection_p, projection_d])
   projections_set = ProjectionsSet(projections=[projection_sp3d2, projection_t2g])

   symmetrizer.set_D_wann_from_projections(projections_set)
   amn = wb.w90files.AMN.from_bandstructure(bandstructure, projections=projections_set)
   w90data.set_symmetrizer(symmetrizer)
   w90data.set_file("amn", amn)

   wb.wannierise(w90data=w90data,
                 init="amn",
                 froz_min=-10,
                 froz_max=20,
                 outer_min=-8,
                 outer_max=50,
                 print_progress_every=10,
                 num_iter=101,
                 conv_tol=1e-10,
                 mix_ratio_z=1.0,
                 sitesym=True,
                 parallel=False
                 )

   system = wb.system.System_w90(w90data=w90data, berry=True)
   # Now do whatever you want with the system, as if it was created with Wannier90


.. |NEW| image:: ../imag/new.png
   :width: 100px
   :alt: NEW!