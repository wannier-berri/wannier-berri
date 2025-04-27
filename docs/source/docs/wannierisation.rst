.. _sec-wannierisation:

========================================
|NEW| Wannierisation inside WannierBerri
========================================

Now WannierBerri can construct wannier functions on its own. Both disentanglement and localisation.

Wannierisation can be performed with both `sitesym=True` and **frozen window** (this feature is currently not available in Wannier90).

see examples in `examples/wannierise+sitesym-Fe` and `examples/wannierise+sitesym-diamond`

.. automethod:: wannierberri.wannierise.wannierise

Example
====================

.. code-block:: python

   # This tutorial shows how to generate a DMN file inside Wanier Berri code (without pw2wannier90)
   # and then use it to generate a Wannier90 output.
   # It may be used with any DFT code that is supported by IrRep (QE, VASP, AINIT, ...)
   # the .dmn file is not needed before the calculation, it is generated on the fly

   import wannierberri as wberri
   from wannierberri.w90files import DMN
   from irrep.bandstructure import BandStructure

   # Create the dmn file
   try:
      dmn_new = DMN("mydmn")
   except:
         # see documentation of irrep for usage with other codes
      bandstructure = BandStructure(code='espresso', prefix='di', Ecut=100,
                                 normalize=False)

      bandstructure.spacegroup.show()
      dmn_new = DMN(empty=True)
      dmn_new.from_irrep(bandstructure)
      pos = [[0,0,0],[0,0,1/2],[0,1/2,0],[1/2,0,0]]
      spacegroup = bandstructure.spacegroup
      proj_s = Projection(position_num=pos, orbital='s', spacegroup=spacegroup)
      dmn_new.set_D_wann_from_projections([proj_s])
      dmn_new.to_w90_file("mydmn")

   # Read the data from the Wanier90 inputs 
   w90data = wberri.w90files.Wannier90data(seedname='diamond')
   w90data.set_file("dmn", dmn_new)

   #Now wannierise with sitesym and frozen window (the part that is not implemented in Wanier90)
   w90data.wannierise(
                  froz_min=0,
                  froz_max=4,
                  num_iter=1000,
                  conv_tol=1e-6,
                  mix_ratio_z=1.0,
                  print_progress_every=20,
                  sitesym=True,
                  localise=True,
                  )

   system = wberri.system.System_w90(w90data=w90data)

   # Now do whatever you want with the system, as if it was created with Wannier90


.. |NEW| image:: ../imag/new.png
   :width: 100px
   :alt: NEW!