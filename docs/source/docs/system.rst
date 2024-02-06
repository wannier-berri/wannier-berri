Initializing a System
======================

The first step in the ``wannierberri`` calculation is initialising the System.  This is done by means of child classes :class:`~wannierberri.__system.System` described below. 
They all have an important common method :func:`~wannierberri.System.set_symmetry`.
The system may come either from :ref:`Wanier functions <sec-wan-fun>`  constructed by `Wannier90 <http://wannier90.org>`_, or from ref:`tight binding <sec-tb-models>` models. 

.. autoclass:: wannierberri.system.System
   :members: set_symmetry
   :undoc-members:
   :show-inheritance:
   :member-order: bysource


Real-space systems
======================
.. autoclass:: wannierberri.system.System_R
   :members: set_structure, set_symmetry_from_structure, set_R_mat, set_spin, set_spin_pairs, set_spin_from_code
   :undoc-members:
   :show-inheritance:
   :member-order: bysource


Symmetrization of the system
-----------------------------

.. automethod:: wannierberri.system.System_R.symmetrize


.. _sec-wan-fun:

From Wannier functions 
-----------------------------

Wannier90
+++++++++++++++++++++++++

.. autoclass:: wannierberri.system.System_w90
   :show-inheritance:


Wanierisation inside WannierBerri
+++++++++++++++++++++++++++++++++++
.. _sec-wannierisation:

.. automethod:: wannierberri.system.disentangle

Example:

.. code:: python

   import wannierberri as wberri
   # assume that files are in folder ./path/to and the seedname is `Fe` (files `Fe.win`, `Fe.chk`, etc)
   w90data = wberri.system.Wannier90data(sedname="path/to/Fe")
   wberri.system.disentangle(w90data,
                 froz_min=-8,
                 froz_max=20,
                 num_iter=2000,
                 conv_tol=5e-7,
                 mix_ratio=0.9,
                 print_progress_every=100
   )
   system=wberri.system.System_w90(w90data=w90data,berry=True, morb=True)
   del w90data # recommended to save memory,  we may not need it anymore



FPLO
+++++++++++++++++++++++++

.. autoclass:: wannierberri.system.System_fplo
   :show-inheritance:


ASE
+++++++++++++++++++++++++

.. autoclass:: wannierberri.system.System_ASE
   :show-inheritance:



.. _sec-tb-models:


From tight-binding models 
----------------------------------

``wannier90_tb.dat`` file
+++++++++++++++++++++++++

.. autoclass:: wannierberri.system.System_tb
   :show-inheritance:

PythTB
+++++++++

.. autoclass:: wannierberri.system.System_PythTB
   :show-inheritance:

TBmodels
+++++++++

.. autoclass:: wannierberri.system.System_TBmodels
   :show-inheritance:


k-space systems
================

.. _sec-kp-model:

:math:`\mathbf{k}\cdot\mathbf{p}` models
------------------------------------------

.. autoclass:: wannierberri.system.SystemKP
   :show-inheritance:



