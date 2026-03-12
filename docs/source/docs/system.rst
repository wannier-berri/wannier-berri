System
======================

The first step in the ``wannierberri`` calculation is initialising the System.  This is done by means of child classes :class:`~wannierberri.system.System` described below. 
The system may either be constructed bu :ref:`wannierisation in WannierBerri <sec-wannierisation>`, or come either from :ref:`Wanier functions <sec-wan-fun>`  constructed by `Wannier90 <http://wannier90.org>`_, 
or from ref:`tight binding <sec-tb-models>` models. Also k.p models are supported, see :class:`~wannierberri.system.SystemKP`.

.. autoclass:: wannierberri.system.System
   :members: set_pointgroup, cell_volume, recip_lattice
   :undoc-members:
   :show-inheritance:
   :member-order: bysource


Real-space systems
======================
.. autoclass:: wannierberri.system.System_R
   :members: set_structure, set_pointgroup_from_structure, set_R_mat, set_spin, set_spin_pairs, set_spin_from_projections
   :undoc-members:
   :show-inheritance:
   :member-order: bysource


Symmetrization of the system
-----------------------------

There are two interfaces: one with explicit specification of the struvcture (old interface) and one with :class:`~wannierberri.symmetry.sawf.SymmetrizerSAWF` (new interface).


.. automethod:: wannierberri.system.System_R.symmetrize


.. automethod:: wannierberri.system.System_R.symmetrize2


.. _sec-wan-fun:

From Wannier functions 
-----------------------------

Wanierisation inside WannierBerri
+++++++++++++++++++++++++++++++++++


Now WannierBerri can construct wannier functions on its own.see :ref:`sec-wannierisation`




Wannier90
+++++++++++++++++++++++++

.. automethod:: wannierberri.system_R.from_w90_data


FPLO
+++++++++++++++++++++++++

.. automethod:: wannierberri.system_R.from_fplo



ASE
+++++++++++++++++++++++++

.. automethod:: wannierberri.system_R.from_ase



.. _sec-tb-models:


From tight-binding models 
----------------------------------

``wannier90_tb.dat`` file
+++++++++++++++++++++++++

.. automethod:: wannierberri.system_R.from_tb_dat

PythTB
+++++++++

.. automethod:: wannierberri.system_R.from_pythtb

TBmodels
+++++++++

.. automethod:: wannierberri.system_R.from_tbmodels

Randomly generated
----------------------------------

.. automethod:: wannierberri.system.random_system

k-space systems
================

.. _sec-kp-model:

:math:`\mathbf{k}\cdot\mathbf{p}` models
------------------------------------------

.. autoclass:: wannierberri.system.SystemKP
   :show-inheritance:



