System
======================

The first step in the ``wannierberri`` calculation is initialising the System.  This is done by means of child classes :class:`~wannierberri.system.System` described below. 
The system may either be constructed by :ref:`wannierisation in WannierBerri <sec-wannierisation>`, or come either from :ref:`Wannier functions <sec-wan-fun>`  constructed by `Wannier90 <http://wannier90.org>`_, 
or from :ref:`tight binding <sec-tb-models>` models. Also k.p models are supported, see :class:`~wannierberri.system.SystemKP`.

.. autoclass:: wannierberri.system.System
   :members: set_pointgroup, cell_volume, recip_lattice
   :undoc-members:
   :show-inheritance:
   :member-order: bysource


Real-space systems
======================
.. autoclass:: wannierberri.system.System_R
   :members: set_structure, set_pointgroup_from_structure, set_R_mat, set_spin_pairs
   :undoc-members:
   :show-inheritance:
   :member-order: bysource


Symmetrization of the system
-----------------------------

There are two interfaces: one with explicit specification of the structure (old interface) and one with :class:`~wannierberri.symmetry.sawf.SymmetrizerSAWF` (new interface).


.. automethod:: wannierberri.system.System_R.symmetrize


.. automethod:: wannierberri.system.System_R.symmetrize2


.. _sec-wan-fun:

From Wannier functions 
-----------------------------

Wannierisation inside WannierBerri
+++++++++++++++++++++++++++++++++++


Now WannierBerri can construct Wannier functions on its own. See :ref:`sec-wannierisation`




Wannier90
+++++++++++++++++++++++++

.. automethod:: wannierberri.system.System_R.from_wannierdata

.. autofunction:: wannierberri.system.system_w90.get_system_w90


FPLO
+++++++++++++++++++++++++

.. automethod:: wannierberri.system.System_R.from_fplo

.. autofunction:: wannierberri.system.system_fplo.get_system_fplo

ASE
+++++++++++++++++++++++++

.. automethod:: wannierberri.system.System_R.from_ase

.. autofunction:: wannierberri.system.system_ase.get_system_ase

.. _sec-tb-models:


From tight-binding models 
----------------------------------

``wannier90_tb.dat`` file
+++++++++++++++++++++++++

.. automethod:: wannierberri.system.System_R.from_tb_dat

.. autofunction:: wannierberri.system.system_tb.system_tb

PythTB
+++++++++

.. automethod:: wannierberri.system.System_R.from_pythtb

.. autofunction:: wannierberri.system.system_tb_py.get_system_pythtb

TBmodels
+++++++++

.. automethod:: wannierberri.system.System_R.from_tbmodels

.. autofunction:: wannierberri.system.system_tb_py.get_system_tbmodels

Randomly generated
----------------------------------

.. automethod:: wannierberri.system.System_R.from_random

.. autofunction:: wannierberri.system.system_random.get_system_random

k-space systems
================

.. _sec-kp-model:

:math:`\mathbf{k}\cdot\mathbf{p}` models
------------------------------------------

.. autoclass:: wannierberri.system.SystemKP
   :show-inheritance:



