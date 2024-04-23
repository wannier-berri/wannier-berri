Calculators
=============

.. _sec-calculators:

.. autoclass:: wannierberri.calculators.Calculator




Static (dependent only on Fermi level)
+++++++++++++++++++++++++++++++++++++++

.. autoclass:: wannierberri.calculators.static.StaticCalculator
   :members: __init__, __call__

In the following `**kwargs` refer to the arguments of :class:`~wannierberri.calculators.static.StaticCalculator`

.. automodule:: wannierberri.calculators.static
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: StaticCalculator




Dynamic (dependent on Fermi level and frequency)
+++++++++++++++++++++++++++++++++++++++++++++++++

.. autoclass:: wannierberri.calculators.dynamic.DynamicCalculator

.. automodule:: wannierberri.calculators.dynamic
   :members: 
   :undoc-members:
   :show-inheritance:
   :exclude-members: DynamicCalculator

Tabulating
+++++++++++++++++++++++++++++++++++++++++++++++++

.. autoclass:: wannierberri.calculators.TabulatorAll
   :show-inheritance:

.. autoclass:: wannierberri.calculators.tabulate.Tabulator
   :show-inheritance:


.. automodule:: wannierberri.calculators.tabulate
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: Tabulator

.. autofunction:: wannierberri.npz_to_fermisurfer
