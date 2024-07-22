|NEW| Wanierisation inside WannierBerri
========================================
.. _sec-wannierisation

Now WannierBerri can construct wannier functions on its own. It does only disentanglement, but not maximal localisation.

However, disentanglement can be performed with both `sitesym=True` and **frozen window** (this feature is currently not available in Wannier90).

Then, WanierBerri can write the disentangled files (such that `num_bands==num_wann`), and those can be used in Wannier90 for maximal localisation ( without need for further disentanglement )

see examples in `examples/sitesym` and `examples/wannierise`

.. automethod:: wannierberri.wannierise.disentangle

Example:
--------

.. literalinclude:: ../../../examples/sitesym/disentangle-sitesym.py
   :language: python
   :linenos:

.. |NEW| image:: ../imag/new.png
   :width: 100px
   :alt: NEW!