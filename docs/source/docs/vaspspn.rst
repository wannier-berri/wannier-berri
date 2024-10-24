
.. _sec-vaspspn:

vaspspn
======================

**Note**: New versions of VASP compute the spn file directly, so this utility is mostly obsolete. see `LWRITE_SPN <https://www.vasp.at/wiki/index.php/LWRITE_SPN>`__ 

To interpolate the spin operator expectation value, the matrix
:math:`s_{mn}({\bf q})=\langle u_{m{\bf q}}\vert\hat{\sigma}\vert u_{n{\bf q}}\rangle`
is needed. To facilitate study of spin-dependent properties within `VASP <https://www.vasp.at/>`_
code, a submodule `wannierberri.utils.vaspspn` is
included, which computes :math:`S_{mn{\bf q}}` based on the normalized
pseudo-wavefunction read from the ``WAVECAR`` file. Note that the use of
pseudo-wavefunction instead of the full PAW (`Blöchl 1994 <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.50.17953>`_) wavefunction
is an approximation, which however in practice gives a rather accurate
interpolation of spin.


.. automodule:: wannierberri.utils.vaspspn
   :members:
   :no-undoc-members:
   :show-inheritance:
