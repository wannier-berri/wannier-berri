.. _sec-mmn2uHu:

mmn2uHu
==========

Wannier interpolation starts from certain matrix elements defined on the
*ab initio* (:math:`{\bf q}`) grid. Those matrix elements should be
evaluated within the *ab initio* code, namely within its interface to
Wannier90. However, only `QuantumEspresso <https://www.quantum-espresso.org/>`_ has the
most complete interface ``pw2wannier90.x``. The other codes provide only
the basic interface, which includes the eigenenergies
:math:`E_{n{\bf q}}` (``.eig`` file) and overlaps

.. math:: M_{mn}^{\mathbf{b}}({\bf q})=\langle u_{m{\bf q}}\vert u_{n{\bf q}+\mathbf{b}}\rangle

(file ``.mmn``), where :math:`\mathbf{b}` vector connects neighbouring
:math:`{\bf q}`-points. This information allows to interpolate the band
energies (and their derivatives of any order) as well as Berry
connections (`Yates et al. 2007 <https://doi.org/10.1103/PhysRevB.75.195121>`_) and Berry curvature (`Wang et al. 2006 <https://doi.org/10.1103/PhysRevB.74.195118.>`_).
However, to evaluate the orbital moment of a Bloch state, one needs
matrix elements of the Hamiltonian (`Lopez et al. 2012 <https://doi.org/10.1103/PhysRevB.85.014435.>`_) (``.uHu`` file)

.. math::
    :label: Cmnq

    C_{mn}^{\mathbf{b}_1,\mathbf{b}_2}({\bf q})=\langle u_{m{\bf q}+\mathbf{b}_1}\vert\hat{H}_{\bf q}\vert u_{n{\bf q}+\mathbf{b}_2}\rangle.

The evaluation of :eq:`Cmnq` is very specific to the
details of the *ab initio* code, and implemented only in
``pw2wannier90.x`` and only for norm-conserving pseudopotentials. To
enable the study of properties related to the orbital moment with other
*ab initio* codes, the following workaround may be employed. By
inserting a complete set of Bloch states at a particular :math:`{\bf q}`
point
:math:`\mathbf{1}=\sum_l^\infty \vert u_{l{\bf q}}\rangle\langle u_{l{\bf q}}\vert`
we can rewrite :eq:`Cmnq` as

.. math::
    :label: Cmnqsum

    C_{mn}^{\mathbf{b}_1,\mathbf{b}_2}({\bf q})\approx\sum_l^{l_{\rm max}}  \left(M_{lm}^{\mathbf{b}_1}({\bf q})\right)^* E_{l{\bf q}}   M_{ln}^{\mathbf{b}_2}({\bf q}).

Similarly, the two quantities below allows to interpolate spin currnet matrices to calculate spin Hall conductivity. In Ryoo's scheme (`Ryoo et al. 2019 <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.235113>`_)
, they are required as input data (``.sHu`` and ``.sIu``)

.. math::
    :label: Smnqsum

    \langle u_{m{\bf q}}\vert\hat{s}\hat{H}_{\bf q}\vert u_{n{\bf q}+\mathbf{b}}\rangle \approx \sum_l^{l_{\rm max}}  \left(s_{lm}({\bf q})\right)^* E_{l{\bf q}}   M_{ln}^{\mathbf{b}}({\bf q}).

    \langle u_{m{\bf q}}\vert\hat{s}\vert u_{n{\bf q}+\mathbf{b}}\rangle \approx \sum_l^{l_{\rm max}}  \left(s_{lm}({\bf q})\right)^*   M_{ln}^{\mathbf{b}}({\bf q}).


These equations are implemented within the |mmn2uHu|
submodule, which allows to generate the ``.uHu``, ``.sHu``, and ``.sIu`` file out of ``.mmn``, ``.spn``, 
and ``.eig`` files. The equality in :eq:`Cmnqsum` and :eq:`Smnqsum` is
exact only in the limit :math:`l_{\rm max}\to\infty` and infinitely
large basis set for the wavefunctions representation. So in practice one
has to check convergence for a particular system. As an example the
bandstructure of bcc Fe was calculated based on the QE code and a
norm-sonserving pseudopotential from the PseudoDojo library(van Setten
et al. 2018; Hamann 2013). Next, the orbital magnetization was
calculated using the ``.uHu`` file computed with ``pw2wannier90.x`` and
using the |mmn2uHu| interface with different summation
limit :math:`l_{\rm max}` in :eq:`Cmnqsum`. As can be
seen in :numref:`figmmn2uHu` already :math:`l_{\rm max}=200`
(corresponding energy :math:`\sim 230` eV) yields a result very close to
that of ``pw2wannier90.x``. However one should bear in mind that
convergence depends on many factors, such as as choice of WFs and
pseudopotentials. In particular, for tellurium we observed (Tsirkin,
Puente, and Souza 2018) that including only a few bands above the
highest :math:`p`-band is enough to obtain accurate results. However for
iron, using a pseudopotential shipped with the examples of Wannier90, we
failed to reach convergence even with :math:`l_{\rm max}=600`.

.. _figmmn2uHu:
.. figure:: ../imag/figures/Fe-Morb.pdf.svg

   Orbital magntization of bcc Fe as a function of the Fermi level
   :math:`E_F` reative to the pristine Fermi level :math:`E_F^0`,
   evaluated using the ``.uHu`` file computed with ``pw2wannier90.x``
   (dashed line) and using the |mmn2uHu| interface (solid
   lines) with different summation limit :math:`l_{\rm max}` in
   :eq:`Cmnqsum` 



.. automodule:: wannierberri.utils.mmn2uHu
   :members:
   :no-undoc-members:
   :show-inheritance:


.. |mmn2uHu| replace:: :obj:`wannierberri.utils.mmn2uHu`
