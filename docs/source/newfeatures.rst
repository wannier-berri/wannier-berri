|NEW| New features in version-0.10.0
#######################################

[ new features will be illustrated here with links to documentation ]

Multi-node parallelization
+++++++++++++++++++++++++++


Old WB could compute only on one node. By means or the `Ray <https://www.ray.io/>`__ module 
now WB can perform calculaions on as many nodes as needed. 
To find how to bind several nodes in a Ray cluster see   :ref:`here  <doc-parallel>`.

Fermi-sea formulas for many quantities
++++++++++++++++++++++++++++++++++++++

External vs internal terms
++++++++++++++++++++++++++++++++++++++

For example, the total Berry curvature of the occupied manifold is
evaluated by means of Wannier interpolation as  
(`Wang et al. 2006 <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.74.195118>`__).

.. math::

   \Omega_\gamma ({\bf k}) =     +\epsilon_{\alpha\beta\gamma}{\rm Im\,}\sum_n^{\text{occ}}\sum_l^{\text{unocc}}D_{nl,\alpha} D_{ln,\beta} +
  {\rm Re\,}\sum_n^{\text{occ}}\overline{\Omega}^{\rm H}_{nn,\gamma}
   -2\epsilon_{\alpha\beta\gamma}{\rm Re\,}\sum_n^{\text{occ}}\sum_l^{\text{unocc}}D_{nl,\alpha}\overline{A}^{\rm H}_{ln,\beta} 

The first term 'internal', and the rest - 'external'. This distinction is motivated 
by the fact that if we employ the tight-binding convention (Convention I, see below) for
the Bloch sums, then the external terms vanish for a tight-binding model, 
where the position matrix elements 
:math:`\langle \mathbf{0}i|\mathbf{r} | \mathbf{R}j\rangle = \delta_{ij}\delta_{\mathbf{R},\mathbf{0}} \mathbf{t}_j`
between the basis orbitals are usually assumed diaginal. 
So, for tight-binding models one can proceed as 

.. code:: python

   system=wberri.System_PythTB(model,berry=False,use_wcc_phase=True)
   wberri.integrate(system, grid, 
               Efermi=np.linspace(12.,13.,1001), 
               smearEf=10, # 10K
               quantities=["ahc"],
               parameters = {'external_terms':False}
               parallel = parallel,
               adpt_num_iter=10,
               fout_name="Fe")

Also, for ab initio Wannier functions it makes sence to separate the evaluation of internal and external terms, 
for better efficiency.
Recall that :math:`D_{ln,\beta}=\langle l| \partial_k H |r \rangle/(E_n-E_l)`
therefore, the internal terms have a very oscillatory behaviour in k-space in regions where degeneracies or avoided crossings
happen near the Fermi level. Therefore, internal terms need a much denser grid to  converge. In turn, the external terms are 
heavier to calculate, but are less oscillating, and hence require a smaller grid.


Tetrahedron method
+++++++++++++++++++

Fermi-surface properties can be evaluated with higher accuracy and without 
need for smeared :math:`\delta`-function. just add ``parameters = {'tetra':True}``


One quantity with differnet parameters in one run
++++++++++++++++++++++++++++++++++++++++++++++++++

Now the same quantity may be evasluated in one run with different sets of parameters, e.g. to compare their effect. 
See :ref:`doc-parameters` for details.


user-defined quantities
+++++++++++++++++++++++++

more quantities to tabulate
++++++++++++++++++++++++++++

two conventions for the Bloch sums
++++++++++++++++++++++++++++++++++++++++++++++++++++++++

We implement both conventions for the Bloch sums, as defined in 
`PythonTB documentation  <http://www.physics.rutgers.edu/pythtb/formalism.html>`__

Basis functions for Bloch states maybe written in one of the two ways:

+ Convention I

.. math::

    |\chi_j^\mathbf{k}\rangle  = 
        \sum_{\mathbf{R}} e^{i\mathbf{k}\cdot(\mathbf{R}+\mathbf{t}_j)} 
            | \mathbf{R}j \rangle


+ Convention II

.. math::

    |\chi_j^\mathbf{k}\rangle = 
        \sum_{\mathbf{R}} e^{i\mathbf{k}\cdot \mathbf{R}} 
            | \mathbf{R}j \rangle

The default (so far) is Convention II (follwoing the tradition of Wannier90), 
while Convention I is activated by ``use_wcc_phase=True`` in :class:`wannierberri.System` 
The two conventions give the same result, but different distribution between internal 
and external terms (see above) With ``use_wcc_phase=True`` the external terms usually become
smaller (or even vanish for the tight-binding models), therefore it is a preferred way
(however, not implemented for all quantities yet)


pre-defined TB models for illustration and testing
++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Haldane model and its 3D stacked version :ref:`see here <doc-models>`



.. |NEW| image:: imag/NEW.jpg
   :width: 50px
   :alt: NEW!
