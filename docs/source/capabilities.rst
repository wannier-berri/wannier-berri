
.. _sec-capabilities:

*********************
Capabilities  
*********************

.. role:: red
.. role:: green


Integration
-----------

The code may be used to evaluate the following quantities, represented
as Brillouin zone integrals (by means of the |integrate| function):

Static (frequency-independent) quantities
++++++++++++++++++++++++++++++++++++++++++

-  ``'ahc'`` :  intrinsic anomalous Hall conductivity
   :math:`\sigma_{\alpha\beta}^{\rm AHE}` (`Nagaosa et al. 2010 <https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.82.1539>`_) via

   .. math:: \sigma_{\alpha\beta}^{\rm AHE}=-\frac{e^2}{\hbar}\epsilon_{\alpha\beta\gamma}\int \frac{d{\bf k}}{(2\pi)^3}\Omega_\gamma({\bf k}).


-  Anomalous Nernst conductivity (`Xiao et al. 2006 <https://doi.org/10.1103/PhysRevLett.97.026603>`_)
   :math:`\alpha_{\alpha\beta}^{\rm ANE}` may be obtained from
   :math:`\sigma_{\alpha\beta}(\epsilon)^{\rm AHE}` evaluated over a
   dense grid of Fermi levels :math:`\epsilon`

   .. math:: 
       :label: eq-ANE

       \alpha_{\alpha\beta}^{\rm ANE}=-\frac{1}{e}\int d\varepsilon \frac{\partial f}{\partial\varepsilon}\sigma_{\alpha\beta}(\varepsilon)\frac{\varepsilon-\mu}{T}, \label{eq:ANE}

   where
   :math:`f(\varepsilon)=1/\left(1+e^\frac{\varepsilon-\mu}{k_{\rm B}T}\right)`;

-  ``'Morb'`` : orbital magnetization (`Lopez et al. 2012 <https://doi.org/10.1103/PhysRevB.85.014435.>`_)

   .. math::

      M^\gamma_n({\bf k})=\frac{e}{2\hbar}{\rm Im\,}\epsilon_{\alpha\beta\gamma}\int[d{\bf k}]\sum_n^{\rm occ}\Bigl[
      \langle\partial_a u_{n{\bf k}}\vert H_{\bf k}+E_{n{\bf k}}-2E_F\vert\partial_b u_{n{\bf k}}\rangle\Bigr];

-  ``'berry_dipole'`` and ``'berry_dipole_fsurf'`` : berry curvature dipole

   .. math::

      D_{\alpha\beta}(\mu)=\int[d{\bf k}]\sum_n^{\rm occ} \partial_\alpha \Omega_n^{\beta}= \int[d{\bf k}]\sum_n^{\rm occ} \partial_\alpha E_{n\mathbf{k}} \Omega_n^{\beta} \delta(E_{n\mathbf{k}}-\mu) 

   which describes nonlinear Hall effect (`Sodemann and Fu 2015 <https://link.aps.org/doi/10.1103/PhysRevLett.115.216806>`_);


-  ``'gyrotropic_Korb'`` and ``'gyrotropic_Kspin' :`` gyrotropic
   magnetoelectric effect (GME) (`Zhong, Moore, and Souza 2016 <https://link.aps.org/doi/10.1103/PhysRevLett.116.077201>`_) tensor
   (orbital and spin contributions) in the Fermi-sea formulation:

   .. math:: K_{\alpha\beta}(\mu)=\int[d{\bf k}]\sum_n^{\rm occ}  \partial_\alpha m_n^{\beta} ; \label{eq:gyro-K}

-  ``'gyrotropic_Korb_fsurf'`` and ``'gyrotropic_Kspin_fsurf'`` :  gyrotropic
   magnetoelectric effect (GME) (`Zhong, Moore, and Souza 2016 <https://link.aps.org/doi/10.1103/PhysRevLett.116.077201>`_) tensor
   (orbital and spin contributions) in the Fermi-surface formulation:

   .. math:: 

      K_{\alpha\beta}(\mu)=\int[d{\bf k}]\sum_n^{\rm occ}  \partial_\alpha E_{n\mathbf{k}} m_n^{\beta} \delta (E_{n{\bf k}}-\mu) 


-  ``'conductivity_Ohmic'`` and  ``'conductivity_Ohmic_fsurf'`` ohmic conductivity within the Boltzmann
   transport theory in constant relaxation time (:math:`\tau`) 
   - Femi-sea and Fermi-surface formula approximation:

   .. math::

      \sigma_{\alpha\beta}^{\rm Ohm}(\mu)
      =\tau\int[d{\bf k}]\sum_n^{E_{n{\bf k}}<\mu} \partial^2_{\alpha\beta} E_{n{\bf k}}
      =\tau\int[d{\bf k}]\sum_n^{\rm occ} \partial_\alpha E_{n{\bf k}}\partial_\beta E_{n{\bf k}} \delta(E_{n{\bf k}}-\mu) 
            ; \label{eq:ohmic}

-  ``'dos'``: density of states :math:`n(E)`

-  ``'cumdos'``: cumulative density of states

   .. math::

      N(E) = \int\limits_{-\infty}^En(\epsilon)d\epsilon.
          \label{eq:cDOS}

-  ``'shc_static_ryoo'`` and ``'shc_static_qiao'``: Kubo-Greenwood formula for static spin Hall conductivity (SHC) (`Ryoo, Park, and Souza 2019 <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.235113>`_) or (`Qiao, Zhou, Yuan, and Zhao 2018 <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.98.214402>`_). Equivalent to setting :math:`\omega=0` in  ``'opt_SHCryoo'`` and ``'opt_SHCqiao'``.

   .. math::

      \sigma^{\gamma}_{\alpha\beta}(\mu) =
      \frac{e\hbar}{N_k\Omega_c} \sum_{\bf k} \sum_n^{\rm occ}
      \Omega^{{\rm spin};\,\gamma}_{\alpha\beta, n}({\bf k}),

   where

   .. math::

      \Omega^{{\rm spin};\,\gamma}_{\alpha\beta, n}({\bf k}) = -2 {\rm Im} \sum_l^{\rm unocc}
      \frac{\langle\psi_{n{\bf k}}\vert \frac{1}{2} \{ s^{\gamma}, v_\alpha \} \vert\psi_{l{\bf k}}\rangle
      \langle\psi_{l{\bf k}}\vert v_\beta\vert\psi_{n{\bf k}}\rangle}
      {(\varepsilon_{n{\bf k}}-\varepsilon_{l{\bf k}})^2}.


Dynamic (frequency-dependent) quantities
++++++++++++++++++++++++++++++++++++++++++

-  ``'opt_conductivity'``: Kubo-greenwood formula for optical conductivity (:ref:`example <sec-optconf-example>`)

   .. math::
      :label: optcondform

      \sigma_{\alpha\beta}(\hbar\omega)=\frac{ie^2\hbar}{N_k\Omega_c}
      \sum_{\bf k}\sum_{n,m}
      \frac{f_{m{\bf k}}-f_{n{\bf k}}}
      {\varepsilon_{m{\bf k}}-\varepsilon_{n{\bf k}}}
      \frac{\langle\psi_{n{\bf k}}\vert v_\alpha\vert\psi_{m{\bf k}}\rangle
      \langle\psi_{m{\bf k}}\vert v_\beta\vert\psi_{n{\bf k}}\rangle}
      {\varepsilon_{m{\bf k}}-\varepsilon_{n{\bf k}}-(\hbar\omega+i\eta)}.


-  ``'opt_shiftcurrent'``: shift photocurrent (`PRB 2018 <https://doi.org/10.1103/PhysRevB.97.245143>`_)

   .. math::
      :label: shiftcurrent

      \sigma^{abc}(0;\omega,-\omega) = -\frac{i\pi e^3}{4\hbar^2}
      \sum_{\bf k}\sum_{n,m}\left( f_{n{\bf k}}-f_{m{\bf k}} \right)
      \left(I^{abc}_{mn}+I^{acb}_{mn}\right)
      \times \left[\delta(\omega_{mn}-\omega)+\delta(\omega_{nm}-\omega)\right].

   where :math:`I^{abc}_{mn}=r^b_{mn}r^{c;a}_{nm}`;  :math:`r^a_{\mathbf{k}nm}=(1-\delta_{nm})A^a_{\mathbf{k} nm}`;
   :math:`r^{a;b}_{\mathbf{k} nm}=\partial_b r^a_{\mathbf{k} nm} -i\left(A^b_{\mathbf{k}nn}-A^b_{\mathbf{k} mm}\right)r^a_{\mathbf{k} nm}`;
   :math:`A^a_{\mathbf{k} nm}=i\langle{u_{\mathbf{k} n}}|{\partial_a u_{\mathbf{k} m}}\rangle`.

-  ``'opt_SHCryoo'`` and ``'opt_SHCqiao'``: Kubo-Greenwood formula for spin Hall conductivity (SHC) under time-reversal symmetry (`Ryoo, Park, and Souza 2019 <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.235113>`_) or (`Qiao, Zhou, Yuan, and Zhao 2018 <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.98.214402>`_)

   .. math::

      \sigma^{\gamma}_{\alpha\beta}(\hbar\omega)=\frac{-e\hbar}{N_k\Omega_c}
      \sum_{\bf k}\sum_{n,m}
      \left(f_{n{\bf k}}-f_{m{\bf k}}\right)
      \frac{\textrm{Im}\left[\langle\psi_{n{\bf k}}\vert \frac{1}{2}\{ s^{\gamma}, v_\alpha \} \vert\psi_{m{\bf k}}\rangle
      \langle\psi_{m{\bf k}}\vert v_\beta\vert\psi_{n{\bf k}}\rangle\right]}
      {(\varepsilon_{n{\bf k}}-\varepsilon_{m{\bf k}})^2-(\hbar\omega+i\eta)^2}.



Tabulating
----------

.. _figFefrmsf:
.. figure:: imag/figures/Fe-berry.pdf.svg

   Fermi surface of bcc iron, colored by the Berry curvature
   :math:`\Omega_z`. Figure produced using `FermiSurfer <https://fermisurfer.osdn.jp/>`_.

``WannerBerri`` can also tabulate certain band-resolved quantities over the
Brillouin zone. This feature is called with |tabulate| function, e.g.

.. code:: python

   WB.tabulate(system, grid,
                quantities=["berry"],
                fout_name="Fe",
                numproc=16,
                ibands=np.arange(4,10),
                Ef0=12.610)

which will produce files ``Fe_berry-?.frmsf``, containing the Energies
and Berry curvature of bands ``4-9`` (band counting starts from zero).
The format of the files allows to be directly passed to the
``FermiSurfer`` visualization tool (Kawamura 2019) which can produce a
plot like :numref:`figFefrmsf`. Transformation of files to other
visualization software is straightforward.

Currently the following quantities are available to tabulate:

-  ``'berry'``: Berry curvature [Å\ :sup:`2`\]

   .. math:: \Omega^\gamma_n({\bf k})=-\epsilon_{\alpha\beta\gamma}{\rm Im\,}\langle\partial_\alpha u_{n{\bf k}}\vert\partial_\beta u_{n{\bf k}}\rangle;

-  ``'morb'``: orbital moment of Bloch states [eV·Å\ :sup:`2`\]

   .. math:: m^\gamma_n({\bf k})=\frac{e}{2\hbar}\epsilon_{\alpha\beta\gamma}{\rm Im\,}\langle\partial_\alpha u_{n{\bf k}}\vert H_{\bf k}-E_{n{\bf k}}\vert\partial_\beta u_{n{\bf k}}\rangle;

-  ``'spin'``: the expectation value of the Pauli operator [ħ]

   .. math:: \mathbf{s}_n({\bf k})=\langle u_{n{\bf k}}\vert\hat{\bf \sigma}\vert u_{n{\bf k}}\rangle;

-  ``'V'``: the band gradients [eV·Å] :math:`\nabla_{\bf k}E_{n{\bf k}}`.

- ``'spin_berry'``: Spin Berry curvature [ħ·Å\ :sup:`2`\]. Requires an additional parameter ``spin_current_type`` which can be ``"ryoo"`` or ``"qiao"``.

   .. math::

      \Omega^{{\rm spin};\,\gamma}_{\alpha\beta, n}({\bf k}) = -2 {\rm Im} \sum_{\substack{l \\ \varepsilon_{l{\bf k}} \neq \varepsilon_{n{\bf k}}}}
      \frac{\langle\psi_{n{\bf k}}\vert \frac{1}{2} \{ s^{\gamma}, v_\alpha \} \vert\psi_{l{\bf k}}\rangle
      \langle\psi_{l{\bf k}}\vert v_\beta\vert\psi_{n{\bf k}}\rangle}
      {(\varepsilon_{n{\bf k}}-\varepsilon_{l{\bf k}})^2}.


Evaluation of additional matrix elements 
-----------------------------------------

In order to produce the matrix elements that are not evaluated by a particular *ab initio* code, the following interfaces
have been developed:

mmn2uHu 
++++++++++++++++++

The |mmn2uHu| module evaluates the (``.uHu`` file) containing the matrix elements needed for orbital moment calculations

.. math::

    C_{mn}^{\mathbf{b}_1,\mathbf{b}_2}({\bf q})=\langle u_{m{\bf q}+\mathbf{b}_1}\vert\hat{H}_{\bf q}\vert u_{n{\bf q}+\mathbf{b}_2}\rangle.

on the basis of the ``.mmn`` and ``.eig`` files by means of the sum-over-states formula

.. math::

    C_{mn}^{\mathbf{b}_1,\mathbf{b}_2}({\bf q})\approx\sum_l^{l_{\rm max}}  \left(M_{lm}^{\mathbf{b}_1}({\bf q})\right)^* E_{l{\bf q}}   M_{ln}^{\mathbf{b}_2}({\bf q}).

and the (``.sHu`` and ``.sIu`` file) containing the matrix elements needed for Ryoo's spin current calculations(`Ryoo, Park, and Souza 2019 <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.235113>`_)
on the basis of the ``.mmn``, ``.spn`` and ``.eig`` files by means of the sum-over-states formula

.. math::

    \langle u_{m{\bf q}}\vert\hat{s}\hat{H}_{\bf q}\vert u_{n{\bf q}+\mathbf{b}}\rangle \approx \sum_l^{l_{\rm max}}  \left(s_{lm}({\bf q})\right)^* E_{l{\bf q}}   M_{ln}^{\mathbf{b}}({\bf q}).
.. math::

    \langle u_{m{\bf q}}\vert\hat{s}\vert u_{n{\bf q}+\mathbf{b}}\rangle \approx \sum_l^{l_{\rm max}}  \left(s_{lm}({\bf q})\right)^*   M_{ln}^{\mathbf{b}}({\bf q}).  

see :ref:`sec-mmn2uHu` for more details

vaspspn 
+++++++

The |vaspspn| computes the spin matrix

.. math:: s_{mn}({\bf q})=\langle u_{m{\bf q}}\vert\hat{\sigma}\vert u_{n{\bf q}}\rangle

based on the normalized pseudo-wavefunction read from the ``WAVECAR`` file written by 
`VASP <https://www.vasp.at/>`_

see :ref:`sec-vaspspn` for more details




The |mmn2uHu| and |vaspspn| modules were initially developed and
used in (`Tsirkin, Puente, and Souza 2018 <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.97.035158>`_) as separate scripts, but were
not published so far. Now they are included in the ``WannierBerri``
package with a hope of being useful for the community.

.. include:: shortcuts.rst