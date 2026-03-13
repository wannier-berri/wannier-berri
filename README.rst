=========
Wannier Berri
=========

.. image:: https://codecov.io/gh/wannier-berri/wannier-berri/branch/master/graph/badge.svg?token=S7CH32VXRP
  :target: https://codecov.io/gh/wannier-berri/wannier-berri


A code for construction pf Wannier functions and Wannier interpolation
-----------------------------------------------------------------------
* Symmetry-adapted Wannier functions
* Symmetrization of the Hamiltonian and matrix elements
* Fast evaluation of k-space integrals
* Berry-type quantities: 
  - Berry curvature
  - Orbital moment
  - Berry curvature dipole
  - gyrotropic magnetoelectric effect
  - Anomalous Hall conductivity 
  - Optical conductivity
  - Low-field Hall effect
  - eMChA
  - many more .... 


Web page
--------
http://wannier-berri.org


Full documentation
------------------
https://docs.wannier-berri.org/

Tutorials
=========
https://tutorial.wannier-berri.org



Feedback:
-------------
Preferrably, Discussions and Issues on GitHub should be used for any consultations.

To subscribe please send an email to  sympa@physik.lists.uzh.ch  with the subject
**subscribe wannier-berri Firstname Lastname**
or visit the list homepage https://physik.lists.uzh.ch/sympa/info/wannier-berri



Improved performance and accuracy:
----------------------------------
Wannier-Berri calculates Brillouin zone integrals very fast with high precision over an 
ultradense k-grid. This is achieved due to :

* Using Fast Fourier Transform
* account of symmetries, to reduce integration to irreducible part of the Brillouin zone
* recursive adaptive refinement algorithm
* optimized Fermi level scan
* optimized minimal distanse replica method (``use_ws_distance``)


Other features:
---------------
* Object oriented structure also makes it potentially easier to implement further features. 
* Calculations may also be performed for any **tight-binding model** or **k.p model**, not only for Wannier functions.
* WannierBerri can run in parallel by means of **ray** module

Installation
------------
``pip install wannierberri[default]``

(optionally, for minimum installation ``pip install wannierberri`` or for full installation with all dependencies ``pip install wannierberri[all]``)

Author
------
Stepan Tsirkin, 
University of Zurich
(At present: EPFL, Lausanne, Switzerland)

License
--------
The code is distributed under the terms of  GNU GENERAL PUBLIC LICENSE  Version 2

