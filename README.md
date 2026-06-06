# Wannier Berri

[![codecov](https://codecov.io/gh/wannier-berri/wannier-berri/branch/master/graph/badge.svg?token=S7CH32VXRP)](https://codecov.io/gh/wannier-berri/wannier-berri)

A code for construction of Wannier functions and Wannier interpolation.

- Symmetry-adapted Wannier functions
- Symmetrization of the Hamiltonian and matrix elements
- Fast evaluation of k-space integrals
- Berry-type quantities:
  - Berry curvature
  - Orbital moment
  - Berry curvature dipole
  - Gyrotropic magnetoelectric effect
  - Anomalous Hall conductivity
  - Optical conductivity
  - Low-field Hall effect
  - eMChA
  - and many more

## Web page

http://wannier-berri.org

## Full documentation

https://docs.wannier-berri.org/

## Tutorials

https://tutorial.wannier-berri.org

## Feedback

Preferably, Discussions and Issues on GitHub should be used for consultations.

To subscribe, send an email to `sympa@physik.lists.uzh.ch` with the subject:

`subscribe wannier-berri Firstname Lastname`

or visit: https://physik.lists.uzh.ch/sympa/info/wannier-berri

## Improved performance and accuracy

Wannier-Berri calculates Brillouin-zone integrals very fast with high precision over an ultra-dense k-grid. This is achieved by:

- Fast Fourier Transform
- Symmetry reduction to the irreducible Brillouin zone
- Recursive adaptive refinement
- Optimized Fermi-level scan
- Optimized minimal-distance replica method (`use_ws_distance`)

## Other features

- Object-oriented structure that makes extending features easier
- Support for tight-binding models and k·p models, not only Wannier functions
- Parallel execution via the **ray** module

## Installation

`pip install wannierberri[default]`

Optionally:

- minimal install: `pip install wannierberri`
- full install: `pip install wannierberri[all]`

## Author

Stepan Tsirkin

University of Zurich  
At present: EPFL, Lausanne, Switzerland

## License

The code is distributed under the terms of GNU GENERAL PUBLIC LICENSE Version 2.

