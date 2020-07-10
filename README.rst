=========
Wannier Berri
=========
(a.k.a. Wannier19)

A code for highly efficient Wannier interpolation. 
----------------------------------------------------------
Evaluation of k-space integrals of Berry curvature, orbital moment and derived quantities by means of MLWFs or tight-binding models.  Compared to postw90.x part of Wannier90 code, it has extended functional and improved performance

=========
Tutorial 
=========
took place on  June 4th 2020 at Electronic Structure Workshop  https://physics.ucmerced.edu/electronic-structure-workshop

The recording may be viewed at:
https://uzh.zoom.us/rec/share/y84qFIzs8WlIY53g-UGYdfUCB6DUaaa80SUZ-fJZy-GyE37OpaVGSfwDqVj43hk

And the presentation slides and input files for exercises are available 
https://www.dropbox.com/sh/8lt0rznh7zetagp/AABGrVWr6-1b9kMR3Wo8H92Na?dl=0


User manual  
-----------
under construction may be viewed here: https://www.overleaf.com/read/kbxxtfbnjvxx

Web page
--------
http://wannier-berri.org

Mailing list:
-------------
To subscribe please send an email to  sympa@physik.lists.uzh.ch  with the subject
**subscribe wannier-berri Firstname Lastname**
or visit the list homepage https://physik.lists.uzh.ch/sympa/info/wannier-berri



This code is intended for highly-efficient wannier interpolation.
Being initially an analog of postw90.x part of Wannier90 code, it has extended functional and improved performance. 


Improved performance and accuracy:
----------------------------------
Wannier-Berri calculates Brillouin zone integrals very fast with high precision over an 
ultradense k-grid. This is achieved due to :

* Using Fast Fourier Transform
* account of symmetries, to reduce integration to irreducible part of the Brillouin zone
* recursive adaptive refinement algorithm
* optimized Fermi level scan
* optimized minimal distanse replica method (``use_ws_distance``)

Implemented functionality:
---------------------
* Anomalous Hall conductivity
* Orbital magnetization (modern theory)
* Ohmic conductivity
* Berry curvature dipole
* gyrotropic magnetoelectric effect
* Hall effect
* Low-Field Hall effect

Other features:
---------------
* Object oriented structure also makes it potentially easier to implement further features. 
* Calculations may also be performed for any tight-binding model, for which a "_tb.dat" file was generated in watever way.
* WannierBerri can run in parallel by means of multiprocessing module

Installation
------------
``pip3 install wannierberri``

Author
------
Stepan Tsirkin, 
University of Zurich
stepan.tsirkin@physik.uzh.ch


License
--------
The code is distributed under the terms of  GNU GENERAL PUBLIC LICENSE  Version 2, the same as Wannier90

Acknowledgements
----------------
The code was inspired by the Wannier90 Fortran code:
http://www.wannier.org/ , https://github.com/wannier-developers/wannier90 . 
Some parts of the code are an adapted translation of postw90 code. 

I acknowledge Ivo Souza for a useful discussion.
