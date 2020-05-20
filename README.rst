=========
Wannier Berri
=========
(a.k.a. Wannier19)

A code for highly efficient Wannier interpolation. 
----------------------------------------------------------
Evaluation of k-space integrals of Berry curvature, orbital moment and derived quantities by means of MLWFs or tight-binding models.  Compared to postw90.x part of Wannier90 code, it has extended functional and improved performance

=========
New: Online tutorial on June 4th 2020
=========
Tutorial at Electronic Structure Workshop  
https://physics.ucmerced.edu/electronic-structure-workshop
Please register by May 22


User manual  
-----------
under construction may be viewed here: https://www.overleaf.com/read/kbxxtfbnjvxx

Web page
--------
https://www.physik.uzh.ch/en/groups/neupert/Software/Wannier-Berri.html

Mailing list:
-------------
To subscribe please send an email to  sympa@physik.lists.uzh.ch  with the subject
**subscribe wannier-berri Firstname Lastname**
or visit the list homepage https://physik.lists.uzh.ch/sympa/info/wannier-berri



This code is intended for highly-efficient wannier interpolation.
Being initially an analog of postw90.x part of Wannier90 code, it has extended functional and improved performance. 

Some parts of the code are an adapted translation of Wannier90 Fortran code:
http://www.wannier.org/
https://github.com/wannier-developers/wannier90

The code is distributed under the terms of  GNU GENERAL PUBLIC LICENSE  Version 2, the same as Wannier90

For example, Wannier-Berri calculates Anomalous Hall conductivity very fast (much faster then wannier90) with high precision over an 
ultradense k-grid. This is achieved due to :

1) Using Fast Fourier Transform
2) account of symmetries, to reduce integration to irreducible part of the Brillouin zone
3) recursive adaptive refinement algorithm
4) optimizing the implementation of scan of Fermi level and 'use_ws_distance' parameter (see wannier90 documentation for details) 


Implemented functionality:
---------------------__
-Anomalous Hall conductivity
-Orbital magnetization (modern theory)
-Ohmic conductivity
-Berry curvature dipole
-gyrotropic magnetoelectric effect
-Hall effect
-Low-Field Hall effect

Object oriented structure also makes it potentially easier to implement further features. 

The usage of the code  is demonstrated by "tutorial/example.py". That will produce AHC and tabulate the berry curvature and velocity of bcc iron.

The user manual is under construction, and can be viewed as a work in progress at https://www.overleaf.com/read/kbxxtfbnjvxx

Calculations may also be performed for any tight-binding model, for which a "_tb.dat" file was generated in watever way.

WannierBerri can be run in parallel by means of multiprocessing module

Installation
------------
``pip3 install wannierberri``

Author
------
Stepan Tsirkin, 
University of Zurich
stepan.tsirkin@physik.uzh.ch
