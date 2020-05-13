.::    .   .:::  :::.     :::.    :::.:::.    :::. :::.,::::::  :::::::..       :::::::.  .,::::::  :::::::..   :::::::..   :::
';;,  ;;  ;;;'   ;;`;;    `;;;;,  `;;;`;;;;,  `;;; ;;;;;;;''''  ;;;;``;;;;       ;;;'';;' ;;;;''''  ;;;;``;;;;  ;;;;``;;;;  ;;;
 '[[, [[, [['   ,[[ '[[,    [[[[[. '[[  [[[[[. '[[ [[[ [[cccc    [[[,/[[['       [[[__[[\. [[cccc    [[[,/[[['   [[[,/[[['  [[[
   Y$c$$$c$P   c$$$cc$$$c   $$$ "Y$c$$  $$$ "Y$c$$ $$$ $$""""    $$$$$$c         $$""""Y$$ $$""""    $$$$$$c     $$$$$$c    $$$
    "88"888     888   888,  888    Y88  888    Y88 888 888oo,__  888b "88bo,    _88o,,od8P 888oo,__  888b "88bo, 888b "88bo,888
     "M "M"     YMM   ""`   MMM     YM  MMM     YM MMM """"YUMMM MMMM   "W"     ""YUMMMP"  """"YUMMM MMMM   "W"  MMMM   "W" MMM

a.k.a. Wannier19
                  __                     ___                  
        |_       (_  |_  _  _   _   _     |   _ .  _ |  .  _  
        |_) \/   __) |_ (- |_) (_| | )    |  _) | |  |( | | ) 
            /              |                                  ```


I started this project by python realization of some of postw90 functional.
Some parts of the code are an adapted translation of Wannier90 Fortran code:
http://www.wannier.org/
https://github.com/wannier-developers/wannier90

The code is distributed under the terms of  GNU GENERAL PUBLIC LICENSE  Version 2, the same as Wannier90

At the moment the code calculates Anomalous Hall conductivity very fast (much faster then wannier90) with high precision over an 
ultradense k-grid. This is achieved due to :

1) Using Fast Fourier Transform
2) account of symmetries, to reduce integration to irreducible part of the Brillouin zone
3) recursive refinement algorithm
4) optimizing the implementation of scan of Fermi level and 'use_ws_distance' parameter (see wannier90 documentation for details) 

Object oriented structure also makes it potentially easier to implement further features. 

The usage of the code  is demonstrated by "tutorial/example.py". That will produce AHC and tabulate the berry curvature and velocity of bcc iron.

The user manual is under construction, and can be viewed as a work in progress at https://www.overleaf.com/read/kbxxtfbnjvxx

AHC can also be calcualted for any tight-binding model, for which a "_tb.dat" file was generated in watever way.

WannierBerri can be run in parallel by means of multiprocessing module

installation is possible by pip

pip3 install wannierberri


The project started on June, 25th 2019. 
Any interest from the community will be motivating for developing.


Stepan Tsirkin, 
University of Zurich
stepan.tsirkin@physik.uzh.ch
