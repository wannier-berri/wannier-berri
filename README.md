```
___       __                        _____                      _____________ 
__ |     / /______ ________ _______ ___(_)_____ ________       __<  /__  __ \
__ | /| / / _  __ `/__  __ \__  __ \__  / _  _ \__  ___/       __  / _  /_/ /
__ |/ |/ /  / /_/ / _  / / /_  / / /_  /  /  __/_  /           _  /  _\__, / 
____/|__/   \__,_/  /_/ /_/ /_/ /_/ /_/   \___/ /_/            /_/   /____/  

                  __                     ___                  
        |_       (_  |_  _  _   _   _     |   _ .  _ |  .  _  
        |_) \/   __) |_ (- |_) (_| | )    |  _) | |  |( | | ) 
            /              |                                  
```

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

The usage of the code  is demonstrated by "exampleFe" in the examples folder.
AHC can also be calcualted for any tight-binding model, for which a "_tb.dat" file was generated in watever way.

wannier19 can be run in parallel by means of multiprocessing module

The project started on June, 25th 2019. 
Any interest from the community will be motivating for developing.


Stepan Tsirkin, 
University of Zurich
stepan.tsirkin@physik.uzh.ch
