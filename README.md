Wannier19

I started this project by python realization of some of postw90 functional.
Some parts of the code are an adapted translation of Wannier90 Fortran code:
http://www.wannier.org/
https://github.com/wannier-developers/wannier90

The code is distributed under the terms of  GNU GENERAL PUBLIC LICENSE  Version 2, the same as Wannier90

My motivation was:
1) to have fun with python
2) to make it potentially easier to implement further features
3) to make it faster.  And it IS already much Faster than the Fortran code of Wannier90, due to usage of fast Fourier transform (FFT). (Of course, I could do FFT in Fortran, but I did not want to. Maybe someone else will be encouraged to do that.)

So far only AHC for one Fermi level is implemented and benchmarked. 
The usage of the code  is demonstrated by "exampleFe" in theexamples folder.
AHC can also be calcualted for any tight-binding model, for which a "_tb.dat" file was generated in watever way.

wannier19 can be run in parallel by means of multiprocessing module

Note: benchmark works correctly only with use_ws_distance=F

The project is in early stage, started on June, 25th 2019. Futher features and explanations will be added.
Any interest from the community will be motivating for developing.


Stepan Tsirkin, 
University of Zurich
stepan.tsirkin@physik.uzh.ch
