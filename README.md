# wannier19
python realization of some of postw90 functional.  

My motivation was:
1) to have fun with python
2) to make it potentially easier to implement further features
3) it is already Faster than the Fortran code of Wannier90, due to usage of FFT.

I could do FFT in Fortran, but I did not want to. Maybe someone else will be encouraged to do that.


to use it one needs to generate HH_R , AA_R files, .. etc. This can be done by postw90.x compiled from the following branch:
https://github.com/stepan-tsirkin/wannier90/tree/saveHH

Sofar only AHC for one Fermi level is implemented and benchmarked. 
Note: benchmark works correctly only with use_ws_distance=F
