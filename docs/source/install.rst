**********************************
Installation and technical remarks
**********************************

To run the ``WannierBerri``\ code only python3 is required, independent of the
operating system. The easy way of installation of the latest stable
version maybe achieved via `pip <https://pypi.org/project/wannierberri/>`_

::

   pip install wannierberri[all]

This will install all needed dependencies. However, some dependencies are heavy
and might be not necessary for your particular use. Moreover, some of them might fail
to install on some systems. To try WannierBerri, one may use a bare installation 
`pip install wannierberri` (with minimal necessary dependencies), which will fail
for some features, or you may specify which features you actually need.

* `parallel` : enables parallel execution via `ray`
* `symmetry` : enables symmetrization, setting symmetry form structure
* `fftw`     : use a faster implementation of FFT in FFTW
* `plot`     : use plotting routines
* `phonons`  : phonons interpolation


For example : 

::

   pip install wannierberri[parallel,fftw]


Parallelization
----------------
One note should be mentioned about the parallel run. ``numpy`` already
includes parallelization over threads. However, if ``wannierberri``\ is
running with the number of processes equal to the number of physical
cores, obviously extra threading may only slow down the calculation.
Generally I recommend to switch off the threading in numpy by setting
the corresponding environent variable. It can be done inside the python
script by

::

   import os
   os.environ['OPENBLAS_NUM_THREADS'] = '1'

or

::

   os.environ['MKL_NUM_THREADS'] = '1'  

Depending whether numpy is linked with openblas or MKL libraries. Note,
this should be done in the beginning of the script, before importing
``numpy`` or ``wannierberri`` for the first time.


Windows OS
----------

(probably outdated, and not needed anymore)

On Windows the subprocesses, initated within WannierBerri  will import (i.e. execute) the main module (the user's python script) at start. 
Hence, to execute WannierBerri in parallel on Windows one has to start the script from a  ``if __name__ == '__main__':`` guard to avoid creating subprocesses recursively : 

::

    if __name__ == '__main__'
        import wannierberri
        import numpy
        ..........


The solution was found in this answer:  `<https://stackoverflow.com/a/18205006>`_


.. _sec-pyfftw:

known bug with pyfftw
----------------------

Under some installations appears a bug with pyfftw:

::

      File "/home/stepan/github/wannier-berri-work/wannier-berri-master/examples/wannierberri/__utility.py", line 152, in fourier_q_to_R
        AA_q_mp = FFT(AA_q_mp, axes=(0, 1, 2), numthreads=numthreads, fft=fft, destroy=False)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/stepan/github/wannier-berri-work/wannier-berri-master/examples/wannierberri/__utility.py", line 138, in FFT
        return fft_W(inp, axes, inverse=inverse, destroy=destroy, numthreads=numthreads)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/stepan/github/wannier-berri-work/wannier-berri-master/examples/wannierberri/__utility.py", line 107, in fft_W
        fft_object = pyfftw.FFTW(
                     ^^^^^^^^^^^^
      File "pyfftw/pyfftw.pyx", line 1426, in pyfftw.pyfftw.FFTW.__cinit__
    RuntimeError: The data has an uncaught error that led to the planner returning NULL. This is a bug.


As mentioned [here](https://github.com/spectralDNS/spectralDNS/issues/29#issuecomment-392537709), this can be avoided by importing `pyfftw` prior to `numpy`.
This is the order of imports inside `wannierberri`, but to avoid this bug do not import `numpy` before `wanierberri`, always after

If the problem is not solved, you may switch to `numpy` for fft transforms.See 

    * :class:`~wannierberri.system.System_w90` (parameter `fft`)

    * :class:`~wannierberri.data_K._Data_K` (`fftlib` passed from :func:`~wannierberri.run` via `parameters_K`)