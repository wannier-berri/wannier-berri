
.. _sec-benchmark:

********************************
Benchmarking with ``postw90.x``
********************************

As you may notice, the results given by WannierBerri, e.g. for anomalous Hall conductivity are not exactly the same as given by postw90.x. 
This is understood, because different numerical algorithms are used for refinement, for symmetry account, and other technicalities.  
And you might want to make sure that it is not a bug, not a misuse of parameters, and that the speed of WannierBerri is achieved honestly without any fraud.
In principle the results should agree, when they are both converged with respect to k-points. However, this convergence is hard to achieve. 
Moreover,  the use of symmetries in WannierBerri does not affect the result only if the Wannier functions are perfectly symmetric, 
which is often not the case due to numerical inaccuracies.
However, **there is a set of parameters under which the two codes do essentially the same**. 
Hence the result should agree within machine precision even for calculations on very coarse grids. 
To facilitate comparison, since version 0.6.0  we have introduced an option ``transl_inv`` in ``System_w90`` , which has a default ``True`` . 
There is a corresponding undocumented  option in postw90.x , where it has a default value `False`. 

Below I list the conditions, under which the results of two codes should be **precisely the same**. 
In the example of bcc iron the absolute difference was within 1e-3 , while the absolute value of AHC reaches order of 1e+3.  
Also the artificial spikes (which disappear in converged result) are the same from the two codes. 
This agreement also should not be affected by how good/bad the Wannier functions are.

Conditions for bench-marking:
 + switch off symmetries in wberri
 + no adaptive refinement (both in wberri and postw90 )
 + set `NKFFT` and `NKdiv` manually, so that their product yields the same grid as used by postw90.
   the result will not depend on `NKFFT` and `NKdiv` individually, only on their produuct. However, for optimal performance 
   of FFT transforms  it is recommended to set NKFFT approximately equal to the ab initio grid.
   also note, that if you set only `NK`, then it may be changed slightly by the code, to get a better 
   factorization into `NKFFT` and `NKdiv`
 + consistent use of parameters `transl_inv` and `use_ws`/`use_ws_distance`.
 + compare the postw90 result to non-smeared result of wberri (first columns in the output)
 + use the version 3.1.0 of postw90.x

For example, the WannierBerri script:

.. code:: python

    import wannierberri as wberri
    system = wberri.System_w90(seedname='Fe',berry=True,transl_inv=False )
    grid   = wberri.Grid ( system,
                           NKFFT = [10,10,10],
                           NKdiv = [5,5,5] )

    wberri.integrate(system,
                    Efermi=np.linspace(11,14,301) ,
                    smearEf=300,
                    quantities=['ahc'],
                    numproc=num_proc,
                    adpt_num_iter=0,
                    fout_name='Fe',
                    restart=False
                    )

Will correspond to Fe.win:

.. code:: 

    fermi_energy_min=11
    fermi_energy_max=14
    fermi_energy_step=0.01
    berry = T
    berry_task = ahc
    berry_kmesh = 50 50 50
    # the followinfg parameters are at default values, so may be omitted in the .win file:
    transl_inv=F             
    use_ws_distance=T        
    berry_curv_adpt_kmesh = 1 

I hope you can reproduce this agreement on any system of your interest. For optical properties also be careful with the smearing options. 
