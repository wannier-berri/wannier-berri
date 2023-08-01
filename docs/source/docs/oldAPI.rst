Old interface
===============

While the following functions still work, they are not guaranteed to be supported in the future.
Please adapt to the new, more versatile interface of the :func:`~wannierberri.run`

Integrating
+++++++++++++++

.. autofunction:: wannierberri.integrate

Tabulating
+++++++++++++++

.. autofunction:: wannierberri.tabulate


.. _doc-parameters:

Specifying parameters
+++++++++++++++++++++++++++


While the basic parameters of the calculation in the :func:`~wannierberri.integrate` and :func:`~wannierberri.tabulte` 
methods determine the general behaviour of the calculation (grid, refinement etc.) , the specific evauation of each 
quantity are determined by the dictionaries `parameters` and `specific_parameters`

`parameters`
---------------

a dictionary like `{'parameter1':value1, 'parameter2':value2, ...}`. Each quantity will look for parameters that it can recognize,
and take the corresponding values from the dictionary. The unrecognized paremeters will be ignored. A complete list of available 
options for each quantity may be obtained by running `wannierberri.__old_API.__main.print_options()`

`specific_parameters` -- same quantity with different parameters in one run
-----------------------------------------------------------------------------

One can specify the same quantity several times in the `quantities` list and to distinguish them,
a label may be added after the `'^'` symbol, e.g. `quantities = ['ahc^int','ahc^ext']`. 
Further, different parameters  may be passed to `'ahc^int'` and `'ahc^ext'` by means of `specific_parameters` option, which
is a dictionary like `{'quantity1^label1':{'parameter1':value1,...},...}`. 
Values found in  `specific_parameters` will always override those found in `parameters`

For example, after a run

.. code:: python

   wberri.integrate(system,
             grid=grid,
             Efermi=Efermi, 
             quantities=['ahc^int','ahc^ext'],
             parameters = {'tetra':True,'internal_terms':False,'external_terms':False},
             specific_parameters = {'ahc^int':{'internal_terms':True},'ahc^ext':{'external_terms':True}},
             fout_name='Fe',
             )

`Fe-ahc^int_iter-0000.dat` will contain contributions from internal terms only, and 
`Fe-ahc^ext_iter-0000.dat` will contain contributions from external terms.

`parameters_K`
-----------------


In addition, there is a dictionary of parameters `parameters_K` that will de passed to the constructor of the 
:class:`~wannierberri.data_K.Data_K` object. Therefore, these parameters will be common for all quantities calculated


