Running a calculation
======================

On a grip or path
+++++++++++++++++++
In the new API with the :func:`~wannierberri.run` method one can do integration and tabulating
in one run. Also, different parameters may be used for different quantities 
and also same qantities may be calculated with different sets of parameters in one run.
This is done by calling the following function:

.. autofunction:: wannierberri.run


Single k-point
+++++++++++++++++++

Sometimes, ot os needed to evaluate some properties at a single k-point. To make it simple a functions is provided:

.. autofunction:: wannierberri.evaluate_k


on a path (simple) 
+++++++++++++++++++


While calculation on a k-path may be done with the `run` method, and a :class:`~wannierberri.grid.path.Path` object,
for simplicity there is a shortcut function:

.. autofunction:: wannierberri.evaluate_k_path