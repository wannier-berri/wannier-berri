.. _doc-parallel:

Parallel execution
===================
To perform execution in parallel, first one needs to create a :class:`~wannierberri.parallel.Parallel` object 
that will describe the parameters of the calculation.

.. autoclass:: wannierberri.parallel.Parallel
   :undoc-members:
   :show-inheritance:

.. autoclass:: wannierberri.parallel.Serial
   :undoc-members:
   :show-inheritance:

**NOTE**: 
Ray will produce a lot of temorary files during running. `/tmp` is the default directory for temporary data files. More information about `temorary files <https://docs.ray.io/en/stable/tempfile.html>`__.

If you are using a cluster, you may have no permission to delete them under `/tmp`. Please store them under the folder which under your control by adding ``ray_init={'_temp_dir': Your_Path}``.
Please keep ``Your_Path`` shorter. There is a problem if your path is long. Please check `temp_dir too long bug <https://github.com/ray-project/ray/issues/7724>`__

multi-node mode
+++++++++++++++++

When more than one node are employed on a cluster, first they should be
connected together into a Ray cluster. This can be done by a script
suggested by `gregSchwartz18 <https://github.com/gregSchwartz18>`__  
`here <https://github.com/ray-project/ray/issues/826#issuecomment-522116599>`__

Such a script can be generated, for example if your cluster uses SLURM ::

    python -m wannierberri.cluster --batch-system slurm --exp-name wb-2nondes --num-nodes 2  --partition cmt --command "python -u example.py 2-nodes" --submit

Or if you are using PBS ::

    python -m wannierberri.cluster --batch-system pbs --exp-name wb-2nondes --num-nodes 2  --partition cmt --command "python -u example.py 2-nodes" --submit

for more info see  ::

    python -m wannierberri.cluster -h


