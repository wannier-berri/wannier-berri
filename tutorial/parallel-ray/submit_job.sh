#!/bin/bash
##  To generate a script to run wannierberri on 2 nodes, and submit thge script to the slurm cluster just run the following command:
##

python -m wannierberri.slurm --exp-name wb-2nondes --num-nodes 2  --partition cmt --command "python -u example.py 2-nodes" --submit

#
#  for more info see 
#
#  python -m wannierberri.slurm -h
#