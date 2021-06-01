#!/bin/bash
##  To generate a script to run wannierberri on 2 nodes, and submit thge script to the slurm or PBS cluster just run the following command:
##

# Using slurm
python -m wannierberri.cluster --batch-system slurm --exp-name wb-2nondes --num-nodes 2  --partition cmt --command "python -u example.py 2-nodes" --submit

# Using PBS torque
python -m wannierberri.cluster --batch-system pbs --exp-name wb-2nondes --num-nodes 2  --partition cmt --command "python -u example.py 2-nodes" --submit

#
#  for more info see 
#
#  python -m wannierberri.cluster -h
#
