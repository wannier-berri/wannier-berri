#!/bin/bash

# THIS FILE IS GENERATED BY AUTOMATION SCRIPT! PLEASE REFER TO ORIGINAL SCRIPT!
# THIS FILE IS MODIFIED AUTOMATICALLY FROM TEMPLATE AND SHOULD BE RUNNABLE!

#SBATCH --partition=express
#SBATCH --job-name=my_first_job_0524-035316
#SBATCH --output=my_first_job_0524-035316.log


### This script works for any number of nodes, Ray will find and manage all resources
#SBATCH --nodes=4
#SBATCH --exclusive

### Give all resources to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=3

# Load modules or your own conda environment here
# module load pytorch/v1.4.0-gpu
# conda activate {{CONDA_ENV}}


################# DON NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###############
# This script is a modification to the implementation suggest by gregSchwartz18 here:
# https://github.com/ray-project/ray/issues/826#issuecomment-522116599
redis_password=$(uuidgen)
export redis_password

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=($nodes)

node_1=${nodes_array[0]}
ip=$(srun --nodes=1 --ntasks=1 -w $node_1 hostname --ip-address) # making redis-address

if [[ $ip == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"$ip"
  if [[ ${#ADDR[0]} > 16 ]]; then
    ip=${ADDR[1]}
  else
    ip=${ADDR[0]}
  fi
  echo "We detect space in ip! You are using IPV6 address. We split the IPV4 address as $ip"
fi

port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "STARTING HEAD at $node_1"
# srun --nodes=1 --ntasks=1 -w $node_1 start-head.sh $ip $redis_password &
srun --nodes=1 --ntasks=1 -w $node_1   ray start --head --node-ip-address=$ip --port=6379 --redis-password=$redis_password --num-cpus=3 --block &
sleep 12.34

worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
for ((i = 1; i <= $worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"
  srun --nodes=1 --ntasks=1 -w $node_i ray start --address $ip_head --redis-password=$redis_password --num-cpus=3 --block &
  sleep 5.0
done

##############################################################################################

#### call your code below
'python -u wb-example.py' 
