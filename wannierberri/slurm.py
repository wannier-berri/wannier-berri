"""
 This script is copied from https://github.com/pengzhenghao/use-ray-with-slurm
 by  PENG Zhenghao, Department of Information Engineering, the Chinese University of Hong Kong.
 slightly modified by Stepan Tsirkin for WannierBerri project


Usage: 
   python -m wannierberri.slurm  --exp-name <Job name>  --num-nodes <number of nodes> --partition cmt --command "<comand to run>"


"""


import argparse
import sys
import time
import os
from pathlib import Path
import subprocess
from .__sbatch_template import text

JOB_NAME = "{{JOB_NAME}}"
NUM_NODES = "{{NUM_NODES}}"
NUM_GPUS_PER_NODE = "{{NUM_GPUS_PER_NODE}}"
PARTITION_NAME = "{{PARTITION_NAME}}"
COMMAND_PLACEHOLDER = "{{COMMAND_PLACEHOLDER}}"
GIVEN_NODE = "{{GIVEN_NODE}}"
COMMAND_SUFFIX = "{{COMMAND_SUFFIX}}"
LOAD_ENV = "{{LOAD_ENV}}"
SLEEP_HEAD = "{{SLEEP_HEAD}}"
SLEEP_WORKER = "{{SLEEP_WORKER}}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name", type=str, required=True,
        help="The job name and path to logging file (exp_name.log)."
    )
    parser.add_argument(
        "--num-nodes", "-n", type=int, default=1,
        help="Number of nodes to use."
    )
    parser.add_argument(
        "--node", "-w", type=str, default="",
        help="The specified nodes to use. Same format as the return of 'sinfo'. Default: ''."
    )
    parser.add_argument(
        "--num-gpus", type=int, default=0,
        help="Number of GPUs to use in each node. (Default: 0)"
    )
    parser.add_argument(
        "--partition", "-p", type=str, default="chpc",
    )
    parser.add_argument(
        "--load-env", type=str, default="",
        help="The script to load your environment, e.g. 'module load cuda/10.1'"
    )
    parser.add_argument(
        "--command", type=str, required=True,
        help="The command you wish to execute. For example: --command 'python "
             "test.py' Note that the command must be a string."
    )
    parser.add_argument(
        "--sleep-head", type=float, default=30.,
        help="Time to wait (sleep) after starting ray on the head node (seconds, Default: 30.0)"
    )
    parser.add_argument(
        "--sleep-worker", type=float, default=5.,
        help="Time to wait (sleep) after starting ray on every worker node (deconds, Default: 5.0)"
    )



    parser.add_argument('--submit', dest='submit', action='store_true',help=" DO submit the generated script with sbatch")
    parser.add_argument('--no-submit', dest='submit', action='store_false',help=" DO NOT submit the generated script with sbatch")
    parser.set_defaults(submit=False)
    args = parser.parse_args()

    if args.node:
        # assert args.num_nodes == 1
        node_info = "#SBATCH -w {}".format(args.node)
    else:
        node_info = ""

    job_name = "{}_{}".format(
        args.exp_name,
        time.strftime("%m%d-%H%M%S", time.localtime())
    )

    text = text.replace(JOB_NAME, job_name)
    text = text.replace(NUM_NODES, str(args.num_nodes))
    text = text.replace(NUM_GPUS_PER_NODE, 
                 f"#SBATCH --gpus-per-task={args.num_gpus}" if args.num_gpus>0 else "")
    text = text.replace(PARTITION_NAME, str(args.partition))
    text = text.replace(COMMAND_PLACEHOLDER, str(args.command))
    text = text.replace(LOAD_ENV, str(args.load_env))
    text = text.replace(GIVEN_NODE, node_info)
    text = text.replace(COMMAND_SUFFIX, "")
    text = text.replace(
        "# THIS FILE IS A TEMPLATE AND IT SHOULD NOT BE DEPLOYED TO "
        "PRODUCTION!",
        "# THIS FILE IS MODIFIED AUTOMATICALLY FROM TEMPLATE AND SHOULD BE "
        "RUNNABLE!"
    )
    text = text.replace(SLEEP_HEAD, str(args.sleep_head))
    text = text.replace(SLEEP_WORKER, str(args.sleep_worker))

    # ===== Save the script =====
    script_file = "{}.sh".format(job_name)
    with open(script_file, "w") as f:
        f.write(text)

    print( f"Script file written to: <{script_file}>. Log file will be at: <{job_name}.log>" )

    # ===== submit the job  =====
    if args.submit:
        print("Start to submit job!")
        subprocess.Popen(["sbatch", script_file])
        print("Job submitted!" )
    else:
        print( f"Now you may submit it to the queue with ' sbatch {script_file} '" )

    sys.exit(0)
