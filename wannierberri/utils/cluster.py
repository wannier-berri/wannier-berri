"""
 This script is copied from https://github.com/pengzhenghao/use-ray-with-slurm
 by  PENG Zhenghao, Department of Information Engineering, the Chinese University of Hong Kong.
 slightly modified by Stepan Tsirkin for WannierBerri project


Usage:
   python -m wannierberri.cluster --batch-system <slurm or pbs> --exp-name <Job name>  --num-nodes <number of nodes> --partition cmt --command "<comand to run>"


"""

import argparse
import sys
import time
import subprocess
from .__cluster_template import slurm_text, pbs_torque_text

JOB_NAME = "{{JOB_NAME}}"
NUM_NODES = "{{NUM_NODES}}"
NUM_CPUS_TEXT = "{{NUM_CPUS_TEXT}}"
NUM_GPUS_PER_NODE = "{{NUM_GPUS_PER_NODE}}"
PARTITION_NAME = "{{PARTITION_NAME}}"
COMMAND_PLACEHOLDER = "{{COMMAND_PLACEHOLDER}}"
GIVEN_NODE = "{{GIVEN_NODE}}"
COMMAND_SUFFIX = "{{COMMAND_SUFFIX}}"
LOAD_ENV = "{{LOAD_ENV}}"
SLEEP_HEAD = "{{SLEEP_HEAD}}"
SLEEP_WORKER = "{{SLEEP_WORKER}}"
SPILLING = "{{SPILLING}}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-system", type=str, required=True, help="The batch system to use. Only Slurm and PBS implemented..")
    parser.add_argument(
        "--exp-name", type=str, required=True, help="The job name and path to logging file (exp_name.log).")
    parser.add_argument("--num-nodes", "-n", type=int, default=1, help="Number of nodes to use.")
    parser.add_argument(
        "--node",
        "-w",
        type=str,
        default="",
        help="The specified nodes to use. Same format as the return of 'sinfo'. Default: ''.")
    parser.add_argument("--num-cpus-per-node", nargs='?', type=int, help="Number of CPUs to use in each node. "
        "(Default: None (use all available cpus))")
    parser.add_argument("--num-gpus", type=int, default=0, help="Number of GPUs to use in each node. (Default: 0)")
    parser.add_argument(
        "--partition",
        "-p",
        type=str,
        default="chpc",
    )
    parser.add_argument(
        "--load-env", type=str, default="", help="The script to load your environment, e.g. 'module load cuda/10.1'")
    parser.add_argument(
        "--command",
        type=str,
        required=True,
        help="The command you wish to execute. For example: --command 'python "
        "test.py' Note that the command must be a string.")
    parser.add_argument(
        "--sleep-head",
        type=float,
        default=30.,
        help="Time to wait (sleep) after starting ray on the head node (seconds, Default: 30.0)")
    parser.add_argument(
        "--sleep-worker",
        type=float,
        default=5.,
        help="Time to wait (sleep) after starting ray on every worker node (deconds, Default: 5.0)")
    parser.add_argument(
        "--spilling-directory", type=str, default="", help="directory to spill objects in case of lack of memory"
    )  # see : https://docs.ray.io/en/master/memory-management.html#object-spilling

    parser.add_argument(
        '--submit', dest='submit', action='store_true', help=" DO submit the generated script with sbatch")
    parser.add_argument(
        '--no-submit', dest='submit', action='store_false', help=" DO NOT submit the generated script with sbatch")
    parser.set_defaults(submit=False)
    args = parser.parse_args()

    if args.node:
        # assert args.num_nodes == 1
        node_info = f"#SBATCH -w {args.node}"
    else:
        node_info = ""

    job_name = args.exp_name + "_" + time.strftime("%m%d-%H%M%S", time.localtime())

    batch_system = args.batch_system.lower()
    if batch_system == "slurm":
        text = slurm_text
        submit_command = "sbatch"
    elif batch_system == "pbs":
        text = pbs_torque_text
        submit_command = "qsub"
    else:
        raise ValueError("Batch system not identified. Only slurm or pbs are currently implemented.")

    if args.spilling_directory == "":
        text = text.replace(SPILLING, "")
    else:
        # Note that `object_spilling_config`'s value should be json format.
        text = text.replace(
            SPILLING, '--system-config=\'{"object_spilling_config":"{"type":"filesystem",'
            '"params":{"directory_path":"' + args.spilling_directory + '"}}"}\'')

    text = text.replace(JOB_NAME, job_name)
    text = text.replace(NUM_NODES, str(args.num_nodes))

    num_cpus_text = ""
    if args.num_cpus_per_node is not None:
        num_cpus_text = f"--num-cpus={args.num_cpus_per_node}"
    text = text.replace(NUM_CPUS_TEXT, num_cpus_text)

    gpu_text = ""
    if args.num_gpus > 0:
        if batch_system == "slurm":
            gpu_text = f"#SBATCH --gpus-per-task={args.num_gpus}"
        elif batch_system == "pbs":
            gpu_text = f":gpus={args.num_gpus}"
    text = text.replace(NUM_GPUS_PER_NODE, gpu_text)

    text = text.replace(PARTITION_NAME, str(args.partition))
    text = text.replace(COMMAND_PLACEHOLDER, str(args.command))
    text = text.replace(LOAD_ENV, str(args.load_env))
    text = text.replace(GIVEN_NODE, node_info)
    text = text.replace(COMMAND_SUFFIX, "")
    text = text.replace(
        "# THIS FILE IS A TEMPLATE AND IT SHOULD NOT BE DEPLOYED TO "
        "PRODUCTION!", "# THIS FILE IS MODIFIED AUTOMATICALLY FROM TEMPLATE AND SHOULD BE "
        "RUNNABLE!")
    text = text.replace(SLEEP_HEAD, str(args.sleep_head))
    text = text.replace(SLEEP_WORKER, str(args.sleep_worker))

    # ===== Save the script =====
    script_file = f"{job_name}.sh"
    with open(script_file, "w") as f:
        f.write(text)

    print(f"Script file written to: <{script_file}>. Log file will be at: <{job_name}.log>")

    # ===== submit the job  =====
    if args.submit:
        print("Start to submit job!")
        subprocess.Popen([submit_command, script_file])
        print("Job submitted!")
    else:
        print(f"Now you may submit it to the queue with ' {submit_command} {script_file} '")

    sys.exit(0)


if __name__ == '__main__':
    main()
