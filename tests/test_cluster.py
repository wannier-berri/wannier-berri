import contextlib
import io
import os
import re

import pytest

from .common import REF_DIR, OUTPUT_DIR
"""
Test creation of submission scripts for slurm and PBS batch systems.
No jobs are submitted in the test.
"""


def compare_texts(script_text, ref_text, variable_strings):
    for l1, l2 in zip(script_text, ref_text):
        l1 = l1.strip()
        l2 = l2.strip()
        print(f"Comparing lines:\n<{l1}>\n and \n<{l2}>\n")
        if l1 != l2:
            for l in variable_strings:
                if l in l1:
                    if l1[:len(l)] == l2[:len(l)]:
                        break
            else:
                raise AssertionError(f"Lines \n<{l1}>\n and \n<{l2}>\n do not match")


@pytest.mark.parametrize("cluster_type", ["slurm", "pbs"])
def test_cluster_script(cluster_type):
    if cluster_type == "slurm":
        variable_strings = ["#SBATCH --job-name=my_first_job_",
                            "#SBATCH --output=my_first_job_"]
    elif cluster_type == "pbs":
        variable_strings = ["#PBS -N my_first_job_",
                            "#PBS -o my_first_job_",
                            "#PBS -e my_first_job_"]
    else:
        raise ValueError("cluster_type not identified. Only slurm or pbs.")

    from wannierberri.utils.cluster import main

    argv = [
        "--batch-system",
        cluster_type,
        "--exp-name",
        "my_first_job_" + cluster_type,
        "--num-nodes",
        "4",
        "--partition",
        "express",
        "--command",
        "'python -u wb-example.py'",
        "--num-gpus=3",
        "--sleep-head",
        "12.34",
        "--sleep-worker=5",
        "--no-submit",
        "--num-cpus-per-node",
        "3",
    ]

    stdout_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer):
        try:
            text = main(argv)
        except SystemExit as exc:
            assert exc.code == 0

    print(f"Captured stdout:\n{text}")
    stdout = stdout_buffer.getvalue()
    match = re.search(r"Script file written to: <([^>]+)>", stdout)
    assert match is not None, "Script file path not found in output."
    script_name = match.group(1)
    script_text = open(script_name, "r").readlines()
    ref_text = open(os.path.join(REF_DIR, f"my_first_job_{cluster_type}.sh"), "r").readlines()
    compare_texts(script_text, ref_text, variable_strings)
    compare_texts(text.splitlines(), ref_text, variable_strings)
    os.replace(script_name, os.path.join(OUTPUT_DIR, script_name))
