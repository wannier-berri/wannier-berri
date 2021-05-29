import os
import pytest


@pytest.fixture(scope="module")
def check_command_output():
    from sys import version_info
    assert version_info.major == 3
    if version_info.minor>=7:
        from subprocess import run
        def _inner(command):
            sp = run(command,capture_output=True)
            return str(sp.stdout)
    else:
        from subprocess import Popen,PIPE,STDOUT
        def _inner(command):
            sp = Popen(["ls", "-l"], stdout=PIPE, stderr=STDOUT,encoding='UTF-8')
            return sp.stdout.read()
    return _inner
    


def test_slurm(rootdir,check_command_output):
    command=['python',
             '-m',
             'wannierberri.slurm',
             '--exp-name',
             'my_first_job',
             '--num-nodes',
             '4',
             '--partition',
             'express',
             '--command',
             "'python -u wb-example.py'",
             '--num-gpus=3',
             '--sleep-head',
             '123.45',
             '--sleep-worker=67',
             '--no-submit']
    stdout=check_command_output(command)    
    print (stdout)
    script_name=stdout.split("'")[-2].split()[-1]
    print (script_name)
    script_text = open(script_name,"r").readlines()
    ref_text    = open(os.path.join(rootdir,"reference","my_first_job.sh"),"r").readlines()
    variable_strings = ["#SBATCH --job-name=my_first_job_","#SBATCH --output=my_first_job_"]
    for l1,l2 in zip(script_text,ref_text):
        if l1 !=l2 :
            try: 
                match=False
                for l in variable_strings:
                    if l in l1:
                        assert l1[:len(l)]==l2[:len(l)]
                        match=True
                        break
                if not match:
                    raise AssertionError()
            except AssertionError:
                raise AssertionError(f"Lines \n<{l1}>\n and \n<{l2}>\n do not match")