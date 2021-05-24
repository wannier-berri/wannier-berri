import subprocess
import os

def test_slurm(rootdir):
    sp = subprocess.run(['python',
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
    , capture_output=True )
    
    print (str(sp.stdout))
    script_name=str(sp.stdout).split("'")[-2].split()[-1] 
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