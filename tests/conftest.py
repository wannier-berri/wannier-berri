"""pytest configuration file for WannierBerri tests."""

import os

import numpy
import pytest

@pytest.fixture(scope="session")
def rootdir():
    return os.path.dirname(os.path.abspath(__file__))

@pytest.fixture(scope="session", autouse=True)
def output_dir(rootdir):
    from pathlib import Path
    directory = os.path.join(rootdir, "_dat_files")
    Path(directory).mkdir(exist_ok=True)
    return directory


@pytest.fixture(scope="session")
def parallel_serial():
    from wannierberri import Parallel
    return Parallel(
                   method="serial",
                   num_cpus=0  ,
                   npar_k = 0 , 
                   progress_step_percent  = 1  ,  #
                   progress_timeout = None  # relevant only for ray, seconds
                 )



@pytest.fixture(scope="session")
def parallel_ray():
    from wannierberri import Parallel
    return Parallel(
                   method="ray",
                   num_cpus=4  ,
                   npar_k = 0 , 
                   ray_init={} ,     # add extra parameters for ray.init()
                   cluster=False , # add parameters for ray.init() for the slurm cluster
                   progress_step_percent  = 1  ,  #
                   progress_timeout = None  # relevant only for ray, seconds
                 )



