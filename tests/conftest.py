"""pytest configuration file for WannierBerri tests."""

import os

import numpy
import pytest

# Root folder containing test scripts
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder containing output dat files of tests
OUTPUT_DIR = os.path.join(ROOT_DIR, "_dat_files")

@pytest.fixture(scope="session", autouse=True)
def create_output_dir():
    # Create folder OUTPUT_DIR
    from pathlib import Path
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

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
    # If multiple ray parallel setups are tested in a single session, the
    # parallel object needs to be shutdown before changing the setup.
    # To do so, one needs to change the scope to function, use yield instead
    # of return, and add parallel.shutdown() after the yield statement.
    # See https://docs.pytest.org/en/6.2.x/fixture.html#yield-fixtures-recommended
    # Currently, only a single ray setup is used, so this is not a problem.
    return Parallel(
                   method="ray",
                   num_cpus=4  ,
                   npar_k = 0 , 
                   ray_init={} ,     # add extra parameters for ray.init()
                   cluster=False , # add parameters for ray.init() for the slurm cluster
                   progress_step_percent  = 1  ,  #
                   progress_timeout = None  # relevant only for ray, seconds
                 )



