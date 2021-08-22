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
#                   ray_init={} ,     # add extra parameters for ray.init()
#                   cluster=False , # add parameters for ray.init() for the slurm cluster
#                   chunksize=None  , # size of chunk in multiprocessing 
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
                   chunksize=None  , # size of chunk in multiprocessing 
                   progress_step_percent  = 1  ,  #
                   progress_timeout = None  # relevant only for ray, seconds
                 )



@pytest.fixture(scope="session")
def parallel_multiprocessing():
    from wannierberri import Parallel
    return Parallel(
                   method="multiprocessing-K",
                   num_cpus=4  ,
                   npar_k = 0 , 
                   ray_init={} ,     # add extra parameters for ray.init()
                   cluster=False , # add parameters for ray.init() for the slurm cluster
                   chunksize=None  , # size of chunk in multiprocessing 
                   progress_step_percent  = 1  ,  #
                   progress_timeout = None  # relevant only for ray, seconds
                 )
