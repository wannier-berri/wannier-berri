"""pytest configuration file for WannierBerri tests."""

import numpy
import pytest

from common import ROOT_DIR, OUTPUT_DIR, REF_DIR

# WannierBerri Systems
from common_systems import (
    create_files_Fe_W90, create_files_GaAs_W90,
    model_tbmodels_Haldane, model_pythtb_Haldane, model_chiral, model_CuMnAs_2d_broken,
    system_Fe_W90, system_Fe_W90_wcc, system_Fe_FPLO, system_Fe_FPLO_wcc, system_GaAs_W90,
    system_GaAs_W90_wcc, system_GaAs_tb, system_GaAs_tb_wcc, system_GaAs_tb_wcc_ws,
    system_Haldane_PythTB, system_Haldane_TBmodels, system_Haldane_TBmodels_internal,
    system_Chiral, system_CuMnAs_2d_broken,
)

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

    # first we just check that the initialization works with cluster=True and some dummy ray_init parameters
    ray_init = {}
    ray_init['address'] = ''
    ray_init['_node_ip_address']  =  "0.0.0.0"
    ray_init['_redis_password']   = 'some_password'

    parallel = Parallel(
                   method="ray",
                   num_cpus=4  ,
                   npar_k = 0 , 
                   ray_init=ray_init ,     # add extra parameters for ray.init()
                   cluster=True , # add parameters for ray.init() for the slurm cluster
                   progress_step_percent  = 1  ,  #
                 )

    parallel.shutdown()


    # Now create a proper parallel environment to be used
    return Parallel(
                   method="ray",
                   num_cpus=4  ,
                   npar_k = 0 , 
                   ray_init={} ,     # add extra parameters for ray.init()
                   cluster=False , # add parameters for ray.init() for the slurm cluster
                   progress_step_percent  = 1  ,  #
                 )


