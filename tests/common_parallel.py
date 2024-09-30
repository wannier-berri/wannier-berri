import pytest
from wannierberri.parallel import Parallel, Serial


@pytest.fixture(scope="session")
def parallel_serial():
    return Serial(
        npar_k=0,
        progress_step_percent=1,
    )


@pytest.fixture(scope="session")
def parallel_ray():
    # If multiple ray parallel setups are tested in a single session, the
    # parallel object needs to be shutdown before changing the setup.
    # To do so, one needs to change the scope to function, use yield instead
    # of return, and add parallel.shutdown() after the yield statement.
    # See https://docs.pytest.org/en/6.2.x/fixture.html#yield-fixtures-recommended
    # Currently, only a single ray setup is used, so this is not a problem.

    # first we just check that the initialization works with cluster=True and some dummy ray_init parameters
    ray_init = {}
    ray_init['address'] = None
    ray_init['_node_ip_address'] = "0.0.0.0"
    ray_init['_redis_password'] = 'some_password'
    ray_init['num_gpus'] = 0  # otherwise failing with NVIDIA-555 driver.

    parallel = Parallel(
        num_cpus=4,
        npar_k=0,
        ray_init=ray_init,  # add extra parameters for ray.init()
        cluster=True,  # add parameters for ray.init() for the slurm cluster
        progress_step_percent=1,
    )

    parallel.shutdown()

    # Now create a proper parallel environment to be used
    ray_init = {}
    ray_init['num_gpus'] = 0  # otherwise failing with NVIDIA-555 driver.


    return Parallel(
        num_cpus=4,
        npar_k=0,
        ray_init=ray_init,  # add extra parameters for ray.init()
        cluster=False,  # add parameters for ray.init() for the slurm cluster
        progress_step_percent=1,
    )
