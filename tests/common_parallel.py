import ray
from wannierberri.parallel import ray_init_cluster



def init_parallel_ray():
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

    ray_init_cluster(
        num_cpus=None,
        ignore_initialized=True,
        **ray_init,
    )


    ray.shutdown()

    # Now create a proper parallel environment to be used
    ray_init = {}
    ray_init['num_gpus'] = 0  # otherwise failing with NVIDIA-555 driver.
    ray.init(**ray_init)
