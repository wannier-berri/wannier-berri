import os
import warnings


def get_ray_cpus_count():
    try:
        import ray
        if ray.is_initialized():
            return int(round(ray.available_resources()['CPU']))
        else:
            return 1
    except ImportError:
        return 1


def ray_init_cluster(
    num_cpus=None,
    ignore_initialized=False,
    **ray_init,
):
    """# The follwoing is done for testing, when __init__ is called with `cluster = True`,
    # but no actual ray cluster was initialized (and hence the needed environmental variables are not set
    """
    import ray
    if ray.is_initialized():
        if not ignore_initialized:
            warnings.warn("Ray is already initialized, using the existing initialization, ignoring the parameters passed to ray_init_cluster")
        return
    ray_init_loc = {}

    def set_opt(opt, def_val):
        if opt not in ray_init:
            ray_init_loc[opt] = def_val()
        else:
            warnings.warn(f"the ray cluster will use '{opt}={ray_init[opt]}' provided in ray_init")
    set_opt('address', lambda: 'auto')
    set_opt('_node_ip_address', lambda: os.environ["ip_head"].split(":")[0])
    set_opt('_redis_password', lambda: os.environ["redis_password"])

    ray_init_loc.update(ray_init)
    ray_init_loc['num_cpus'] = num_cpus
    print("initializing ray with ", ray_init_loc)
    ray.init(**ray_init_loc)


def ray_init(**kwargs):
    """tries to import and initialize ray, but does nothing if it cannot be imported, or if it is already initialized"""
    try:
        import ray
        if ray.is_initialized():
            print(f"ray is already initialized with {get_ray_cpus_count()} cpus")
        else:
            ray.init(**kwargs)
    except ImportError as err:
        print(f"write : unable to import ray, no initialization performed:{err}")


def ray_shutdown():
    """tries to shutdown ray, but does nothing if it cannot be imported, or if it is not initialized"""
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
    except ImportError:
        pass
