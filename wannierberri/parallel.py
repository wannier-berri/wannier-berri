import os
import warnings


def get_ray_runtime_env(runtime_env=None, use_current_checkout=True):
    """Build a Ray runtime_env that ships the local wannierberri checkout.

    Parameters
    ----------
    runtime_env : dict or None
        Existing Ray runtime environment configuration.
    use_current_checkout : bool
        If ``True``, include the local ``wannierberri`` package directory in
        ``py_modules`` so workers import the same checkout as the driver.
    """
    if runtime_env is None:
        runtime_env_loc = {}
    else:
        runtime_env_loc = dict(runtime_env)

    if not use_current_checkout:
        return runtime_env_loc or None

    package_dir = os.path.dirname(os.path.abspath(__file__))
    py_modules = list(runtime_env_loc.get("py_modules", []))
    if package_dir not in py_modules:
        py_modules.append(package_dir)
    runtime_env_loc["py_modules"] = py_modules
    return runtime_env_loc


def get_ray_cpus_count():
    try:
        import ray
        if ray.is_initialized():
            return int(round(ray.cluster_resources()['CPU']))
        else:
            return 1
    except ImportError:
        return 1


def ray_init_cluster(
    num_cpus=None,
    ignore_initialized=False,
    use_current_checkout=True,
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
    runtime_env = get_ray_runtime_env(ray_init_loc.get('runtime_env'), use_current_checkout=use_current_checkout)
    if runtime_env is not None:
        ray_init_loc['runtime_env'] = runtime_env
    ray_init_loc['num_cpus'] = num_cpus
    print("initializing ray with ", ray_init_loc)
    ray.init(**ray_init_loc)


def ray_init(ignore_missing=True, use_current_checkout=True, **kwargs):
    """tries to import and initialize ray, but does nothing if it cannot be imported, 
    or if it is already initialized

    Parameters
    ----------
    kwargs:
        parameters to be passed to ray.init() function. Please refer to ray `documentation <https://docs.ray.io/en/latest/>`__ for more options of ray.init() function.
        Use ``use_current_checkout=False`` to disable automatic shipping of the
        local ``wannierberri`` checkout to Ray workers.
    """
    try:
        import ray
        if ray.is_initialized():
            print(f"ray is already initialized with {get_ray_cpus_count()} cpus")
        else:
            runtime_env = get_ray_runtime_env(kwargs.get('runtime_env'), use_current_checkout=use_current_checkout)
            if runtime_env is not None:
                kwargs['runtime_env'] = runtime_env
            ray.init(**kwargs)
    except ImportError as err:
        if ignore_missing:
            print(f"write : unable to import ray, no initialization performed:{err}")
        else:
            raise RuntimeError(f"write : unable to import ray, no initialization performed:{err}. Exiting")



def ray_shutdown():
    """tries to shutdown ray, but does nothing if it cannot be imported, or if it is not initialized"""
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
    except ImportError:
        pass


def check_ray_initialized():
    try:
        import ray
        if ray.is_initialized():
            return True
        else:
            warnings.warn("ray package found, but ray is not initialized, running in serial mode")
            return False
    except ImportError:
        warnings.warn("Ray is not installed, running in serial mode")
        return False
