import os


class Parallel():
    """ a class to store parameters of parallel evaluation

    Parameters
    -----------
    num_cpus : int
        number of parallel processes. If `None` - automatically chose by `Ray` (1 per CPU)
    npar_k : int
        additional parallelisation over k-points inside the FFT grid
    cluster : bool
        set to `True` to use a multi-node ray cluster ( see also `wannierberri.cluster <file:///home/stepan/github/wannier-berri-org/html/docs/parallel.html#multi-node-mode>`__  module)
    ray_init : dict
        parameters to be passed to `ray.init()`. Use only if you know wwhat you are doing.
    progress_step_percent : int or float
        progress (and estimated time to end) will be printed after each percent is completed
"""

    def __init__(
            self,
            num_cpus=None,
            npar_k=0,
            ray_init={},  # add extra parameters for ray.init()
            cluster=False,  # add parameters for ray.init() for the slurm cluster
            progress_step_percent=1,
                 ):

        self.method = "ray"
        self.progress_step_percent = progress_step_percent

        ray_init_loc = {}
        if cluster:
            # The follwoing is done for testing, when __init__ is called with `cluster = True`,
            # but no actual ray cluster was initialized (and hence the needed environmental variables are not set
            def set_opt(opt, def_val):
                if opt not in ray_init:
                    ray_init_loc[opt] = def_val()
                else:
                    print(f"WARNING: the ray cluster will use '{ray_init[opt]}' provided in ray_init")
            set_opt('address', lambda: 'auto')
            set_opt('_node_ip_address', lambda: os.environ["ip_head"].split(":")[0])
            set_opt('_redis_password', lambda: os.environ["redis_password"])

        ray_init_loc.update(ray_init)
        ray_init_loc['num_cpus'] = num_cpus
        import ray
        ray.init(**ray_init_loc)
        self.num_cpus = int(round(ray.available_resources()['CPU']))
        self.ray = ray
        _, self.npar_k = pool(npar_k)
        self.npar_K = int(round(self.num_cpus / self.npar_k))

    def progress_step(self, n_tasks, npar):
        return max(1, npar, int(round(n_tasks * self.progress_step_percent / 100)))

    def shutdown(self):
        self.ray.shutdown()


class Serial(Parallel):
    """ a class defining the serial execution (although `npar_k` is allowed)

    Parameters
    -----------
    npar_k : int
        additional parallelisation ove k-points inside the FFT grid
    progress_step_percent : int or float
        progress (and estimated time to end) will be printed after each percent is completed
"""

    def __init__(self, npar_k=None, progress_step_percent=1):
        self.progress_step_percent = progress_step_percent
        self.method = "serial"
        self.num_cpus = 1
        if npar_k is None:
            _, self.npar_k = pool(0)
        else:
            self.npar_k = npar_k
        self.npar_K = 1

    def shutdown(self):
        print("No need to shutdown Serial()")



def pool(npar):
    if npar > 1:
        try:
            from multiprocessing import Pool
            pool = Pool(npar).imap
            print('created a pool of {} workers'.format(npar))
            return pool, npar
        except Exception as err:
            print('failed to create a pool of {} workers : {}\n doing in serial'.format(npar, err))
    return (lambda fun, lst: [fun(x) for x in lst]), 1
