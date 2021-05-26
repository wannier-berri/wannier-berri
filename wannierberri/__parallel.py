



class Parallel()
    """ a class to store parameters of parallel evaluation

    Parameters
    -----------
    method : str
        a method to be used for parallelization 'serial', 'multiprocessing'  or 'ray'
    num_cus : int 
        number of parallel processes. If <=0  - serial execution 
    chunksize : int
        chunksize for distributing K points among processes. If not set or if <=0, set to max(1, min(int(numK / num_proc / 200), 10)). Relevant only if num_proc > 0.
"""

    def __init__(self,
                   method=None ,
                   num_cpus=0  ,
                   npar_k = 0 , 
                   ray_init={} ,     # add extra parameters for ray.init()
                   ray_slurm=False , # add parameters for ray.init() for the slurm cluster
                   chunksize=None  , # size of chunk in multiprocessing 
                   progress_step_percent  = 1    ,  #
                   progress_timeout = None,  # relevant only for ray, seconds
                 )

        if method is None:
            if num_cpus <= 0:
                method = "serial"
            else:
                method = "multiprocessing-K"
        self.method=method
        self.progress_step_percent  = progress_step_percent

        if  self.method == "serial":
            self.num_cpus = 1
            self.pool_k,self.npar_k=pool(0)
            self.pool_K,self.npar_K=pool(0)
        elif self.method == "ray" : 
            ray_init_loc={}
            ray_init_loc.update(ray_init)
            if num_cpus>0:
                ray_init[num_cpus]=num_cpus
            import ray
            ray.init()
            self.num_cpus=int(round(ray.available_resources()['CPU']))
            self.ray=ray
            self.pool_k,self.npar_k=pool(npark)
            self.npar_K=int(round(self.num_cpus/self.npar_k))
        elif self.method == 'multiprocessing-k'
            self.pool_k,self.npar_k=pool(self.num_cpus)
            self.pool_K,self.npar_K=pool(0)
        elif self.method == 'multiprocessing-K'
            self.pool_k,self.npar_k=pool(0)
            self.pool_K,self.npar_K=pool(self.num_cpus)
        else :
            raise ValueError ("Unknown parallelization method:{}".format(self.method))






    def progress_step(self,n_tasks,npar=None):
        if npar is None: 
            npar = self.num_cpus
        return step_print = max ( 1, 
                                  npar,
                                  self.npar_progress, 
                                  int(round(n_tasks*progress_step_procent / 100)) 
                                )


    def __del__(self):
        if self.method == "ray":
            ray.shutdown()


def pool(npar):
    if npar>0:
        try:
            from  multiprocessing import Pool
            pool = multiprocessing.Pool(npar).map
            print ('created a pool of {} workers'.format(npar))
            return pool , npar
        except Exception as err:
            print ('failed to create a pool of {} workers : {}\n doing in serial'.format(npar,err))
    return   (lambda fun,lst : [fun(x) for x in lst]) , 1
