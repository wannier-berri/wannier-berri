



class Parallel():
    """ a class to store parameters of parallel evaluation

    Parameters
    -----------
    method : str
        a method to be used for parallelization 'serial' or 'ray'
    num_cpus : int 
        number of parallel processes. If <=0  - serial execution 
    chunksize : int
        chunksize for distributing K points among processes. If not set or if <=0, set to max(1, min(int(numK / num_proc / 200), 10)). Relevant only if num_proc > 0.
"""

    def __init__(self,
                   method=None ,
                   num_cpus=0  ,
                   npar_k = 0 , 
                   ray_init={} ,     # add extra parameters for ray.init()
                   cluster=False , # add parameters for ray.init() for the slurm cluster
                   progress_step_percent  = 1  ,  #
                   progress_timeout = None  # relevant only for ray, seconds
                 ):

        if method is None:
            if num_cpus <= 0:
                method = "serial"
            else:
                method = "ray"

        self.method=method
        self.progress_step_percent  = progress_step_percent

        if cluster:
            if self.method == "ray" :
                ray_init_loc['address']          = 'auto'
                ray_init_loc['_node_ip_address'] = os.environ["ip_head"].split(":")[0]
                ray_init_loc['_redis_password']  = os.environ["redis_password"]
            else :
                print ("WARNING: cluster (multinode) computation is possible only with 'ray' parallelization")

        if  self.method == "serial":
            self.num_cpus = 1
            _,self.npar_k=pool(0)
            self.pool_K,self.npar_K=pool(0)
        elif self.method == "ray" : 
            ray_init_loc={}
            ray_init_loc.update(ray_init)
            if num_cpus>0:
                ray_init_loc['num_cpus']=num_cpus
            import ray
            ray.init(**ray_init_loc)
            self.num_cpus=int(round(ray.available_resources()['CPU']))
            self.ray=ray
            _,self.npar_k=pool(npar_k)
            self.npar_K=int(round(self.num_cpus/self.npar_k))
        else :
            raise ValueError ("Unknown parallelization method:{}".format(self.method))



    def progress_step(self,n_tasks,npar):
        return max ( 1,
                      npar,
                      int(round(n_tasks*self.progress_step_percent / 100)) 
                    )


    def shutdown(self):
        if self.method == "ray":
            self.ray.shutdown()


def pool(npar):
    if npar>1:
        try:
            from  multiprocessing import Pool
            pool = Pool(npar).imap
            print ('created a pool of {} workers'.format(npar))
            return pool , npar
        except Exception as err:
            print ('failed to create a pool of {} workers : {}\n doing in serial'.format(npar,err))
    return   (lambda fun,lst : [fun(x) for x in lst]) , 1
