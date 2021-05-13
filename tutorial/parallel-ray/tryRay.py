import ray,os
import time

# Start Ray.
t0=time.time()
print(os.environ["ip_head"], os.environ["redis_password"])
ray.init(address='auto', _node_ip_address=os.environ["ip_head"].split(":")[0], _redis_password=os.environ["redis_password"])

print (ray.nodes())

t01=time.time()



@ray.remote
def f(x):
    time.sleep(1)
    return x

# Start 4 tasks in parallel.
print (ray.get(f.remote(-1)))

# Wait for the tasks to complete and retrieve the results.
# With at least 4 cores, this will take 1 second.

t02=time.time()
for n in 1,4,8,16,32,64,127,128,256:
    t1=time.time()
    result_ids = []
    for i in range(n):
        result_ids.append(f.remote(i))
    results = ray.get(result_ids)  # [0, 1, 2, 3]
    t2=time.time()
    print ( "{:5d} : {:8.3f}  s ".format(n, t2-t1) )

ray.shutdown()
