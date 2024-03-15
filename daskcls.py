# from dask.distributed import Client
# from dask_jobqueue import SLURMCluster
# import multiprocessing 

# # Define your computation function
# def compute(Xs_root, iter_list):
#     # Example computation that utilizes close to 100% of cores
#     result = []
#     cpu = multiprocessing.cpu_count()
#     # for x, i in zip(Xs_root, iter_list):
#         # Perform computation here
#         # Example: Append the result of some computation to the result list
#     result.append((Xs_root, iter_list, cpu))
#     return result

# # Define your lists
# iter_list = [1, 2, 3, 4, 5]
# Xs_root = ['a', 'b', 'c', 'd', 'e']

# # Configure SLURMCluster
# cluster = SLURMCluster(cores=1, memory='500MB', processes=1, queue='htc', walltime='00:15:00')

# # Scale the cluster to desired number of workers
# cluster.scale(5)  # Scale to 20 workers (adjust as needed)

# # Connect a client to the cluster
# client = Client(cluster)

# # Scatter Xs_root and iter_list among workers
# # Xs_root_future = client.scatter(Xs_root)
# # iter_future = client.scatter(iter_list)

# results = []
# for idx, Xs_root_item in enumerate(Xs_root):
#     result = client.submit(compute, Xs_root_item, iter_list[idx])
#     print('result: ', result)
#     results.append(result)

# # Define the computation function and submit it to the cluster
# # result_future = client.submit(compute, Xs_root_future, iter_future)

# # Gather the result
# # result = result_future.result()
# results = client.gather(results)

# # Print the result
# print("Result:", results)

# # Close the client and cluster
# client.close()
# cluster.close()




from dask.distributed import Client
from dask_jobqueue import SLURMCluster
import joblib
from joblib import parallel_backend, parallel_config
from tqdm import tqdm

def my_parallel_function(x):
    # Example parallelizable function
    res = pll(x)
    return res

def pll(n):
    results = joblib.Parallel(n_jobs=-1)(joblib.delayed(intern)(x) for x in tqdm(range(n)))
    return results
    
def intern(x):
    
    return x*x

# Configure SLURMCluster
cluster = SLURMCluster(cores=1, memory='500MB', processes=1, queue='htc', walltime='01:00:00')

# Scale the cluster to desired number of workers
cluster.scale(5)  # Scale to 20 workers (adjust as needed)

# Connect a client to the cluster
client = Client(cluster)


# Use Dask as the backend for joblib
with parallel_backend('dask', scheduler_host=client.scheduler.address):
    parallel_config(backend='dask', wait_for_workers_timeout=200)
    # Define the range of values to process
    values = [10,]*100000000 #range(10)
    
    # Perform parallel computation using joblib
    results = joblib.Parallel(n_jobs=-1)(joblib.delayed(my_parallel_function)(x) for x in tqdm(values))

# Close the client and cluster
client.close()
cluster.close()

# Print the results
print(results)

