
# import dask
# from dask.distributed import Client, LocalCluster
# from joblib import Parallel, delayed

# dask.config.set({'distributed.worker.daemon': False})
# cluster = LocalCluster(memory_limit='2GB')
# client = Client(cluster)

# def genSamples_in_parallel(x):
#             x += 1
#             return  x

#         # Execute the evaluation function in parallel for each Xs_root item
# results = Parallel(n_jobs=-1)(delayed(genSamples_in_parallel)(x) for x in range(100))
        

# # results = client.submit(lambda x: x + 1, 4).result()

# # Gather results
# results = client.gather(results)

# # Close the client and cluster
# client.close()
# cluster.close()

from dask.distributed import Client, LocalCluster
import dask.array as da
import numpy as np

# Create a local cluster with 4 workers
cluster = LocalCluster(n_workers=4)
client = Client(cluster)

# Define a simple computation function
def compute_sum(array):
    return array.sum()

# Generate a large random array
array_size = (1000, 1000)
random_array = da.random.random(array_size, chunks=(100, 100))

# Perform the computation in parallel
result = compute_sum(random_array)

# Print the result
print("Result:", result.compute())

# Close the client and cluster
client.close()
cluster.close()
