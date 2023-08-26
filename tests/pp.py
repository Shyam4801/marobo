import multiprocessing
import numpy as np

class YourClass:
    def __init__(self):
        self.num_processes = multiprocessing.cpu_count()  # Number of available CPU cores
        self.numthreads = 4
        self.mc_iters = 5

    def perf(self):
        self.get_arrays()
        res = self.parallel_processing()
        return res

    def get_arrays(self):
        # Your logic to get arrays here
        arrays = np.arange(24).reshape((6,2,2))  # List of arrays with shape (6, m, n)
        self.arrays = arrays
        # return arrays

    

    def parallel_processing(self):
#         multiprocessing.freeze_support()  # Only for Windows
        
        arrays = self.arrays
        
        # with multiprocessing.Pool(processes=self.num_processes) as pool:
        #     processed_arrays = pool.map(self.process_array, arrays)
        
        if self.numthreads > 1:
            serial_mc_iters = [int(self.mc_iters/self.numthreads)] * self.numthreads
            pool = multiprocessing.Pool(processes=self.numthreads)
            processed_arrays = pool.map(self.process_array, arrays)
            pool.close()
            pool.join()
            
        # for idx, processed_array in enumerate(processed_arrays):
        #     print(f"Processed Array {idx}: {processed_array}")
        return processed_arrays

    def process_array(self, arr):
        # Your array processing logic here
        result = arr * 0.5  # Just an example
        return result

# if __name__ == '__main__':
#     your_instance = YourClass()
#     # your_instance.get_arrays()
#     # your_instance.parallel_processing()
#     res = your_instance.perf()
#     print(res)



import multiprocessing

def parallelize(func):
    def wrapper(*args, **kwargs):
        num_processes = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(func, *args, **kwargs)
        return results
    return wrapper

@parallelize
def process_array(arr):
    # Your array processing logic here
    result = arr + 1  # Just an example
    return result

if __name__ == '__main__':
    arrays = [...]  # List of arrays with shape (6, m, n)
    processed_arrays = process_array(arrays)
    print(processed_arrays)
