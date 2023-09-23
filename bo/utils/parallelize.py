

# def parallelise(point_to_evaluate, numthreads):
#     def multi(func):
#         # This function shows the execution time of 
#         # the function object passed
#         def wrap_func(*args, **kwargs):
#             if numthreads > 1:
#                 serial_mc_iters = [int(int(self.numthreads)/self.numthreads)] * self.numthreads
#                 print('serial_mc_iters',serial_mc_iters, self.numthreads)
#                 pool = Pool(processes=self.numthreads)
#                 rewards = pool.map(self.get_pt_reward, serial_mc_iters)
#                 pool.close()
#                 pool.join()
#             else:
#                 rewards = self.get_pt_reward()
#             rewards = np.hstack((rewards))
#         return wrap_func
#     return timer_func


# def _evaluate_at_point_list(self, point_to_evaluate):
#         self.point_current = point_to_evaluate
#         if self.numthreads > 1:
#             serial_mc_iters = [int(int(self.numthreads)/self.numthreads)] * self.numthreads
#             print('serial_mc_iters',serial_mc_iters, self.numthreads)
#             pool = Pool(processes=self.numthreads)
#             rewards = pool.map(self.get_pt_reward, serial_mc_iters)
#             pool.close()
#             pool.join()
#         else:
#             rewards = self.get_pt_reward()
#         rewards = np.hstack((rewards))
#         # print('rewards: ', rewards)
#         return np.sum(rewards)/self.numthreads


# import multiprocessing

# def process_data(item):
#     # Your processing logic here
#     result = item * 2
#     return result

# def parallel_process(cls, data_list,process_data, num_processes):
#         pool = multiprocessing.Pool(processes=num_processes)
#         results = pool.map(process_data, data_list)
#         pool.close()
#         pool.join()
#         return results

# class MyClass:
#     # @classmethod
#     def __init__(self, data):
#          self.data = data    

#     def process():
         
#     # @classmethod
#     # def another_class_method(cls, data_list, num_processes):
#     #     # You can call the parallel_process method from here
#     #     processed_data = cls.parallel_process(data_list,process_data, num_processes)
#     #     print(cls)
#     #     return processed_data

# if __name__ == "__main__":
#     data_list = [1, 2, 3, 4, 5]
#     num_processes = 2  # Adjust this as needed

#     # Call the parallel_process method without creating an instance
#     processed_data = MyClass.parallel_process(data_list,process_data, num_processes)

#     print(processed_data)

from joblib import Parallel, delayed
# from timerf import logtime, LOGPATH
import time

def unwrap_self(arg, **kwarg):
    return square_class.square_int(*arg, **kwarg)

class square_class:
    def square_int(self, i):
        return i * i
    
    def run(self, num):
        results = []
        results = Parallel(n_jobs= -1, backend="loky")\
            (delayed(unwrap_self)(i) for i in zip([self]*len(num), num))
        print(results)

strt = time.time()
square_int = square_class()

[square_int.square_int(i) for i in range(1000000)]
# square_int.run(num = range(1000000))

end = time.time()
print(end-strt)