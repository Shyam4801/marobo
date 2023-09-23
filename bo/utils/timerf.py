from time import time
LOGPATH = 'results/timerlog.txt'

def logtime(path_to_logfile):
    def timer_func(func):
        # This function shows the execution time of 
        # the function object passed
        def wrap_func(*args, **kwargs):
            t1 = time()
            result = func(*args, **kwargs)
            t2 = time()
            with open(path_to_logfile, 'a') as f:
                f.write(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s'+ "\n")
            return result
        return wrap_func
    return timer_func
    

# @logtime('timelog.txt')
# def testiter(n):
#     for i in range(n):
#         pass

# def process_data(item):
#     # Your processing logic here
#     result = item * 2
#     return result


# # testiter(10)
# from parallelize import MyClass

# data_list = [1, 2, 3, 4, 5]
# num_processes = 2  # Adjust this as needed

# arr = MyClass(process_data)
# # Call the parallel_process method without creating an instance
# processed_data = arr.parallel_process(data_list, num_processes)

# print(processed_data)