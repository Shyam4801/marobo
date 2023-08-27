from time import time
LOGPATH = '/Users/shyamsundar/ASU/sem2/RA/partma/results/timerlog.txt'

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


# testiter(10)