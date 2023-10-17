from time import time
import pandas as pd
import numpy as np 
import datetime
import os
import logging
import threading
import csv 

LOGPATH = 'results/timerlog.txt'
LOGRESULTSPATH = 'results/logDir/'
# logging.basicConfig(filename=LOGRESULTSPATH+'test.txt', level=logging.INFO, format="%(asctime)s - Variable: %(message)s")
metadata = ""
    

def logtime(path_to_logfile):
    def timer_func(func):
        # This function shows the execution time of 
        # the function object passed
        def wrap_func(*args, **kwargs):
            t1 = time()
            result = func(*args, **kwargs)
            t2 = time()
            with open(path_to_logfile, 'w') as f:
                f.write(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s'+ "\n")
            return result
        return wrap_func
    return timer_func

def logMeta(func, init, maxsmp):
    # mt = meta(func, init, maxsmp)    
    # mt.setMeta()
    global metadata
    metadata = func+"_"+str(init)+"_"+str(maxsmp)

def log_periodically(interval):
    def decorator(method):
        def wrapper(self, *args, **kwargs):
            def log_variable_periodically():
                name = metadata #mt.funcName+"_"+str(mt.initSamp)+"_"+str(mt.maxSmp)
                while not self.stop_logging:
                    variable_length = len(self.point_history)
                    if variable_length % interval == 0:
                        batch = self.point_history
                        # self.variable = self.point_history[5:]
                        write_to_csv(batch, name)

                # Log any remaining elements when the method finishes
                if self.point_history:
                    write_to_csv(self.point_history, name)

            def write_to_csv(data, name):     
                timestmp = LOGRESULTSPATH+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
                if not os.path.exists(timestmp):
                    print(timestmp)
                    os.makedirs(timestmp)           
                with open(timestmp+'/'+name+'.csv', "w", newline="") as file:
                    writer = csv.writer(file)
                    for row in data:
                        tmp = [i for i in row[1]]
                        tmp.append(row[2])
                        writer.writerow(tmp)

            self.stop_logging = False
            logging_thread = threading.Thread(target=log_variable_periodically)
            logging_thread.start()

            try:
                return method(self, *args, **kwargs)
            finally:
                self.stop_logging = True
                logging_thread.join()

        return wrapper
    return decorator

