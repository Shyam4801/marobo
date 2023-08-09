import time
import numpy as np

class Fn:
    def __init__(self, func):
        self.func = func
        self.count = 0
        self.agent_count = 0
        self.point_history = []
        self.simultation_time = []
        self.agent_point_history = []


    def __call__(self, *args, **kwargs):
        from_agent = kwargs['from_agent']
        print(kwargs['from_agent'])
        sim_time_start = time.perf_counter()
        rob_val = self.func(*args, **kwargs)
        time_elapsed = time.perf_counter() - sim_time_start
        self.simultation_time.append(time_elapsed)
        min_rob = np.inf
        min_args = np.inf
        if not from_agent:
            self.count = self.count + 1
        #     self.agent_point_history.append([*args, rob_val])
        # else:
        self.point_history.append([self.count, *args, rob_val])
        # for k,v in kwargs.items():
        #     print(k,v)
        
        # agent_samples = np.array(list(agent_samples))
        # else:
        #     self.agent_count = self.agent_count + 1
        #     self.agent_point_history.append(*args)
        return rob_val
