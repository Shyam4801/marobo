import numpy as np
from bo import Behavior, PerformBO
from bo.bayesianOptimization import InternalBO
from bo.bayesianOptimization.marlBO import RolloutBO
from bo.gprInterface import InternalGPR
from bo.interface import BOResult
import time
import os

# logger = MyLogger("expLog.log").get_logger()
# def internal_function(X):
#             # print('internal func :', X,X.shape)
#             return X[0] ** 2 + X[1] ** 2 -1

# def internal_function(X):
#             return (X[0] - 2)**2 + (X[1] - 2)**2 

# def internal_function(X, from_agent = None):
#     return (X[0]**2+X[1]-11)**2 + (X[0]+X[1]**2-7)**2

# def internal_function(X, from_agent = None):
#             return X[0]**4 + X[1]**4 - 4*X[0]*X[1] + 1

# def internal_function(X, from_agent = None): #Branin with unique glob min -  9.42, 2.475 local min (3.14, 12.27) and (3.14, 2.275)
#             x1 = X[0]
#             x2 = X[1]
#             t = 1 / (8 * np.pi)
#             s = 10
#             r = 6
#             c = 5 / np.pi
#             b = 5.1 / (4 * np.pi ** 2)
#             a = 1
#             term1 = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2
#             term2 = s * (1 - t) * np.cos(x1)
#             l1 = 5 * np.exp(-5 * ((x1+3.14)**2 + (x2-12.27)**2))
#             l2 = 5 * np.exp(-5 * ((x1+3.14)**2 + (x2-2.275)**2))
#             return term1 + term2 + s + l1 + l2

# def internal_function(x, lb=None, ub=None, from_agent = None): #ackley
#     n = len(x)
#     sum_sq_term = -0.2 * np.sqrt((1/n) * np.sum(x**2))
#     cos_term = np.sum(np.cos(2 * np.pi * x))
#     return -20 * np.exp(sum_sq_term) - np.exp(cos_term) + 20 + np.exp(1)

def internal_function(x, lb=None, ub=None, from_agent = None):
    # print('x',x.reshape((10,1)))
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

range_array = np.array([[-2.5, 3]])  # Range [-4, 5] as a 1x2 array
init_reg_sup = np.tile(range_array, (10, 1))  # Replicate the range 10 times along axis 0


task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
sd = int(time.time())

optimizer = PerformBO(
            test_function=internal_function,
            init_budget=5,
            max_budget=6,
            region_support=init_reg_sup,
            seed=task_id,
            num_agents= 4,
            behavior=Behavior.MINIMIZATION,
            init_sampling_type="lhs_sampling"
            # logger = logger
        )

z, rg, plot_res = optimizer(RolloutBO(), InternalGPR())
# z = optimizer(bo_model=RolloutBO(4), gpr_model=InternalGPR())
# minobs, timestmp = logdf(data,init_samp,maxbud, name+str(sd)+"_"+str(task_id), y_of_mins, rollout=True)
history = z.history
time = z.optimization_time

print(np.array(history, dtype=object))
print(f"Time taken to finish iterations: {round(time, 3)}s.")

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

def logdf(data,init_samp,maxbud, name, rollout=False):
    # print(np.array(data.history, dtype='object'))
    df = pd.DataFrame(np.array(data.history, dtype='object'))
    # df = df.iloc[:,1].apply(lambda x: x[0])
    print(df)
    print('_______________________________')
    # print('yofmins :',yofmins)
    xcoord = pd.DataFrame(df.iloc[:,1].to_list())
    xcoord['y'] = df.iloc[:,2]
    xcoord['ysofar'] = [min(xcoord['y'].iloc[:i]) for i in range(1,len(xcoord)+1)]
    if rollout:
        rl='rollout'
    else:
        rl = 'n'
    xcoord.to_csv('results/rollout/rastrigin50300/'+str(name)+'_'+str(init_samp)+'_'+str(maxbud)+rl+'.csv')
    # plot_convergence(xcoord, name+str(maxbud)+'_'+rl)
    plt.plot(xcoord.index,xcoord['ysofar'])
    
    plt.savefig('results/rollout/rastrigin50300/'+str(name)+'_'+str(init_samp)+'_'+str(maxbud)+rl+'.png')
    plt.show()


# logdf(z,50,300,'rastrigin_rl_'+str(sd)+'_'+str(task_id))