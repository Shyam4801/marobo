# import numpy as np
# from bo import Behavior, PerformBO
# from bo.bayesianOptimization import InternalBO
# from bo.gprInterface import InternalGPR
# from bo.interface import BOResult

import pickle
import numpy as np
import pandas as pd 
from numpy.random import default_rng
import pickle
import unittest
from bo.utils.volume import compute_volume

from bo.gprInterface import InternalGPR
from bo.utils import Fn, compute_robustness
from bo.sampling import uniform_sampling
from bo.bayesianOptimization.internalBO import InternalBO
# from bo.bayesianOptimization.rolloutBO import RolloutBO
from bo.bayesianOptimization.rolloutAllatOnce import RolloutBO
# from bo.bayesianOptimization.rolloutIsolated import RolloutBO
from bo import Behavior, PerformBO
from matplotlib import pyplot as plt
from bo.utils.visualize import *
from bo.utils.logged import *
import time , datetime, os

# def internal_function(X):
#             return X[0] ** 2 + X[1] ** 2 -1

# init_reg_sup = np.array([[-1, 1], [-2, 2]])                          # cartesian prod is -1,-2 ; -1,2 ; 1,-2 ; 1,2
# def internal_function(X):
#     return (X[0] - 2)**2 + (X[1] - 2)**2   #glob min (2,2) Local minimum at (0.5, 0.5)

# init_reg_sup = np.array([[-5, 5], [-5, 5]])   

# def internal_function(X): #Branin with unique glob min -  9.42, 2.475
#         # if X.shape[1] != 2:
#         #     raise Exception('Dimension must be 2')
#         # d = 2
#         # if lb is None or ub is None:
#         #     lb = np.full((d,), 0)
#         #     ub = np.full((d,), 0)
#         #     lb[0] = -5
#         #     lb[1] = 0
#         #     ub[0] = 10
#         #     ub[1] = 15
#         # x = from_unit_box(x, lb, ub)
#         x1 = X[0]
#         x2 = X[1]
#         t = 1 / (8 * np.pi)
#         s = 10
#         r = 6
#         c = 5 / np.pi
#         b = 5.1 / (4 * np.pi ** 2)
#         a = 1
#         term1 = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2
#         term2 = s * (1 - t) * np.cos(x1)
#         l1 = 5 * np.exp(-5 * ((x1+3.14)**2 + (x2-12.27)**2))
#         l2 = 5 * np.exp(-5 * ((x1+3.14)**2 + (x2-2.275)**2))
#         return term1 + term2 + s + l1 + l2

# init_reg_sup = np.array([[-5, 10], [-5, 15]])

# Evaluated by default from [-4, 5]^d
# def from_unit_box(x, lb, ub):
#     return lb + (ub - lb) * x

# def internal_function(x, lb=None, ub=None): #ackley
#     n = len(x)
#     sum_sq_term = -0.2 * np.sqrt((1/n) * np.sum(x**2))
#     cos_term = np.sum(np.cos(2 * np.pi * x))
#     return -20 * np.exp(sum_sq_term) - np.exp(cos_term) + 20 + np.exp(1)

# range_array = np.array([[-2, 2]])  # Range [-4, 5] as a 1x2 array
# init_reg_sup = np.tile(range_array, (10, 1))  # Replicate the range 10 times along axis 0

# # def internal_function(X):
# #     return X[0]**4 + X[1]**4 - 4*X[0]*X[1] + 1

# # init_reg_sup = np.array([[-5, 5], [-5, 5]]) 



# optimizer = PerformBO(
#     test_function=internal_function,
#     init_budget=10,
#     max_budget=80,
#     region_support=init_reg_sup,
#     seed=12345,
#     behavior=Behavior.MINIMIZATION,
#     init_sampling_type="lhs_sampling"
# )

# z = optimizer(bo_model=InternalBO(), gpr_model=InternalGPR())
# history = z.history
# time = z.optimization_time

# print(np.array(history, dtype=object))
# print(f"Time taken to finish iterations: {round(time, 3)}s.")


def logdf(data,init_samp,maxbud, name, yofmins, rollout=False):
    df = pd.DataFrame(np.array(data.history, dtype='object'))
    # df = df.iloc[:,1].apply(lambda x: x[0])
    print(df)
    print('_______________________________')
    print('yofmins :',yofmins)
    xcoord = pd.DataFrame(df.iloc[:,1].to_list())
    xcoord['y'] = df.iloc[:,2]
    xcoord['ysofar'] = [min(xcoord['y'].iloc[:i]) for i in range(1,len(xcoord)+1)] #xcoord['y'].apply(lambda x : min([x - y for y in yofmins]))
    if rollout:
        rl='rollout'
    else:
        rl = 'n'
    timestmp = 'results/macreps/'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    if not os.path.exists(timestmp):
        os.makedirs(timestmp)
    xcoord.iloc[init_samp:].to_csv(timestmp+'/'+str(name)+'_'+str(init_samp)+'_'+str(maxbud)+rl+'.csv')
    plot_convergence(xcoord.iloc[init_samp:], timestmp+'/'+name+str(maxbud)+'_'+rl)
    xcoord = xcoord.to_numpy()
    print('_______________ Min Observed ________________')
    print(xcoord[np.argmin(xcoord[:,2]), :])
    
    return xcoord[np.argmin(xcoord[:,2]), :], timestmp
    
logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s %(module)s %(levelname)s: %(message)s',
                    datefmt = '%m/%d/%Y %I:%M:%S %p', level = logging.INFO, filename='expLog.log')
# def himmelblau(self):
# def internal_function(x, lb=None, ub=None, from_agent = None): #ackley
#     n = len(x)
#     sum_sq_term = -0.2 * np.sqrt((1/n) * np.sum(x**2))
#     cos_term = np.sum(np.cos(2 * np.pi * x))
#     return -20 * np.exp(sum_sq_term) - np.exp(cos_term) + 20 + np.exp(1)

# range_array = np.array([[-4, 5]])  # Range [-4, 5] as a 1x2 array
# region_support = np.tile(range_array, (10, 1))  # Replicate the range 10 times along axis 0

def internal_function(X, from_agent = None):
    return (X[0]**2+X[1]-11)**2 + (X[0]+X[1]**2-7)**2

task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
# Minimum 1: f(3.0, 2.0) = 0.0
# Minimum 2: f(-2.805118, 3.131312) = 0.0
# Minimum 3: f(-3.779310, -3.283186) = 0.0
# Minimum 4: f(3.584428, -1.848126) = 0.0
glob_mins = np.array([[3,2],[-2.805118, 3.131312],[-3.779310, -3.283186],[3.584428, -1.848126]])
y_of_mins = np.array([internal_function(i) for i in glob_mins])

region_support = np.array([[-5, 5], [-5, 5]]) 

seeds = []

sd = int(time.time())
seeds.append(sd)
seed = task_id #12345

gpr_model = InternalGPR()
bo = RolloutBO()

init_samp = 10
maxbud = 15
opt = PerformBO(
    test_function=internal_function,
    init_budget=init_samp,
    max_budget=maxbud,
    region_support=region_support,
    seed=seed,
    num_agents= 4,
    behavior=Behavior.MINIMIZATION,
    init_sampling_type="lhs_sampling",
    logger = logger
)

data, rg, plot_res = opt(bo, gpr_model)
name = 'himmelblau' #Test_internalBO.himmelblau.__name__
minobs, timestmp = logdf(data,init_samp,maxbud, name+str(sd)+str(task_id), y_of_mins, rollout=True)

print('seeds :',seeds)
sdf = pd.DataFrame(seeds)
sdf.to_csv(timestmp+'/sdf.csv')
init_vol = compute_volume(region_support)
final_vol = compute_volume(rg)
reduction = ((init_vol - final_vol)/init_vol)* 100
print('_______________________________')
print('reduced ', reduction)
print('_______________________________')
print('Bounds of final partition: ',rg)
print('_______________________________')
print()
print('Plotting')