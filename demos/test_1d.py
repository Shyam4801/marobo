import numpy as np
from bo import Behavior, PerformBO
from bo.bayesianOptimization import InternalBO
from bo.bayesianOptimization.rolloutBO import RolloutBO
from bo.gprInterface import InternalGPR
from bo.interface import BOResult
import time
from bo.utils.loggScript import *
import os

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s %(module)s %(levelname)s: %(message)s',
                    datefmt = '%m/%d/%Y %I:%M:%S %p', level = logging.INFO, filename='expLog.log')

# modified Branin function
def internal_function(X, from_agent = None): #Branin with unique glob min -  9.42, 2.475 local min (3.14, 12.27) and (3.14, 2.275)
    x1 = X[0]
    x2 = X[1]
    t = 1 / (8 * np.pi)
    s = 10
    r = 6
    c = 5 / np.pi
    b = 5.1 / (4 * np.pi ** 2)
    a = 1
    term1 = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2
    term2 = s * (1 - t) * np.cos(x1)
    l1 = 5 * np.exp(-5 * ((x1+3.14)**2 + (x2-12.27)**2))
    l2 = 5 * np.exp(-5 * ((x1+3.14)**2 + (x2-2.275)**2))
    return term1 + term2 + s + l1 + l2

region_support = np.array([[-5, 10], [-5, 15]])
seed = 123

gpr_model = InternalGPR()
bo = RolloutBO()

init_samp = 20
maxbud = 22
num_agents = 4
opt = PerformBO(
    test_function=internal_function,
    init_budget=init_samp,
    max_budget=maxbud,
    region_support=region_support,
    seed=seed,
    num_agents= num_agents,
    behavior=Behavior.MINIMIZATION,
    init_sampling_type="lhs_sampling",
    logger = logger
)

data = opt(bo, gpr_model)
history = data.history
time = data.optimization_time

agentSamples = (maxbud - init_samp) * num_agents 
print(np.array(history[-agentSamples:], dtype=object))
print('-'*20) 
print(f"Time taken to finish iterations: {round(time, 3)}s.")