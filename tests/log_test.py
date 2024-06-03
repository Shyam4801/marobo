import pickle
import numpy as np
import pickle
import unittest

from bo.gprInterface import InternalGPR
from bo.bayesianOptimization.internalBO import InternalBO
from bo.bayesianOptimization.rolloutBO import RolloutBO
from bo import Behavior, PerformBO
from bo.utils.loggScript import *
import datetime, os, time
from bo.utils.logger import logMeta
import math

def from_unit_box(x, lb, ub):
    return lb + (ub - lb) * x

class Test_internalBO(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(format = '%(asctime)s %(module)s %(levelname)s: %(message)s',
                    datefmt = '%m/%d/%Y %I:%M:%S %p', level = logging.INFO, filename='expLog.log')

    def x2y2(self):
        def internal_function(X, from_agent = None):
            return X[0] ** 2 + X[1] ** 2 -1
            # return X[0] ** 2 + X[1] ** 2 + X[2] ** 2

        seed = 12345
        region_support = np.array([[-1, 1],[-2, 2]])

        gpr_model = InternalGPR()
        bo = RolloutBO()

        seeds = []

        gpr_model = InternalGPR()
        bo = RolloutBO()
        num_agents = 4

        init_samp = 1
        maxbud = 11
        name = Test_internalBO.x2y2.__name__

        opt = PerformBO(
            test_function=internal_function,
            init_budget=init_samp,
            max_budget=maxbud,
            region_support=region_support,
            seed=seed,
            num_agents= num_agents,
            behavior=Behavior.MINIMIZATION,
            init_sampling_type="lhs_sampling",
            logger = self.logger
        )

        data = opt(bo, gpr_model)
        history = data.history
        time = data.optimization_time

        agentSamples = (maxbud - init_samp) * num_agents 
        print(np.array(history[-agentSamples:], dtype=object))
        print('-'*20) 
        print(f"Time taken to finish iterations: {round(time, 3)}s.")

    def rastrigin(self):
        def internal_function(x, lb=None, ub=None, from_agent = None):
            # print('x inside rastrigin ',x.shape)#,x.reshape((10,1)))
            A = 10
            n = len(x)
            return 20 + x[0]**2 - A * np.cos(2 * np.pi * x[0]) + x[1]**2 - A * np.cos(2 * np.pi * x[1])
        
        range_array = np.array([[-2.5, 3]])  # Range [-4, 5] as a 1x2 array
        region_support = np.tile(range_array, (4, 1))  # Replicate the range 10 times along axis 0

        seed = 12345

        gpr_model = InternalGPR()
        bo = RolloutBO()

        init_samp = 20
        maxbud = 22
        name = Test_internalBO.rastrigin.__name__
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
            logger = self.logger
        )

        data = opt(bo, gpr_model)
        history = data.history
        time = data.optimization_time

        agentSamples = (maxbud - init_samp) * num_agents 
        print(np.array(history[-agentSamples:], dtype=object))
        print('-'*20) 
        print(f"Time taken to finish iterations: {round(time, 3)}s.")

    def ackley(self):
        def internal_function(x, lb=None, ub=None, from_agent = None): #ackley
            n = len(x)
            sum_sq_term = -0.2 * np.sqrt((1/n) * np.sum(x**2)/n)
            cos_term = np.sum(np.cos(2 * np.pi * x))/n
            return -20 * np.exp(sum_sq_term) - np.exp(cos_term) + 20 + np.exp(1)
        
        range_array = np.array([[-4, 5]])  # Range [-4, 5] as a 1x2 array
        region_support = np.tile(range_array, (100, 1))

        seed = 12345

        gpr_model = InternalGPR()
        bo = RolloutBO()
        num_agents = 4

        init_samp = 5
        maxbud = 10
        name = Test_internalBO.ackley.__name__

        opt = PerformBO(
            test_function=internal_function,
            init_budget=init_samp,
            max_budget=maxbud,
            region_support=region_support,
            seed=seed,
            num_agents= num_agents,
            behavior=Behavior.MINIMIZATION,
            init_sampling_type="lhs_sampling",
            logger = self.logger
        )

        data = opt(bo, gpr_model)
        history = data.history
        time = data.optimization_time

        agentSamples = (maxbud - init_samp) * num_agents 
        print(np.array(history[-agentSamples:], dtype=object))
        print('-'*20) 
        print(f"Time taken to finish iterations: {round(time, 3)}s.")
        
    def mod_branin(self):
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
        name = Test_internalBO.mod_branin.__name__
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
            logger = self.logger
        )

        data = opt(bo, gpr_model)
        history = data.history
        time = data.optimization_time

        agentSamples = (maxbud - init_samp) * num_agents 
        print(np.array(history[-agentSamples:], dtype=object))
        print('-'*20) 
        print(f"Time taken to finish iterations: {round(time, 3)}s.")
       

    # @log(my_logger=logger)
    def x22y22(self):
        def internal_function(X, from_agent = None):
            return (X[0] - 2)**2 + (X[1] - 2)**2   #glob min (2,2) Local minimum at (0.5, 0.5)
        

        region_support = np.array([[-5, 5], [-5, 5]]) 

        seed = 12345

        gpr_model = InternalGPR()
        bo = RolloutBO()
        num_agents = 4

        init_samp = 1
        maxbud = 11
        opt = PerformBO(
            test_function=internal_function,
            init_budget=init_samp,
            max_budget=maxbud,
            region_support=region_support,
            seed=seed,
            num_agents= num_agents,
            behavior=Behavior.MINIMIZATION,
            init_sampling_type="lhs_sampling",
            logger = self.logger
        )

        data = opt(bo, gpr_model)
        history = data.history
        time = data.optimization_time

        agentSamples = (maxbud - init_samp) * num_agents 
        print(np.array(history[-agentSamples:], dtype=object))
        print('-'*20) 
        print(f"Time taken to finish iterations: {round(time, 3)}s.")

    def x4y4(self):
        def internal_function(X, from_agent = None):
            return X[0]**4 + X[1]**4 - 4*X[0]*X[1] + 1
        
        region_support = np.array([[-3, 3], [-3, 3]]) 

        seed = 12345

        gpr_model = InternalGPR()
        bo = RolloutBO()
        num_agents = 4

        init_samp = 1
        maxbud = 5
        opt = PerformBO(
            test_function=internal_function,
            init_budget=init_samp,
            max_budget=maxbud,
            region_support=region_support,
            seed=seed,
            num_agents= num_agents,
            behavior=Behavior.MINIMIZATION,
            init_sampling_type="lhs_sampling",
            logger = self.logger
        )

        data = opt(bo, gpr_model)
        name = Test_internalBO.test3_internalBO.__name__
        history = data.history
        time = data.optimization_time

        agentSamples = (maxbud - init_samp) * num_agents 
        print(np.array(history[-agentSamples:], dtype=object))
        print('-'*20) 
        print(f"Time taken to finish iterations: {round(time, 3)}s.")


    def himmelblau(self):
        def internal_function(X, from_agent = None):
            return (X[0]**2+X[1]-11)**2 + (X[0]+X[1]**2-7)**2
        
        # Minimum 1: f(3.0, 2.0) = 0.0
        # Minimum 2: f(-2.805118, 3.131312) = 0.0
        # Minimum 3: f(-3.779310, -3.283186) = 0.0
        # Minimum 4: f(3.584428, -1.848126) = 0.0
        # glob_mins = np.array([[3,2],[-2.805118, 3.131312],[-3.779310, -3.283186],[3.584428, -1.848126]])
        # y_of_mins = np.array([internal_function(i) for i in glob_mins])

        region_support = np.array([[-5, 5], [-5, 5]]) 

        seed = 12345
        
        gpr_model = InternalGPR()
        bo = RolloutBO()
        num_agents = 4

        init_samp = 1
        maxbud = 11
        opt = PerformBO(
            test_function=internal_function,
            init_budget=init_samp,
            max_budget=maxbud,
            region_support=region_support,
            seed=seed,
            num_agents= num_agents,
            behavior=Behavior.MINIMIZATION,
            init_sampling_type="lhs_sampling",
            logger = self.logger
        )

        data = opt(bo, gpr_model)
        history = data.history
        time = data.optimization_time

        agentSamples = (maxbud - init_samp) * num_agents 
        print(np.array(history[-agentSamples:], dtype=object))
        print('-'*20) 
        print(f"Time taken to finish iterations: {round(time, 3)}s.")


    def hartmann6d(self):
        def internal_function(x, from_agent = None):
            x = x.reshape((1,6))
            if x.shape[1] != 6:
                raise Exception('Dimension must be 6')
            d = 6
            # if lb is None or ub is None:
            #     lb = np.full((d,), 0)
            #     ub = np.full((d,), 1)
            # x = from_unit_box(x, lb, ub)
            alpha = np.array([1.0, 1.2, 3.0, 3.2])
            A = np.array([[10.0, 3.0,  17.0, 3.5,  1.7,  8.0],
                        [0.05, 10.0, 17.0, 0.1,  8.0,  14.0],
                        [3.0,  3.5,  1.7,  10.0, 17.0, 8.0],
                        [17.0, 8.0,  0.05, 10.0, 0.1,  14.0]])
            P = 1e-4 * np.array([[1312.0, 1696.0, 5569.0, 124.0,  8283.0, 5886.0],
                                [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
                                [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
                                [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0]])
            outer = 0
            for ii in range(4):
                inner = 0
                for jj in range(6):
                    xj = x[:, jj]
                    Aij = A[ii, jj]
                    Pij = P[ii, jj]
                    inner += Aij * ((xj - Pij) ** 2)
                outer += alpha[ii] * np.exp(-inner)
            res = -outer
            return res[0]
        
        range_array = np.array([[-5, 5]])  # Range [-4, 5] as a 1x2 array
        region_support = np.tile(range_array, (6, 1))
        
        seed = 12345
        
        gpr_model = InternalGPR()
        bo = RolloutBO()
        num_agents = 4

        init_samp = 1
        maxbud = 6
        opt = PerformBO(
            test_function=internal_function,
            init_budget=init_samp,
            max_budget=maxbud,
            region_support=region_support,
            seed=seed,
            num_agents= num_agents,
            behavior=Behavior.MINIMIZATION,
            init_sampling_type="lhs_sampling",
            logger = self.logger
        )

        data = opt(bo, gpr_model)

        history = data.history
        time = data.optimization_time

        agentSamples = (maxbud - init_samp) * num_agents 
        print(np.array(history[-agentSamples:], dtype=object))
        print('-'*20) 
        print(f"Time taken to finish iterations: {round(time, 3)}s.")

    def sixhump(self):
        def internal_function(x, lb=None, ub=None, from_agent = None):
            # x = x.reshape((1,2))
            # if x.shape[1] != 2:
            #     raise Exception('Dimension must be 2')
            d = 2
            res = (4.0 - 2.1*x[0]**2 + (x[0]**4)/3.0)*x[0]**2 + x[0]*x[1] + (-4 + 4*x[1]**2) * x[1]**2
            return res#[0]
        
        # range_array = np.array([[0, 1]])  # Range [-4, 5] as a 1x2 array
        region_support = np.array([[-3, 3],[-2, 2]])
        
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
        glob_mins = np.array([[-0.0898,0.7126]])
        y_of_mins = []
        # glob_mins=[]
        seed = 12345
        sd = task_id
        # region_support = np.array([[-1, 1],[-2, 2]])

        gpr_model = InternalGPR()
        bo = RolloutBO()
        num_agents = 4

        init_samp = 1
        maxbud = 3
        opt = PerformBO(
            test_function=internal_function,
            init_budget=init_samp,
            max_budget=maxbud,
            region_support=region_support,
            seed=seed,
            num_agents= num_agents,
            behavior=Behavior.MINIMIZATION,
            init_sampling_type="lhs_sampling",
            logger = self.logger
        )

        data = opt(bo, gpr_model)

        history = data.history
        time = data.optimization_time

        agentSamples = (maxbud - init_samp) * num_agents 
        print(np.array(history[-agentSamples:], dtype=object))
        print('-'*20) 
        print(f"Time taken to finish iterations: {round(time, 3)}s.")

    def griewank(self):
        def internal_function(x, from_agent = None): #Branin with unique glob min -  9.42, 2.475 local min (3.14, 12.27) and (3.14, 2.275)
            # x = x[0]
            # print(x)
            sum_sq = sum(xi ** 2 for xi in x)
            prod_cos = math.prod(math.cos(xi / math.sqrt(i + 1)) for i, xi in enumerate(x))
            return 1 + (1 / 4000) * sum_sq - prod_cos

        range_array = np.array([[-600, 600]])  # Range [-4, 5] as a 1x2 array
        region_support = np.tile(range_array, (10, 1))
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
        # glob_mins = np.array([[3]*10,[-2.805118]*10,[-3.779310]*10,[3.584428]*10])
        y_of_mins = []
        # glob_mins=[]
        seed = task_id + 2#123
        sd = task_id
        # region_support = np.array([[-1, 1],[-2, 2]])

        gpr_model = InternalGPR()
        bo = RolloutBO()
        num_agents = 4
        

        init_samp = 20
        maxbud = 22
        name = Test_internalBO.griewank.__name__
        logMeta(name+"_"+str(task_id), init_samp, maxbud, str(task_id))

        opt = PerformBO(
            test_function=internal_function,
            init_budget=init_samp,
            max_budget=maxbud,
            region_support=region_support,
            seed=seed,
            num_agents= num_agents,
            behavior=Behavior.MINIMIZATION,
            init_sampling_type="lhs_sampling",
            logger = self.logger
        )

        data, rg, plot_res = opt(bo, gpr_model)

        history = data.history
        time = data.optimization_time

        agentSamples = (maxbud - init_samp) * num_agents 
        print(np.array(history[-agentSamples:], dtype=object))
        print('-'*20) 
        print(f"Time taken to finish iterations: {round(time, 3)}s.")


    def schwefel(self):
        def internal_function(x, from_agent = None): #Branin with unique glob min -  9.42, 2.475 local min (3.14, 12.27) and (3.14, 2.275)
            d = len(x)
            return (418.9829 * d) - (np.sum(x * np.sin(np.sqrt(np.abs(x)))))

        range_array = np.array([[-500, 500]])  # Range [-4, 5] as a 1x2 array
        region_support = np.tile(range_array, (10, 1))
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
        # glob_mins = np.array([[3]*10,[-2.805118]*10,[-3.779310]*10,[3.584428]*10])
        y_of_mins = []
        # glob_mins=[]
        seed = task_id #123
        sd = task_id
        # region_support = np.array([[-1, 1],[-2, 2]])

        gpr_model = InternalGPR()
        bo = RolloutBO()

        num_agents = 4

        init_samp = 20
        maxbud = 22
        name = Test_internalBO.griewank.__name__
        logMeta(name+"_"+str(task_id), init_samp, maxbud, str(task_id))

        opt = PerformBO(
            test_function=internal_function,
            init_budget=init_samp,
            max_budget=maxbud,
            region_support=region_support,
            seed=seed,
            num_agents= num_agents,
            behavior=Behavior.MINIMIZATION,
            init_sampling_type="lhs_sampling",
            logger = self.logger
        )

        data = opt(bo, gpr_model)

        history = data.history
        time = data.optimization_time

        agentSamples = (maxbud - init_samp) * num_agents 
        print(np.array(history[-agentSamples:], dtype=object))
        print('-'*20) 
        print(f"Time taken to finish iterations: {round(time, 3)}s.")
        

if __name__ == "__main__":
    unittest.main()