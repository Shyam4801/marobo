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
from bo.bayesianOptimization.rolloutBO import RolloutBO
from bo import Behavior, PerformBO
from matplotlib import pyplot as plt
from bo.utils.visualize import *
from bo.utils.logged import *

# logger = MyLogger("expLog.log").get_logger()

def logdf(data,init_samp,maxbud, name):
    df = pd.DataFrame(np.array(data.history, dtype='object'))
    # df = df.iloc[:,1].apply(lambda x: x[0])
    print(df)
    print('_______________________________')
    xcoord = pd.DataFrame(df.iloc[:,1].to_list())
    xcoord['y'] = df.iloc[:,2]
    xcoord.to_csv('results/'+str(name)+'_'+str(init_samp)+'_'+str(maxbud)+'.csv')
    xcoord = xcoord.to_numpy()
    print('_______________ Min Observed ________________')
    print(xcoord[np.argmin(xcoord[:,2]), :])
    return xcoord[np.argmin(xcoord[:,2]), :]
    
    # plot_1d(xcoord,self.tf,0.25,0.07,0.8,self.init_budget,self.max_budget - self.init_budget)
    # plot_obj(xcoord,internal_function,[9.42, 2.475],[-5,10],[0,15],init_samp,maxbud - init_samp)

class Test_internalBO(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(format = '%(asctime)s %(module)s %(levelname)s: %(message)s',
                    datefmt = '%m/%d/%Y %I:%M:%S %p', level = logging.INFO, filename='expLog.log')
    # logging.FileHandler(filename='expLog.log')
        # formatter = logging.Formatter(
        #                 fmt = '%(asctime)s :: %(message)s', datefmt = '%a, %d %b %Y %H:%M:%S'
        #                 )

        # self.fh.setFormatter(formatter)

    def test1_internalBO(self):
        def internal_function(X, agent_samples = None):
            return X[0] ** 2 + X[1] ** 2 -1
            # return X[0] ** 2 + X[1] ** 2 + X[2] ** 2

        seed = 12345
        region_support = np.array([[-1, 1],[-2, 2]])

        gpr_model = InternalGPR()
        bo = InternalBO()

        init_samp = 5
        maxbud = 10
        opt = PerformBO(
            test_function=internal_function,
            init_budget=init_samp,
            max_budget=maxbud,
            region_support=region_support,
            seed=seed,
            behavior=Behavior.MINIMIZATION,
            init_sampling_type="lhs_sampling"
        )

        data, rg = opt(bo, gpr_model)

        name = Test_internalBO.test1_internalBO.__name__
        logdf(data,init_samp,maxbud, name)

        init_vol = compute_volume(region_support)
        final_vol = compute_volume(rg)
        reduction = ((init_vol - final_vol)/init_vol)* 100
        print('_______________________________')
        print('reduced ', reduction)
        print('_______________________________')
        print('Bounds of final partition: ',rg)
        print('_______________________________')

        assert np.array(data.history, dtype=object).shape[0] == maxbud
        assert np.array(data.history, dtype=object).shape[1] == 3


    def ackley(self):
        def internal_function(x, lb=None, ub=None, from_agent = None): #ackley
            n = len(x)
            sum_sq_term = -0.2 * np.sqrt((1/n) * np.sum(x**2))
            cos_term = np.sum(np.cos(2 * np.pi * x))
            return -20 * np.exp(sum_sq_term) - np.exp(cos_term) + 20 + np.exp(1)

        range_array = np.array([[-4, 5]])  # Range [-4, 5] as a 1x2 array
        region_support = np.tile(range_array, (10, 1))  # Replicate the range 10 times along axis 0

        glob_mins = np.array([[2,2],[0.5,0.5]])
        y_of_mins = np.array([internal_function(i) for i in glob_mins])

        seed = 12345
        # region_support = np.array([[-1, 1],[-2, 2]])

        gpr_model = InternalGPR()
        bo = RolloutBO()

        init_samp = 5
        maxbud = 6
        opt = PerformBO(
            test_function=internal_function,
            init_budget=init_samp,
            max_budget=maxbud,
            region_support=region_support,
            seed=seed,
            num_agents= 4,
            behavior=Behavior.MINIMIZATION,
            init_sampling_type="lhs_sampling"
        )

        data, rg, plot_res = opt(bo, gpr_model)
        name = Test_internalBO.himmelblau.__name__
        minobs = logdf(data,init_samp,maxbud, name)
        
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
        
        # minobs = data.history[np.argmin(data.history[:,2]), :]
        
        contour(plot_res['agents'], plot_res['assignments'], plot_res['region_support'], plot_res['test_function'],plot_res['inactive_subregion_samples'], plot_res['sample'], [glob_mins,y_of_mins], minobs)
        assert np.array(data.history, dtype=object).shape[0] == maxbud
        assert np.array(data.history, dtype=object).shape[1] == 3
        
    def mod_branin(self):
        def internal_function(X, agent_samples = None): #Branin with unique glob min -  9.42, 2.475 local min (3.14, 12.27) and (3.14, 2.275)
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

        seed = 12345
        # region_support = np.array([[-1, 1],[-2, 2]])

        gpr_model = InternalGPR()
        bo = InternalBO()

        init_samp = 5
        maxbud = 10
        opt = PerformBO(
            test_function=internal_function,
            init_budget=init_samp,
            max_budget=maxbud,
            region_support=region_support,
            seed=seed,
            behavior=Behavior.MINIMIZATION,
            init_sampling_type="lhs_sampling"
        )

        data, rg = opt(bo, gpr_model)

        name = Test_internalBO.mod_branin.__name__
        logdf(data,init_samp,maxbud, name)

        init_vol = compute_volume(region_support)
        final_vol = compute_volume(rg)
        reduction = ((init_vol - final_vol)/init_vol)* 100
        print('_______________________________')
        print('reduced ', reduction)
        print('_______________________________')
        print('Bounds of final partition: ',rg)
        print('_______________________________')

        assert np.array(data.history, dtype=object).shape[0] == maxbud
        assert np.array(data.history, dtype=object).shape[1] == 3

    # @log(my_logger=logger)
    def test2_internalBO(self):
        def internal_function(X, from_agent = None):
            return (X[0] - 2)**2 + (X[1] - 2)**2   #glob min (2,2) Local minimum at (0.5, 0.5)
        
        glob_mins = np.array([[2,2],[0.5,0.5]])
        y_of_mins = np.array([internal_function(i) for i in glob_mins])

        region_support = np.array([[-5, 5], [-5, 5]]) 

        seed = 12345

        gpr_model = InternalGPR()
        bo = RolloutBO()

        init_samp = 5
        maxbud = 17
        opt = PerformBO(
            test_function=internal_function,
            init_budget=init_samp,
            max_budget=maxbud,
            region_support=region_support,
            seed=seed,
            num_agents= 4,
            behavior=Behavior.MINIMIZATION,
            init_sampling_type="lhs_sampling",
            logger = self.logger
        )

        data, rg, plot_res = opt(bo, gpr_model)
        # print('data: ',data.agent_history)

        name = Test_internalBO.test2_internalBO.__name__
        minobs = logdf(data,init_samp,maxbud, name)
        
        init_vol = compute_volume(region_support)
        final_vol = compute_volume(rg)
        reduction = ((init_vol - final_vol)/init_vol)* 100
        print('_______________________________')
        print('reduced ', reduction)
        print('_______________________________')
        print('Bounds of final partition: ',rg)
        print('_______________________________')

        contour(plot_res['agents'],plot_res['assignments'], plot_res['region_support'], plot_res['test_function'],plot_res['inactive_subregion_samples'], plot_res['sample'], [glob_mins,y_of_mins], minobs)

        self.logger.info("init_samp: {}".format(init_samp))

        assert np.array(data.history, dtype=object).shape[0] == maxbud
        assert np.array(data.history, dtype=object).shape[1] == 3

    def test3_internalBO(self):
        def internal_function(X, from_agent = None):
            return X[0]**4 + X[1]**4 - 4*X[0]*X[1] + 1
        
        glob_mins = np.array([[1,1],[-1, -1]])
        y_of_mins = np.array([internal_function(i) for i in glob_mins])

        region_support = np.array([[-3, 3], [-3, 3]]) 

        seed = 12345

        gpr_model = InternalGPR()
        bo = RolloutBO()

        init_samp = 5
        maxbud = 30
        opt = PerformBO(
            test_function=internal_function,
            init_budget=init_samp,
            max_budget=maxbud,
            region_support=region_support,
            seed=seed,
            num_agents= 4,
            behavior=Behavior.MINIMIZATION,
            init_sampling_type="lhs_sampling",
            logger = self.logger
        )

        data, rg, plot_res = opt(bo, gpr_model)
        name = Test_internalBO.test3_internalBO.__name__
        minobs = logdf(data,init_samp,maxbud, name)
        
        init_vol = compute_volume(region_support)
        final_vol = compute_volume(rg)
        reduction = ((init_vol - final_vol)/init_vol)* 100
        print('_______________________________')
        print('reduced ', reduction)
        print('_______________________________')
        print('Bounds of final partition: ',rg)
        print('_______________________________')

        contour(plot_res['agents'],plot_res['assignments'], plot_res['region_support'], plot_res['test_function'],plot_res['inactive_subregion_samples'], plot_res['sample'], [glob_mins,y_of_mins], minobs)

        assert np.array(data.history, dtype=object).shape[0] == maxbud
        assert np.array(data.history, dtype=object).shape[1] == 3


    def himmelblau(self):
        def internal_function(X, from_agent = None):
            return (X[0]**2+X[1]-11)**2 + (X[0]+X[1]**2-7)**2
        
        # Minimum 1: f(3.0, 2.0) = 0.0
        # Minimum 2: f(-2.805118, 3.131312) = 0.0
        # Minimum 3: f(-3.779310, -3.283186) = 0.0
        # Minimum 4: f(3.584428, -1.848126) = 0.0
        glob_mins = np.array([[3,2],[-2.805118, 3.131312],[-3.779310, -3.283186],[3.584428, -1.848126]])
        y_of_mins = np.array([internal_function(i) for i in glob_mins])

        region_support = np.array([[-5, 5], [-5, 5]]) 

        seed = 124

        gpr_model = InternalGPR()
        bo = RolloutBO()

        init_samp = 5
        maxbud = 7
        opt = PerformBO(
            test_function=internal_function,
            init_budget=init_samp,
            max_budget=maxbud,
            region_support=region_support,
            seed=seed,
            num_agents= 4,
            behavior=Behavior.MINIMIZATION,
            init_sampling_type="lhs_sampling",
            logger = self.logger
        )

        data, rg, plot_res = opt(bo, gpr_model)
        name = Test_internalBO.himmelblau.__name__
        minobs = logdf(data,init_samp,maxbud, name)
        
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
        
        # minobs = data.history[np.argmin(data.history[:,2]), :]
        
        # contour(plot_res['agents'], plot_res['assignments'], plot_res['region_support'], plot_res['test_function'],plot_res['inactive_subregion_samples'], plot_res['sample'], [glob_mins,y_of_mins], minobs)
        assert np.array(data.history, dtype=object).shape[0] == (maxbud - init_samp)*4 + init_samp
        assert np.array(data.history, dtype=object).shape[1] == 3


if __name__ == "__main__":
    unittest.main()