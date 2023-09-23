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
import datetime, os

# logger = MyLogger("expLog.log").get_logger()

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
    # tot_samples = xcoord['y'][:10]
    # tot_samples.append(xcoord['y'][init_samp:].rolling(window=4).min())
    # reduced_df = pd.DataFrame(tot_samples)
    xcoord.to_csv(timestmp+'/'+str(name)+'_'+str(init_samp)+'_'+str(maxbud)+rl+'.csv')
    # plot_convergence(xcoord.iloc[init_samp:], timestmp+'/'+name+str(maxbud)+'_'+rl)
    xcoord = xcoord.to_numpy()
    print('_______________ Min Observed ________________')
    print(xcoord[np.argmin(xcoord[:,2]), :])
    
    return xcoord[np.argmin(xcoord[:,2]), :], timestmp

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

    def rastrigin(self):
        def internal_function(x, lb=None, ub=None, from_agent = None):
            # print('x',x.reshape((10,1)))
            A = 10
            n = len(x)
            return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
            # x = x.reshape((10,1))
            # d = x.shape
            # # if lb is None or ub is None:
            # #     lb = np.full((d,), -2.5)
            # #     ub = np.full((d,), 3)
            # # x = from_unit_box(x, lb, ub)
            # return 10 * d + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=0)

        range_array = np.array([[-2.5, 3]])  # Range [-4, 5] as a 1x2 array
        region_support = np.tile(range_array, (2, 1))  # Replicate the range 10 times along axis 0

        # task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
        glob_mins = np.array([[3]*10,[-2.805118]*10,[-3.779310]*10,[3.584428]*10])
        y_of_mins = np.array([internal_function(i) for i in glob_mins])

        # print('region_support: ',region_support)
        # region_support = np.array([[-5, 5], [-5, 5]]) 

        seeds = []
        
        # sd = int(time.time())
        # seeds.append(sd)
        seed = 123 #task_id #12345

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
            num_agents= 7,
            behavior=Behavior.MINIMIZATION,
            init_sampling_type="lhs_sampling",
            logger = self.logger
        )

        data, rg, plot_res = opt(bo, gpr_model)
        name = Test_internalBO.rastrigin.__name__
        minobs, timestmp = logdf(data,init_samp,maxbud, name+str(seed)+"_"+str(12), y_of_mins, rollout=True)
        
        print('seeds :',seeds)
        sdf = pd.DataFrame(seeds)
        # sdf.to_csv(timestmp+'/sdf.csv')
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
        # print(np.array(data.history, dtype=object).shape)
        # contour(plot_res['agents'], plot_res['assignments'], plot_res['region_support'], plot_res['test_function'],plot_res['inactive_subregion_samples'], plot_res['sample'], [glob_mins,y_of_mins], minobs)
        assert np.array(data.history, dtype=object).shape[0] == (maxbud - init_samp)*4 + init_samp
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
    
    def branin(self):
        def internal_function(x, from_agent = None):
            a = 1.0
            b = 5.1 / (4 * np.pi ** 2)
            c = 5.0 / np.pi
            r = 6.0
            s = 10.0
            t = 1.0 / (8 * np.pi)

            result = 0
            for i in range(98):  # 100 dimensions, but loop from 0 to 98
                result += a * (x[i + 1] - b * x[i] ** 2 + c * x[i] - r) ** 2 + s * (1 - t) * np.cos(x[i]) + s

            return result

        range_array = np.array([[-5, 15]])  # Range [-4, 5] as a 1x2 array
        region_support = np.tile(range_array, (100, 1))  # Replicate the range 10 times along axis 0

        # task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
        glob_mins = np.array([[3]*10,[-2.805118]*10,[-3.779310]*10,[3.584428]*10])
        # y_of_mins = np.array([internal_function(i) for i in glob_mins])

        # print('region_support: ',region_support)
        # region_support = np.array([[-5, 5], [-5, 5]]) 

        seeds = []
        
        # sd = int(time.time())
        # seeds.append(sd)
        seed = 123#task_id #12345

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
        name = Test_internalBO.rastrigin.__name__
        # minobs, timestmp = logdf(data,init_samp,maxbud, name+str(seed)+"_"+str(sd), y_of_mins, rollout=True)
        
        print('seeds :',seeds)
        sdf = pd.DataFrame(seeds)
        # sdf.to_csv(timestmp+'/sdf.csv')
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
        # print(np.array(data.history, dtype=object).shape)
        # contour(plot_res['agents'], plot_res['assignments'], plot_res['region_support'], plot_res['test_function'],plot_res['inactive_subregion_samples'], plot_res['sample'], [glob_mins,y_of_mins], minobs)
        assert np.array(data.history, dtype=object).shape[0] == (maxbud - init_samp)*4 + init_samp
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