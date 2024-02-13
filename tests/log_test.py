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
from bo.utils.loggScript import *
import datetime, os, time
from bo.utils.logger import logMeta

def from_unit_box(x, lb, ub):
    return lb + (ub - lb) * x

def logrolldf(xtr,ytr,aidx,h,init_samp, rollout=True):
    # df = pd.DataFrame(np.array(data.history, dtype='object'))
    # df = df.iloc[:,1].apply(lambda x: x[0])
    # print(df)
    # print('_______________________________')
    # print('yofmins :',yofmins)
    xcoord = pd.DataFrame([xtr, ytr])
    # xcoord['y'] = df.iloc[:,2]
    xcoord['ysofar'] = [min(xcoord.iloc[:i,-1]) for i in range(1,len(xcoord)+1)] #xcoord['y'].apply(lambda x : min([x - y for y in yofmins]))
    print('df : ', xcoord)
    if rollout:
        rl='rollout'
    else:
        rl = 'n'
    timestmp = 'results/rollResults/'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    if not os.path.exists(timestmp):
        os.makedirs(timestmp)
    
    xcoord.to_csv(timestmp+'/'+str(aidx)+'/'+str(h)+'_'+rl+'.csv')
    plot_convergence(xcoord.iloc[init_samp:], timestmp+'/'+str(aidx)+'/'+str(h)+'_'+rl)
    xcoord = xcoord.to_numpy()
    # agentSamples = (maxbud - init_samp) * 4
    # print(xcoord[np.argmin(xcoord[-agentSamples:,-2]), :])
    # print(f'_______________ Index of Min Observed _{agentSamples}_______________')
    # print(np.argmin(xcoord[-agentSamples:,-2]))
    
    # return xcoord[np.argmin(xcoord[:,-2]), :], timestmp

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
    timestmp = 'results/rollResults/'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    if not os.path.exists(timestmp):
        os.makedirs(timestmp)
    # tot_samples = xcoord['y'][:10]
    # tot_samples.append(xcoord['y'][init_samp:].rolling(window=4).min())
    # reduced_df = pd.DataFrame(tot_samples)
    dfdic = xcoord.iloc[:40,:-2].to_dict()
    initpath = '/Users/shyamsundar/MS/resume/gitrepo/non-myopic_bo/results/fromagents'
    with open(initpath+f'/initsmp.pickle', 'wb') as handle:
        pickle.dump(dfdic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    xcoord.to_csv(timestmp+'/'+str(name)+'_'+str(init_samp)+'_'+str(maxbud)+rl+'.csv')
    plot_convergence(xcoord.iloc[init_samp:], timestmp+'/'+name+str(maxbud)+'_'+rl)
    xcoord = xcoord.to_numpy()
    agentSamples = (maxbud - init_samp) * 4
    print(xcoord[np.argmin(xcoord[-agentSamples:,-2]), :])
    print(f'_______________ Index of Min Observed _{agentSamples}_______________')
    print(np.argmin(xcoord[-agentSamples:,-2]))
    
    return xcoord[np.argmin(xcoord[:,-2]), :], timestmp

class Test_internalBO(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(format = '%(asctime)s %(module)s %(levelname)s: %(message)s',
                    datefmt = '%m/%d/%Y %I:%M:%S %p', level = logging.INFO, filename='expLog.log')
    # logging.FileHandler(filename='expLog.log')
        # formatter = logging.Formatter(
        #                 fmt = '%(asctime)s :: %(message)s', datefmt = '%a, %d %b %Y %H:%M:%S'
        #                 )

        # self.fh.setFormatter(formatter)

    def x2y2(self):
        def internal_function(X, from_agent = None):
            return X[0] ** 2 + X[1] ** 2 -1
            # return X[0] ** 2 + X[1] ** 2 + X[2] ** 2

        seed = 12345
        region_support = np.array([[-1, 1],[-2, 2]])

        gpr_model = InternalGPR()
        bo = RolloutBO()

        seeds = []

        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
        glob_mins = np.array([[3]*10,[-2.805118]*10,[-3.779310]*10,[3.584428]*10])
        y_of_mins = []

        # print('region_support: ',region_support)
        # region_support = np.array([[-5, 5], [-5, 5]]) 

        seeds = []
        
        sd = int(time.time())
        # seeds.append(sd)
        seed = task_id + 2#12345

        gpr_model = InternalGPR()
        bo = RolloutBO()

        init_samp = 1
        maxbud = 11
        name = Test_internalBO.x2y2.__name__
        logMeta(name+"_"+str(task_id), init_samp, maxbud, str(task_id))

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
        
        minobs, timestmp = logdf(data,init_samp,maxbud, name+str(sd)+"_"+str(task_id), y_of_mins, rollout=True)
        
        print('seeds :',seeds, 'minobs: ', minobs)
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
        
        # minobs = data.history[np.argmin(data.history[:,2]), :]
        # print(np.array(data.history, dtype=object).shape)
        if len(region_support) == 2:
            contour(plot_res['agents'], plot_res['assignments'], plot_res['status'], plot_res['region_support'], plot_res['test_function'],plot_res['inactive_subregion_samples'], plot_res['sample'], [glob_mins,y_of_mins], minobs)
        # assert np.array(data.history, dtype=object).shape[0] == maxbud
        # assert np.array(data.history, dtype=object).shape[1] == 3

    def rastrigin(self):
        def internal_function(x, lb=None, ub=None, from_agent = None):
            # print('x inside rastrigin ',x.shape)#,x.reshape((10,1)))
            A = 10
            n = len(x)
            # return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
            return 20 + x[0]**2 - A * np.cos(2 * np.pi * x[0]) + x[1]**2 - A * np.cos(2 * np.pi * x[1])
            # x = x.reshape((10,1))
            # d = x.shape
            # # if lb is None or ub is None:
            # #     lb = np.full((d,), -2.5)
            # #     ub = np.full((d,), 3)
            # # x = from_unit_box(x, lb, ub)
            # return 10 * d + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=0)

        range_array = np.array([[-2.5, 3]])  # Range [-4, 5] as a 1x2 array
        region_support = np.tile(range_array, (4, 1))  # Replicate the range 10 times along axis 0

        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
        glob_mins = np.array([[3]*10,[-2.805118]*10,[-3.779310]*10,[3.584428]*10])
        y_of_mins = []

        # print('region_support: ',region_support)
        # region_support = np.array([[-5, 5], [-5, 5]]) 

        seeds = []
        
        sd = int(time.time())
        # seeds.append(sd)
        seed = task_id + 2#12345

        gpr_model = InternalGPR()
        bo = RolloutBO()

        init_samp = 20
        maxbud = 22
        name = Test_internalBO.rastrigin.__name__
        logMeta(name+"_"+str(task_id), init_samp, maxbud, str(task_id))

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
        
        minobs, timestmp = logdf(data,init_samp,maxbud, name+str(sd)+"_"+str(task_id), y_of_mins, rollout=True)
        
        print('seeds :',seeds, 'minobs: ', minobs)
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
        
        # minobs = data.history[np.argmin(data.history[:,2]), :]
        # print(np.array(data.history, dtype=object).shape)
        # if len(region_support) == 2:
        #     contour(plot_res['agents'], plot_res['assignments'], plot_res['status'], plot_res['region_support'], plot_res['test_function'],plot_res['inactive_subregion_samples'], plot_res['sample'], [glob_mins,y_of_mins], minobs)
        # assert np.array(data.history, dtype=object).shape[0] == (maxbud - init_samp)*4 + init_samp
        assert np.array(data.history, dtype=object).shape[1] == 3

    def ackley(self):
        def internal_function(x, lb=None, ub=None, from_agent = None): #ackley
            n = len(x)
            sum_sq_term = -0.2 * np.sqrt((1/n) * np.sum(x**2)/n)
            cos_term = np.sum(np.cos(2 * np.pi * x))/n
            return -20 * np.exp(sum_sq_term) - np.exp(cos_term) + 20 + np.exp(1)
        
        range_array = np.array([[-4, 5]])  # Range [-4, 5] as a 1x2 array
        region_support = np.tile(range_array, (100, 1))

        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
        # glob_mins = np.array([[3]*10,[-2.805118]*10,[-3.779310]*10,[3.584428]*10])
        y_of_mins = []

        # print('region_support: ',region_support)
        # region_support = np.array([[-5, 5], [-5, 5]]) 

        seeds = []
        
        sd = int(time.time())
        # seeds.append(sd)
        seed = task_id #12345

        gpr_model = InternalGPR()
        bo = RolloutBO()

        init_samp = 5
        maxbud = 10
        name = Test_internalBO.ackley.__name__
        logMeta(name+"_"+str(task_id), init_samp, maxbud)

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
        minobs, timestmp = logdf(data,init_samp,maxbud, name+str(sd)+"_"+str(task_id), y_of_mins, rollout=True)
        
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
        
        # minobs = data.history[np.argmin(data.history[:,2]), :]
        # print(np.array(data.history, dtype=object).shape)
        # contour(plot_res['agents'], plot_res['assignments'], plot_res['region_support'], plot_res['test_function'],plot_res['inactive_subregion_samples'], plot_res['sample'], [glob_mins,y_of_mins], minobs)
        assert np.array(data.history, dtype=object).shape[0] == (maxbud - init_samp)*4 + init_samp
        assert np.array(data.history, dtype=object).shape[1] == 3
        
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
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
        glob_mins = np.array([[3]*10,[-2.805118]*10,[-3.779310]*10,[3.584428]*10])
        y_of_mins = []
        # glob_mins=[]
        seed = 12345
        sd = task_id
        # region_support = np.array([[-1, 1],[-2, 2]])

        gpr_model = InternalGPR()
        bo = RolloutBO()

        init_samp = 20
        maxbud = 25
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

        name = Test_internalBO.mod_branin.__name__
        minobs, timestmp = logdf(data,init_samp,maxbud, name+str(sd)+"_"+str(task_id), y_of_mins, rollout=True)

        init_vol = compute_volume(region_support)
        final_vol = compute_volume(rg)
        reduction = ((init_vol - final_vol)/init_vol)* 100
        print('_______________________________')
        print('reduced ', reduction)
        print('_______________________________')
        print('Bounds of final partition: ',rg)
        print('_______________________________')
        print(minobs)

        contour(plot_res['agents'], plot_res['assignments'], plot_res['status'], plot_res['region_support'], plot_res['test_function'],plot_res['inactive_subregion_samples'], plot_res['sample'], [glob_mins,y_of_mins], minobs)
        
        # assert np.array(data.history, dtype=object).shape[0] == maxbud
        # assert np.array(data.history, dtype=object).shape[1] == 3

    # @log(my_logger=logger)
    def x22y22(self):
        def internal_function(X, from_agent = None):
            return (X[0] - 2)**2 + (X[1] - 2)**2   #glob min (2,2) Local minimum at (0.5, 0.5)
        
        glob_mins = np.array([[2,2],[0.5,0.5]])
        y_of_mins = np.array([internal_function(i) for i in glob_mins])

        region_support = np.array([[-5, 5], [-5, 5]]) 

        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
        glob_mins = np.array([[3]*10,[-2.805118]*10,[-3.779310]*10,[3.584428]*10])
        y_of_mins = []
        # glob_mins=[]
        seed = 12345
        sd = task_id
        # region_support = np.array([[-1, 1],[-2, 2]])

        gpr_model = InternalGPR()
        bo = RolloutBO()

        init_samp = 1
        maxbud = 11
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

        name = Test_internalBO.x22y22.__name__
        minobs, timestmp = logdf(data,init_samp,maxbud, name+str(sd)+"_"+str(task_id), y_of_mins, rollout=True)

        init_vol = compute_volume(region_support)
        final_vol = compute_volume(rg)
        reduction = ((init_vol - final_vol)/init_vol)* 100
        print('_______________________________')
        print('reduced ', reduction)
        print('_______________________________')
        print('Bounds of final partition: ',rg)
        print('_______________________________')
        print(minobs)

        contour(plot_res['agents'], plot_res['assignments'], plot_res['status'], plot_res['region_support'], plot_res['test_function'],plot_res['inactive_subregion_samples'], plot_res['sample'], [glob_mins,y_of_mins], minobs)
        

    def x4y4(self):
        def internal_function(X, from_agent = None):
            return X[0]**4 + X[1]**4 - 4*X[0]*X[1] + 1
        
        glob_mins = np.array([[1,1],[-1, -1]])
        y_of_mins = np.array([internal_function(i) for i in glob_mins])

        region_support = np.array([[-3, 3], [-3, 3]]) 

        seed = 12345

        gpr_model = InternalGPR()
        bo = RolloutBO()

        init_samp = 1
        maxbud = 5
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
        # glob_mins = np.array([[3,2],[-2.805118, 3.131312],[-3.779310, -3.283186],[3.584428, -1.848126]])
        # y_of_mins = np.array([internal_function(i) for i in glob_mins])

        region_support = np.array([[-5, 5], [-5, 5]]) 

        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
        glob_mins = np.array([[3]*10,[-2.805118]*10,[-3.779310]*10,[3.584428]*10])
        y_of_mins = []
        # glob_mins=[]
        seed = 12345
        sd = task_id
        # region_support = np.array([[-1, 1],[-2, 2]])

        gpr_model = InternalGPR()
        bo = RolloutBO()

        init_samp = 1
        maxbud = 11
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
        minobs, timestmp = logdf(data,init_samp,maxbud, name+str(sd)+"_"+str(task_id), y_of_mins, rollout=True)

        init_vol = compute_volume(region_support)
        final_vol = compute_volume(rg)
        reduction = ((init_vol - final_vol)/init_vol)* 100
        print('_______________________________')
        print('reduced ', reduction)
        print('_______________________________')
        print('Bounds of final partition: ',rg)
        print('_______________________________')
        print(minobs)

        contour(plot_res['agents'], plot_res['assignments'], plot_res['status'], plot_res['region_support'], plot_res['test_function'],plot_res['inactive_subregion_samples'], plot_res['sample'], [glob_mins,y_of_mins], minobs)
        # assert np.array(data.history, dtype=object).shape[0] == (maxbud - init_samp)*4 + init_samp
        # assert np.array(data.history, dtype=object).shape[1] == 3


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
        
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
        glob_mins = np.array([[0.114614,0.555649,0.852547,0.653893]])
        y_of_mins = []
        # glob_mins=[]
        seed = 12345
        sd = task_id
        # region_support = np.array([[-1, 1],[-2, 2]])

        gpr_model = InternalGPR()
        bo = RolloutBO()

        init_samp = 1
        maxbud = 6
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

        name = Test_internalBO.mod_branin.__name__
        minobs, timestmp = logdf(data,init_samp,maxbud, name+str(sd)+"_"+str(task_id), y_of_mins, rollout=True)

        init_vol = compute_volume(region_support)
        final_vol = compute_volume(rg)
        reduction = ((init_vol - final_vol)/init_vol)* 100
        print('_______________________________')
        print('reduced ', reduction)
        print('_______________________________')
        print('Bounds of final partition: ',rg)
        print('_______________________________')
        print(minobs)

        # contour(plot_res['agents'], plot_res['assignments'], plot_res['status'], plot_res['region_support'], plot_res['test_function'],plot_res['inactive_subregion_samples'], plot_res['sample'], [glob_mins,y_of_mins], minobs)
        

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

        init_samp = 1
        maxbud = 3
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

        name = Test_internalBO.sixhump.__name__
        minobs, timestmp = logdf(data,init_samp,maxbud, name+str(sd)+"_"+str(task_id), y_of_mins, rollout=True)

        init_vol = compute_volume(region_support)
        final_vol = compute_volume(rg)
        reduction = ((init_vol - final_vol)/init_vol)* 100
        print('_______________________________')
        print('reduced ', reduction)
        print('_______________________________')
        print('Bounds of final partition: ',rg)
        print('_______________________________')
        print(minobs)

        # contour(plot_res['agents'], plot_res['assignments'], plot_res['status'], plot_res['region_support'], plot_res['test_function'],plot_res['inactive_subregion_samples'], plot_res['sample'], [glob_mins,y_of_mins], minobs)
        
if __name__ == "__main__":
    unittest.main()