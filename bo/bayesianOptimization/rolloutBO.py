from typing import Callable, Tuple
from copy import deepcopy
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import norm
from tqdm import tqdm
from ..utils.visualize import contour
import plotly.graph_objects as go
import yaml

from .bointerface import BO_Interface
from .rolloutEI import RolloutEI
from ..gprInterface import GPR
from ..sampling import uniform_sampling, lhs_sampling
from ..utils import compute_robustness
from ..behavior import Behavior

from ..utils.volume import compute_volume
from ..agent.treeOperations import * 
from ..agent.partition import Node
from ..agent.agent import Agent
from ..agent.constants import *
from ..agent.prior import Prior
from ..utils.logger import logtime, LOGPATH
from ..utils.savestuff import *
import random
import os
from itertools import permutations
import multiprocessing as mp
from multiprocessing import Pool
from joblib import Parallel, delayed
from ..utils.plotlyExport import exportTreeUsingPlotly
from memory_profiler import profile

from dask.distributed import Client, LocalCluster

from joblib import parallel_backend, parallel_config
import joblib
from joblib import wrap_non_picklable_objects

import dask, time, json
# from dask_jobqueue import SLURMCluster
import concurrent.futures

from ..utils.function import Fn

from ..utils.cpr import cprofile

with open('config.yml', 'r') as file:
    configs = yaml.safe_load(file)


def min_diagonal(matrix):
    # Number of 4x4 matrices in the nx4x4 matrix
    n = matrix.shape[0]
    
    # Initialize the resulting 4x4 matrix with max values
    result_matrix = np.full((4, 4), np.inf)
    index_matrix = np.zeros((4,), dtype=int)
    
    # Iterate over each 4x4 matrix
    for i in range(n):
        # Extract the current 4x4 matrix
        current_matrix = matrix[i]
        
        # Iterate over each row
        for j in range(4):
            # Compare diagonal elements and retain the row with the minimum diagonal element
            if current_matrix[j, j] < result_matrix[j, j]:
                result_matrix[j] = current_matrix[j]
                index_matrix[j] = i
    
    return result_matrix, index_matrix

def find_min_diagonal_sum_matrix(matrix):
    n, _, _ = matrix.shape
    min_sum = float('inf')
    min_matrix = None
    
    for i in range(n):
        sums = np.trace(matrix[i])
        if min_sum > sums:
            min_sum = sums
            min_matrix = i

    return min_matrix

def find_min_among_diagonal_matrix(matrix):
    n, _, _ = matrix.shape
    min_sum = float('inf')
    min_matrix = None
    
    for i in range(n):
        sums = min(np.diagonal(matrix[i])) #np.trace(matrix[i])
        if min_sum > sums:
            min_sum = sums
            min_matrix = i

    return min_matrix

def unwrap_self(arg, **kwarg):
    return RolloutBO.sample(*arg, **kwarg)

class RolloutBO(BO_Interface):
    def __init__(self):
        pass
    
    # @logtime(LOGPATH)
    def sample(
        self,
        test_function: Callable,
        num_samples: int,
        x_train: NDArray,
        y_train: NDArray,
        region_support: NDArray,
        gpr_model: Callable,
        rng,
        num_agents: int,
        # init_sampling_type,
        # init_budget,
        behavior:Behavior = Behavior.MINIMIZATION
    ) -> Tuple[NDArray]:

        """Internal BO Model

        Args:
            test_function: Function of System Under Test.
            num_samples: Number of samples to generate from BO.
            x_train: Samples from Training set.
            y_train: Evaluated values of samples from Trainig set.
            region_support: Min and Max of all dimensions
            gpr_model: Gaussian Process Regressor Model developed using Factory
            rng: RNG object from numpy

        Raises:
            TypeError: If x_train is not 2 dimensional numpy array or does not match dimensions
            TypeError: If y_train is not (n,) numpy array
            TypeError: If there is a mismatch between x_train and y_train

        Returns:
            x_complete
            y_complete
            x_new
            y_new
        """
        self.horizon = 4
        falsified = False
        self.tf = test_function
        self.agent_point_hist = []
        # Domain reduces after each BO iteration 
        self.region_support = region_support
        self.assignments = []
        self.assignmentsDict = []
        self.status = []
        self.behavior = behavior
        self.x_train = x_train
        self.y_train = y_train
        maintrace = []
        self.numthreads = int(mp.cpu_count()/2)
        self.mc_iters = configs['sampling']['mc_iters']

        tf_dim = region_support.shape[0]
        X_root = Node(self.region_support, 1)
        # factorized = find_prime_factors(num_agents) #sorted(find_close_factor_pairs(num_agents), reverse=True)
        # agents_to_subregion = get_subregion(deepcopy(X_root), num_agents, factorized, np.random.randint(tf_dim))
        # X_root.add_child(agents_to_subregion)
        # # assignments = {value: 1 for value in agents_to_subregion} 
        # testv=0
        # for i in agents_to_subregion:
        #     testv += i.getVolume()

        # assert X_root.getVolume() == testv
        # # print('set([i.getVolume() for i in agents_to_subregion]) :', set([i.getVolume() for i in agents_to_subregion]))
        
        # # agents = []
        # mainAgents = []
        # globalXtrain = np.empty((1,tf_dim))
        # globalYtrain = np.empty((1))

        init_sampling_type = "lhs_sampling"
        # init_budget = configs['sampling']['initBudget'] * tf_dim#x_train.shape[0]

        # for id, l in enumerate(agents_to_subregion):
        #     l.setRoutine(MAIN)
        #     if l.getStatus(MAIN) == 1:
        #         # xtr, ytr = self.initAgents(l.input_space, init_sampling_type, int(init_budget/num_agents), tf_dim, rng, store=False)
        #         # print(f'agent xtr ', xtr, ytr)
        #         # globalXtrain = np.vstack((globalXtrain, xtr))
        #         # globalYtrain = np.hstack((globalYtrain, ytr))

        # # for id, l in enumerate(agents_to_subregion):
        # #     l.setRoutine(MAIN)
        # #     if l.getStatus(MAIN) == 1:
        #         mainag = Agent(id, None, x_train, y_train, l)
        #         # mainag.updateModel()
        #         mainag(MAIN)
        #         mainAgents.append(mainag)
        #         l.addAgentList(mainag, MAIN)
        #         # l.region_suport = region_support
        #         # print('new subr : ', l.input_space, 'agent : ', l.agent.id)

        # # split initial obs 
        # # mainAgents = splitObs(mainAgents, tf_dim, rng, MAIN, self.tf, self.behavior)
        # for a in mainAgents:
        #     a.updateModel()
        #     a.resetModel()
        
        globalXtrain = x_train #globalXtrain[1:]
        globalYtrain = y_train #globalYtrain[1:]
        if not os.path.exists(f'results/'+configs['testfunc']):
            os.makedirs(f'results/'+configs['testfunc'])
        writetocsv(f'results/'+configs['testfunc']+'/initSmp',[[globalXtrain[:,0], globalXtrain[:,1],  globalYtrain]])

        agentAftereachSample = []
        agentsWithminSmps = 0
        agdic = {0:0,1:0,2:0,3:0}
        # client = self.getWorkers()
        for sample in tqdm(range(num_samples)):
            print('globalXtrain, globalYtrain :', min(globalYtrain))
            # for sample in tqdm(range(num_samples)):
            print('_____________________________________', sample)
            # print(f"INPUT SPACE : {GREEN}{self.region_support}{END}")
            print('_____________________________________')
            print('global dataset : ', globalXtrain.shape, globalYtrain.shape)
            print('_____________________________________')
            model = GPR(gpr_model)
            model.fit(globalXtrain, globalYtrain)
            X_root.model = deepcopy(model)

            # exportTreeUsingPlotly(X_root, MAIN)
            
            avgrewards = np.zeros((1,num_agents,num_agents))
            # agentModels = []
            # xroots = []
            
            roots = self.getRootConfigs(X_root, model, sample, num_agents, tf_dim, agentsWithminSmps)
            xroots, agentModels = self.genSamplesForConfigsinParallel(configs['configs']['smp'], num_agents, roots, init_sampling_type, tf_dim, self.tf, self.behavior, rng)
            xroots  = np.hstack((xroots))
            agentModels  = np.hstack((agentModels))

            # print('xroots : ', xroots)
            for i in xroots:
                print_tree(i, MAIN)
                for id, l in enumerate(i.find_leaves()):
                    try:
                        assert l.checkFootprint() == True
                    except AssertionError:
                        print(l.__dict__)
            #         # l.setRoutine(MAIN)
            #         if l.getStatus(MAIN) == 1:
            #             a = l.agent
            #             print(f'agent xtr config sample', i, a.x_train, a.y_train, a.id, a.region_support.input_space)
            #             # print('agent dict : ', a.__dict__)
            #             # print('rregion dict : ', l.__dict__)
            # exit(1)
                    
            # Xs_roots = self.evalConfigsinParallel(roots) #, sample, Xs_root, agents, num_agents, globalXtrain, globalYtrain, region_support, model, rng)
            print('xroots and agents b4 joblib : ', len(xroots), len(agentModels))
            Xs_roots = self.evalConfigs(xroots, sample, agentModels, num_agents, globalXtrain, globalYtrain, region_support, model, rng)
            print("Xs_root from joblib ",len(Xs_roots))
            # save_node(Xs_roots, f'/Users/shyamsundar/ASU/sem2/RA/partmahpc/partma/results/'+configs['testfunc']+f'/Xs_roots_{sample}.pkl')
            # exit(1)
            # if sample==1:
            #     exit(1)
            # print('Xs_roots: ', Xs_roots)

            for x in Xs_roots:
                print('$'*100)
                print_tree(x[1], MAIN)
                print('$'*100)
                agents = [] #x[2]
                for id, l in enumerate(x[1].find_leaves()):
                        if l.getStatus(MAIN) == 1:
                            agents.append(l.agent)
                # exit(1)
                # for smp in range(samples):
                    # self.smp = smp
                avgAgentrewards = np.zeros((1,num_agents))
                # self.root = self._evaluate_at_point_list(agents)
                agents = sorted(agents, key=lambda x: x.id)
                for a in agents:
                    # print('aid inside reward acc: ', a.id)
                    # avgAgentrewards.append(a.region_support.avgRewardDist)
                    avgAgentrewards = np.vstack((avgAgentrewards, a.region_support.avgRewardDist.reshape((1,num_agents))))
                    # print(avgAgentrewards.shape)
                    # a.appendevalReward(avgAgentrewards[a.id][a.id])
                    # print('a.pointsToeval[:smp]: ',a.pointsToeval[:smp+1], a.pointsToeval[:smp+1].shape )
                    # a.x_train = np.vstack((a.x_train, a.pointsToeval[:smp+1]))
                    # a.y_train = np.hstack((a.y_train, np.array(a.evalRewards[:smp+1])))
                    # a.updateModel()
                    # a.resetModel()
                    # print('aid xtrain ,ytrain: ',a.id, a.x_train, a.y_train)
                    # print('a.pointsToeval[:smp+1]: ',a.id , a.pointsToeval[:smp+1], a.evalRewards[:smp+1])

                # print(avgAgentrewards, avgAgentrewards.shape)
                avgrewards = np.vstack((avgrewards, avgAgentrewards[1:].reshape((1,num_agents,num_agents))))
                # agentModels.append(agents)
                # xroots.append(Xs_root)

            # X_root = deepcopy(Xs_root)
                
            avgrewards = avgrewards[1:]
            # print('avgrewards: ',avgrewards)
            # avgrewards = np.hstack((avgrewards))
            # minrewardDist, minrewardDistIdx = min_diagonal(avgrewards)
            mincumRewardIdx = find_min_diagonal_sum_matrix(avgrewards)
            print('mincumRewardIdx: ', mincumRewardIdx)
            # exit(1)
            # print('minrewardDist, minrewardDistIdx: ',minrewardDist, minrewardDistIdx, minrewardDist.shape, minrewardDistIdx.shape)
            # minregtojump = np.argmin(minrewardDist[:len(agents)], axis=0)
            # print('minregtojump: ',minregtojump)
            # exit(1)
            # print('b4 reassign MAIN [i.region_support for i in agents]: ',[i.region_support.input_space for i in agents])
            # agents = reassign(root, MAIN, agents, currentAgentIdx, gpr_model, xtr, ytr)
            # mainAgents = sorted(mainAgents, key=lambda x: x.id)

            agentsWithminSmps = Xs_roots[mincumRewardIdx][2] #agentModels[mincumRewardIdx] #self.get_nextXY(agentModels, minrewardDistIdx)
            # print('agentsWithminSmps: ',[(i.id, i.region_support.input_space )for i in agentsWithminSmps], len(agentsWithminSmps))
            agentsWithminSmps = sorted(agentsWithminSmps, key=lambda x: x.id)

            # partition the actual obs 
            # agentsWithminSmps = splitObs(agentsWithminSmps, tf_dim, rng, ACTUAL, self.tf, self.behavior)
            #get the corresponding Xroot
            X_root = Xs_roots[mincumRewardIdx][1] #xroots[mincumRewardIdx]

            # for a in agentsWithminSmps:
            #     minx = a.region_support.smpXtr[np.argmin(a.region_support.smpYtr),:]
            #     miny = np.min(a.region_support.smpYtr)
            #     miny , _ = compute_robustness(np.array([minx]), test_function, behavior, agent_sample=True)
            #     # print(a.x_train, a.y_train)
            #     # print('a.x_train.all() == a.ActualXtrain.all() ',a.x_train, a.ActualXtrain)
            #     # a.resetActual()

            #     a.x_train = np.vstack((a.x_train, minx))
            #     a.y_train = np.hstack((a.y_train, miny))

            #     a.updateModel()

            minytrval = float('inf')
            minytr = []
            for ix, a in enumerate(agentsWithminSmps):
                model = a.model
                for ia in agents:
                    if min(ia.y_train) < minytrval:
                        minytrval = min(ia.y_train)
                        minytr = ia.y_train
            ytr = minytr
            # print('?'*100)  
            # print_tree(X_root, MAIN)
            # print_tree(X_root, ROLLOUT)
            # print('?'*100)
            x_opt_from_all = []
            for i,a in enumerate(agentsWithminSmps):
                # minx = a.region_support.smpXtr[np.argmin(a.region_support.smpYtr),:]
                # miny = np.min(a.region_support.smpYtr)
                # miny , _ = compute_robustness(np.array([minx]), test_function, behavior, agent_sample=True)
                # # print(a.x_train, a.y_train)
                # # print('a.x_train.all() == a.ActualXtrain.all() ',a.x_train, a.ActualXtrain)
                # # a.resetActual()

                # a.x_train = np.vstack((a.x_train, minx))
                # a.y_train = np.hstack((a.y_train, miny))

                # a.updateModel()
                # assert a.x_train.all() == a.ActualXtrain.all()
                # print('after rest actual : ', a.id, a.x_train, a.y_train)
                #get the new set of points by EI
                assert check_points(a, MAIN) == True
                x_opt = self.ei_roll._opt_acquisition(ytr, a.model, a.region_support.input_space, rng)
                yofEI, _ = compute_robustness(np.array([x_opt]), test_function, behavior, agent_sample=True)

                # print('end of rollout avg smps check ', a.region_support.input_space ,a.region_support.smpXtr, a.region_support.smpYtr)
                # print('end of rollout smps check ', a.region_support.avgsmpXtr, a.region_support.avgsmpYtr)
                # smpxtr = a.region_support.smpXtr[np.argmin(a.region_support.smpYtr),:] 
                k = 10
                idx = np.argpartition(a.region_support.smpYtr, k)[:k]  # Indices not sorted

                minK = idx[np.argsort(a.region_support.smpYtr[idx])]  # Indices sorted by value from smallest to largest

                smpxtr = a.region_support.smpXtr[minK] 

                yofs = np.array([smpxtr[0]])
                yofsmpxtr, _ = compute_robustness(yofs, test_function, behavior, agent_sample=True) #np.array([smpxtr])
                
                print('yofEI, miny from minK: ',x_opt, yofEI, smpxtr, yofsmpxtr, yofs)
                if sample >= 2 : #yofEI > np.min(yofsmpxtr):
                    x_opt = smpxtr[np.argmin(yofsmpxtr),:] 
                    # x_opt = smpxtr
                else:
                    agdic[i] +=  1
                x_opt_from_all.append(x_opt)
            # exit(1)
            subx = np.hstack((x_opt_from_all)).reshape((num_agents,tf_dim))
            pred_sample_x = subx[:num_agents]

            
            pred_sample_y, falsified = compute_robustness(pred_sample_x, test_function, behavior, agent_sample=False)
            # if pred_sample_y > agentminytr:


            print('pred_sample_x, pred_sample_y: ', pred_sample_x, pred_sample_y)
            globalXtrain = np.vstack((globalXtrain, pred_sample_x))
            globalYtrain = np.hstack((globalYtrain, (pred_sample_y)))
            print('min obs so far : ', pred_sample_x[np.argmin(pred_sample_y),:], np.min(pred_sample_y))

            for i,a in enumerate(agentsWithminSmps):
                # a.resetActual()
                # a.ActualXtrain = np.vstack((a.ActualXtrain, np.asarray([pred_sample_x[i]])))
                # a.ActualYtrain = np.hstack((a.ActualYtrain, pred_sample_y[i]))
                a.x_train = np.vstack((a.x_train, np.asarray([pred_sample_x[i]])))
                a.y_train = np.hstack((a.y_train, pred_sample_y[i]))
                a.updateModel()
                # a.resetActual()
                # add actual footprint 
                # actPrior = Prior(a.ActualXtrain, a.ActualXtrain, a.model, MAIN)
                a.region_support.addFootprint(a.x_train, a.y_train, a.model)
                # a.region_support.addFootprint(a.ActualXtrain, a.ActualYtrain, a.model)
            
            print_tree(X_root, MAIN)
            # exit(1)
        #     inactive_subregion_samples = []   
        #     self.agent_point_hist.extend(pred_sample_x)
        # #     # writetocsv(f'results/reghist/aSim{i}_{sample}',a.simregHist)

        # # plot_dict = {'agents':self.agent_point_hist,'assignments' : self.assignments, 'status': status, 'region_support':region_support, 'test_function' : test_function, 'inactive_subregion_samples' : inactive_subregion_samples, 'sample': num_samples}
        # save_plot_dict = {'agentmodels':agentAftereachSample, 'agents':self.agent_point_hist,'assignments' : self.assignmentsDict, 'status': status, 'region_support':region_support, 'test_function' : [], 'inactive_subregion_samples' : inactive_subregion_samples, 'sample': num_samples}
        
        # # print(plot_dict)
        # save_node(save_plot_dict, '/Users/shyamsundar/ASU/sem2/RA/partmahpc/partma/results/'+configs['testfunc']+f'/plot_dict.pkl')
        print()
        print('times when EI pt was chosen',agdic)
        print()
        return falsified, self.region_support , None #plot_dict



    def getRootConfigs(self, X_root, rootModel, sample, num_agents, tf_dim, agentsWithminSmps):
        # Generate permutations of the numbers 0 to 3
        permutations_list = list(permutations(range(num_agents)))

        roots = []
        for dim in range(tf_dim):
            if sample == 0: 
                print(dim)
                root = deepcopy(X_root)
                factorized = find_prime_factors(num_agents) #sorted(find_close_factor_pairs(num_agents), reverse=True)
                print(factorized)
                agents_to_subregion = get_subregion(deepcopy(root), num_agents, factorized, dim)
                root.add_child(agents_to_subregion)
                for id, l in enumerate(root.find_leaves()):
                    l.setRoutine(MAIN)
                    if l.getStatus(MAIN) == 1:
                        mainag = Agent(id, deepcopy(root.model), self.x_train, self.y_train, l)
                        mainag(MAIN)
                roots.append(root)
                
            else:
                jump = random.random()
                root = deepcopy(X_root)
                agents =[]
                for id, l in enumerate(root.find_leaves()):
                    if l.getStatus(MAIN) == 1:
                        agents.append(l.agent)
                    # print(f'agent x_train b4 partitioning  ', l.agent.x_train, l.agent.y_train, l.agent.id, l.input_space)
                subregions = reassignUsingRewardDist( root, MAIN, agents, jump_prob=jump)
                agents = partitionRegions(root, subregions, MAIN, dim)
                print('after moving and partitioning ')
                print_tree(root, MAIN)
                for a in agents:
                    a.resetRegions()
                    if a.region_support.getStatus(MAIN) == 0:
                        a.region_support.agentList = []
                    a.region_support.resetavgRewardDist(num_agents)
                    a.resetModel()
                    # a.region_support.addFootprint(a.ActualXtrain, a.ActualYtrain, a.model)
                    # assert a.region_support.checkFootprint() == True
                
                
                roots.append(root)
            
        # testv=0
        # for i in agents_to_subregion:
        #     testv += i.getVolume()

        # assert X_root.getVolume() == testv
        print('roots after dim: ', roots)

        # permutations_list = permutations_list[:2]
        # agents = []
        # moreRoots = []
        # for rt in roots:
        #     print_tree(rt, MAIN)
        #     for i in range(len(permutations_list)):
        #         copyrts = deepcopy(rt)
        #         moreRoots.append(copyrts)

        # print('roots b4 perm: ', moreRoots)

        # moreRoots = []
        # agents = []
        # nid = 0
        # for rt in roots:
        #     for perm in range(len(permutations_list)):
        #         for id, l in enumerate(rt.find_leaves()):
        #             l.setRoutine(MAIN)
        #             nid = nid % num_agents
        #             if l.getStatus(MAIN) == 1:
        #                 if sample == 0:
        #                     # print(idx, id,i, len(permutations_list))
        #                     mainag = Agent(permutations_list[perm][nid], rootModel, self.x_train, self.y_train, l)
        #                     mainag(MAIN)
        #                     l.addAgentList(mainag, MAIN)
        #                 else:
        #                     l.agent.id = permutations_list[perm][nid]
        #                     mainag = l.agent
        #                 nid += 1
        #                 agents.append(mainag)
        #         moreRoots.append(deepcopy(rt))

        # for idx, mrt in enumerate(moreRoots):        
        #     # for perm in permutations_list:
        #     # print_tree(mrt, MAIN)
        #     idx = idx % len(permutations_list)
        #     i = 0
        #     for id, l in enumerate(mrt.find_leaves()):
        #         l.setRoutine(MAIN)
        #         if l.getStatus(MAIN) == 1:
        #             if sample == 0:
        #                 # print(idx, id,i, len(permutations_list))
        #                 mainag = Agent(permutations_list[idx][i], rootModel, self.x_train, self.y_train, l)
        #                 mainag(MAIN)
        #                 l.addAgentList(mainag, MAIN)
        #             else:
        #                 l.agent.id = permutations_list[idx][i]
        #                 mainag = l.agent
        #             agents.append(mainag)
        #             i += 1

        # print('roots after perm: ', roots)
        # for i in moreRoots:
        #     print_tree(i, MAIN) 

        # exit(1)

        return roots
    
    # @profile
    def genSamplesForConfigsinParallel(self, configSamples, num_agents, roots, init_sampling_type, tf_dim, tf, behavior, rng):
        self.ei_roll = RolloutEI()

        # Define a helper function to be executed in parallel
        def genSamples_in_parallel(num_agents, roots, init_sampling_type, tf_dim, tf, behavior, rng):
            return genSamplesForConfigs(self.ei_roll, num_agents, roots, init_sampling_type, tf_dim, tf, behavior, rng)

        # Execute the evaluation function in parallel for each Xs_root item
        results = Parallel(n_jobs=8)(delayed(genSamples_in_parallel)(num_agents, roots, init_sampling_type, tf_dim, tf, behavior, np.random.default_rng(csmp+1)) for csmp in tqdm(range(configSamples)))
        
        roots = [results[i][0] for i in range(configSamples)]
        agents = [results[i][1] for i in range(configSamples)]

        return roots , agents
    
    # @profile
    # def evalConfigs(self, Xs_root, sample, agents, num_agents, globalXtrain, globalYtrain, region_support, model, rng):
    #     self.ei_roll = RolloutEI()
    #     # cluster = LocalCluster()
    #     # # connect client to your cluster
    #     # client = Client(cluster)

    #     # # Monitor your computation with the Dask dashboard
    #     # print(client.dashboard_link)
    #     # print('inside evalConfigs')
    #     # Define a helper function to be executed in parallel
    #     def evaluate_in_parallel(Xs_root_item, sample, num_agents, globalXtrain, globalYtrain, region_support, model, rng):
    #         # print('Xs_root_item in eval config : ',Xs_root_item)
    #         agents = []
    #         return self.ei_roll.sample(sample, Xs_root_item, agents, num_agents, self.tf, globalXtrain, self.horizon, globalYtrain, region_support, model, rng)

    #     # def daskcompute():
    #         # Execute the evaluation function in parallel for each Xs_root item
    #     results = Parallel(n_jobs=-1)(delayed(evaluate_in_parallel)(Xs_root_item, sample, num_agents, globalXtrain, globalYtrain, region_support, model, rng) for (Xs_root_item) in tqdm(Xs_root))

    #     # with joblib.parallel_backend('dask'):
    #     #     results = daskcompute()

    #     return results
    
    @cprofile
    def evalConfigs(self, Xs_root, sample, agents, num_agents, globalXtrain, globalYtrain, region_support, model, rng):
        ei_roll = RolloutEI()
        # srt = pickle.dump(Xs_root)
        
        # def internal_function(X, from_agent = None): #Branin with unique glob min -  9.42, 2.475 local min (3.14, 12.27) and (3.14, 2.275)
        #     x1 = X[0]
        #     x2 = X[1]
        #     t = 1 / (8 * np.pi)
        #     s = 10
        #     r = 6
        #     c = 5 / np.pi
        #     b = 5.1 / (4 * np.pi ** 2)
        #     a = 1
        #     term1 = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2
        #     term2 = s * (1 - t) * np.cos(x1)
        #     l1 = 5 * np.exp(-5 * ((x1+3.14)**2 + (x2-12.27)**2))
        #     l2 = 5 * np.exp(-5 * ((x1+3.14)**2 + (x2-2.275)**2))
        #     return term1 + term2 + s + l1 + l2
        
        # tf = Fn(internal_function)
        
        # print('tf : ', tf)
        horizon = self.horizon
        # print('inside evalConfigs')
        # Define a helper function to be executed in parallel
        def evaluate_in_parallel(Xs_root_item): #, sample, num_agents, globalXtrain, globalYtrain, region_support, model, rng):
            ei_roll = RolloutEI()
            # Xs_root_item = pickle.load(Xs_root_item)
            # print('Xs_root_item in eval config : ',Xs_root_item)
            agents = []
            return ei_roll.sample(sample, Xs_root_item, agents, num_agents, globalXtrain, self.horizon, globalYtrain, region_support, model, rng)

        # Execute the evaluation function in parallel for each Xs_root item
        results = Parallel(n_jobs=-1)(delayed(evaluate_in_parallel)(Xs_root_item) for (Xs_root_item) in tqdm(Xs_root))

        # results = Parallel(n_jobs= 2, backend="loky")\
        #     (delayed(unwrap_self)(i) for i in zip([self]*2, sample, Xs_root, agents, num_agents, tf, globalXtrain, horizon, globalYtrain, region_support, model, rng))
    
        # with Pool(processes=4) as pool:  
        #     args = [(ei_roll,Xs_root[i], sample, num_agents, globalXtrain, globalYtrain, region_support, model,horizon, rng) for i in range(len(Xs_root))]
        #     # issue multiple tasks each with multiple arguments
        #     results = pool.map(evalWrapper, args)

        # with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        #     # Submit tasks for each tree node
        #     futures = [executor.submit(evaluate_in_parallel, node) for node in tqdm(Xs_root)]
            
        #     # Gather results
        #     results = [future.result() for future in concurrent.futures.as_completed(futures)]
        # arg = (Xs_root, sample, num_agents, globalXtrain, self.horizon, globalYtrain, region_support, model, rng)
        # with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        #     results = list(executor.map(evaluate_in_parallel, [ei_roll, arg]))

        return results
    
    # @profile
    # def evalConfigs(self, Xs_root, sample, agents, num_agents, globalXtrain, globalYtrain, region_support, model, rng):
    #     self.ei_roll = RolloutEI()

    #     # Set up a local cluster
    #     # dask.config.set({'distributed.worker.daemon': False})
    #     cluster = LocalCluster(processes=50)
    #     client = Client(cluster)

    #     # Retrieve logs from all workers
    #     worker_logs = client.get_worker_logs()
    
    #     # Define a helper function to be executed in parallel
    #     def evaluate_in_parallel(Xs_root_item, sample, num_agents, globalXtrain, globalYtrain, region_support, model, rng):
    #         agents = []  # Do you need to redefine agents here?
    #         return self.ei_roll.sample(sample, Xs_root_item, agents, num_agents, self.tf, globalXtrain, self.horizon, globalYtrain, region_support, model, rng)

    #     # for worker, logs in worker_logs.items():
    #     #     print(f"Logs from worker {worker}:")
    #     #     for log in logs:
    #     #         print(log)
                
    #     # Execute the evaluation function in parallel for each Xs_root item
    #     results = []
    #     for Xs_root_item in tqdm(Xs_root):
    #         result = client.submit(evaluate_in_parallel, Xs_root_item, sample, num_agents, globalXtrain, globalYtrain, region_support, model, rng)
    #         results.append(result)
    
    #     # Gather results
    #     results = client.gather(results)
    
    #     # Close the client and cluster
    #     client.close()
    #     cluster.close()
    
    #     return results

    def getWorkers(self):
        cluster = LocalCluster(processes=1000)
        cluster.scale(5) 
        client = Client(cluster)
        
        nb_workers = 0
        while True:
            nb_workers = len(client.scheduler_info()["workers"])
            print('Got {} workers'.format(nb_workers))
            if nb_workers >= 5:
                break
            time.sleep(1)
        
        return client

    # def evalConfigs(self, client, Xs_root, sample, agents, num_agents, globalXtrain, globalYtrain, region_support, model, rng):
    #     self.ei_roll = RolloutEI()
        
    #     # Configure SLURMCluster

    #     # cluster = SLURMCluster(cores=32, memory='10GB', processes=30, queue='htc', walltime='01:30:00') # 16 cores 5GB 12 processes 45 min 15 workers | 25 workers , cores=32, memory='10GB', processes=30, queue='htc', walltime='00:45:00'
        
    #     # # Scale the cluster to desired number of workers
    #     # cluster.scale(25)  # Scale to 20 workers (adjust as needed)
    #     # # cluster.adapt(maximum_jobs=100)
    #     # # Connect a client to the cluster
    #     # client = Client(cluster)
        
        

    
    #     # Define a helper function to be executed in parallel
    #     def evaluate_in_parallel(Xs_root_item): #, sample, num_agents, globalXtrain, globalYtrain, region_support, model, rng):
    #         agents = []  # Do you need to redefine agents here?
    #         return self.ei_roll.sample(sample, Xs_root_item, agents, num_agents, self.tf, globalXtrain, self.horizon, globalYtrain, region_support, model, rng)
    
    #     # Use Dask as the backend for joblib
    #     # with parallel_backend('dask', scheduler_host=client.scheduler.address):
    #     #     dask.config.set({'distributed.worker.daemon': False})
    #     #     parallel_config(backend='dask', wait_for_workers_timeout=500)
            
    #     #     results = Parallel(n_jobs=-1, verbose=100)(delayed(evaluate_in_parallel)(Xs_root_item, sample, num_agents, globalXtrain, globalYtrain, region_support, model, rng) for (Xs_root_item) in tqdm(Xs_root))

    #     # print('Xs_roots:',Xs_roots, clients)
    #     result = [dask.delayed(evaluate_in_parallel)(Xs_root_item) for Xs_root_item in tqdm(Xs_root)]
    #     futures = client.compute(result)
    #     print('futures: ', futures)
    #     results = client.gather(futures)
        
    #     print(results)

    #     # client.close()
    #     # cluster.close()
        
    #     return results

    def get_nextXY(self, agentmodels, minRewardIdx): #, rng, test_function, behavior):
        agents = []
        for i, minidx in enumerate(minRewardIdx):
            agents.append(agentmodels[minidx][i])

        return agents

        # for a in agents:
        #     x_opt = self.ei_roll._opt_acquisition(a.y_train, a.model, a.region_support.input_space, rng)
        #     pred_sample_y, falsified = compute_robustness(x_opt, test_function, behavior, agent_sample=False)
        

    
