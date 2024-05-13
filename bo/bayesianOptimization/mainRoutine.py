

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
# from ..agent.agent import Agent
from ..agent.constants import *
from ..agent.localgp import Prior
from ..agent.observation import Observations
from ..utils.logger import logtime, LOGPATH
from ..utils.savestuff import *
import random
import os
from itertools import permutations
import multiprocessing as mp
from multiprocessing import Pool
from joblib import Parallel, delayed
# from ..utils.plotlyExport import exportTreeUsingPlotly
from memory_profiler import profile

from dask.distributed import Client, LocalCluster

from joblib import parallel_backend, parallel_config
import joblib
import dask, time, json

from .rolloutRoutine import RolloutRoutine

# Responsible for sampling locations one agent at a time 
class MainRoutine:
    def __init__(self) -> None:
         pass
    
    def sample(self, xroots, globalGP, num_agents, tf_dim, test_function, behavior, rng):
        ei_roll = RolloutEI()
        for m in range(num_agents):
            print(f'MAIN minz {m}')

            # Rollout the configurations and get the config with min f_ro
            minXroot = self.run(m, xroots, globalGP, num_agents, tf_dim, test_function, behavior, rng)

            if m == num_agents-1:
                break
            
            # Get the next location from agent m and its true function value
            roots = getRootConfigs(m, minXroot, globalGP, 1, num_agents, tf_dim)
            for i in roots:
                for id, l in enumerate(i.find_leaves()):    
                    if l.getStatus() == RegionState.ACTIVE.value and l.agentId == m:
                        minidx = globalGP.dataset._getMinIdx(l.obsIndices)
                        fmin = globalGP.dataset.y_train[minidx]

                        x_opt = ei_roll._opt_acquisition(fmin, l.model, l.input_space, rng)
                        yofEI, _ = compute_robustness(np.array([x_opt]), test_function, behavior, agent_sample=True)

                        print('agent actual sample decision : ',fmin, l.agentId, x_opt, yofEI)

                        globalGP.dataset = globalGP.dataset.appendSamples(x_opt, yofEI)
                        
                        localGP = Prior(globalGP.dataset, l.input_space)
                        # X_root.gp = globalGP
                        model , indices = localGP.buildModel()
                        l.updateModel(indices, model)
            
            # Fix the configuration for agents 1 to (i-1) using rollout policy and get configs for agents i to m
            xroots, agentModels, tmpGP = genSamplesForConfigsinParallel(m, globalGP, configs['configs']['smp'], num_agents, roots, "lhs_sampling", tf_dim, test_function, behavior, rng)
            xroots  = np.hstack((xroots))
            agentModels  = np.hstack((agentModels))

            print('xroots : ', xroots)
            for i in xroots:
                print(f'tree after {m} minz')
                print_tree(i)
                # print(globalGP)
                for id, l in enumerate(i.find_leaves()): 
                    print('l.avgRewardDist: ', l.avgRewardDist, l.input_space)
                    localGP = Prior(tmpGP.dataset, l.input_space)
                    # print(l.__dict__)
                    try:
                        assert localGP.checkPoints(tmpGP.dataset.x_train[l.obsIndices]) == True
                    except AssertionError:
                        print(l.__dict__)
                        exit(1)

        return minXroot

    # Responsible for rolling out the configurations and get the config with min f_ro
    def run(self, m, Xs_roots, globalGP, num_agents, tf_dim, test_function, behavior, rng):
        print('res of rollout', Xs_roots)
        
        rollout  = RolloutRoutine()
        Xs_roots, F_nc = rollout.run(m, Xs_roots, globalGP, num_agents, test_function, tf_dim, behavior, rng)
        print('Fnc : ',F_nc)
        minXroot = Xs_roots[np.argmin(F_nc)]

        
                
        
        return minXroot
    
        # for x in Xs_roots:
        #     print('$'*100)
        #     print_tree(x)
        #     print('$'*100)
        #     agents = [] #x[2]
        #     for id, l in enumerate(x.find_leaves()):
        #             if l.status == 1:
        #                 agents.append(l)
        #     # exit(1)
        #     # for smp in range(samples):
        #         # self.smp = smp
        #     avgAgentrewards = np.zeros((1,num_agents))
        #     # self.root = self._evaluate_at_point_list(agents)
        #     agents = sorted(agents, key=lambda x: x.agentId)
        #     for a in agents:
        #         avgAgentrewards = np.vstack((avgAgentrewards, a.rewardDist.reshape((1,num_agents))))

        #     # print(avgAgentrewards, avgAgentrewards.shape)
        #     avgrewards = np.vstack((avgrewards, avgAgentrewards[1:].reshape((1,num_agents,num_agents))))

                
        #     avgrewards = avgrewards[1:]
        #     print('avgrewards: ',avgrewards)
        #     mincumRewardIdx = find_min_diagonal_sum_matrix(avgrewards)

        #     agentsWithminSmps = []#Xs_roots[mincumRewardIdx][2] #agentModels[mincumRewardIdx] #self.get_nextXY(agentModels, minrewardDistIdx)
        #     for id, l in enumerate(Xs_roots[mincumRewardIdx].find_leaves()):
        #             if l.status == 1:
        #                 agentsWithminSmps.append(l)
        #     print('agentsWithminSmps: ',[(i.agentId, i.input_space )for i in agentsWithminSmps], len(agentsWithminSmps))
        #     agentsWithminSmps = sorted(agentsWithminSmps, key=lambda x: x.agentId)

        #     X_root = Xs_roots[mincumRewardIdx]#[1] 

        #     minytrval = float('inf')
        #     minytr = []
        #     # for ix, a in enumerate(agents):
        #     #     # print('indside getsmpConfigs', 'status:', a.__dict__)
        #     #     for ia in agents:
        #     #         minidx = globalGP.dataset._getMinIdx(ia.obsIndices)
        #     #         if min(globalGP.dataset.y_train[minidx]) < minytrval:
        #     #             minytrval = globalGP.dataset.y_train[minidx]
        #     #             minytrval = min(globalGP.dataset.y_train[minidx])
            
        #     ytr = min(globalGP.dataset.y_train) # minytr
        #     print('? most min'*100)  
        #     print_tree(X_root)
        #     # print_tree(X_root, ROLLOUT)
        #     print('?'*100)
        #     x_opt_from_all = []
        #     for i,a in enumerate(agentsWithminSmps):
        #         # minx = a.region_support.smpXtr[np.argmin(a.region_support.smpYtr),:]
        #         # miny = np.min(a.region_support.smpYtr)
        #         # miny , _ = compute_robustness(np.array([minx]), test_function, behavior, agent_sample=True)
        #         # # print(a.x_train, a.y_train)
        #         # # print('a.x_train.all() == a.ActualXtrain.all() ',a.x_train, a.ActualXtrain)
        #         # # a.resetActual()

        #         # a.x_train = np.vstack((a.x_train, minx))
        #         # a.y_train = np.hstack((a.y_train, miny))

        #         # a.updateModel()
        #         # assert a.x_train.all() == a.ActualXtrain.all()
        #         # print('after rest actual : ', a.id, a.x_train, a.y_train)
        #         #get the new set of points by EI
        #         # assert check_points(a, MAIN) == True
        #         x_opt = ei_roll._opt_acquisition(ytr, a.model, a.input_space, rng)
        #         yofEI, _ = compute_robustness(np.array([x_opt]), test_function, behavior, agent_sample=True)

        #         print('end of rollout avg smps check ', a.input_space ,a.samples)
        #         # print('end of rollout smps check ', a.region_support.avgsmpXtr, a.region_support.avgsmpYtr)
        #         # smpxtr = a.samples.x_train[np.argmin(a.smpIndices[a.samples.y_train[a.smpIndices]]),:] 
        #         # yofsmpxtr, _ = compute_robustness(np.array([smpxtr]), test_function, behavior, agent_sample=True) #np.array([smpxtr])
                
        #         # print('yofEI, miny: ',x_opt, yofEI, smpxtr, yofsmpxtr)
        #         # if yofEI > yofsmpxtr:
        #         # # x_opt = smpxtr[np.argmin(yofsmpxtr),:] 
        #         #     x_opt = smpxtr
        #         # else:
        #         #     agdic[i] +=  1
        #         x_opt_from_all.append(x_opt)
        #     # exit(1)
        #     subx = np.hstack((x_opt_from_all)).reshape((num_agents,tf_dim))
        #     pred_sample_x = subx[:num_agents]

            
        #     pred_sample_y, falsified = compute_robustness(pred_sample_x, test_function, behavior, agent_sample=False)
        #     # if pred_sample_y > agentminytr:


        #     print('pred_sample_x, pred_sample_y: ', pred_sample_x, pred_sample_y)
        #     globalGP.dataset = globalGP.dataset.appendSamples(pred_sample_x, pred_sample_y)
        #     # globalXtrain = np.vstack((globalXtrain, pred_sample_x))
        #     # globalYtrain = np.hstack((globalYtrain, (pred_sample_y)))
        #     print('min obs so far : ', pred_sample_x[np.argmin(pred_sample_y),:], np.min(pred_sample_y))

        #     for i,a in enumerate(agentsWithminSmps):
        #         localGP = Prior(globalGP.dataset, a.input_space)
        #         # X_root.gp = globalGP
        #         model , indices = localGP.buildModel()
        #         a.updateModel(indices, model)
        

        # return X_root, agdic


