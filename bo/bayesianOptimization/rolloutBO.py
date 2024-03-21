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
# from dask_jobqueue import SLURMCluster

from .routine import Routine
from .mainRoutine import MainRoutine
from .rolloutRoutine import RolloutRoutine



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
    return RolloutBO.evalConfigs(*arg, **kwarg)

class RolloutBO(BO_Interface):
    def __init__(self):
        pass
    
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
        self.region_support = region_support
        self.behavior = behavior
        self.x_train = x_train
        self.y_train = y_train
        self.numthreads = int(mp.cpu_count()/2)
        self.mc_iters = configs['sampling']['mc_iters']

        tf_dim = region_support.shape[0]
        X_root = Node(self.region_support, RegionState.ACTIVE)

        init_sampling_type = "lhs_sampling"
        globalObs = Observations(x_train, y_train)
        

        agentsWithminSmps = 0
        agdic = {0:0,1:0,2:0,3:0}
        # client = self.getWorkers()
        for sample in tqdm(range(num_samples)):
            print('globalXtrain, globalYtrain :', min(globalObs.y_train))
            # for sample in tqdm(range(num_samples)):
            print('_____________________________________', sample)
            # print(f"INPUT SPACE : {GREEN}{self.region_support}{END}")
            print('_____________________________________')
            print('global dataset : ', globalObs.x_train.shape, globalObs.y_train.shape)
            print('_____________________________________')
            # model = GPR(gpr_model)
            # model.fit(globalObs.x_train, globalObs.y_train)
            globalGP = Prior(globalObs, region_support)
            print(globalGP)
            # X_root.gp = globalGP
            model , indices = globalGP.buildModel()
            X_root.updateModel(indices, model)
            
            avgrewards = np.zeros((1,num_agents,num_agents))
            # agentModels = []
            # xroots = []
            
            roots = self.getRootConfigs(X_root, globalGP, sample, num_agents, tf_dim, agentsWithminSmps)
            xroots, agentModels, globalGP = self.genSamplesForConfigsinParallel(globalGP, configs['configs']['smp'], num_agents, roots, init_sampling_type, tf_dim, self.tf, self.behavior, rng)
            xroots  = np.hstack((xroots))
            agentModels  = np.hstack((agentModels))

            # print('xroots : ', xroots)
            for i in xroots:
                print_tree(i)
                print(globalGP)
                for id, l in enumerate(i.find_leaves()):
                    localGP = Prior(globalGP.dataset, l.input_space)
                    # print(l.__dict__)
                    try:
                        assert localGP.checkPoints(globalGP.dataset.x_train[l.obsIndices]) == True
                    except AssertionError:
                        print(l.__dict__)
                        exit(1)
            
            rollout  = RolloutRoutine()
            Xs_roots = rollout.run(xroots, globalGP, num_agents, region_support, rng)

            main = MainRoutine()
            X_root = main.run(Xs_roots, globalGP, num_agents, tf_dim, test_function, behavior, rng, agdic)
            
            
            # print_tree(X_root, RegionState.ACTIVE)

        print()
        print('times when EI pt was chosen',agdic)
        print()
        return falsified, self.region_support , None #plot_dict



    def getRootConfigs(self, X_root, globalGP, sample, num_agents, tf_dim, agentsWithminSmps):
        # Generate permutations of the numbers 0 to 3
        permutations_list = list(permutations(range(num_agents)))

        roots = []
        for dim in range(tf_dim):
            if sample == 0: 
                print(dim)
                root = deepcopy(X_root)
                factorized = find_prime_factors(num_agents) #sorted(find_close_factor_pairs(num_agents), reverse=True)
                print('num_agents, factorized: ',num_agents, factorized)
                agents_to_subregion = get_subregion(deepcopy(root), num_agents, factorized, dim)
                root.add_child(agents_to_subregion)
                for id, l in enumerate(root.find_leaves()):
                    # print('indside getRootConfigs', 'status:', root.__dict__)
                    if l.getStatus() == RegionState.ACTIVE.value:
                        
                        localGP = Prior(globalGP.dataset, l.input_space)
                        # l.samples = root.samples
                        
                        l.model , l.obsIndices = localGP.buildModel()

                        
                roots.append(root)
                
            else:
                jump = random.random()
                root = deepcopy(X_root)
                agents =[]
                for id, l in enumerate(root.find_leaves()):
                    if l.getStatus() == RegionState.ACTIVE.value:
                        l.rewardDist = l.avgRewardDist.reshape((num_agents))
                        agents.append(l)

                    print(f'agent x_train b4 partitioning  ', globalGP.dataset.x_train[l.obsIndices], globalGP.dataset.y_train[l.obsIndices], l.agentIdx, l.input_space)
                subregions = reassignUsingRewardDist( root, RegionState, agents, jump_prob=jump)
                root = partitionRegions(root, globalGP, subregions, RegionState, dim)
                print('after moving and partitioning ')
                # print_tree(root, RegionState.ACTIVE)
                for a in agents:
                    # a.resetRegions()
                    if a.getStatus() == 0:
                        a.agentList = []
                    a.resetavgRewardDist(num_agents)
                    # a.resetModel()
                    # a.region_support.addFootprint(a.ActualXtrain, a.ActualYtrain, a.model)
                    # assert a.region_support.checkFootprint() == True
                
                
                roots.append(root)
            
        # testv=0
        # for i in agents_to_subregion:
        #     testv += i.getVolume()

        # assert X_root.getVolume() == testv
        print('roots after dim: ', roots)
        for i in roots:
            print_tree(i)

        return roots
    
    # @profile
    def genSamplesForConfigsinParallel(self, globalGP, configSamples, num_agents, roots, init_sampling_type, tf_dim, tf, behavior, rng):
        self.ei_roll = RolloutEI()

        # Define a helper function to be executed in parallel
        def genSamples_in_parallel(num_agents, roots, init_sampling_type, tf_dim, tf, behavior, rng):
            return genSamplesForConfigs(self.ei_roll, globalGP, num_agents, roots, init_sampling_type, tf_dim, tf, behavior, rng)

        # Execute the evaluation function in parallel for each Xs_root item
        results = Parallel(n_jobs=-1)(delayed(genSamples_in_parallel)(num_agents, roots, init_sampling_type, tf_dim, tf, behavior, np.random.default_rng(csmp+1)) for csmp in tqdm(range(configSamples)))
        
        roots = [results[i][0] for i in range(configSamples)]
        agents = [results[i][1] for i in range(configSamples)]
        globalGP = results[0][2]

        return roots , agents, globalGP
    