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
# from .rolloutRoutine import RolloutRoutine



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
        m = -1
        # Sample points using the multi agent routine 
        for sample in tqdm(range(num_samples)):
            print('globalXtrain, globalYtrain :', min(globalObs.y_train))
            print('_____________________________________', sample)
            print('_____________________________________')
            print('global dataset : ', globalObs.x_train.shape, globalObs.y_train.shape)
            print('_____________________________________')
            # Build the model over the entire region
            globalGP = Prior(globalObs, region_support)
            print(globalGP)
            model , indices = globalGP.buildModel()
            X_root.updateModel(indices, model)
            
            # Get the initial set of configs by partitioning among m agents
            roots = getRootConfigs(m, X_root, globalGP, sample, num_agents, tf_dim)
            xroots, agentModels, globalGP = genSamplesForConfigsinParallel(m, globalGP, configs['configs']['smp'], num_agents, roots, init_sampling_type, tf_dim, self.tf, self.behavior, rng)
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
            
            # Calling the main routine to get the configuration as a result of multi agent rollout
            main = MainRoutine()
            X_root = main.sample(xroots, globalGP, num_agents, tf_dim, test_function, behavior, rng)
            print('roll bo :',X_root)

            ei_roll = RolloutEI()
            agents = []
            for id, l in enumerate(X_root.find_leaves()):    
                if l.getStatus() == RegionState.ACTIVE.value:
                    agents.append(l)

            agents = sorted(agents, key=lambda x: x.agentId)
            # Sample new set of agent locations from the winning configuration
            for l in agents: 
                minidx = globalGP.dataset._getMinIdx(l.obsIndices)
                fmin = globalGP.dataset.y_train[minidx]

                x_opt = ei_roll._opt_acquisition(fmin, l.model, l.input_space, rng)
                yofEI, _ = compute_robustness(np.array([x_opt]), test_function, behavior, agent_sample=False)

                print('%'*100)
                print('pred x, pred y: ', fmin, l.agentId, x_opt, yofEI)
                print('%'*100)
                globalGP.dataset = globalGP.dataset.appendSamples(x_opt, yofEI)
                
                # Update the respective local GPs
                localGP = Prior(globalGP.dataset, l.input_space)
                model , indices = localGP.buildModel()
                l.updateModel(indices, model)

            # exit(1)
            print_tree(X_root) #, RegionState.ACTIVE)
            if sample == 1:
                exit(1)

        print()
        print('times when EI pt was chosen',agdic)
        print()
        return falsified, self.region_support , None #plot_dict



    
    