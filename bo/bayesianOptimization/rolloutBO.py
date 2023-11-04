from typing import Callable, Tuple
from copy import deepcopy
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import norm
from tqdm import tqdm
from ..utils.visualize import contour
import plotly.graph_objects as go


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
from ..utils.logger import logtime, LOGPATH

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
        self.behavior = behavior

        tf_dim = region_support.shape[0]
        X_root = Node(self.region_support, 1)
        factorized = find_prime_factors(num_agents) #sorted(find_close_factor_pairs(num_agents), reverse=True)
        agents_to_subregion = get_subregion(deepcopy(X_root), num_agents, factorized, np.random.randint(tf_dim))
        X_root.add_child(agents_to_subregion)
        # assignments = {value: 1 for value in agents_to_subregion} 
        testv=0
        for i in agents_to_subregion:
            testv += i.getVolume()

        assert X_root.getVolume() == testv
        print('set([i.getVolume() for i in agents_to_subregion]) :', set([i.getVolume() for i in agents_to_subregion]))
        
        agents = []
        globalXtrain = np.empty((1,tf_dim))
        globalYtrain = np.empty((1))

        init_sampling_type = "uniform_sampling"
        init_budget = 8

        for l in agents_to_subregion:
            l.setRoutine(MAIN)
            if l.getStatus(MAIN) == 1:
                xtr, ytr = self.initAgents(l.input_space, init_sampling_type, int(init_budget/num_agents), tf_dim, rng)
                # print(f'agent xtr ', xtr, ytr)
                globalXtrain = np.vstack((globalXtrain, xtr))
                globalYtrain = np.hstack((globalYtrain, ytr))
                ag = Agent(None, xtr, ytr, l)
                ag.updateModel()
                ag(MAIN)
                agents.append(ag)
        
        globalXtrain = globalXtrain[1:]
        globalYtrain = globalYtrain[1:]
        # print('globalXtrain, globalYtrain :', globalXtrain, globalYtrain)
        for sample in tqdm(range(num_samples)):
            print('_____________________________________', sample)
            # print(f"INPUT SPACE : {GREEN}{self.region_support}{END}")
            print('_____________________________________')
            print('global dataset : ', x_train.shape, y_train.shape)
            print('_____________________________________')
            model = GPR(gpr_model)
            model.fit(globalXtrain, globalYtrain)
            self.ei_roll = RolloutEI()
                
            pred_sample_x, X_root, agents = self.ei_roll.sample(X_root, agents, num_agents, self.tf, x_train, self.horizon, y_train, region_support, model, rng) #self._opt_acquisition(agent.y_train, agent.model, agent.region_support, rng) 
            pred_sample_y, falsified = compute_robustness(pred_sample_x, test_function, behavior, agent_sample=True)
            
            x_train = np.vstack((x_train, pred_sample_x))
            y_train = np.hstack((y_train, (pred_sample_y)))
            print('np.asarray([pred_sample_x[i]]).shape : ', np.asarray([pred_sample_x[0]]).shape)
            for i,a in enumerate(agents):
                print(f'b4 appendign agent {i} xtrain :', a.x_train)
                a.x_train = np.vstack((a.x_train, np.asarray([pred_sample_x[i]])))
                print(f'agent {i} xtrain :', a.x_train)
                a.y_train = np.hstack((a.y_train, pred_sample_y[i]))
                a.updateModel()
        plot_dict = {} #{'agents':self.agent_point_hist,'assignments' : self.assignments, 'region_support':region_support, 'test_function' : test_function, 'inactive_subregion_samples' : self.inactive_subregion_samples, 'sample': num_samples}
            # X_root.add_child(self.region_support)
        return falsified, self.region_support, plot_dict


    def initAgents(self, region_support, init_sampling_type, init_budget, tf_dim, rng):
        if init_sampling_type == "lhs_sampling":
            x_train = lhs_sampling(init_budget, region_support, tf_dim, rng)
        elif init_sampling_type == "uniform_sampling":
            x_train = uniform_sampling(init_budget, region_support, tf_dim, rng)
        else:
            raise ValueError(f"{init_sampling_type} not defined. Currently only Latin Hypercube Sampling and Uniform Sampling is supported.")
        
        y_train, falsified = compute_robustness(x_train, self.tf, self.behavior)
        if not falsified:
            print("No falsification in Initial Samples. Performing BO now")

        return x_train, y_train