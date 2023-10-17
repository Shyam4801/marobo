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

        tf_dim = region_support.shape[0]
        X_root = Node(self.region_support, 1)
        factorized = sorted(find_close_factor_pairs(num_agents), reverse=True)
        agents_to_subregion = get_subregion(deepcopy(X_root), num_agents, factorized, dim=0)
        X_root.add_child(agents_to_subregion)
        # assignments = {value: 1 for value in agents_to_subregion} 

        
        for sample in tqdm(range(num_samples)):
            print('_____________________________________', sample)
            # print(f"INPUT SPACE : {GREEN}{self.region_support}{END}")
            print('_____________________________________')
            print('global dataset : ', x_train.shape, y_train.shape)
            print('_____________________________________')
            model = GPR(gpr_model)
            model.fit(x_train, y_train)
            self.ei_roll = RolloutEI()
                
            pred_sample_x, X_root = self.ei_roll.sample(X_root, num_agents, self.tf, x_train, self.horizon, y_train, region_support, model, rng) #self._opt_acquisition(agent.y_train, agent.model, agent.region_support, rng) 
            pred_sample_y, falsified = compute_robustness(pred_sample_x, test_function, behavior, agent_sample=True)
            
            x_train = np.vstack((x_train, pred_sample_x))
            y_train = np.hstack((y_train, (pred_sample_y)))
        plot_dict = {} #{'agents':self.agent_point_hist,'assignments' : self.assignments, 'region_support':region_support, 'test_function' : test_function, 'inactive_subregion_samples' : self.inactive_subregion_samples, 'sample': num_samples}
            # X_root.add_child(self.region_support)
        return falsified, self.region_support, plot_dict

