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
from ..agent.partition import find_close_factor_pairs, Node, print_tree
from ..agent.agent import Agent
from ..agent.constants import *
from ..utils.timerf import logtime, LOGPATH

class RolloutBO(BO_Interface):
    def __init__(self):
        pass
    
    def split_region(self,root,dim,num_agents):
        if num_agents % 2 == 0:
            splits = num_agents+1
        else:
            splits = num_agents #+ 1
        region = np.linspace(root.input_space[dim][0], root.input_space[dim][1], num = splits)
        final = []

        for i in range(len(region)-1):
            final.append([region[i], region[i+1]])
        regs = []
        for i in range(len(final)):
            org = root.input_space.copy()
            org[dim] = final[i]
            regs.append(org)

        regs = [Node(i, 1) for i in regs]
        return regs
    
    def get_subregion(self,root, num_agents,dic, dim=0):
        q=[root]
        while(len(q) < num_agents):
            if len(q) % 2 == 0:
                dim = (dim+1)% len(root.input_space)
            curr = q.pop(0)
            print('dim curr queue_size', dim, curr.input_space, len(q))
            ch = self.split_region(curr,dim, dic[dim])
            # print('ch',ch)
            curr.add_child(ch)
            q.extend(ch)
        print([i.input_space for i in q])
        return q
    
    @logtime(LOGPATH)
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

        # num_agents = 5
        tf_dim = region_support.shape[0]
        # split the input space among m agents initially
        # print('just b4 X root: ',region_support)
        X_root = Node(self.region_support, 1)
        # agents_to_subregion = self.split_space(X_root.input_space,num_agents, tf_dim)
        # print('b4 agents_to_subregion')
        factorized = sorted(find_close_factor_pairs(num_agents), reverse=True)
        print('factorized among agents',factorized)
        agents_to_subregion = self.get_subregion(deepcopy(X_root), num_agents, factorized, dim=0)
        X_root.add_child(agents_to_subregion)
        assignments = {value: 1 for value in agents_to_subregion} 

        
        for sample in tqdm(range(num_samples)):
            print('_____________________________________', sample)
            print(f"INPUT SPACE : {GREEN}{self.region_support}{END}")
            print('_____________________________________')
            print('global dataset : ', x_train, y_train)
            print('_____________________________________')
            model = GPR(gpr_model)
            model.fit(x_train, y_train)
            self.ei_roll = RolloutEI()
            
            
            # lf = X_root.find_leaves()
            # print('______________below find leaves_______________________')
            # print_tree(X_root)
            # print('_____________________________________')
            # assignments = {}
            # agents_to_subregion = []
            # internal_inactive_subregion=[]
            # for l in lf:
            #     if l.status == 1:
            #         assignments.update({l : l.status})
            #         agents_to_subregion.append(l)
            #     elif l.status == 0:
            #         internal_inactive_subregion.append(l)
            
            # print('assignments: ',[{str(k.input_space) : v} for k,v in assignments.items()])
            # print('lf size: ', len(lf))
            
            # xtr = deepcopy(x_train)
            # ytr = deepcopy(y_train)
            # agents = [Agent(model, xtr, ytr, agents_to_subregion[a].input_space) for a in range(num_agents)]
            # print('agents_to_subregion : ',agents_to_subregion)
            # for i,sub in enumerate(agents_to_subregion):
            #     sub.update_agent(agents[i])
            

            # internal_inactive_subregion_not_node = [i.input_space for i in internal_inactive_subregion]
            # print('internal inactive: ',internal_inactive_subregion_not_node, internal_inactive_subregion)
            # if internal_inactive_subregion != []:
            #     self.inactive_subregion = (internal_inactive_subregion_not_node)
            
            # print('_______________________________ AGENTS AT WORK ___________________________________')           
                
            pred_sample_x, assignments, X_root, agents = self.ei_roll.sample(X_root, num_agents, self.tf, x_train, self.horizon, y_train, region_support, model, rng) #self._opt_acquisition(agent.y_train, agent.model, agent.region_support, rng) 
            pred_sample_y, falsified = compute_robustness(pred_sample_x, test_function, behavior, agent_sample=True)
            
            x_train = np.vstack((x_train, pred_sample_x))
            y_train = np.hstack((y_train, (pred_sample_y)))
            print('pred x,y appended: ', x_train, y_train)

        self.assignments.append(assignments)
        self.agent_point_hist.extend([i.point_history[-1] for i in agents])
        print('_______________________________ TREE _______________________________')
        print_tree(X_root)
        print('____________________________________________________________________')
        print('_______________________________')
        # print("Aggregate action: ", final_predx)
        print('_______________________________')
        print('Active partitions: ', [{str(k.input_space) : v} for k,v in assignments.items()])
        print('_______________________________')
        # print("End of BO iter: ", test_function.agent_point_history)
        plot_dict = {} #{'agents':self.agent_point_hist,'assignments' : self.assignments, 'region_support':region_support, 'test_function' : test_function, 'inactive_subregion_samples' : self.inactive_subregion_samples, 'sample': num_samples}
            # X_root.add_child(self.region_support)
        return falsified, self.region_support, plot_dict

