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
from ..gprInterface import GPR
from ..sampling import uniform_sampling, lhs_sampling
from ..utils import compute_robustness
from ..behavior import Behavior

from ..utils.volume import compute_volume
from ..agent.partition import find_close_factor_pairs

# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Input, Output
# import plotly.graph_objs as go

RED = '\033[91m'
GREEN = '\033[92m'
END = '\033[0m'

class Node:
    def __init__(self, input_space, status) -> None:
        self.input_space = np.asarray(input_space, dtype=np.float32)
        self.child = []
        self.status = status
        self.agent = 0

    def update_agent(self, agent):
        self.agent = agent 
    
    def add_child(self,c):
        for i in c:
            # print("i,i.input_space: ",i.input_space)
            self.child.append(i)
    
    def find_leaves(self):
        leaves = []
        self._find_leaves_helper(self, leaves)
        return leaves

    def _find_leaves_helper(self, node, leaves):
        if not node.child:
            if node != self:  # Exclude the root node
                leaves.append(node)
        else:
            for child in node.child:
                self._find_leaves_helper(child, leaves)
    
def print_tree(node, level=0, prefix=''):
    # RED = '\033[91m'
    # GREEN = '\033[92m'
    # END = '\033[0m'
    if node.status:
        color = GREEN
    else:
        color = RED
    if node is None:
        return

    for i, child in enumerate(node.child):
        print_tree(child, level + 1, '|   ' + prefix if i < len(node.child) - 1 else '    ' + prefix)
    
    print('    ' * level + prefix + f'-- {color}{node.input_space.flatten()}{END}')

class Agent():
    def __init__(self, model, x_train, y_train, region_support) -> None:
        self.model = model
        self.point_history = []
        self.x_train = x_train
        self.y_train = y_train
        self.region_support = region_support
        
    def add_point(self, point):
        self.point_history.append(point)

    def update_model(self, model):
        self.model = model

    def update_bounds(self, region_support):
        self.region_support = region_support


class RolloutBO(BO_Interface):
    def __init__(self):
        pass
    
    # def split_space(self,input_space,num_agents, tf_dim):
    #     reg = np.zeros((num_agents, tf_dim), dtype=np.int64)
    #     for dim in input_space:
    #         region = np.linspace(dim[0], dim[1], num = num_agents+1)
    #         final = []
    #         for i in range(len(region)-1):
    #             final.append([region[i], region[i+1]])

    #         region = np.asarray(final)
    #         reg = np.hstack((reg,region))
    #     child_nodes = []
    #     for child_reg in reg[:,tf_dim:]:
    #         child_nodes.append(Node(child_reg.reshape((tf_dim,2)),1))
    #     return child_nodes
    
    # def split_region(self,root,dim):
    #     region = np.linspace(root.input_space[dim][0], root.input_space[dim][1], num = 3)
    #     final = []

    #     for i in range(len(region)-1):
    #         final.append([region[i], region[i+1]])
    #     regs = []
    #     for i in range(len(final)):
    #         org = root.input_space.copy()
    #         org[dim] = final[i]
    #         regs.append(org)

    #     regs = [Node(i, 1) for i in regs]
    #     return regs

    # def get_subregion(self, root, num_agents, dim = 0):
    #     q=[root]
    #     while(len(q) < num_agents):
    #         if len(q) % 2 == 0:
    #             dim = (dim+1)% len(root.input_space)
    #         curr = q.pop(0)
    #         # print('dim curr queue_size', dim, curr.input_space, len(q))
    #         ch = self.split_region(curr,dim)
    #         curr.add_child(ch)
    #         q.extend(ch)
    #     # print([i.input_space for i in q])
    #     return q
    
    def split_region(self,root,dim,num_agents):
        region = np.linspace(root.input_space[dim][0], root.input_space[dim][1], num = num_agents+1)
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
            # print('dim curr queue_size', dim, curr.input_space, len(q))
            ch = self.split_region(curr,dim, dic[dim])
            # print('ch',ch)
            curr.add_child(ch)
            q.extend(ch)
        # print([i.input_space for i in q])
        return q

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
        falsified = False
        self.agent_point_hist = []
        # Domain reduces after each BO iteration 
        self.region_support = region_support
        # X_root = Node(region_support)
        self.inactive_subregion = [] #np.empty((0)).reshape((0,2,2))
        # self.total_pred_sample_x = []
        self.inactive_subregion_samples = [np.array([None]),]
        internal_inactive_subregion = []
            # internal_inactive_subregion = {value: 0 for value in internal_inactive_subregion}
        internal_inactive_subregion_samples = []
        self.assignments = []
        for sample in tqdm(range(num_samples)):
            print('_____________________________________', sample)
            print(f"INPUT SPACE : {GREEN}{self.region_support}{END}")
            print('_____________________________________')
            print('global dataset : ', x_train, y_train)
            print('_____________________________________')
            model = GPR(gpr_model)
            model.fit(x_train, y_train)

            tf_dim = region_support.shape[0]
            
            self.internal_region_support = self.region_support
            # num_agents = 5
            action = np.zeros((num_agents,tf_dim))
            # split the input space among m agents initially
            # print('just b4 X root: ',region_support)
            X_root = Node(self.internal_region_support, 1)
            # agents_to_subregion = self.split_space(X_root.input_space,num_agents, tf_dim)
            # print('b4 agents_to_subregion')
            factorized = sorted(find_close_factor_pairs(num_agents), reverse=True)
            agents_to_subregion = self.get_subregion(deepcopy(X_root), num_agents, factorized, dim=0)
            # self.inactive_subregion_samples = []
            # self.inactive_subregion = []
            # print('after agents_to_subregion', self.inactive_subregion)
            X_root.add_child(agents_to_subregion)
            assignments = {value: 1 for value in agents_to_subregion} 
            print('assignments: ',[{str(k.input_space) : v} for k,v in assignments.items()])
            
            xtr = deepcopy(x_train)
            ytr = deepcopy(y_train)
            final_agents_predictions_val = []

            agents = [Agent(model, xtr, ytr, agents_to_subregion[a].input_space) for a in range(num_agents)]
            for i,sub in enumerate(agents_to_subregion):
                sub.update_agent(agents[i])
            # for i in agents:
            #     i.x_train = deepcopy(x_train)
            #     i.y_train = deepcopy(y_train)
            # for agent in num_agents:
            
            
            final_sample_val = np.inf
            print()
            print('_______________________________ AGENTS AT WORK ___________________________________')
            if (np.size(self.inactive_subregion) != 0):
                print('at the start inactive subregion: ',self.inactive_subregion)
                sample_from_inactive_region = uniform_sampling(5, self.inactive_subregion, tf_dim, rng)
                print('samples from inactive: ',sample_from_inactive_region)
                # random_samples = np.vstack((random_samples,sample_from_inactive_region))
                # print(random_samples.shape)
                internal_inactive_subregion_samples.append(sample_from_inactive_region)
                self.inactive_subregion_samples.append(sample_from_inactive_region)
            # print('random_samples, sample_from_inactive_region----->',random_samples, sample_from_inactive_region)

            for agent_idx, agent in enumerate(agents):
                rollout_samples = lhs_sampling(5, agent.region_support, tf_dim, rng)
                print('below rollout sample: ', agent.region_support)
                # if agent_idx == 0 and (np.size(self.inactive_subregion) != 0):
                #     rollout_samples = np.vstack((rollout_samples,sample_from_inactive_region))

                if agent_idx != 0 :
                    print('inside appending :', rollout_samples.shape, np.array([agents[agent_idx-1].point_history[-1]]).shape)
                    rollout_samples = np.vstack((rollout_samples, np.array([agents[agent_idx-1].point_history[-1]])))
                print('agent id : ',agent_idx, 'agent samples: ',rollout_samples )
                best_sample = self.rollout_acquisition(agent, gpr_model, rollout_samples, test_function, behavior, rng)
                print('best sample : ',best_sample)
                ysmp, falsified = compute_robustness(np.array([best_sample]), test_function, behavior, agent_sample=True)

                agent.x_train = np.vstack((agent.x_train, np.array([best_sample])))
                agent.y_train = np.hstack((agent.y_train, ysmp))

                internal_model = GPR(gpr_model)
                internal_model.fit(agent.x_train, agent.y_train)
                # print('model: -----s',internal_model)
                agent.model = internal_model

                agent_predictions = self.get_most_min_sample(agents,rng)
                agent_with_best_sample = np.argmin(agent_predictions)
                final_sample_val = min(final_sample_val, min(agent_predictions))
                final_agents_predictions_val.append(final_sample_val)
                final_predx = agents[agent_with_best_sample].point_history[-1]
                print('final sample: ',final_sample_val, final_predx, 'agent: ',agent_idx,'agent_with_best_sample:',agent_with_best_sample,'agents[agent_with_best_sample].point_history: ',agents[agent_with_best_sample].point_history )

                point_pred = self.get_point_pred(agent_predictions)
                print('point_pred: ',point_pred)
                min_bo = final_predx
                lower_bound_theta = np.ndarray.flatten(self.region_support[:, 0])
                upper_bound_theta = np.ndarray.flatten(self.region_support[:, 1])
                fun = lambda x_: -1 * self._acquisition(agent.y_train, x_, agent.model)     
                ei_constraint = {'type': 'ineq', 'fun': lambda x: self._acquisition(agent.y_train, x, agent.model) - point_pred}
                # for _ in range(9):
                #     new_params = minimize(
                #         fun,
                #         bounds=list(zip(lower_bound_theta, upper_bound_theta)),
                #         x0=min_bo,
                #         constraints=[ei_constraint],
                #     )

                #     if not new_params.success:
                #         continue

                #     if min_bo is None or fun(new_params.x) < min_bo_val:
                #         min_bo = new_params.x
                #         min_bo_val = fun(min_bo)
                new_params = minimize(
                    fun, bounds=list(zip(lower_bound_theta, upper_bound_theta)), x0=min_bo,constraints=[ei_constraint],
                )
                min_bo = new_params.x
                print('point_pred after gradient opt: ',min_bo)
                # agent.update_bounds(agents_to_subregion[agent_with_best_sample].input_space)
                
                # , 
                assignments, self.internal_region_support, not_inside_any = self.modify_subregion(min_bo, agents_to_subregion, internal_inactive_subregion, assignments, agent_idx,agent, agent_with_best_sample)
                print('self.internal_region_support after each agent: ',self.internal_region_support)
                if not_inside_any:
                    X_root = Node(self.internal_region_support, 1)
                    # agents_to_subregion = self.split_space(X_root.input_space,num_agents, tf_dim)
                    # print('b4 agents_to_subregion')
                    internal_factorized = sorted(find_close_factor_pairs(num_agents), reverse=True)
                    agents_to_subregion = self.get_subregion(deepcopy(X_root), num_agents,internal_factorized,  dim=0)
                    # self.inactive_subregion_samples = []
                    # self.inactive_subregion = []
                    # print('after agents_to_subregion', self.inactive_subregion)
                    X_root.add_child(agents_to_subregion)
                    assignments = {value: 1 for value in agents_to_subregion} 
                    print('not inside any assignments: ',[{str(k.input_space) : v} for k,v in assignments.items()])
                
            pred_sample_y, falsified = compute_robustness(np.array([final_predx]), test_function, behavior)

            x_train = np.vstack((x_train, np.array([final_predx])))
            y_train = np.hstack((y_train, pred_sample_y))
            
            for i in assignments.keys():
                if assignments[i] == 1:
                    i.status = 1
                if assignments[i] > 1:
                    # ch = self.split_space(i.input_space, assignments[i], tf_dim)
                    internal_factorized = sorted(find_close_factor_pairs(assignments[i]), reverse=True)
                    ch = self.get_subregion(deepcopy(i), assignments[i], internal_factorized)
                    i.add_child(ch)
            self.region_support = self.internal_region_support
            print('End of agents work region bounds: ',self.region_support)
            self.assignments.append(assignments)
            internal_inactive_subregion = [i.input_space for i in assignments.keys() if assignments[i] == 0]
            print('internal_inactive_subregion: ',len(internal_inactive_subregion), 'all inactive: ',len(self.inactive_subregion))
            if internal_inactive_subregion != []:
                # self.inactive_subregion = np.vstack((self.inactive_subregion, internal_inactive_subregion))
                self.inactive_subregion = np.hstack(internal_inactive_subregion)
                # print('just b4 hstack :',self.inactive_subregion, internal_inactive_subregion)
                # self.inactive_subregion.extend(internal_inactive_subregion)
                # print('after extend :', self.inactive_subregion)
                # self.inactive_subregion = np.hstack(self.inactive_subregion)
                # self.inactive_subregion = np.concatenate([self.inactive_subregion, internal_inactive_subregion], axis=0)
                # self.inactive_subregion = np.array([i.input_space for i in assignments.keys() if assignments[i] == 0])[0]#.reshape((2,2))
                print('check from ass: ',self.inactive_subregion, np.size(self.inactive_subregion))

                # self.inactive_subregion = np.hstack(self.inactive_subregion)
                # print('b4 inactive_subregion: ',self.inactive_subregion)
                lb = np.min(self.inactive_subregion,axis=1)
                ub = np.max(self.inactive_subregion,axis=1)
                print('lbub: ',lb,ub)
                self.inactive_subregion = np.vstack((lb,ub)).T
            print('below lbub inactive_subregion: ',self.inactive_subregion)
            # contour(self.assignments, region_support, test_function, self.inactive_subregion_samples, sample, random_samples)
            self.agent_point_hist.extend([i.point_history[-1] for i in agents])
            # print('self.agent_point_hist: ',self.agent_point_hist)
        print('_______________________________ TREE _______________________________')
        print_tree(X_root)
        print('____________________________________________________________________')
        print('_______________________________')
        print("Aggregate action: ", final_predx)
        print('_______________________________')
        print('Active partitions: ', [{str(k.input_space) : v} for k,v in assignments.items()])
        print('_______________________________')
        # print("End of BO iter: ", test_function.agent_point_history)
        plot_dict = {'agents':self.agent_point_hist,'assignments' : self.assignments, 'region_support':region_support, 'test_function' : test_function, 'inactive_subregion_samples' : self.inactive_subregion_samples, 'sample': num_samples}
            # X_root.add_child(self.region_support)
        return falsified, self.region_support, plot_dict

    def modify_subregion(self, sample_point, agents_to_subregion, inactive_subregion, assignments,agent_idx, agent, agent_with_best_sample):
        # check each sub region
        not_inside_any = False
        for i, subregion in enumerate(agents_to_subregion):
            # print(subregion.input_space.shape)
            inside = True
            for d in range(len(subregion.input_space)):
                lb,ub = subregion.input_space[d]
                if sample_point[d] < lb or sample_point[d] > ub:
                    inside = False
            
            if inside: 
                assignments[subregion] += 1
                assignments[agents_to_subregion[agent_idx]] -= 1
                agents_to_subregion[agent_idx].status = 0
                # agent.update_bounds(subregion.input_space)

        if agent_idx == 0 and inside:
            fall_under = []
            for i, subregion in enumerate(inactive_subregion):
            # print(subregion.input_space.shape)
                inside_inactive = True
                for d in range(len(subregion)):
                    lb,ub = subregion[d]
                    if sample_point[d] < lb or sample_point[d] > ub:
                        inside_inactive = False
                
                if inside_inactive: 
                    fall_under.append(subregion)
                    # inactive_subregion[subregion] += 1
                    assignments[agents_to_subregion[agent_idx]] -= 1
                    agents_to_subregion[agent_idx].status = 0
                    # agent.update_bounds(subregion)
                    not_inside_any = True

        # agents_to_subregion[agent_with_best_sample].update_agent(agent)
        # assignments[agents_to_subregion[agent_with_best_sample]] += 1
        # assignments[agents_to_subregion[agent]] -= 1
        
        print('assignments: ',[{str(k.input_space) : v} for k,v in assignments.items()])
        print('inactive_subregion: ',[k for k in inactive_subregion])
        
        # modify the region support
        region_support = [i.input_space for i in assignments.keys() if assignments[i] >= 1]
        # from_inactive_subregion = [i for i in inactive_subregion.keys() if inactive_subregion[i] >= 1]
        if not_inside_any:
            print('b4 hstack region_support: ',region_support, fall_under)
            print('b4 hstack shapes: ',len(region_support), len(fall_under))
            region_support = np.vstack((region_support,fall_under))
        region_support = np.hstack(region_support)
        print('b4 region_support: ',region_support)
        lb = np.min(region_support,axis=1)
        ub = np.max(region_support,axis=1)
        region_support = np.vstack((lb,ub)).T
        print('region_support: ',region_support)
        
        return assignments, region_support, not_inside_any
        


    def rollout_acquisition(self, agent, gpr_model, rollout_samples, test_function, behavior, rng):
        agent_pred = []
        sample_min = np.inf
        agent_min_pred = np.inf
        print('agent xtrain, ytrain', agent.x_train, agent.y_train )
        for smp in rollout_samples:
            xtr = deepcopy(agent.x_train)
            ytr = deepcopy(agent.y_train)
            ysmp, falsified = compute_robustness(np.array([smp]), test_function, behavior, agent_sample=True)

            xtr = np.vstack((xtr, np.array([smp])))
            ytr = np.hstack((ytr, ysmp))
            print('agent xtr ytr after appending each sample: ', xtr, ytr)
            model = GPR(gpr_model)
            model.fit(xtr, ytr)
            

            pred_sample_x, min_bo_val = self._opt_acquisition(ytr, model, agent.region_support, rng)
            pred_mean, _ = self._surrogate(model, np.array([pred_sample_x]))
            print('agent min pred val: ', agent_min_pred,"agent min pred x: ",pred_sample_x, "curr agent minboval surr mean: ",pred_mean)
            if agent_min_pred > pred_mean:
                agent_min_pred = pred_mean
                sample_min = smp
                agent.model = model
            agent_pred.append([smp, pred_mean])
        print('one step EI pred and sample used from each posterior: ', agent_pred)

        return sample_min

    def get_most_min_sample(self, agent_posterior, rng):
        agent_predictions = []
        # print('region support inside get_most_min_sample: ',agent.region_support)
        for agent in agent_posterior:
            print('region support inside get_most_min_sample: ',agent.region_support)
            predx, min_bo_val = self._opt_acquisition(agent.y_train, agent.model, agent.region_support, rng) 
            agent_predictions.append(min_bo_val)
            agent.add_point(predx)
        print('inside get most min sample : ',agent_predictions)
        return agent_predictions

    def get_point_pred(self, agent_predictions):
        mag = [abs(x) for x in agent_predictions]
        total_sum = sum(mag)
        weights = [x / total_sum for x in mag]

        weighted_sum = sum(x * w for x, w in zip(agent_predictions, weights))

        return weighted_sum


    def _opt_acquisition(self, y_train: NDArray, gpr_model: Callable, region_support: NDArray, rng) -> NDArray:
        """Get the sample points

        Args:
            X: sample points
            y: corresponding robustness values
            model: the GP models
            sbo: sample points to construct the robustness values
            test_function_dimension: The dimensionality of the region. (Dimensionality of the test function)
            region_support: The bounds of the region within which the sampling is to be done.
                                        Region Bounds is M x N x O where;
                                            M = number of regions;
                                            N = test_function_dimension (Dimensionality of the test function);
                                            O = Lower and Upper bound. Should be of length 2;

        Returns:
            The new sample points by BO
        """

        tf_dim = region_support.shape[0]
        lower_bound_theta = np.ndarray.flatten(region_support[:, 0])
        upper_bound_theta = np.ndarray.flatten(region_support[:, 1])
        
        curr_best = np.min(y_train)

        

        fun = lambda x_: -1 * self._acquisition(y_train, x_, gpr_model)
        
        # for agent in range(num_agents):
            # lower_bound_theta = np.ndarray.flatten(region_support[:, 0])
            # upper_bound_theta = np.ndarray.flatten(region_support[:, 1])
        random_samples = uniform_sampling(2000, region_support, tf_dim, rng)
        # random_samples = np.vstack((random_samples,action[:agent]))
        # 
        min_bo_val = -1 * self._acquisition(
            y_train, random_samples, gpr_model, "multiple"
        )
        min_bo = np.array(random_samples[np.argmin(min_bo_val), :])
        # print('min bo b4 BFGS :',min_bo, min_bo_val[-3],random_samples[-3])
        min_bo_val = np.min(min_bo_val)
        # print('lower_bound_theta: ',list(zip(lower_bound_theta, upper_bound_theta)))
        for _ in range(9):
            new_params = minimize(
                fun,
                bounds=list(zip(lower_bound_theta, upper_bound_theta)),
                x0=min_bo,
            )

            if not new_params.success:
                continue

            if min_bo is None or fun(new_params.x) < min_bo_val:
                min_bo = new_params.x
                min_bo_val = fun(min_bo)
        new_params = minimize(
            fun, bounds=list(zip(lower_bound_theta, upper_bound_theta)), x0=min_bo
        )
        min_bo = new_params.x
            # action[agent] = min_bo
            # print('agent: ',agent, 'minbo: ',min_bo, min_bo_val)
            
        # print('shape: ...........', self.inactive_subregion,np.array([i.input_space for i in assignments.keys() if assignments[i] == 0]).shape )
        # self.inactive_subregion = np.array(self.inactive_subregion)
        


        # get the num of agents in each subregion 
        return np.array(min_bo), min_bo_val  #action, assignments, random_samples #

    def _surrogate(self, gpr_model: Callable, x_train: NDArray):
        """_surrogate Model function

        Args:
            model: Gaussian process model
            X: Input points

        Returns:
            Predicted values of points using gaussian process model
        """

        return gpr_model.predict(x_train)

    def _acquisition(self, y_train: NDArray, sample: NDArray, gpr_model: Callable, sample_type:str ="single") -> NDArray:
        """Acquisition Model: Expected Improvement

        Args:
            y_train: corresponding robustness values
            sample: Sample(s) whose EI is to be calculated
            gpr_model: GPR model
            sample_type: Single sample or list of model. Defaults to "single". other options is "multiple".

        Returns:
            EI of samples
        """
        # print(f"Sample shape is {sample.shape}")
        curr_best = np.min(y_train)
        # print('curr_best: ',curr_best)
        if sample_type == "multiple":
            mu, std = self._surrogate(gpr_model, sample)
            ei_list = []
            for mu_iter, std_iter in zip(mu, std):
                pred_var = std_iter
                if pred_var > 0:
                    var_1 = curr_best - mu_iter
                    var_2 = var_1 / pred_var

                    ei = (var_1 * norm.cdf(var_2)) + (
                        pred_var * norm.pdf(var_2)
                    )
                else:
                    ei = 0.0

                ei_list.append(ei)
            # print(np.array(ei_list).shape)
            # print("*****")
            # return np.array(ei_list)
        elif sample_type == "single":
            # print("kfkf")
            mu, std = self._surrogate(gpr_model, sample.reshape(1, -1))
            pred_var = std[0]
            if pred_var > 0:
                var_1 = curr_best - mu[0]
                var_2 = var_1 / pred_var    # var of the ybest from the mean as a percentage of pred var 

                ei = (var_1 * norm.cdf(var_2)) + (          # cdf - prob of getting a value <= this percentage ; pdf - pdf - prob of getting exactly this percentage 
                    pred_var * norm.pdf(var_2)
                )
            else:
                ei = 0.0
            # return ei

        if sample_type == "multiple":
            return_ei = np.array(ei_list)
        elif sample_type == "single":
            return_ei = ei

        return return_ei