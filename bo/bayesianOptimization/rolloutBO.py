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

'''
    def sample_from_discontinuous_region(self, num_samples, regions, region_support, tf_dim, rng, volume=True ):
        filtered_samples = np.empty((0,tf_dim))
        # threshold = 0.3
        total_volume = compute_volume(region_support)
        vol_dic = {}
        for reg in regions:
            print('inside vol dict ', reg)
            vol_dic[reg] = compute_volume(reg.input_space) / total_volume

        # def target_region(regions, samples):
        #     print('inside target region , ', samples )
        #     for sample_point in samples:
        #         for i, subregion in enumerate(regions):
        #             # print(subregion.input_space.shape)
        #             inside = True
        #             for d in range(len(subregion)):
        #                 lb,ub = subregion[d]
        #                 if sample_point[d] >= lb or sample_point[d] <= ub:
        #                     weight = compute_volume(subregion) / total_volume
        #                     # print('sample, wt, subregion: ', sample_point, weight, subregion)
        #                     if bool(int(num_samples * weight)):
        #                         # filtered_samples = np.vstack((filtered_samples,sample_point)) 
        #                         return True, sample_point
        #     return False, None
        vol_dic_items = sorted(vol_dic.items(), key=lambda x:x[1])
        print('vol dict :',vol_dic_items, total_volume)
        for v in vol_dic_items:
            tsmp = uniform_sampling(int(num_samples*v[1]), v[0].input_space, tf_dim, rng)
            
            filtered_samples = np.vstack((filtered_samples,tsmp))

        # while len(filtered_samples) < num_samples:
        #     # Generate random points in the larger bounding box [-5,-5][5,5]
        #     test_samples = uniform_sampling(10, self.inactive_subregion, tf_dim, rng)
        #     print('uniform samples :',test_samples)
        #     # Check if the point is within the desired region
        #     isin, pt = target_region(regions, test_samples)
        #     if isin:
        #         filtered_samples = np.vstack((filtered_samples,pt)) 
        #     print('len(filtered_samples): ',len(filtered_samples))
        # print('filtered_samples: ,', filtered_samples, regions)

        return filtered_samples
    
    def check_inactive(self, sample_point, agents_to_subregion, inactive_subregion, assignments,agent_idx):
        if agent_idx == 0:
            fall_under = []
            print('inside check inactive')
            # for sample_point in sample_points:
            for i, subregion in enumerate(inactive_subregion):
            # print(subregion.input_space.shape)
                inside_inactive = True
                for d in range(len(subregion.input_space)):
                    lb,ub = subregion.input_space[d]
                    if sample_point[d] < lb or sample_point[d] > ub:
                        inside_inactive = False
                
                if inside_inactive: 
                    fall_under.append(subregion)
                    print('fall under : ',fall_under, inactive_subregion[i].input_space)
                    # inactive_subregion[subregion] += 1
                    # assignments[agents_to_subregion[agent_idx]] -= 1
                    if assignments[agents_to_subregion[agent_idx]] == 0:
                        agents_to_subregion[agent_idx].status = 0
                    inactive_subregion[i].status = 1
                    # agent.update_bounds(subregion)
                    not_inside_any = True
                    # return sample_point

    def reassign_subregion(self, agents, agents_predictions, sample_from_inactive_region, gpr_model, tf_dim, rng):
        reassignments = []
        for a_idx, a_ymin in enumerate(agents_predictions):
            print('a ymin : ', a_ymin)
            region_ei_min = np.inf
            agent_idx = -1
            idx_a1 = -1
            for i, agent in enumerate(agents):
                print('agent region: ', agent.region_support)
                random_samples = uniform_sampling(2000, agent.region_support, tf_dim, rng)
                min_bo_val = -1 * self._acquisition(
                    a_ymin, random_samples, gpr_model, "multiple"
                )
                if a_idx == 0 and (np.size(self.inactive_subregion) != 0):
                    ei_inactive = -1 * self._acquisition(
                    a_ymin, sample_from_inactive_region, gpr_model, "multiple"
                )
                    min_bo_val = np.hstack((min_bo_val, ei_inactive))
                print('argmin(min_bo_val) :',i, np.argmin(min_bo_val), random_samples, min_bo_val, min(min_bo_val))
                if region_ei_min > min(min_bo_val):
                    region_ei_min = min(min_bo_val)
                    agent_idx = i
                    if a_idx == 0 and (np.size(self.inactive_subregion) != 0) and np.argmin(min_bo_val) > 2000:
                        agent_idx = np.argmin(min_bo_val)

                    print('region_ei_min, i: ',region_ei_min, i)
            reassignments.append(agent_idx)
        return reassignments


    def get_most_min_sample(self, agent_posterior, x_train, rng):
        agent_predictions = []
        # print('region support inside get_most_min_sample: ',agent.region_support)
        for agent in agent_posterior:
            print('region support inside get_most_min_sample: ',agent.region_support)
            predx, min_bo_val = self.ei_roll.sample(agent, x_train, self.tf, self.horizon, agent.y_train, agent.region_support, agent.model, rng) #self._opt_acquisition(agent.y_train, agent.model, agent.region_support, rng) 
            # predx, min_bo_val = self._opt_acquisition(agent.y_train, agent.model, agent.region_support, rng) 
            agent_predictions.append(min_bo_val)
            agent.add_point(predx[0])
        print('inside get most min sample : ',agent_predictions)
        return agent_predictions
    
    def get_ei_across_regions(self, agent_idx, agent_posterior, rng):
        agent_eis = []
        # print('region support inside get_most_min_sample: ',agent.region_support)
        for agent in agent_posterior:
            print('agent idx inside across regions: ',agent_idx, 'curr region :',agent.region_support)
        # predx, min_bo_val = self.ei_roll.sample(agent, self.tf, self.horizon, agent.y_train, agent.region_support, agent.model, rng) #self._opt_acquisition(agent.y_train, agent.model, agent.region_support, rng) 
            predx, min_bo_val = self._opt_acquisition(agent_posterior[agent_idx].y_train, agent_posterior[agent_idx].model, agent.region_support, rng) 
            agent_eis.append(min_bo_val)
            # agent.add_point(predx)
        print('inside get most min sample : ',agent_eis)
        return agent_eis

    def get_point_pred(self, agent_predictions):
        mag = [abs(x) for x in agent_predictions]
        total_sum = sum(mag)
        weights = [x / (total_sum + 0.0000001) for x in mag]

        # weighted_sum = sum(x * w for x, w in zip(agent_predictions, weights))

        return weights
    
    # def modify_subregion(self, sample_point, agents_to_subregion, inactive_subregion, assignments,agent_idx, agent, agent_with_best_sample):
    #     # check each sub region
    #     not_inside_any = False
    #     for i, subregion in enumerate(agents_to_subregion):
    #         # print(subregion.input_space.shape)
    #         inside = True
    #         for d in range(len(subregion.input_space)):
    #             lb,ub = subregion.input_space[d]
    #             if sample_point[d] < lb or sample_point[d] > ub:
    #                 inside = False
            
    #         if inside: 
    #             assignments[subregion] += 1
    #             assignments[agents_to_subregion[agent_idx]] -= 1
    #             agents_to_subregion[agent_idx].status = 0
    #             # agent.update_bounds(subregion.input_space)

    #     if agent_idx == 0 and inside:
    #         fall_under = []
    #         for i, subregion in enumerate(inactive_subregion):
    #         # print(subregion.input_space.shape)
    #             inside_inactive = True
    #             for d in range(len(subregion.input_space)):
    #                 lb,ub = subregion.input_space[d]
    #                 if sample_point[d] < lb or sample_point[d] > ub:
    #                     inside_inactive = False
                
    #             if inside_inactive: 
    #                 fall_under.append(subregion.input_space)
    #                 # inactive_subregion[subregion] += 1
    #                 assignments[agents_to_subregion[agent_idx]] -= 1
    #                 agents_to_subregion[agent_idx].status = 0
    #                 inactive_subregion[i].status = 1
    #                 # agent.update_bounds(subregion)
    #                 not_inside_any = True

    #     # agents_to_subregion[agent_with_best_sample].update_agent(agent)
    #     # assignments[agents_to_subregion[agent_with_best_sample]] += 1
    #     # assignments[agents_to_subregion[agent]] -= 1
        
    #     print('assignments: ',[{str(k.input_space) : v} for k,v in assignments.items()])
    #     print('inactive_subregion: ',[k for k in inactive_subregion])
        
    #     # modify the region support
    #     region_support = [i.input_space for i in assignments.keys() if assignments[i] >= 1]
    #     # from_inactive_subregion = [i for i in inactive_subregion.keys() if inactive_subregion[i] >= 1]
    #     if not_inside_any:
    #         print('b4 hstack region_support: ',region_support, fall_under)
    #         print('b4 hstack shapes: ',len(region_support), len(fall_under))
    #         region_support = np.vstack((region_support,fall_under))
    #         # assignments.update({fall_under: 1})
    #     region_support = np.hstack(region_support)
    #     print('b4 region_support: ',region_support)
    #     lb = np.min(region_support,axis=1)
    #     ub = np.max(region_support,axis=1)
    #     region_support = np.vstack((lb,ub)).T
    #     print('region_support: ',region_support)
        
    #     return assignments, region_support, not_inside_any
        


    # def rollout_acquisition(self, agent, gpr_model, rollout_samples, test_function, behavior, rng):
    #     agent_pred = []
    #     sample_min = np.inf
    #     agent_min_pred = np.inf
    #     print('agent xtrain, ytrain', agent.x_train, agent.y_train )
    #     for smp in rollout_samples:
    #         xtr = deepcopy(agent.x_train)
    #         ytr = deepcopy(agent.y_train)
    #         ysmp, falsified = compute_robustness(np.array([smp]), test_function, behavior, agent_sample=True)

    #         xtr = np.vstack((xtr, np.array([smp])))
    #         ytr = np.hstack((ytr, ysmp))
    #         print('agent xtr ytr after appending each sample: ', xtr, ytr)
    #         model = GPR(gpr_model)
    #         model.fit(xtr, ytr)
            

    #         pred_sample_x, min_bo_val = self._opt_acquisition(ytr, model, agent.region_support, rng)
    #         pred_mean, _ = self._surrogate(model, np.array([pred_sample_x]))
    #         print('agent min pred val: ', agent_min_pred,"agent min pred x: ",pred_sample_x, "curr agent minboval surr mean: ",pred_mean)
    #         if agent_min_pred > pred_mean:
    #             agent_min_pred = pred_mean
    #             sample_min = smp
    #             agent.model = model
    #         agent_pred.append([smp, pred_mean])
    #     print('one step EI pred and sample used from each posterior: ', agent_pred)

    #     return sample_min

    

    @logtime(LOGPATH)
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
        return np.array(min_bo)#, min_bo_val  #action, assignments, random_samples #

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
        # print('inside acqusition: ',y_train)
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
    
    '''