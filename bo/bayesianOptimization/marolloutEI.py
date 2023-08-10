from typing import Callable, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import norm
from tqdm import tqdm
import copy
from copy import deepcopy
import time 

from .internalBO import InternalBO
# from .rolloutAllatOnce import RolloutBO
from ..agent.partition import find_close_factor_pairs, Node, print_tree
from ..gprInterface import GPR
from bo.gprInterface import InternalGPR
from ..sampling import uniform_sampling
from ..utils import compute_robustness
from ..behavior import Behavior
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial

from ..agent.agent import Agent
# from ..agent.constants import *

def unwrap_self_f(arg, **kwarg):
    return RolloutEI.get_pt_reward(*arg, **kwarg)

class RolloutEI(InternalBO):
    def __init__(self) -> None:
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

    def sample(
        self,
        root,
        agent,
        agents_to_subregion,
        assignments,
        internal_inactive_subregion, 
        sample_from_inactive_region,
        test_function: Callable,
        x_train,
        horizon: int,
        y_train: NDArray,
        region_support: NDArray,
        gpr_model: Callable,
        rng
    ) -> Tuple[NDArray]:

        """Rollout with EI

        Args:
            test_function: Function of System Under Test.
            horizon: Number of steps to look ahead
            y_train: Evaluated values of samples from Training set.
            region_support: Min and Max of all dimensions
            gpr_model: Gaussian Process Regressor Model developed using Factory
            rng: RNG object from numpy

        Raises:
            TypeError: If y_train is not (n,) numpy array

        Returns:
            next x values that have minimum h step reward 
        """
        self.mc_iters = 5
        self.root = root
        self.agent = agent
        self.assignments = assignments
        self.agents_to_subregion = agents_to_subregion
        self.internal_inactive_subregion = internal_inactive_subregion, 
        self.sample_from_inactive_region = sample_from_inactive_region,
        self.numthreads = int(mp.cpu_count()/2)
        self.tf = test_function
        self.x_train = x_train
        self.gpr_model = gpr_model
        self.horizon = horizon
        self.region_support = region_support
        self.rng = rng
        self.tf_dim = region_support.shape[0]
        num_samples = 5
        self.y_train = y_train #copy.deepcopy(np.asarray(test_function.point_history,dtype=object)[:,-1])

        # Choosing the next point
        x_opt_from_all = []
        for i,a in enumerate(agent):
            # smp = uniform_sampling(5, a.region_support, tf_dim, rng)
            x_opt = self._opt_acquisition(y_train, gpr_model, a.region_support, rng) 
            # smp = np.vstack((smp, x_opt))
            x_opt_from_all.append(x_opt)
            # for i, preds in enumerate(pred_sample_x):
            self.agent[i].point_history.append(x_opt)
        # print('x_opt_from_all: ', np.hstack((x_opt_from_all)).reshape((6,4,2)))
        # Generate a sample dataset to rollout and find h step observations
        # exit()
        subx = np.hstack((x_opt_from_all)).reshape((4,self.tf_dim)) #np.asarray(x_opt_from_all)
        # subx = np.vstack([subx,np.array([x_opt_from_all])]) #np.array([x_opt]) #np.vstack([subx,x_opt])
        print('subx : ',subx)
        # print('subx :',subx)
        # Rollout and get h step observations
        
        # suby = -1 * self.get_exp_values(subx)
        predx = self.get_h_step_xt(subx)
        print('############################################################################')
        print()
        print('predx: rollout for 2 horizons with 6 sampled points  :',predx, ' subx:', subx)
        print()
        print('############################################################################')

        # Build Gaussian prior model with the sample dataset 
        # for sample in tqdm(range(num_samples)):
        #     model = GPR(InternalGPR())
        #     model.fit(subx, suby)
        #     # Get the next point using EI 
        #     pred_sub_sample_x = self._opt_acquisition(suby, model, region_support, rng)  # EI for the outer BO 
        #     print('pred_sub_sample_x:  for {sample}: ',pred_sub_sample_x)
        #     # Rollout and get h step obs for the above point 
        #     pred_sub_sample_y = -1 * self.get_exp_values( pred_sub_sample_x)    # this uses the inner MC to rollout EI and get the accumulated the reward or yt+1
        #     print('pred_sub_sample_y:  for {sample} rolled out for 2 horizons : ',pred_sub_sample_y)
        #     print()
        #     # Stack to update the Gaussian prior
        #     subx = np.vstack((subx, pred_sub_sample_x))
        #     suby = np.hstack((suby, pred_sub_sample_y))
        # Find and return the next point with min obs val among this sample dataset 
        # min_idx = np.argmin(suby)

        # lower_bound_theta = np.ndarray.flatten(agent.region_support[:, 0])
        # upper_bound_theta = np.ndarray.flatten(agent.region_support[:, 1])
        # fun = lambda x_: -1 * self.get_exp_values(x_)
        # # min_bo_val = min(suby)
        # min_bo = np.array(subx[np.argmin(suby), :])
        # # print('min bo b4 BFGS :',min_bo, min_bo_val[-3],random_samples[-3])
        # min_bo_val = np.min(suby)
        # # print('lower_bound_theta: ',list(zip(lower_bound_theta, upper_bound_theta)))
        # # for _ in range(9):
        # #     new_params = minimize(
        # #         fun,
        # #         bounds=list(zip(lower_bound_theta, upper_bound_theta)),
        # #         x0=min_bo,
        # #     )

        # #     if not new_params.success:
        # #         continue

        # #     if min_bo is None or fun(new_params.x) < min_bo_val:
        # #         min_bo = new_params.x
        # #         min_bo_val = fun(min_bo)
        # new_params = minimize(
        #     fun, bounds=list(zip(lower_bound_theta, upper_bound_theta)), x0=min_bo
        # )
        # min_bo = new_params.x
        # min_bo_val = fun(min_bo)
        # print('inside rollout optimise :', min_bo,min_bo_val)
        return predx, self.assignments, root #subx[[min_idx],:][0], np.min(suby) #np.array(min_bo), min_bo_val #[[min_idx],:]

    # Get expected value for each point after rolled out for h steps 
    def get_exp_values(self,eval_pts):
        # print('eval_pts shape: ',eval_pts.shape)
        # eval_pts = eval_pts.reshape((-1,2))
        num_pts = eval_pts.shape[0]
        exp_val = np.zeros(num_pts)
        for i in range(num_pts):
        # print()
            exp_val[i] = self.get_pt_reward(eval_pts[i])
        return exp_val
    
    # def _evaluate_at_point_list(self, point_to_evaluate):
    #     self.point_current = point_to_evaluate
    #     my_list = [0]*int(self.numthreads/2) + [1]*int(self.numthreads/2)
    #     th = np.random.shuffle(my_list)
    #     # print('th ---------',th)
    #     if self.numthreads > 1:
    #         serial_mc_iters = [int(int(self.numthreads)/self.numthreads)] * self.numthreads
    #         print('serial_mc_iters',serial_mc_iters, self.numthreads)
    #         pool = Pool(processes=self.numthreads)
    #         rewards = pool.map(self.get_pt_reward, serial_mc_iters)
    #         pool.close()
    #         pool.join()
    #     else:
    #         rewards = self.get_pt_reward()
    #     rewards = np.hstack((rewards))
    #     # print('rewards: ', rewards)
    #     return np.sum(rewards)/self.numthreads

    # Perform Monte carlo itegration
    def get_pt_reward(self, iters=5):
        # reward = 0
        # for i in range(iters):
        reward = self.get_h_step_reward(self.point_current)
        print('reward after each MC iter: ', reward)
        return (reward)
    
    def get_h_step_xt(self,current_point):
        reward = 0
        internal_inactive_subregion = []
        # Temporary Gaussian prior 
        tmp_gpr = copy.deepcopy(self.gpr_model)
        xtr = copy.deepcopy(self.x_train)  
        # xtr = np.asarray([i.tolist() for i in xtr])
        # print('xtr: ', xtr.shape)
        # xtr = [i.tolist() for i in xtr]
        ytr = copy.deepcopy(self.y_train)
        h = self.horizon
        xt = current_point
        idx = -1
        
        while(True):
            # print('xt : ', xt)
            np.random.seed(123)
            print('internal_inactive_subregion: ',internal_inactive_subregion)
            if internal_inactive_subregion != []:
                smp = uniform_sampling(5, internal_inactive_subregion[0].input_space, self.tf_dim, self.rng)
                xt = np.vstack((xt, smp))
            mu, std = self._surrogate(tmp_gpr, xt)
            # in_mu, in_std = self._surrogate(tmp_gpr, smp)
            ri = np.empty(0) #-1 * np.inf
            f_xts = []
            for i in range(len(xt)):
                f_xt = np.random.normal(mu[i],std[i],1)
                f_xts.append(f_xt[0])
                ri = np.hstack((ri, self.reward(f_xt,ytr)))
            # print('fxts : ',np.asarray(f_xts), xtr)
            # reward += (ri)
            idx += 1
            h -= 1
            if h <= 0 :
                break
            print('ri : ', ri)
            minidx = np.argmax(ri)
            print('xt[minidx]: ',minidx, xt[minidx])
            xtr = np.vstack((xtr,xt[minidx]))
            ytr = np.hstack((ytr,np.asarray(f_xts[minidx])))
            # print('xtr, ytr shape ',xtr, ytr )
            tmp_gpr.fit(xtr,ytr)
            tmp_xt = []
            internal_inactive_subregion = self.reassign(self.root,internal_inactive_subregion, tmp_gpr,  idx, minidx)
            for a in self.agent:
                next_xt = self._opt_acquisition(self.y_train,tmp_gpr,a.region_support,self.rng)
                tmp_xt.append(next_xt)
            xt = np.asarray(tmp_xt)
        return xt[:4]
    
    def reassign(self, X_root,internal_inactive_subregion, tmp_gpr, h, minidx):
        # print(self.agents_to_subregion)
        self.assignments[self.agents_to_subregion[h]] -= 1
        if minidx >= len(self.agents_to_subregion):
            self.assignments.update({internal_inactive_subregion[0] : 1})
            internal_inactive_subregion[0].status = 1
        else:
            self.assignments[self.agents_to_subregion[minidx]] += 1

        for i in self.assignments.keys():
            if self.assignments[i] == 1:
                i.status = 1
            if self.assignments[i] == 0:
                i.status = 0
            if self.assignments[i] > 1:
                # ch = self.split_space(i.input_space, assignments[i], tf_dim)
                internal_factorized = sorted(find_close_factor_pairs(self.assignments[i]), reverse=True)
                ch = self.get_subregion(deepcopy(i), self.assignments[i], internal_factorized)
                i.add_child(ch)

        lf = X_root.find_leaves()
        print('______________MA step below find leaves_______________________')
        print_tree(X_root)
        print('_____________________________________')
        self.assignments = {}
        self.agents_to_subregion = []
        internal_inactive_subregion=[]
        for l in lf:
            if l.status == 1:
                self.assignments.update({l : l.status})
                self.agents_to_subregion.append(l)
            elif l.status == 0:
                internal_inactive_subregion.append(l)
        # print('after find leaves: ',{l.input_space: l.status for l in lf})
        # assignments = {l: l.status for l in lf}
        # self.inactive_subregion_samples = []
        # self.inactive_subregion = []
        # print('after agents_to_subregion', self.inactive_subregion)
        
        print('assignments: ',[{str(k.input_space) : v} for k,v in self.assignments.items()])
        print('lf size: ', len(lf))
        
        self.agent = [Agent(tmp_gpr, self.x_train, self.y_train, self.agents_to_subregion[a].input_space) for a in range(self.horizon)]
        print('agents_to_subregion : ',self.agents_to_subregion)
        for i,sub in enumerate(self.agents_to_subregion):
            sub.update_agent(self.agent[i])
        return internal_inactive_subregion

    # Rollout for h steps 
    def get_h_step_reward(self,current_point):
        reward = 0
        # Temporary Gaussian prior 
        tmp_gpr = copy.deepcopy(self.gpr_model)
        xtr = copy.deepcopy(self.x_train)  
        # xtr = np.asarray([i.tolist() for i in xtr])
        # print('xtr: ', xtr.shape)
        print()
        # xtr = [i.tolist() for i in xtr]
        ytr = copy.deepcopy(self.y_train)
        h = self.horizon
        xt = current_point
        
        
        while(True):
            # print('xt : ', xt)
            np.random.seed(int(time.time()))
            mu, std = self._surrogate(tmp_gpr, xt)
            ri = -1 * np.inf
            f_xts = []
            for i in range(4):
                f_xt = np.random.normal(mu[i],std[i],1)
                f_xts.append(f_xt[0])
                ri = max(ri, self.reward(f_xt,ytr))
            # print('fxts : ',np.asarray(f_xts), xtr)
            reward += (ri)
            h -= 1
            if h <= 0 :
                break
            
            xtr = np.vstack((xtr,xt))
            ytr = np.hstack((ytr,np.asarray(f_xts)))
            # print('xtr, ytr shape ',xtr, ytr )
            tmp_gpr.fit(xtr,ytr)
            tmp_xt = []
            for a in self.agent:
                next_xt = self._opt_acquisition(self.y_train,tmp_gpr,a.region_support,self.rng)
                tmp_xt.append(next_xt)
            xt = np.asarray(tmp_xt)
        return reward
    
    # Reward is the difference between the observed min and the obs from the posterior
    def reward(self,f_xt,ytr):
        ymin = np.min(ytr)
        r = max(ymin - f_xt, 0)
        # print(' each regret : ', r)
        return r
    
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
    
    