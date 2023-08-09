from typing import Callable, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import norm
from tqdm import tqdm
import copy
import time 


from .internalBO import InternalBO
# from .rolloutIsolated import RolloutBO
from ..gprInterface import GPR
from bo.gprInterface import InternalGPR
from ..sampling import uniform_sampling
from ..utils import compute_robustness
from ..behavior import Behavior
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial

def unwrap_self_f(arg, **kwarg):
    return RolloutEI.get_pt_reward(*arg, **kwarg)

class RolloutEI(InternalBO):
    def __init__(self) -> None:
        pass

    def sample(
        self,
        agent,
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
        self.agent = agent
        self.numthreads = int(mp.cpu_count()/2)
        self.tf = test_function
        self.x_train = x_train
        self.gpr_model = gpr_model
        self.horizon = horizon
        self.region_support = region_support
        self.rng = rng
        tf_dim = region_support.shape[0]
        num_samples = 5
        self.y_train = y_train #copy.deepcopy(np.asarray(test_function.point_history,dtype=object)[:,-1])

        # Choosing the next point
        x_opt_from_all = []
        for a in agent:
            smp = uniform_sampling(5, a.region_support, tf_dim, rng)
            x_opt = self._opt_acquisition(y_train, gpr_model, a.region_support, rng) 
            smp = np.vstack((smp, x_opt))
            x_opt_from_all.append(smp)
        # print('x_opt_from_all: ', np.hstack((x_opt_from_all)).reshape((6,4,2)))
        # Generate a sample dataset to rollout and find h step observations
        # exit()
        subx = np.hstack((x_opt_from_all)).reshape((6,4,tf_dim)) #np.asarray(x_opt_from_all)
        # subx = np.vstack([subx,np.array([x_opt_from_all])]) #np.array([x_opt]) #np.vstack([subx,x_opt])
        # print('subx : ',subx)
        # print('subx :',subx)
        # Rollout and get h step observations
        suby = -1 * self.get_exp_values(subx)
        print('############################################################################')
        print()
        print('suby: rollout for 2 horizons with 6 sampled points  :',suby, ' subx:', subx)
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
        min_idx = np.argmin(suby)

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
        return subx[[min_idx],:][0], np.min(suby) #np.array(min_bo), min_bo_val #[[min_idx],:]

    # Get expected value for each point after rolled out for h steps 
    def get_exp_values(self,eval_pts):
        # print('eval_pts shape: ',eval_pts.shape)
        # eval_pts = eval_pts.reshape((-1,2))
        num_pts = eval_pts.shape[0]
        exp_val = np.zeros(num_pts)
        for i in range(num_pts):
        # print()
            exp_val[i] = self._evaluate_at_point_list(eval_pts[i])
        return exp_val
    
    def _evaluate_at_point_list(self, point_to_evaluate):
        self.point_current = point_to_evaluate
        my_list = [0]*int(self.numthreads/2) + [1]*int(self.numthreads/2)
        th = np.random.shuffle(my_list)
        # print('th ---------',th)
        if self.numthreads > 1:
            serial_mc_iters = [int(int(self.numthreads)/self.numthreads)] * 5 #self.numthreads
            print('serial_mc_iters',serial_mc_iters, self.numthreads)
            pool = Pool(processes=self.numthreads)
            rewards = pool.map(self.get_pt_reward, serial_mc_iters)
            pool.close()
            pool.join()
        else:
            rewards = self.get_pt_reward()
        rewards = np.hstack((rewards))
        # print('rewards: ', rewards)
        return np.sum(rewards)/self.numthreads

    # Perform Monte carlo itegration
    def get_pt_reward(self, iters=5):
        reward = 0
        for i in range(iters):
            reward += self.get_h_step_reward(self.point_current)
            print('reward after each MC iter: ', self.point_current, reward)
        return (reward/iters)
    
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
    
    def _top5_acquisition(self, y_train: NDArray, gpr_model: Callable, region_support: NDArray, rng) -> NDArray:
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
        # min_bo = np.array(random_samples[np.argmin(min_bo_val), :])
        # print('min bo b4 BFGS :',min_bo, min_bo_val[-3],random_samples[-3])
        # min_bo_val = np.min(min_bo_val)
        print('before sorting : ',np.min(min_bo_val), np.array(random_samples[np.argmin(min_bo_val), :]))

        min_bo_val, random_samples = zip(*sorted(zip(min_bo_val, random_samples), key=lambda x: x[0]))
        
        # print('after sorting :', np.asarray(random_samples[:5]), min_bo_val[:5])
        min_bo = np.asarray(random_samples[:5])


        # min_bo = np.array(random_samples[np.argmin(min_bo_val), :])
        # # print('min bo b4 BFGS :',min_bo, min_bo_val[-3],random_samples[-3])
        # min_bo_val = np.min(min_bo_val)
        # print('lower_bound_theta: ',list(zip(lower_bound_theta, upper_bound_theta)))
        # for _ in range(9):
        #     new_params = minimize(
        #         fun,
        #         bounds=list(zip(lower_bound_theta, upper_bound_theta)),
        #         x0=min_bo,
        #     )

        #     if not new_params.success:
        #         continue

        #     if min_bo is None or fun(new_params.x) < min_bo_val:
        #         min_bo = new_params.x
        #         min_bo_val = fun(min_bo)
        # new_params = minimize(
        #     fun, bounds=list(zip(lower_bound_theta, upper_bound_theta)), x0=min_bo
        # )
        # min_bo = new_params.x
       # get the num of agents in each subregion 
        return min_bo#, min_bo_val[:5]