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
        self.gpr_model = gpr_model
        self.horizon = horizon
        self.region_support = region_support
        self.rng = rng
        tf_dim = region_support.shape[0]
        num_samples = 5
        self.y_train = y_train #copy.deepcopy(np.asarray(test_function.point_history,dtype=object)[:,-1])

        # Choosing the next point
        x_opt = self._opt_acquisition(y_train, gpr_model, region_support, rng) 
        # Generate a sample dataset to rollout and find h step observations
        subx = uniform_sampling(5, region_support, tf_dim, rng)
        subx = np.vstack([subx,x_opt]) #np.array([x_opt]) #np.vstack([subx,x_opt])
        print('subx : ',subx)
        # print('subx :',subx)
        # Rollout and get h step observations
        suby = -1 * self.get_exp_values(subx)
        print('############################################################################')
        print()
        print('suby: rollout for 2 horizons with 6 sampled points  :',suby, ' subx:', subx)
        print()
        print('############################################################################')

        # Build Gaussian prior model with the sample dataset 
        for sample in tqdm(range(num_samples)):
            model = GPR(InternalGPR())
            model.fit(subx, suby)
            # Get the next point using EI 
            pred_sub_sample_x = self._opt_acquisition(suby, model, region_support, rng)  # EI for the outer BO 
            print('pred_sub_sample_x:  for {sample}: ',pred_sub_sample_x)
            # Rollout and get h step obs for the above point 
            pred_sub_sample_y = -1 * self.get_exp_values( pred_sub_sample_x)    # this uses the inner MC to rollout EI and get the accumulated the reward or yt+1
            print('pred_sub_sample_y:  for {sample} rolled out for 2 horizons : ',pred_sub_sample_y)
            print()
            # Stack to update the Gaussian prior
            subx = np.vstack((subx, pred_sub_sample_x))
            suby = np.hstack((suby, pred_sub_sample_y))
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
        return subx[[min_idx],:], np.min(suby) #np.array(min_bo), min_bo_val

    # Get expected value for each point after rolled out for h steps 
    def get_exp_values(self,eval_pts):
        eval_pts = eval_pts.reshape((-1,2))
        num_pts = eval_pts.shape[0]
        exp_val = np.zeros(num_pts)
        for i in range(num_pts):
            exp_val[i] = self.get_pt_reward(eval_pts[[i],:])
        return exp_val
    
    def _evaluate_at_point_list(self, point_to_evaluate):
        self.point_current = point_to_evaluate
        partial_getptreard = partial(self.get_pt_reward)
        # getptreward = RolloutEI()
        # getptreward.point_current = point_to_evaluate
        # getptreward.gpr_model = self.gpr_model
        # getptreward.tf = self.tf
        
        if self.numthreads > 1:
            serial_mc_iters = [int(self.mc_iters/self.numthreads)] * self.numthreads
            pool = Pool(processes=self.numthreads)
            rewards = pool.map(unwrap_self_f, ([self]*5,serial_mc_iters))
            pool.close()
            pool.join()
        else:
            rewards = self.get_pt_reward(self.point_current, self.mc_iters)

        return np.sum(rewards)/self.numthreads

    # Perform Monte carlo itegration
    def get_pt_reward(self,current_point, iters=5):
        reward = 0
        # for i in range(iters):
        reward += self.get_h_step_reward(current_point)
        return reward #(reward/iters)
    
    # Rollout for h steps 
    def get_h_step_reward(self,current_point):
        reward = 0
        # Temporary Gaussian prior 
        tmp_gpr = copy.deepcopy(self.gpr_model)
        xtr = copy.deepcopy(self.agent.x_train)   #np.asarray(self.tf.point_history,dtype=object)[:,1]
        print()
        print('xtr: ', xtr)
        print()
        # xtr = [i.tolist() for i in xtr]
        ytr = copy.deepcopy(self.y_train)
        h = self.horizon
        xt = current_point

        while(True):
            np.random.seed(int(time.time()))
            mu, std = self._surrogate(tmp_gpr, xt.reshape(1, -1))
            f_xt = np.random.normal(mu,std,1)
            ri = self.reward(f_xt,ytr)
            reward += ri
            h -= 1
            if h <= 0 :
                break
            
            xt = self._opt_acquisition(self.y_train,tmp_gpr,self.region_support,self.rng)
            np.append(xtr,[xt])
            np.append(ytr,[f_xt])
        return reward
    
    # Reward is the difference between the observed min and the obs from the posterior
    def reward(self,f_xt,ytr):
        ymin = np.min(ytr)
        r = max(ymin - f_xt, 0)
        return r
    
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