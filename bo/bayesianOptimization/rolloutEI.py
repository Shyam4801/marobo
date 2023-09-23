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
from ..sampling import uniform_sampling, sample_from_discontinuous_region
from ..utils import compute_robustness
from ..behavior import Behavior
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial

from ..agent.agent import Agent
# from ..agent.constants import *
from ..utils.volume import compute_volume
from ..utils.timerf import logtime, LOGPATH
import yaml
from joblib import Parallel, delayed

with open('config.yml', 'r') as file:
    configs = yaml.safe_load(file)
  
def unwrap_self(arg, **kwarg):
    return RolloutEI.get_pt_reward(*arg, **kwarg)

class RolloutEI(InternalBO):
    def __init__(self) -> None:
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
            # print('dim curr queue_size', dim, curr.input_space, len(q))
            ch = self.split_region(curr,dim, dic[dim])
            print('ch',ch)
            curr.add_child(ch)
            q.extend(ch)
        # print([i.input_space for i in q])
        return q

    @logtime(LOGPATH)
    def sample(
        self,
        root,
        num_agents,
        # root,
        # agent,
        # agents_to_subregion,
        # assignments,
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
        self.mc_iters = configs['sampling']['mc_iters']
        print('below yml read mc_iters',self.mc_iters)
        self.root = root
        # self.agent = agent
        # self.assignments = assignments
        # self.agents_to_subregion = agents_to_subregion
        self.numthreads = int(mp.cpu_count()/2)
        self.tf = test_function
        self.x_train = x_train
        self.gpr_model = gpr_model
        self.horizon = horizon
        self.region_support = region_support
        self.rng = rng
        self.tf_dim = region_support.shape[0]
        self.y_train = y_train #copy.deepcopy(np.asarray(test_function.point_history,dtype=object)[:,-1])


        lf = self.root.find_leaves()
        print('______________below find leaves_______________________')
        print_tree(self.root)
        print('_____________________________________')
        assignments = {}
        agents_to_subregion = []
        internal_inactive_subregion=[]
        for l in lf:
            if l.status == 1:
                assignments.update({l : l.status})
                agents_to_subregion.append(l)
            elif l.status == 0:
                internal_inactive_subregion.append(l)
        
        print('assignments: ',[{str(k.input_space) : v} for k,v in assignments.items()])
        print('lf size: ', len(lf))
        
        xtr = deepcopy(x_train)
        ytr = deepcopy(y_train)
        agents = [Agent(gpr_model, xtr, ytr, agents_to_subregion[a].input_space) for a in range(num_agents)]
        print('agents_to_subregion : ',agents_to_subregion)
        for i,sub in enumerate(agents_to_subregion):
            print('i',i, len(agents), len(agents_to_subregion),len(assignments))
            sub.update_agent(agents[i])
        

        # internal_inactive_subregion_not_node = [i.input_space for i in internal_inactive_subregion]
        print('internal inactive: ', internal_inactive_subregion)
        # if internal_inactive_subregion != []:
        #     self.inactive_subregion = (internal_inactive_subregion_not_node)
        
        print('_______________________________ AGENTS AT WORK ___________________________________')  

        self.internal_inactive_subregion = internal_inactive_subregion
        print('self.internal_inactive_subregion: ',self.internal_inactive_subregion)
        self.agent = agents
        self.assignments = assignments
        self.agents_to_subregion = agents_to_subregion
        self.num_agents = num_agents

        for na in range(len(agents)):
            
            # Choosing the next point
            x_opt_from_all = []
            for i,a in enumerate(agents):
                # smp = uniform_sampling(5, a.region_support, tf_dim, rng)
                x_opt = self._opt_acquisition(y_train, gpr_model, a.region_support, rng) 
                # smp = np.vstack((smp, x_opt))
                x_opt_from_all.append(x_opt)
                # for i, preds in enumerate(pred_sample_x):
                self.agent[i].point_history.append(x_opt)
            # print('x_opt_from_all: ', np.hstack((x_opt_from_all)).reshape((6,4,2)))
            # Generate a sample dataset to rollout and find h step observations
            # exit()
            subx = np.hstack((x_opt_from_all)).reshape((num_agents,self.tf_dim))
            if np.size(self.internal_inactive_subregion) != 0:
                print('self.internal_inactive_subregion inside sampling: ',[i.input_space for i in self.internal_inactive_subregion])
                smp = sample_from_discontinuous_region(configs['sampling']['num_inactive'], self.internal_inactive_subregion, region_support, self.tf_dim, self.rng, volume=True ) #uniform_sampling(5, internal_inactive_subregion[0].input_space, self.tf_dim, self.rng)
                subx = np.vstack((subx, smp))
            # print('inside for loop subx ', subx)
            suby = -1 * self.get_exp_values(subx)
            print('############################################################################')
            print()
            print('suby: rollout for 2 horizons with 6 sampled points  :',suby, ' subx:', subx)
            print()
            print('############################################################################')
            minidx = np.argmin(suby)
            self.internal_inactive_subregion = self.reassign(self.root,self.internal_inactive_subregion, gpr_model,  na, minidx, subx)
            # self.internal_inactive_subregion.extend(internal_inactive_subregion)
        print('########################### End of MA #################################################')
        print('final subx : ',subx)
        print('############################################################################')
        for i, preds in enumerate(subx[:num_agents]):
                self.agent[i].point_history.append(preds)

        return subx[:num_agents], self.assignments, root, self.agent

    # Get expected value for each point after rolled out for h steps 
    @logtime(LOGPATH)
    def get_exp_values(self,eval_pts):
        # print('eval_pts shape: ',eval_pts.shape)
        # eval_pts = eval_pts.reshape((-1,2))
        num_pts = eval_pts.shape[0]
        exp_val = np.zeros(num_pts)
        # for i in range(num_pts):
        # print()
        exp_val = self._evaluate_at_point_list(eval_pts)
        return exp_val
    
    # def _evaluate_at_point_list(self, point_to_evaluate):
    #     self.point_current = point_to_evaluate
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
    
    def _evaluate_at_point_list(self, point_to_evaluate):
        results = []
        self.point_current = point_to_evaluate
        serial_mc_iters = [int(int(self.mc_iters)/self.numthreads)] * self.numthreads
        print('serial_mc_iters',serial_mc_iters)
        results = Parallel(n_jobs= -1, backend="loky")\
            (delayed(unwrap_self)(i) for i in zip([self]*len(serial_mc_iters), serial_mc_iters))
        print('_evaluate_at_point_list results',results)
        rewards = np.hstack((results))
        return np.sum(rewards)/self.numthreads

    # Perform Monte carlo itegration
    # @logtime(LOGPATH)
    # @numba.jit(nopython=True, parallel=True)
    def get_pt_reward(self,iters):
        reward = []
        for i in range(iters):
            rw = self.get_h_step_all(self.point_current)
            reward.append(rw)
            print('reward after each MC iter: ', reward)
            print(f'########################### Next MC iter {i} #################################################')
        reward = np.array(reward, dtype='object')
        print('end of MC iter: ',reward)
        return np.mean(reward, axis=0)
    
    @logtime(LOGPATH)
    def get_h_step_all(self,current_point):
        reward = 0
        # Temporary Gaussian prior 
        tmp_gpr = copy.deepcopy(self.gpr_model)
        xtr = copy.deepcopy(self.x_train)  
        # xtr = np.asarray([i.tolist() for i in xtr])
        # print('xtr: ', xtr.shape)
        print('current pt : ',current_point)
        # xtr = [i.tolist() for i in xtr]
        ytr = copy.deepcopy(self.y_train)
        h = self.horizon
        xt = current_point
        
        reward = []
        # print('empty reward: ',reward)
        while(True):
            # print('xt : ', xt)
            # np.random.seed(int(time.time()))
            mu, std = self._surrogate(tmp_gpr, xt)
            ri = -1 * np.inf
            f_xts = []
            rws = np.zeros((len(xt)))
            for i in range(len(xt)):
                f_xt = np.random.normal(mu[i],std[i],1)
                f_xts.append(f_xt[0])
                rws[i] = (self.reward(f_xt,ytr))
            # print('fxts : ',np.asarray(f_xts), xtr)
            reward.append(rws)
            # print('h ; rw :',h, reward)
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
            if np.size(self.internal_inactive_subregion) != 0:
                # print('len(self.internal_inactive_subregion)', len(self.internal_inactive_subregion), configs['sampling']['num_inactive'])
                smp = sample_from_discontinuous_region(configs['sampling']['num_inactive'], self.internal_inactive_subregion, self.region_support, self.tf_dim, self.rng, volume=True ) #uniform_sampling(5, internal_inactive_subregion[0].input_space, self.tf_dim, self.rng)
                xt = np.vstack((xt, smp))
        # print('rewards b4 summing up: ',np.array(reward, dtype='object'))
        reward = np.array(reward, dtype='object')
        return np.sum(reward,axis=0)
    
    
    @logtime(LOGPATH)
    def reassign(self, X_root,internal_inactive_subregion, tmp_gpr, h, minidx, subx):
        print('curent agent idx ', h)
        
        self.assignments[self.agents_to_subregion[h]] -= 1
        if minidx >= len(self.agents_to_subregion):
            # print('minidx : ',minidx, self.internal_inactive_subregion)
            inactive_reg_idx = self.check_inactive(subx[minidx], internal_inactive_subregion)
            self.assignments.update({internal_inactive_subregion[inactive_reg_idx] : 1})
            print('agent moved to this inactive region : ', internal_inactive_subregion[inactive_reg_idx].input_space)
            internal_inactive_subregion[inactive_reg_idx].status = 1
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
        
        self.agent = [Agent(tmp_gpr, self.x_train, self.y_train, self.agents_to_subregion[a].input_space) for a in range(len(self.agent))]
        print('agents_to_subregion : ',self.agents_to_subregion)
        for i,sub in enumerate(self.agents_to_subregion):
            sub.update_agent(self.agent[i])
        return internal_inactive_subregion

   
    
    # Reward is the difference between the observed min and the obs from the posterior
    def reward(self,f_xt,ytr):
        ymin = np.min(ytr)
        r = max(ymin - f_xt, 0)
        # print(' each regret : ', r)
        return r
    
    
    def check_inactive(self,sample_point, inactive_subregion):
        
        # print('inside check inactive')
        # for sample_point in sample_points:
        for i, subregion in enumerate(inactive_subregion):
        # print(subregion.input_space.shape)
            inside_inactive = True
            for d in range(len(subregion.input_space)):
                lb,ub = subregion.input_space[d]
                if sample_point[d] < lb or sample_point[d] > ub:
                    inside_inactive = False
            
            if inside_inactive: 
                # print('inside sub reg :', inactive_subregion[i].input_space)
                return i
                # fall_under.append(subregion)
                # print('fall under : ',fall_under, inactive_subregion[i].input_space)
                # # inactive_subregion[subregion] += 1
                # # assignments[agents_to_subregion[agent_idx]] -= 1
                # if assignments[agents_to_subregion[agent_idx]] == 0:
                #     agents_to_subregion[agent_idx].status = 0
                # inactive_subregion[i].status = 1
                # # agent.update_bounds(subregion)
                # not_inside_any = True