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
from ..agent.partition import find_close_factor_pairs, Node, print_tree, accumulate_rewards_and_update, find_min_leaf
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
from ..agent.constants import ROLLOUT, MAIN

def unwrap_self(arg, **kwarg):
    return RolloutEI.get_pt_reward(*arg, **kwarg)

with open('config.yml', 'r') as file:
    configs = yaml.safe_load(file)
  
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

    # @logtime(LOGPATH)
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
        self.num_agents = num_agents

        lf = self.root.find_leaves()
        # print('______________below find leaves_______________________')
        # print_tree(self.root)
        # print('_____________________________________')
        assignments = {}
        agents_to_subregion = []
        internal_inactive_subregion=[]
        for l in lf:
            l.setRoutine(MAIN)
            if l.getStatus(MAIN) == 1:
                assignments.update({l : l.getStatus(MAIN)})
                agents_to_subregion.append(l)
            elif l.getStatus(MAIN) == 0:
                internal_inactive_subregion.append(l)
        
        # print('assignments: ',[{str(k.input_space) : v} for k,v in assignments.items()])
        # print('lf size: ', len(lf))
        
        xtr = deepcopy(x_train)
        ytr = deepcopy(y_train)
        agents = [Agent(gpr_model, xtr, ytr, agents_to_subregion[a]) for a in range(num_agents)]
        # print('agents_to_subregion : ',agents_to_subregion)
        for i,sub in enumerate(agents_to_subregion):
            sub.update_agent(agents[i])
        

        # internal_inactive_subregion_not_node = [i.input_space for i in internal_inactive_subregion]
        # print('internal inactive: ', internal_inactive_subregion)
        # if internal_inactive_subregion != []:
        #     self.inactive_subregion = (internal_inactive_subregion_not_node)
        
        print('_______________________________ AGENTS AT WORK ___________________________________')  

        self.internal_inactive_subregion = internal_inactive_subregion
        # print('self.internal_inactive_subregion: ',self.internal_inactive_subregion)
        self.agent = agents
        self.assignments = assignments
        self.agents_to_subregion = agents_to_subregion
        self.num_agents = num_agents

        print(' initial agents obj: ',self.agent)
        for na in range(len(agents)):
            self.get_exp_values(agents)
            minRegval = find_min_leaf(self.root)
            minReg = minRegval[1]
            val = minRegval[0]
            print('region with min reward: find_min_leaf ',minReg.input_space)

            if agents[na].region_support != minReg:
                if minReg.mainStatus == 1:
                    internal_factorized = sorted(find_close_factor_pairs(2), reverse=True)
                    ch = self.get_subregion(deepcopy(minReg), 2, internal_factorized)
                    minReg.add_child(ch)
                    minReg.updateStatus(0, MAIN)
                    
                else:
                    minReg.updateStatus(1, MAIN)

                agents[na].region_support.updateStatus(0, MAIN)

                newAgents = []
                newLeaves = self.root.find_leaves()
                for reg in newLeaves:
                    reg.setRoutine(MAIN)
                    if reg.mainStatus == 1:
                        newAgents.append(Agent(gpr_model, xtr, ytr, reg))
                    reg.resetStatus()
                    # if len(agents[i].child) != 0:
                    #     newAgents.extend(agents[i].child)

                agents = newAgents
                assert len(agents) == num_agents
            print('<<<<<<<<<<<<<<<<<<<<<<<< Main routine tree <<<<<<<<<<<<<<<<<<<<<<<<')
            print_tree(self.root, MAIN)
            print_tree(self.root, ROLLOUT)
            print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        print('########################### End of MA #################################################')
        # print('final subx : ',subx)
        print('############################################################################')
        # for i, preds in enumerate(subx[:num_agents]):
        #         self.agent[i].point_history.append(preds)

        x_opt_from_all = []
        print('final agent regions ', [i.region_support.input_space for i in agents])
        for i,a in enumerate(agents):
            # smp = uniform_sampling(5, a.region_support, tf_dim, rng)
            x_opt = self._opt_acquisition(y_train, gpr_model, a.region_support.input_space, rng) 
            # smp = np.vstack((smp, x_opt))
            x_opt_from_all.append(x_opt)
        subx = np.hstack((x_opt_from_all)).reshape((num_agents,self.tf_dim))

        for i, preds in enumerate(subx[:num_agents]):
                self.agent[i].point_history.append(preds)
        return subx[:num_agents], self.assignments, root, self.agent

    # Get expected value for each point after rolled out for h steps 
    def get_exp_values(self, agents):
        self.get_pt_reward(2, agents)
        # return exp_val
    
    def _evaluate_at_point_list(self, point_to_evaluate):
        results = []
        self.point_current = point_to_evaluate
        serial_mc_iters = [int(int(self.mc_iters)/self.numthreads)] * self.numthreads
        print('serial_mc_iters using job lib',serial_mc_iters)
        results = Parallel(n_jobs= -1, backend="loky")\
            (delayed(unwrap_self)(i) for i in zip([self]*len(serial_mc_iters), serial_mc_iters))
        # print('_evaluate_at_point_list results',results)
        rewards = np.hstack((results))
        return np.sum(rewards)/self.numthreads

    # Perform Monte carlo itegration
    # @logtime(LOGPATH)
    # @numba.jit(nopython=True, parallel=True)
    def get_pt_reward(self,iters, agents):
        reward = []
        lf = self.root.find_leaves()
        for i in range(iters):
            # rw = 
            print('agents in mc iter : ', [i.region_support.input_space for i in agents])
            self.get_h_step_with_part(agents)
            assert(len(agents) == self.num_agents)
            
            for sima in lf:
                print('sima region and status, ',sima.input_space, sima.mainStatus, sima.rolloutStatus)
                assert sima.routine == MAIN
                accumulate_rewards_and_update(sima)
                # sima.region_support.resetStatus()
                sima.child = []
                sima.resetStatus()

            print(f'########################### End of MC iter {i} #################################################')
        reward = np.array(reward, dtype='object')
        print(">>>>>>>>>>>>>>>>>>>>>>>> Tree just b4 avg reward update >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print_tree(self.root, ROLLOUT)
        (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        for i in range(len(lf)):
            lf[i].reward = lf[i].reward / iters
        
        print()
        print(">>>>>>>>>>>>>>>>>>>>>>>> Tree after avg reward update >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print_tree(self.root, ROLLOUT)
        (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
           
    
    def get_h_step_with_part(self, agents):
        reward = 0
        # Temporary Gaussian prior 
        tmp_gpr = copy.deepcopy(self.gpr_model)
        ytr = copy.deepcopy(self.y_train)
        h = self.horizon
        rl_root = self.root #copy.deepcopy(self.root)
        xt = rl_root.find_leaves()  #self.subregions
        print()
        print('b4 while agents_to_subregion',[(i.input_space,i.rolloutStatus) for i in xt])
        print()

        reward = []
        # print('empty reward: ',reward)
        while(True):
            print(f">>>>>>>>>>>>>>>>>>>>>>>> horizon: {h} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print_tree(rl_root, ROLLOUT)
            print_tree(rl_root, MAIN)
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            for reg in xt:
                if reg.rolloutStatus == 1:
                    next_xt = self._opt_acquisition(self.y_train,tmp_gpr,reg.input_space,self.rng)
                    next_xt = np.asarray([next_xt])
                    mu, std = self._surrogate(tmp_gpr, next_xt)
                    f_xt = np.random.normal(mu,std,1)
                    reg.reward = -1 * self.reward(f_xt,ytr)
                
                else:
                    smp = sample_from_discontinuous_region(configs['sampling']['num_inactive'], [reg], self.region_support, self.tf_dim, self.rng, volume=True ) #uniform_sampling(5, internal_inactive_subregion[0].input_space, self.tf_dim, self.rng)
                    mu, std = self._surrogate(tmp_gpr, smp)
                    for i in range(len(smp)):
                        f_xt = np.random.normal(mu[i],std[i],1)
                        smp_reward = self.reward(f_xt,ytr)
                        if reg.reward > -1*smp_reward:
                            reg.reward = -1 * smp_reward
                            reg.addSample(smp[i])
                    
            print()
            print('Rollout reassignments in the tree directly ')
            print()
            minRegval = find_min_leaf(rl_root)
            minReg = minRegval[1]
            val = minRegval[0]
            print('curr agents[self.num_agents-h].region_support: ',agents[self.num_agents-h].region_support.input_space, agents[self.num_agents-h].region_support)
            print('region with min reward: find_min_leaf ',minReg.input_space, val, minReg)
            
            na = self.num_agents - h
            if agents[na].region_support != minReg:
                if minReg.rolloutStatus == 1:
                    internal_factorized = sorted(find_close_factor_pairs(2), reverse=True)
                    ch = self.get_subregion(deepcopy(minReg), 2, internal_factorized)
                    minReg.add_child(ch)
                    minReg.updateStatus(0, ROLLOUT)
                    
                else:
                    minReg.updateStatus(1, ROLLOUT)

                agents[na].region_support.updateStatus(0, ROLLOUT)
                # agents[na].region_support.setRoutine(None)


                newAgents = []
                newLeaves = rl_root.find_leaves()
                for reg in newLeaves:
                    if reg.rolloutStatus == 1:
                        newAgents.append(Agent(tmp_gpr, self.x_train, ytr, reg))
                    # if len(agents[i].region_support.child) != 0:
                    #     newAgents.extend(agents[i].region_support.child)

                agents = newAgents

            
            xt = rl_root.find_leaves() 
            h -= 1
            if h <= 0 :
                break
        
    
    # @logtime(LOGPATH)
    def get_h_step_all(self,current_point):
        reward = 0
        # Temporary Gaussian prior 
        tmp_gpr = copy.deepcopy(self.gpr_model)
        xtr = copy.deepcopy(self.x_train)  
        # xtr = np.asarray([i.tolist() for i in xtr])
        # print('xtr: ', xtr.shape)
        # print('current pt : ',current_point)
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
    
    
    # @logtime(LOGPATH)
    def reassign(self, routine, X_root, assignments, agents_to_subregion, internal_inactive_subregion, tmp_gpr, h, minidx, subx):
        print('curent agent idx ', h, minidx, agents_to_subregion)
        
        assignments[agents_to_subregion[h]] -= 1
        if minidx >= self.num_agents:
            print('minidx : ',minidx, internal_inactive_subregion, subx)
            # inactive_reg_idx = self.check_inactive(subx[minidx], internal_inactive_subregion)
            assignments.update({subx[minidx] : 1})
            # print('agent moved to this inactive region : ', internal_inactive_subregion[inactive_reg_idx].input_space)
            subx[minidx].updateStatus(1, routine)
        else:
            assignments[agents_to_subregion[minidx]] += 1

        for i in assignments.keys():
            if assignments[i] == 1:
                i.updateStatus(1, routine)
            if assignments[i] == 0:
                i.updateStatus(0, routine)
            if assignments[i] > 1:
                # ch = self.split_space(i.input_space, assignments[i], tf_dim)
                internal_factorized = sorted(find_close_factor_pairs(assignments[i]), reverse=True)
                ch = self.get_subregion(deepcopy(i), assignments[i], internal_factorized)
                i.add_child(ch)

        lf = X_root.find_leaves()
        # print('______________MA step below find leaves_______________________')
        # print_tree(X_root)
        # print('_____________________________________')
        assignments = {}
        agents_to_subregion = []
        internal_inactive_subregion=[]
        for l in lf:
            if l.getStatus(routine) == 1:
                assignments.update({l : l.getStatus(routine)})
                agents_to_subregion.append(l)
            elif l.getStatus(routine) == 0:
                internal_inactive_subregion.append(l)
        # print('after find leaves: ',{l.input_space: l.status for l in lf})
        # assignments = {l: l.status for l in lf}
        # self.inactive_subregion_samples = []
        # self.inactive_subregion = []
        # print('after agents_to_subregion', self.inactive_subregion)
        
        # print('assignments: ',[{str(k.input_space) : v} for k,v in self.assignments.items()])
        # print('lf size: ', len(lf))
        
        agent = [Agent(tmp_gpr, self.x_train, self.y_train, agents_to_subregion[a].input_space) for a in range(self.num_agents)]
        print('agents_to_subregion inside reassign : ',[i.input_space for i in agents_to_subregion], 'agent: ', agent)
        for i,sub in enumerate(agents_to_subregion):
            sub.update_agent(agent[i])
        return internal_inactive_subregion, agent, assignments, agents_to_subregion

   
    
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
