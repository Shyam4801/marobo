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
            # self.subregions = self.root.find_leaves()
            # print('assignments: ',[{str(k.input_space) : v} for k,v in self.assignments.items()])
            # print('lf size: ', [i.input_space for i in self.agents_to_subregion])
            # Choosing the next point
            # x_opt_from_all = []
            # for i,a in enumerate(agents):
            #     # smp = uniform_sampling(5, a.region_support, tf_dim, rng)
            #     x_opt = self._opt_acquisition(y_train, gpr_model, a.region_support, rng) 
            #     # smp = np.vstack((smp, x_opt))
            #     x_opt_from_all.append(x_opt)
            #     # for i, preds in enumerate(pred_sample_x):
            #     self.agent[i].point_history.append(x_opt)
            # # print('x_opt_from_all: ', np.hstack((x_opt_from_all)).reshape((6,4,2)))
            # # Generate a sample dataset to rollout and find h step observations
            # # exit()
            # subx = np.hstack((x_opt_from_all)).reshape((num_agents,self.tf_dim))
            # if np.size(self.internal_inactive_subregion) != 0:
            #     # print('self.internal_inactive_subregion inside sampling: ',[i.input_space for i in self.internal_inactive_subregion])
            #     smp = sample_from_discontinuous_region(configs['sampling']['num_inactive'], self.internal_inactive_subregion, region_support, self.tf_dim, self.rng, volume=True ) #uniform_sampling(5, internal_inactive_subregion[0].input_space, self.tf_dim, self.rng)
            #     subx = np.vstack((subx, smp))
            # print('inside for loop subx ', subx)
            # suby = -1 * self.get_exp_values([self.agents_to_subregion, self.internal_inactive_subregion])
            self.get_exp_values(agents)
            # lf = self.root.find_leaves()
            # # print('______________below find leaves_______________________')
            # # print_tree(self.root)
            # # print('_____________________________________')
            # assignments = {}
            # agents_to_subregion = []
            # internal_inactive_subregion=[]
            # for l in lf:
            #     if l.getStatus(MAIN) == 1:
            #         assignments.update({l : l.getStatus(MAIN)})
            #         agents_to_subregion.append(l)
            #     elif l.getStatus(MAIN) == 0:
            #         internal_inactive_subregion.append(l)
            
            
            
            # xtr = deepcopy(x_train)
            # ytr = deepcopy(y_train)
            # agents = [Agent(gpr_model, xtr, ytr, agents_to_subregion[a].input_space) for a in range(num_agents)]
            # # print('agents_to_subregion : ',agents_to_subregion)
            # for i,sub in enumerate(agents_to_subregion):
            #     sub.update_agent(agents[i])

            # self.internal_inactive_subregion = internal_inactive_subregion
            # # print('self.internal_inactive_subregion: ',self.internal_inactive_subregion)
            # self.agent = agents
            # self.assignments = assignments
            # self.agents_to_subregion = agents_to_subregion
            # # print('############################################################################')
            # # print()
            # # print('suby: rollout for 2 horizons with 6 sampled points  :',suby, ' subx:', subx)
            # # print()
            # # print('############################################################################')
            # # for sr in self.subregions:
            # #     sr.child = []
            # subx = self.subregions
            # if len(self.internal_inactive_subregion) != 0:
            #     subx.extend(self.internal_inactive_subregion)
            # suby = [r.reward for r in subx]
            # suby = np.asarray(suby, dtype='object')
            # suby = np.hstack(suby)
            # print('main h step reward :', suby)
            # minidx = np.argmin(suby)
            minReg = find_min_leaf(self.root)
            print('region with min reward: find_min_leaf ',minReg.input_space)
            if minReg.agent != 0:
                if self.agent[na].region_support != minReg: 
                    internal_factorized = sorted(find_close_factor_pairs(2), reverse=True)
                    ch = self.get_subregion(deepcopy(minReg), 2, internal_factorized)
                    minReg.add_child(ch)
                    print('self.agent[na].region_support.agent :',self.agent[na].region_support.agent)
                    self.agent[na].region_support.agent = 0
                    self.agent[na].region_support.mainStatus = 0
                    self.agent[na].region_support = ch[0]
                    print('new reg of curr agent : should be eq to minreg: ',self.agent[na].region_support)
                    minReg.agent.region_support = ch[1]
                    print('modify reg of agent in the minreg which is = minreg, after split minReg.agent.region_support  : ',minReg.agent.region_support )
                    minReg.agent = 0
                    print('remove the agent in minreg :', minReg.agent)
                    minReg.mainStatus = 0
                    minReg.routine = None
                    self.agent[na].region_support.mainStatus = 1
                    minReg.agent.region_support.mainStatus = 1
                    self.agent[na].region_support.routine = MAIN
                    minReg.agent.region_support.routine = MAIN
                else:
                    continue
            else:
                self.agent[na].region_support.agent = 0
                self.agent[na].region_support.mainStatus = 0
                self.agent[na].region_support.routine = None
                self.agent[na].region_support = minReg
                minReg.agent = self.agent[na]
                minReg.mainStatus = 1
                minReg.routine = MAIN

            # self.internal_inactive_subregion, self.agent, self.assignments, self.agents_to_subregion = self.reassign(MAIN, self.root, self.assignments, self.agents_to_subregion, self.internal_inactive_subregion, gpr_model,  na, minidx, subx)
            # self.internal_inactive_subregion.extend(internal_inactive_subregion)
            print('<<<<<<<<<<<<<<<<<<<<<<<< Main routine tree <<<<<<<<<<<<<<<<<<<<<<<<')
            print_tree(self.root, MAIN)
            print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        print('########################### End of MA #################################################')
        # print('final subx : ',subx)
        print('############################################################################')
        for i, preds in enumerate(subx[:num_agents]):
                self.agent[i].point_history.append(preds)

        x_opt_from_all = []
        for i,a in enumerate(self.agent):
            # smp = uniform_sampling(5, a.region_support, tf_dim, rng)
            x_opt = self._opt_acquisition(y_train, gpr_model, a.region_support, rng) 
            # smp = np.vstack((smp, x_opt))
            x_opt_from_all.append(x_opt)
        subx = np.hstack((x_opt_from_all)).reshape((num_agents,self.tf_dim))

        return subx[:num_agents], self.assignments, root, self.agent

    # Get expected value for each point after rolled out for h steps 
    def get_exp_values(self, agents):
        self.subregions = self.root.find_leaves() 
        # print('eval_pts shape: ',eval_pts.shape)
        # eval_pts = eval_pts.reshape((-1,2))
        # num_pts = eval_pts.shape[0]
        # exp_val = np.zeros(num_pts)
        # for i in range(num_pts):
        # print()
        # exp_val = 
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
        # agents = [Agent(self.gpr_model, self.x_train, self.y_train, self.subregions[a].input_space) for a in range(self.num_agents)]
        # for i,sub in enumerate(self.subregions):
        #     sub.update_agent(agents[i])
        # activeReg = point_current.find_leaves()
        for i in range(iters):
            # rw = 
            print('agents in mc iter : ', [i.region_support for i in agents])
            self.get_h_step_with_part(agents)
            for sr in self.subregions:
                accumulate_rewards_and_update(sr)
                print()
                print('accumulated reward after h horizons  : ',sr.input_space,'act/inact: ',sr.mainStatus, sr.reward)
                print()
                sr.resetStatus()
                sr.child = []
            # reward.append(rw)
            # print('reward after each MC iter: ', [i.reward for i in self.agents_to_subregion],'inactive reg: ',[i.reward for i in self.internal_inactive_subregion])
            print(f'########################### End of MC iter {i} #################################################')
        reward = np.array(reward, dtype='object')
        # print('end of MC iter: ',reward)
        

        # lf = self.root.find_leaves()
        # # print('______________below find leaves_______________________')
        # # print_tree(self.root)
        # # print('_____________________________________')
        # assignments = {}
        # agents_to_subregion = []
        # internal_inactive_subregion=[]
        # for l in lf:
        #     if l.getStatus(MAIN) == 1:
        #         assignments.update({l : l.getStatus(MAIN)})
        #         agents_to_subregion.append(l)
        #     elif l.getStatus(MAIN) == 0:
        #         internal_inactive_subregion.append(l)
        
        
        
        # xtr = deepcopy(self.x_train)
        # ytr = deepcopy(self.y_train)
        # agents = [Agent(self.gpr_model, xtr, ytr, agents_to_subregion[a].input_space) for a in range(self.num_agents)]
        # # print('agents_to_subregion : ',agents_to_subregion)
        # for i,sub in enumerate(agents_to_subregion):
        #     sub.update_agent(agents[i])

        # self.internal_inactive_subregion = internal_inactive_subregion
        # # print('self.internal_inactive_subregion: ',self.internal_inactive_subregion)
        # self.agent = agents
        # self.assignments = assignments
        # self.agents_to_subregion = agents_to_subregion
        for i in range(len(self.subregions)):
            self.subregions[i].reward = self.subregions[i].reward / iters
        
        # for i in range(len(self.agents_to_subregion)):
        #     self.agents_to_subregion[i].reward = self.agents_to_subregion[i].reward / iters

        # for i in range(len(self.internal_inactive_subregion)):
        #     self.internal_inactive_subregion[i].reward = self.internal_inactive_subregion[i].reward / iters
        # return np.mean(reward, axis=0)
        print()
        print(">>>>>>>>>>>>>>>>>>>>>>>> Tree after avg reward update >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print_tree(self.root, ROLLOUT)
        (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # print('Avg reward after MC iter: ', [i.reward for i in self.agents_to_subregion],'inactive reg: ',[i.reward for i in self.internal_inactive_subregion])
            
    
    def get_h_step_with_part(self, agents):
        reward = 0
        # Temporary Gaussian prior 
        tmp_gpr = copy.deepcopy(self.gpr_model)
        # xtr = copy.deepcopy(self.x_train)  
        # xtr = np.asarray([i.tolist() for i in xtr])
        # print('xtr: ', xtr.shape)
        # print('current pt : ',current_point)
        # xtr = [i.tolist() for i in xtr]
        ytr = copy.deepcopy(self.y_train)
        h = self.horizon
        # xt = current_point
        # act, inact = current_point[0], current_point[1]
        # rl_inactive_subregion = self.internal_inactive_subregion #copy.deepcopy(self.internal_inactive_subregion)
        rl_root = self.root #copy.deepcopy(self.root)
        # assignments = self.assignments #copy.deepcopy(self.assignments)
        # agents_to_subregion = self.agents_to_subregion #list(assignments.keys())
        agents = agents #self.agent #[Agent(tmp_gpr, self.x_train, ytr, agents_to_subregion[a].input_space) for a in range(self.num_agents)]

        # inact_dict = {o : d for o,d in zip(self.internal_inactive_subregion,rl_inactive_subregion)}
        # act_dict = {o : d for o,d in zip(self.agents_to_subregion, agents_to_subregion)}
        xt = rl_root.find_leaves()  #self.subregions
        # for reg in agents_to_subregion:
        #     next_xt = self._opt_acquisition(self.y_train,tmp_gpr,reg.input_space,self.rng)
        #     xt.append(next_xt)
        # xt = np.asarray(xt)
        # if np.size(rl_inactive_subregion) != 0:
        #         # print('len(self.internal_inactive_subregion)', len(self.internal_inactive_subregion), configs['sampling']['num_inactive'])
        #         smp = sample_from_discontinuous_region(configs['sampling']['num_inactive'], rl_inactive_subregion, self.region_support, self.tf_dim, self.rng, volume=True ) #uniform_sampling(5, internal_inactive_subregion[0].input_space, self.tf_dim, self.rng)
        #         xt = np.vstack((xt, smp))
        # print('inside rollout :', assignments, agents_to_subregion)
        # print('before deep copy : ', self.assignments, self.agents_to_subregion)
        print()
        print('b4 while agents_to_subregion',[i.input_space for i in xt])
        print()
        # print([i.input_space for i in agents_to_subregion],'rl_inactive_subregion',[i.input_space for i in self.internal_inactive_subregion],[i.input_space for i in rl_inactive_subregion])
        # print(rl_root.find_leaves())
        # print()

        reward = []
        # print('empty reward: ',reward)
        while(True):
            print(f">>>>>>>>>>>>>>>>>>>>>>>> horizon: {h} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print_tree(rl_root, ROLLOUT)
            print_tree(rl_root, MAIN)
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            # np.random.seed(int(time.time()))
            # print('inside while agents_to_subregion',[i.input_space for i in agents_to_subregion],'rl_inactive_subregion',[i.input_space for i in rl_inactive_subregion])
            
            for reg in xt:
                if reg.rolloutStatus == 1:
                    next_xt = self._opt_acquisition(self.y_train,tmp_gpr,reg.input_space,self.rng)
                    next_xt = np.asarray([next_xt])
                    mu, std = self._surrogate(tmp_gpr, next_xt)
                    f_xt = np.random.normal(mu,std,1)
                    reg.reward = -1 * self.reward(f_xt,ytr)
                
                else:
                    # if np.size(rl_inactive_subregion) != 0:
                    #     for reg in rl_inactive_subregion:
                        # print('len(self.internal_inactive_subregion)', len(self.internal_inactive_subregion), configs['sampling']['num_inactive'])
                    smp = sample_from_discontinuous_region(configs['sampling']['num_inactive'], [reg], self.region_support, self.tf_dim, self.rng, volume=True ) #uniform_sampling(5, internal_inactive_subregion[0].input_space, self.tf_dim, self.rng)
                    mu, std = self._surrogate(tmp_gpr, smp)
                    for i in range(len(smp)):
                        f_xt = np.random.normal(mu[i],std[i],1)
                        smp_reward = self.reward(f_xt,ytr)
                        if reg.reward > -1*smp_reward:
                            reg.reward = -1 * smp_reward
                            reg.addSample(smp[i])
                    
            # ri = -1 * np.inf
            # f_xts = []
            # rws = np.zeros((len(xt)))
            # for i in range(len(xt)):
            #     f_xt = np.random.normal(mu[i],std[i],1)
            #     f_xts.append(f_xt[0])
            #     rws[i] = (self.reward(f_xt,ytr))
            # # print('fxts : ',np.asarray(f_xts), xtr)
            # reward.append(rws)
            # print('h ; rw :',h, reward)
            
            
            # xtr = np.vstack((xtr,xt[:self.num_agents]))
            # ytr = np.hstack((ytr,np.asarray(f_xts[:self.num_agents])))
            # print('xtr, ytr shape ',xtr.shape, ytr )
            # tmp_gpr.fit(xtr,ytr)
            
            
            # xt = agents_to_subregion
            # print('xt b4 , ',xt)
            # if len(rl_inactive_subregion) != 0:
            #     xt.extend(rl_inactive_subregion)
            # print()
            # # print('xt , ',xt,'agents_to_subregion',agents_to_subregion)
            # print()
            # rsuby = [r.reward for r in xt]
            # # print('rsuby b4 , ',rsuby)
            # rsuby = np.asarray(rsuby, dtype='object')
            # rsuby = np.hstack(rsuby)
            # print()
            # print('rsuby , ',rsuby, 'agents_to_subregion',agents_to_subregion)
            # minidx = np.argmin(rsuby)

            # rl_inactive_subregion, agent, assignments, agents_to_subregion = self.reassign(ROLLOUT, rl_root, assignments, agents_to_subregion, rl_inactive_subregion, tmp_gpr,  self.num_agents-h, minidx, xt)
            print()
            print('Rollout reassignments in the tree directly ')
            print()
            minRegval = find_min_leaf(rl_root)
            minReg = minRegval[1]
            val = minRegval[0]
            print('curr agents[self.num_agents-h].region_support: ',agents[self.num_agents-h].region_support.input_space, agents[self.num_agents-h].region_support)
            print('region with min reward: find_min_leaf ',minReg.input_space, val, minReg)
            print('not agents[self.num_agents-h].region_support is minReg: ',not agents[self.num_agents-h].region_support is minReg)
            print('agents[self.num_agents-h].region_support != minReg', agents[self.num_agents-h].region_support != minReg)
            if minReg.agent != 0:
                print('agents[self.num_agents-h].region_support: ',agents[self.num_agents-h].region_support.input_space)
                if not agents[self.num_agents-h].region_support is minReg: 
                    internal_factorized = sorted(find_close_factor_pairs(2), reverse=True)
                    ch = self.get_subregion(deepcopy(minReg), 2, internal_factorized)
                    minReg.add_child(ch)
                    print('agent[na].region_support.agent :',agents[self.num_agents-h].region_support.agent)
                    agents[self.num_agents-h].region_support.agent = 0
                    agents[self.num_agents-h].region_support.rolloutStatus = 0
                    agents[self.num_agents-h].region_support = ch[0]
                    print('new reg of curr agent : should be eq to minreg: ',agents[self.num_agents-h].region_support.input_space)
                    minReg.agent.region_support = ch[1]
                    minReg.agent.region_support.routine = ROLLOUT
                    minReg.agent.region_support.rolloutStatus = 1
                    print('modify reg of agent in the minreg which is = minreg, after split minReg.agent.region_support  : ',minReg.agent.region_support.input_space )
                    minReg.agent = 0
                    print('remove the agent in minreg :', minReg.agent)
                    minReg.rolloutStatus = 0
                    minReg.routine = None
                    agents[self.num_agents-h].region_support.rolloutStatus = 1
                    # minReg.agent.region_support.rolloutStatus = 1
                    agents[self.num_agents-h].region_support.routine = ROLLOUT
                    # minReg.agent.region_support.routine = ROLLOUT
                else:
                    print('inside else')
                    # continue
            else:
                agents[self.num_agents-h].region_support.agent = 0
                agents[self.num_agents-h].region_support.rolloutStatus = 0
                agents[self.num_agents-h].region_support.routine = None
                agents[self.num_agents-h].region_support = minReg
                minReg.agent = agents[self.num_agents-h]
                minReg.rolloutStatus = 1
                minReg.routine = ROLLOUT
            
            xt = rl_root.find_leaves() 
            h -= 1
            if h <= 0 :
                break
        
        # for reg in self.subregions:
            
            

        # for act in act_dict.keys():
        #     accumulate_rewards_and_update(act_dict[act])
        #     # act.reward += act_dict[act].reward
        #     # print('act reward copied to original: ', act.reward , act_dict[act].reward)

        # # print([i.input_space for i in rl_inactive_subregion])
        # if len(inact_dict.keys()) != 0:
        #     for inact in inact_dict.keys():
        #         accumulate_rewards_and_update(inact_dict[inact])
                # inact.reward += inact_dict[inact].reward
                # print('inact reward copied to original: ', inact.reward , inact_dict[inact].reward)

        # print()
        # print('act_dict, inact_dict : ',act_dict, inact_dict)
        # print()
        # print('all rewards : ', np.array(reward, dtype='object'))    
        # print('rewards b4 summing up: ',np.array(reward[-1], dtype='object'))
        # reward = np.array(reward[-1], dtype='object')
        # return rl_inactive_subregion, agent, assignments, agents_to_subregion #reward #np.sum(reward,axis=0)
    
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
